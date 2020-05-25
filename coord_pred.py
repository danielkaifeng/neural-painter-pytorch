import argparse
import sys
import torch
import torch as T
import torch.nn.functional as F
from torch.optim import Adam
from torch import nn, autograd, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader

from model.discriminator import discriminator_2d 
from model.generator import linear_net, Encoder
from model.axis_net import Axis, Pos
from utils import *

import os
import numpy as np
import cv2
import random
from dataset import HR_IMG

path = "/data/ai-datasets/202-DIV2K/High-Resolution/DIV2K_train_HR"
eps = 1e-8

NONLIN_TABLE = dict(
    relu=F.relu,
    tanh=T.tanh,
    abs_tanh=lambda x: abs(T.tanh(x)),
    sigmoid=T.sigmoid,
    softplus=F.softplus,
    sin=T.sin,
    cos=T.cos,
    sgn=T.sign,
    #sort=lambda x: T.sort(x, dim=1),
    abs=abs,
    log_abs=lambda x: T.log(abs(x) + eps),  # this is awesome
    log_abs_p1=lambda x: T.log(abs(x) + 1),
    log_relu=lambda x: T.log(F.relu(x) + eps),
    log_square=lambda x: T.log(x**2 + eps),  # just a scalar
    softmax=lambda x: F.softmax(x, dim=1),
    logsoftmax=lambda x: T.log(F.softmax(x, dim=1)),
    identity=lambda x: x,
    square=lambda x: x**2
)

NONLIN_TABLE_small = dict(
    relu=F.relu,
    tanh=T.tanh,
    sigmoid=T.sigmoid,
    softmax=lambda x: F.softmax(x, dim=1),
    square=lambda x: x**2
)




def get_nonlin(name):
    if name == 'random_every_time':
        def nonlin(x):
            return NONLIN_TABLE[random.choice(list(NONLIN_TABLE))](x)
        return nonlin

    if name == 'random_once':
        return NONLIN_TABLE[random.choice(list(NONLIN_TABLE))]

    return NONLIN_TABLE[name]


def sanitize_str(x):
    x = x.replace('/', '-')
    i = 0
    while i < len(x) and x[i] == '-':
        i += 1
    return x[i:]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=42)
    parser.add_argument('--image_size', help='wxh', default='100x100')
    parser.add_argument('--hidden_size', default=100, type=int)
    parser.add_argument('--nr_hidden', default=3, type=int)
    parser.add_argument('--recurrent', action='store_true')
    parser.add_argument('--coord_bias', action='store_true')
    parser.add_argument('--nr_channel', default=1, type=int, choices={1, 3})
    parser.add_argument('--nonlin', default='tanh',
                        choices=list(NONLIN_TABLE) + [
                            'random_once', 'random_every_time'])

    parser.add_argument('--d_lr', default=0.0001, type=float)
    parser.add_argument('--g_lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--step', default=90000, type=int)


    parser.add_argument('--output_nonlin', default='identity',
                        choices=list(NONLIN_TABLE))
    parser.add_argument('--batch_norm', action='store_true')
    parser.add_argument('--use_bias', action='store_true',
                        help='use bias in hidden layer')
    parser.add_argument('--batch_norm_position',
                        choices={'before_nonlin', 'after_nonlin'},
                        default='before_nonlin')
    parser.add_argument('--output', '-o', help='output image path')
    parser.add_argument('--auto_name', action='store_true',
                        help='append generation parameters'
                        ' to the name of the output')

    return parser.parse_args()

args = get_args()
w, h = map(int, args.image_size.split('x'))

def get_coord_rgb(img, coord):
    rgb_list = []
    for pos in coord:
        x, y = pos
        x = int(h * x)
        y = int(w * y)
        rgb_list.append(img[:,x,y])
    return rgb_list

def get_coord(w, h):
    coords = np.array(np.meshgrid(np.arange(h), np.arange(w))[::-1],
                  dtype='float32').reshape((2, -1)).swapaxes(0, 1) / [w, h]
    return coords

def gg_tensor(gen, w, h):
    coords = np.array(np.meshgrid(np.arange(h), np.arange(w))[::-1],
                  dtype='float32').reshape((2, -1)).swapaxes(0, 1) / [w, h]
    #coords = np.random.normal(1., 6, size=[w*h, 2])
    if args.coord_bias:
        coords = np.concatenate((coords, np.ones((coords.shape[0], 1))), axis=1)

    z = coords
    #z = np.random.uniform(-1., 1., size=[w*h, 3])

    z = z.astype('float32')
    z = Variable(T.from_numpy(z).cuda(), requires_grad=True)

    fake_data = gen(z) 
    img_data = fake_data.view(w, h, 3)
    fake_data = img_data.permute(2, 0, 1)

    return fake_data, img_data

def draw(w, h, img_tensor):
    img = img_tensor.data.cpu().numpy()
    img *= 255
    img = np.uint8(img)
    return img

def save_img(img, n):
    outpath = "graph"
    output = args.output
    name, ext = os.path.splitext(output)
    if args.auto_name:
        name = name + '-' + args2name(args)
    cv2.imwrite(os.path.join(outpath, name + str(n) + ext), img)

def run(args):
    #rng = np.random.RandomState(args.seed)

    crop_size = (w,h)
    dset = HR_IMG(path, crop_size)
    loader = DataLoader(dset, 1, shuffle=True, num_workers=10)

    nonlin = get_nonlin(args.nonlin)
    output_nonlin = get_nonlin(args.output_nonlin)

    if args.batch_norm:
        def add_bn(nonlin):
            def func(x):
                if args.batch_norm_position == 'before_nonlin':
                    x = F.batch_norm(x)
                x = nonlin(x)
                if args.batch_norm_position == 'after_nonlin':
                    x = F.batch_norm(x)
                return x
            return func
        nonlin = add_bn(nonlin)

    input_dim = 2

    coords = get_coord(w, h)

    print('Compiling...')
    gen = linear_net(nonlin, hidden_size=args.hidden_size,
                    w=w, h=h,
                    nr_hidden=args.nr_hidden,
                    input_dim=input_dim,
                    output_dim=args.nr_channel,
                    recurrent=args.recurrent,
                    output_nonlin=output_nonlin)
    dis = discriminator_2d(w, h)
    pos_net = Pos(nonlin,
            hidden_size=args.hidden_size, 
            w=h, h=h,
            nr_hidden=args.nr_hidden,
            input_dim=input_dim,
            output_dim=args.nr_channel
            )
    axis = Axis(args.hidden_size, w, h)

    axis.cuda()
    #gen.cuda()
    #dis.cuda()
    pos_net.cuda()
    optD = Adam(dis.parameters(), lr=args.d_lr, betas=(0.5, 0.9))
    optG = Adam(gen.parameters(), lr=args.g_lr, betas=(0.5, 0.9))
    optE = Adam(axis.parameters(), lr=args.g_lr, betas=(0.5, 0.9))
    optP = Adam(pos_net.parameters(), lr=args.g_lr, betas=(0.5, 0.9))

    axis.train()
    pos_net.train()
    

    loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
    print('cuda available: ', torch.cuda.is_available())
    loader_iterator = iter(loader)
    for step in range(args.step):
        if step % 5 == 0:
            try:
                img_ori = next(loader_iterator)
                if img_ori.size()[0] != args.batch_size:
                    loader_iterator = iter(loader)
            except StopIteration:
                loader_iterator = iter(loader)

        optE.zero_grad()
        optP.zero_grad()
        #requires_grad(pos_net, False)
        #requires_grad(axis, True)
        

        #rgb = get_coord_rgb(img[0], coords) # [w*h, 3]
        #rgb = np.array(rgb)
        img = img_ori.data.numpy()
        v = torch.from_numpy(coords.astype('float32')).cuda()
        

        real_img = img_ori.cuda()
        weight = axis(real_img)
        #bias = Variable(torch.zeros(args.hidden_size), requires_grad=True).cuda()
        #v = F.relu(torch.mm(v, weight) + bias)

        #img_pred = pos_net(v, real_img)
        #with torch.no_grad():
        img_pred = pos_net(v, weight)

        img_data = img_pred.view(w, h, 3)
        img_pred = img_pred.permute(0, 3, 1, 2)

        if step % 200 == 0:
            img = draw(w, h, img_data)
            save_img(img, step)

        loss = loss_fn(img_pred, real_img)
        loss.backward()
        optE.step()
        optP.step()

        #l2 = loss.mean().data.cpu().numpy()
        l2 = loss.data.cpu().numpy()
        sys.stdout.write("\r" + "Step "+ str(step) + " Loss %.4f" % l2)

        #l1 = d_loss.data.cpu().numpy()
        #l2 = gen_loss.mean().data.cpu().numpy()
        #sys.stdout.write("\r" + "Step "+ str(step) + " | Loss D/G %.4f\t%.4f prob T/F: %.4f\t%.4f" % (l1, l2, true_prob, fake_prob))
        #sys.stdout.flush()
        
            
def main():
    run(get_args())


if __name__ == '__main__':
    main()

