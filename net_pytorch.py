import argparse

#import theano
#import theano.tensor as T
import torch as T
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader

import os
import numpy as np
import cv2


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


#coords = T.matrix()
class linear_net(nn.Module):
    def __init__(self, nonlin, hidden_size=100, nr_hidden=3,
             input_dim=2,
             output_dim=1, recurrent=False,
             output_nonlin=lambda x: x):

        super(linear_net, self).__init__()

        self.nonlin = nonlin
        self.hidden_size = hidden_size
        self.nr_hidden = nr_hidden
        self.output_nonlin = output_nonlin

        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        v = self.fc1(x)
        v = self.nonlin(v)
        
        for i in range(self.nr_hidden):
            v = self.fc2(v)
            v = self.nonlin(v)

        v = self.fc3(v)
        v = self.output_nonlin(v)
        v = (v - v.min(dim=0, keepdim=True).values) / (
            v.max(dim=0).values - v.min(dim=0).values + 1e-8)
        #v = T.sigmoid(v)

        return v

    #return theano.function([coords], v)


def draw(func, w, h, coord_bias=False):
    coords = np.array(np.meshgrid(np.arange(h), np.arange(w))[::-1],
                      dtype='float32').reshape((2, -1)).swapaxes(0, 1) / [w, h]
    #coords = np.random.uniform(-1., 1., size=[w*h, 2])
    #coords = np.random.normal(1., 6, size=[w*h, 2])

    if coord_bias:
        coords = np.concatenate((coords, np.ones((coords.shape[0], 1))), axis=1)
    coords = coords.astype('float32')

    print(coords.shape)
    x = T.from_numpy(coords.copy()).cuda()
    #x = Variable(x)
    out = func(x)
    out = out.data.cpu().numpy()

    img = (out.reshape((w, h, -1)) * 255).astype('uint8')
    print(img.shape)

    if img.shape[2] == 1:
        img = img[:,:]
    return img


def cvpause():
    while True:
        if (cv2.waitKey(0) & 0xff) == ord('q'):
            break
        print('press `q` to close this window')


def get_nonlin(name, rng):
    if name == 'random_every_time':
        def nonlin(x):
            return NONLIN_TABLE[rng.choice(list(NONLIN_TABLE))](x)
        return nonlin

    if name == 'random_once':
        return NONLIN_TABLE[rng.choice(list(NONLIN_TABLE))]

    return NONLIN_TABLE[name]


def sanitize_str(x):
    x = x.replace('/', '-')
    i = 0
    while i < len(x) and x[i] == '-':
        i += 1
    return x[i:]


def args2name(args):
    black_list = ['output', 'auto_name']

    return '-'.join(['{}:{}'.format(key, sanitize_str(str(value)))
                     for key, value in sorted(args._get_kwargs())
                     if key not in black_list and value is not None])


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


def run(args):
    outpath = "graph"
    rng = np.random.RandomState(args.seed)

    w, h = map(int, args.image_size.split('x'))

    nonlin = get_nonlin(args.nonlin, rng)
    output_nonlin = get_nonlin(args.output_nonlin, rng)

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
    if args.coord_bias:
        input_dim += 1

    if 1:
        print('Compiling...')
        func = linear_net(nonlin, hidden_size=args.hidden_size,
                        nr_hidden=args.nr_hidden,
                        input_dim=input_dim,
                        output_dim=args.nr_channel,
                        recurrent=args.recurrent,
                        output_nonlin=output_nonlin)

        func.cuda()

    optimizerD = Adam(dis.parameters(), lr=args.d_lr, betas=(0.5, 0.9))
    optimizerG = Adam(gen.parameters(), lr=args.g_lr, betas=(0.5, 0.9))

    for nnn in range(20):
        print('Drawing...')
        img = draw(func, w, h, coord_bias=args.coord_bias)

        if args.output:
            output = args.output
            name, ext = os.path.splitext(output)
            if args.auto_name:
                name = name + '-' + args2name(args)
            cv2.imwrite(os.path.join(outpath, name + str(nnn) + ext), img)
        else:
            cv2.imshow('img', img)
            cvpause()


def main():
    run(get_args())


if __name__ == '__main__':
    main()

