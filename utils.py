import argparse
import yaml
import time
import numpy as np
import sys
import os

import torch
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn, autograd, optim
from torch.nn import functional as F

from torchvision.utils import make_grid, save_image

#sys.path.append('models')

if 0:
    state_dict = torch.load(args.gckpt)
    #model.load_generator_state_dict(state_dict)
    model.generator.load_state_dict(state_dict)

    state_dict = torch.load(args.dckpt)
    model.discriminator.load_state_dict(state_dict)
    #model.load_discriminator_state_dict(state_dict)
    model.s_dis.load_state_dict(ckpt['d'])

def args2name(args):
    black_list = ['output', 'auto_name']
    return '-'.join(['{}:{}'.format(key, sanitize_str(str(value)))
                     for key, value in sorted(args._get_kwargs())
                     if key not in black_list and value is not None])

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )   
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(opt['batch_size'], 1, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.cuda() if opt['use_cuda'] else alpha

    #print(alpha.size(), real_data.size(), fake_data.size())
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt['use_cuda']:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size()).cuda() if opt['use_cuda'] else torch.ones(disc_interpolates.size()), create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) **2).mean() * opt['lambda']
    return gradient_penalty


if __name__ == "__main__":
    #ckpt = torch.load("models/550000.pt")
    #gen.s_gen.load_state_dict(ckpt['g'], strict=False)


    one = torch.FloatTensor([1]* opt['batch_size'])
    mone = one  -1
    if opt['use_cuda']:
        one = one.cuda()
        mone = mone.cuda()



    for epoch in range(start_epoch, opt['epochs']):
        start_time = time.time()

        requires_grad(dis, True)
        requires_grad(gen, False)
        
        for i in range(opt['dis_iters']):
            dis.zero_grad()
            
            #retrieve another iterator and start a new epoch if dataset is exhausted
            try:
                real_data = next(loader_iterator)
                if real_data.size()[0] != opt["batch_size"]:
                    loader_iterator = iter(loader)
                    break
            except StopIteration:
                loader_iterator = iter(loader)
                break

            if opt['use_cuda']:
                real_data = real_data.cuda()
            real_data = Variable(real_data)


            """
            d_real_loss = dis(real_data)
            d_real_loss.backward(mone)   
            """
        
            z = gen.generate_input(opt["batch_size"])
            if opt['use_cuda']:
                z = z.cuda()
            #Disable computation of gradients in generator thanks to volatile
            #volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
            #z = Variable(z, volatile=True)
            z = Variable(z)
        
            with torch.no_grad():
                fake_data = Variable(gen(z).data)

            """
            d_fake_loss = dis(fake_data)
            d_fake_loss.backward(one)
            """

            real_pred = dis(real_data)
            fake_pred = dis(fake_data)
            true_prob = real_pred.mean().data.cpu().numpy()
            fake_prob = fake_pred.mean().data.cpu().numpy()


            #wgan-gp gradient penalty computation
            #gradient_penalty = calc_gradient_penalty(dis, real_data.data, fake_data.data)
            #gradient_penalty.backward()

            #loss = d_fake_loss - d_real_loss# + gradient_penalty
            #loss = loss.mean()

            d_loss = d_logistic_loss(real_pred, fake_pred)
            d_loss.backward()
            optimizerD.step()
        
        ######Generator training######
        requires_grad(dis, False)
        requires_grad(gen, True)
       
        if epoch < 1000:
            g_iter = 1
            optG = opt_g_slow
        else:
            optG = optimizerG
            g_iter = 2 

        if epoch % 100 == 0:
            g_iter = 10


        for nn in range(g_iter):
            gen.zero_grad()
            z = gen.generate_input(opt['batch_size'])
            if opt['use_cuda']:
                z = z.cuda()
            z = Variable(z)
            fake_data = gen(z)
            gen_logit = dis(fake_data)
            gen_loss = g_nonsaturating_loss(gen_logit) 
            gen_loss.backward()
            optG.step()


        l1 = d_loss.data.cpu().numpy()
        l2 = gen_loss.mean().data.cpu().numpy()
        sys.stdout.write("\r" + "Epoch "+ str(epoch) + " | Loss D/G %f\t%f prob T/F: %f\t%f" % (l1, l2, true_prob, fake_prob))
        sys.stdout.flush()

        
        end_time = time.time()
        epoch_time = end_time - start_time
        
        ###Savings###
        if (epoch - start_epoch) % opt['save_every'] == 0:
            #print()
            #print("Saving for epoch {}. Gen loss: {:.2f} | Dis loss: {:.2f}".format(epoch, gen_loss.mean().data, d_loss.data))
            
            #Save summary image
            samples_path = '.' if opt['samples_path'] == "" else opt['samples_path']
            videos = fake_data.data
            #video_seq = videos.view(videos.size(0)*videos.size(1), videos.size(2), videos.size(3), videos.size(4))
            #video_grid = make_grid(video_seq, nrow=6)
            
            #save_image(denormalizer(real_data_grid).cpu(), samples_path+"/real_data_"+str(epoch)+".png")
            #save_image(denormalizer(video_grid), samples_path+"/"+str(epoch)+".png")
            #save_image(video_grid, samples_path+"/"+str(epoch)+".png", normalize=True, range=(-1, 1))

            for n in range(videos.size(1)):
                img = videos[:,n]
                save_image(img, "gif/%d_" % epoch + "%03d.png" % n, nrow=4)
            #os.system("ffmpeg -f image2 -r 10 -i gif/%03d.png" + " motion_%d.gif" % epoch)

            #real_seq = real_data.data.view(videos.size(0)*videos.size(1), videos.size(2), videos.size(3), videos.size(4))
            #real_data_grid = make_grid(real_seq, nrow = videos.size(0))
            #save_image(real_data_grid, samples_path+"/real_data_"+str(epoch)+".png", normalize=True, range=(-1, 1))

            #Save checkpoint
            base_path = "." if opt['checkpoint_base'] == "" else opt['checkpoint_base']
            cp_data = {'epoch': epoch, 
                    "epoch_time":epoch_time, 
                    "gen_state_dict":gen.state_dict(),
                    "dis_state_dict":dis.state_dict(), 
                    "last_generated_videos":fake_data.data, 
                    "dis_loss": d_loss, 
                    "gen_loss": gen_loss}

            torch.save(cp_data, base_path+"/"+"checkpoint"+str(epoch)+".pth")




