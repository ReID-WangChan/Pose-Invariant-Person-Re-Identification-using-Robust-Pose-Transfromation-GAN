# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:30:08 2019

@author: VR LAB PC3
"""

import torch
import torch.nn as nn
from torchvision import transforms, utils
import torch.utils.data as Data
from torch.autograd import Variable
from torch.optim import lr_scheduler
from config import cfg
from tensorboardX import SummaryWriter
import os, itertools
import network
import dataset
import time
# import matplotlib.pyplot as plt
import sys
import logger
import random

## Set random seem for reproducibility
#manualSeed = 1
#manualSeed = random.randint(1, 10000) # use if you want new results
manualSeed = 123 
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
#%%

# Save model
def save_model(epoch, path, nets, optimizers, net_name):
    netG, netD = nets
    print(net_name)
    optimizer_G, optimizer_D = optimizers
    print ("Saving model ------------------------------->")
#    if not os.path.exists(os.path.join(path, net_name)):
#        os.mkdir(os.path.join(path, net_name))
    os.makedirs(path + '/' + net_name, exist_ok=True)
    torch.save({'epoch': epoch, 'state_dict': netG.state_dict(), 'optimizer' : optimizer_G.state_dict(), },
                f='%s/%s/%s_%d.pkl' % (path, net_name, 'G', epoch))
    torch.save({'epoch': epoch, 'state_dict': netD.state_dict(), 'optimizer' : optimizer_D.state_dict(), },
                f='%s/%s/%s_%d.pkl' % (path, net_name, 'D', epoch))
    print ("Finished ----------------------------------->")


# Save images
def save_images(net_name, epoch, PATH, src_img, pose, tgt_img, fake_img, summary):
    n, c, h, w = src_img.size()
    samples = torch.FloatTensor(3*n, c, h, w).zero_()
    for i in range(n):
        samples[4*i+0] = src_img[i].data
        samples[4*i+1] = tgt_img[i]
        samples[4*i+2] = fake_img[i].data
#        samples[4*i+3] = pose[i].data

    images = utils.make_grid(samples, nrow=8, padding=30, normalize=True)
    summary.add_image('samples', images, epoch)
    file_name = os.path.join(PATH, net_name)
    if not os.path.exists(file_name):
        os.mkdir(file_name)
    utils.save_image(samples, '%s/samples_%d.png' % (file_name, epoch), nrow=8, padding=30, normalize=True)


    
# Load Data
def load_data():
    train_data = dataset.Market_DataLoader(imgs_path=cfg.TRAIN.imgs_path, pose_path=cfg.TRAIN.pose_path, idx_path=cfg.TRAIN.idx_path,
                                           transform=dataset.train_transform(), loader=dataset.val_loader, mode='train')
    train_loader = Data.DataLoader(train_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                   num_workers=cfg.TRAIN.NUM_WORKERS, drop_last=True)

    val_data = dataset.Market_DataLoader(imgs_path=cfg.TRAIN.imgs_path, pose_path=cfg.TRAIN.pose_path, idx_path=cfg.TEST.idx_path,
                                         transform=dataset.val_transform(), loader=dataset.val_loader, mode='test')
    val_loader = Data.DataLoader(val_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                                 num_workers=cfg.TRAIN.NUM_WORKERS)

    train = [train_data, train_loader]
    val = [val_data, val_loader]
    return train, val


def load_network():
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID
    print ('###################################')
    print ("#####      Build Network      #####")
    print ('###################################')

    netG = network.Res_Generator(ngf=64, nz=(2048+50))
    # for name, param in netG.named_parameters():
    #     print(f'{name} -> {param.size()}')
    # exit()

    netD = network.DC_Discriminator(ndf=64)
    
    netRN = network.ResNet50()
    
    netE = network.Ensemble(netRN, netG)
    
    nets = []
    nets.append(netE)
    nets.append(netD)
    
#    print_networks(nets, debug=True)
    
    for net in nets:
        net.cuda()
        
    return nets
        
        

def print_networks(model_names, debug):
    print ('---------------- Network initialized ----------------')
    names = ['netG', 'netD']
    for i, net in enumerate(model_names):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        if debug:
            print ('=========== %s ===========' % names[i])
            print (net)
        print ('[Network %s] Total number of parameters: %.3f M' % (names[i], num_params / 1e6))
    print ('-----------------------------------------------------')



# define optimizers
def Optimizer(nets):
    netG, netD = nets

    optimizer_G = torch.optim.Adam(netG.parameters(), lr=cfg.TRAIN.LR, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=cfg.TRAIN.LR, betas=(0.5, 0.999))
    optimizers = [optimizer_G, optimizer_D]

    lr_policy = lambda epoch: (1 - 1 * max(0, epoch-cfg.TRAIN.LR_DECAY) / cfg.TRAIN.LR_DECAY)
    scheduler_G = lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_policy)
    scheduler_D = lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lr_policy)
    schedulers = [scheduler_G, scheduler_D]

    summary = SummaryWriter(log_dir='%s/%s' % (os.path.join(cfg.FILE_PATH, 'log'), cfg.NET), comment='')

    return optimizers, schedulers, summary


def loss_func():
    criterionGAN = torch.nn.MSELoss().cuda()
    criterionIdt = torch.nn.L1Loss().cuda()
    criterionAtt = torch.nn.CrossEntropyLoss().cuda()
    criterion = [criterionGAN, criterionIdt, criterionAtt]

    return criterion



#%%
if __name__ == '__main__':
    
    sys.stdout = logger.Logger('./log_GAN_ep2.txt')
    
    train_data = dataset.Market_DataLoader(imgs_path=cfg.TRAIN.imgs_path, pose_path=cfg.TRAIN.pose_path, idx_path=cfg.TRAIN.idx_path,
                                           transform=dataset.train_transform(), img_loader=dataset.val_loader, pose_loader=dataset.pose_loader)
    
    train_loader = Data.DataLoader(train_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS, drop_last=True)
    
    print(f'torch.cuda.is_available() = {torch.cuda.is_available()}')

    val_data = dataset.Market_DataLoader(imgs_path=cfg.TEST.imgs_path, pose_path=cfg.TEST.pose_path, idx_path=cfg.TEST.idx_path,
                                         transform=dataset.val_transform(), img_loader=dataset.val_loader, pose_loader=dataset.pose_loader)
    
    print('after val_data')
    val_loader = Data.DataLoader(val_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                                 num_workers=cfg.TRAIN.NUM_WORKERS)

    print(f'after val_loader')

    train_file = [train_data, train_loader]
    val_file = [val_data, val_loader]
    
    print(f'start load_network')
#    train_file, val_file = load_data()
    nets = load_network()
    optimizers, schedulers, summary = Optimizer(nets)
    criterion = loss_func()
    
#    train(train_file, val_file, nets, optimizers, schedulers, summary, criterion)
    
    print ('\n###################################')
    print ("#####      Start Traning      #####")
    print ('###################################')

    train_data, train_loader = train_file
    val_data, val_loader = val_file
    netG, netD = nets
    optimizer_G, optimizer_D = optimizers
    scheduler_G, scheduler_D = schedulers
    criterionGAN, criterionIdt, criterionAtt = criterion
    
    if os.path.exists('./model/GAN/G_1.pkl'):
        print('===========REUSING EARLIER RESULT============')
        checkpoint = torch.load('./model/GAN/G_2.pkl')
        netG.load_state_dict(checkpoint['state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer'])
        checkpoint = torch.load('./model/GAN/D_2.pkl')
        netD.load_state_dict(checkpoint['state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer'])
        netG.train()
        netD.train()

#%%
    count = 0
    for epoch in range(1, cfg.TRAIN.MAX_EPOCH+1):
        scheduler_G.step()
        scheduler_D.step()
        for step, (src_img, tgt_img, pose, label) in enumerate(train_loader):
            begin = time.time()

            # #######################################################
            # (1) Data process
            # #######################################################

            src_img = Variable(src_img).cuda()      # N x 3 x H x W
            tgt_img = Variable(tgt_img).cuda()      # N x 3 x H x W
            pose = Variable(pose).cuda()            # N x 50
            label = Variable(label).cuda()

            # #######################################################
            # (2) Generate images
            # #######################################################
            print(f'src_img = {src_img.shape}')
            print(f'pose = {pose.shape}')
            # exit()
            fake_img = netG(src_img, pose)
#            print(src_img.size())
#            print(fake_img.size())
#            print(tgt_img.size())
            # #######################################################
            # (3) Update Generators
            # #######################################################
            D_fake_img, _ = netD(fake_img)
            G_loss = criterionGAN(D_fake_img, torch.ones_like(D_fake_img))
            idt_loss = criterionIdt(fake_img, tgt_img) * cfg.TRAIN.lambda_idt

            loss_G = G_loss + idt_loss
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()


            # #######################################################
            # (4) Update Discriminators
            # #######################################################
            D_fake_img, att_fake = netD(fake_img.detach())
            D_real_img, att_real = netD(src_img)

            D_fake_loss = criterionGAN(D_fake_img, torch.zeros_like(D_fake_img))
            
            #D_real_loss = criterionGAN(D_real_img, torch.ones_like(D_real_img))
            # label smoothing for real samples
            l = 0.8
            r = 1.1
            label_real = l + torch.rand(D_real_img.size()) * (r - l)
            random.shuffle(label_real)
            D_real_loss = criterionGAN(D_real_img, label_real.cuda())
            
            D_class_att_loss = criterionAtt(att_real, label) + criterionAtt(att_fake, label)
            

            loss_D = (D_fake_loss + D_real_loss)*10 + D_class_att_loss
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()


            # #######################################################
            # (5) Update Log and Display loss info.
            # #######################################################
            count += 1
            summary.add_scalar('G_loss', G_loss.item(), count)
            summary.add_scalar('Idt_loss', idt_loss.item(), count)
            summary.add_scalar('netG_loss', loss_G.item(), count)
            summary.add_scalar('netD_loss', loss_D.item(), count)

            print ('Epoch: {}/{}  |  Step: {}/{}  |  lr: {:.6f}  |  G_loss: {:.6f}  |  D_loss: {:.6f}  |  Idt_loss: {:.6f}  |  Time: {:.3f}'
               .format(epoch, cfg.TRAIN.MAX_EPOCH, step+1, len(train_loader), optimizer_G.param_groups[0]['lr'],
                       loss_G.item(), loss_D.item(), idt_loss.item(), time.time()-begin))


        # #######################################################
        # (7) Validation
        # #######################################################
        netG.eval()
        PATH = os.path.join(cfg.FILE_PATH, 'images')
        
        os.makedirs(PATH, exist_ok=True)
        
        for _, (src_img, tgt_img, pose, label) in enumerate(val_loader):
            src_img = Variable(src_img, volatile=True).cuda()
            pose = Variable(pose, volatile=True).cuda()

            fake_img = netG(src_img, pose)

            save_images(cfg.NET, epoch, PATH, src_img, pose, tgt_img, fake_img, summary)
            break    # For better comparison, we only test one specfic batch images
        netG.train()


        # #######################################################
        # (7) Save models per epoch
        # #######################################################
        MODEL_PATH = os.path.join(cfg.FILE_PATH, 'model')
        print(MODEL_PATH)
        save_model(epoch, MODEL_PATH, nets, optimizers, cfg.NET)

    summary.close()
    
#    from torchsummary import summary
#    summary(nets, input_size=(3, 128, 128))



