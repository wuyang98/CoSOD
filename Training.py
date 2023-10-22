import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.multiprocessing as mp
import torch.distributed as dist
from tools import custom_print
from dataset import get_loader
import math
# from Models.ImageDepthNet import ImageDepthNet
from Models.ImageDepthNet import VQVAE, PixelCNNWithEmbedding, ImageDepthNet
import os
import numpy as np
from val import validation
import pytorch_toolbelt.losses as PTL
import datetime
import random
import torchvision.transforms as transforms
val_datapath = [os.path.join('/home/hsl/Co-Saliency/data/', i) for i in ['CoCA']]

def _iou(pred, target, size_average = True):

    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:]*pred[i,:,:])
        Ior1 = torch.sum(target[i,:,:]) + torch.sum(pred[i,:,:])-Iand1
        IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b


class IoU_loss(torch.nn.Module):
    def __init__(self):
        super(IoU_loss, self).__init__()

    def forward(self, pred, target):
        # pred = torch.tensor(pred)
        b = np.array(pred.cpu().detach().numpy()).shape[0]
        IoU = 0.0
        for i in range(0, b):
            # compute the IoU of the foreground
            Iand1 = torch.sum(target[i, :, :, :] * pred[i])
            Ior1 = torch.sum(target[i, :, :, :] + pred[i]) - Iand1
            IoU1 = Iand1 / Ior1

            # IoU loss is (1-IoU1)
            IoU = IoU + (1 - IoU1)

        # return IoU/b
        return IoU

class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
    
        pred = torch.sigmoid(pred)

        return _iou(pred, target, self.size_average)

def collate(data):
    out=[]
    zero_map = []
    size = [88, 224, 14, 28, 56, 112, 224, 14, 28, 56, 112]
    data = list(zip(*data))
    for i in range(11):
        group=[]
        for j in range(len(data[0])):
            group.extend(data[i][j])
            
        subs = torch.stack(group, dim=0)
        out.append(subs)
    random_num = random.choice(range(2))
    # random_num = 1
    for o in range(len(size)):
        zero_map.append(torch.zeros(size=[1, size[o], size[o]]))
    for m in range(random_num):
        out[0][m: m + 1, :, :, :] = out[0][m: m + 1, :, :, :] + out[0][m + 5: m + 6, :, :, :]
        out[0][m + 5: m + 6, :, :, :] = out[0][m: m + 1, :, :, :] - out[0][m + 5: m + 6, :, :, :]
        out[0][m: m + 1, :, :, :] = out[0][m: m + 1, :, :, :] - out[0][m + 5: m + 6, :, :, :]
        for n in range(10):  # to 9
            out[n + 1][m] = zero_map[n + 1]
            out[n + 1][m + 5] = zero_map[n + 1]
        
    return out
    
    
def save_loss(whole_iter_num, epoch_total_loss, epoch_loss,log_txt_file):
    custom_print(datetime.datetime.now().strftime('%F %T') +'   '+str(whole_iter_num) +'_iter_total_loss:  [%.4f]'%epoch_total_loss + ', iter_loss:   [%.4f]'%epoch_loss, log_txt_file,'a+')


def adjust_learning_rate(optimizer, decay_rate=.1):
    update_lr_group = optimizer.param_groups
    for param_group in update_lr_group:
        print('before lr: ', param_group['lr'])
        param_group['lr'] = param_group['lr'] * decay_rate
        print('after lr: ', param_group['lr'])
    return optimizer


def save_lr(log_txt_file, optimizer):
    update_lr_group = optimizer.param_groups[1]
    custom_print('encode:update:lr    [%1.1e]'%update_lr_group['lr'] + '\n'+'decode:update:lr   [%1.1e]'%update_lr_group['lr'], log_txt_file,'a+')


def train_net(num_gpus, args):

    mp.spawn(main, nprocs=num_gpus, args=(num_gpus, args, ))


def train_vqvae(model:VQVAE, dataloader, device='cuda', ckpt_path='./checkpoints/VQVAE/',
                lr=1e-3, n_epochs=100, l_w_embedding=1, l_w_commitment=0.25):
    print('train VQVAE!!!')
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    mse_loss = nn.MSELoss()
    tic = time.time()
    best_loss = 99999999999999999999.0

    for e in range(n_epochs):
        total_loss = 0

        for i, data_batch in enumerate(dataloader):
            x, _, _, _, _, _, _, _, _, _, _ = data_batch
            current_batch_size = x.shape[0]
            x = x.to(device)

            x_hat, ze, zq = model(x)
            l_reconstruct = mse_loss(x, x_hat)
            l_embedding = mse_loss(ze.detach(), zq)
            l_commitment = mse_loss(ze, zq.detach())
            loss = l_reconstruct + \
                l_w_embedding*l_embedding + l_w_commitment*l_commitment

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        last_path = os.path.join(ckpt_path, 'last_vqvae.pth')
        best_path = os.path.join(ckpt_path, 'checkpoints/VQVAE/best_vqvae.pth')
        torch.save(model.state_dict(), last_path)
        if total_loss < best_loss:
            torch.save(model.state_dict(), best_path)
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
        print('Done!!!')


def train_generative_model(vqvae: VQVAE, model, dataloader, device='cuda', ckpt_path='./checkpoints/VQVAE/gen_model5.pth',
                             n_epochs=50):
    print('train gen_model')
    vqvae.to(device)
    vqvae.eval()
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    loss_fn = nn.CrossEntropyLoss()
    tic = time.time()
    best_loss = 99999999999.0
    for e in range(n_epochs):
        print(f'epoch:{e}')
        total_loss = 0
        for i, data_batch in enumerate(dataloader):
            x, _, _, _, _, _, _, _, _, _, _ = data_batch
            current_batch_size = x.shape[0]
            # print(f'current_batch_size:{current_batch_size}')
            with torch.no_grad():
                x = x.to(device)
                x = vqvae.encode(x)

            predict_x = model(x)
            loss = loss_fn(predict_x, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        last_path = os.path.join(ckpt_path, 'last_gen.pth')
        best_path = os.path.join(ckpt_path, 'best_gen.pth')
        torch.save(model.state_dict(), last_path)
        if total_loss < best_loss:
            torch.save(model.state_dict(), best_path)
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('Done')


def main(rank, num_gpus, args):

    rank += 2
    # create log dir
    log_root = './result/logs'
    if not os.path.exists(log_root):
        os.makedirs(log_root)

    # create log txt
    project_name = 'baseline'
    i = 0
    while (os.path.exists(os.path.join(log_root, project_name + '_log_%d.txt'%i))):
        i = i+1
    log_txt_file = os.path.join(log_root, project_name + '_log_%d.txt'%i)
    custom_print(project_name, log_txt_file, 'w')

    models_root = './result/models/'
    if not os.path.exists(models_root):
        os.makedirs(models_root)

    cudnn.benchmark = True

    # dist.init_process_group(backend='nccl', init_method=args.init_method, world_size=num_gpus, rank=local_rank)
    # print(local_rank)
    # print(rank)
    # torch.cuda.set_device(rank)
    # torch.cuda.set_device(2)
    vqvae = VQVAE(dim=args.n_dim, n_embedding=args.n_embedding)
    pixelcnn = PixelCNNWithEmbedding(n_blocks=15, p=256, linear_dim=args.linear_dim, bn=True, color_level=args.color_level)

    train_dataset = get_loader(args.trainset, args.data_root, args.img_size, group_size=args.group_size)

    # sampler = torch.utils.data.distributed.DistributedSampler(
    #     train_dataset,
    #     num_replicas=num_gpus,
    #     rank=rank,
    # )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=6,
                                               pin_memory=True,
                                               drop_last=True,
                                               collate_fn=collate,
                                               )
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=6,
    #                                            pin_memory=True,
    #                                            sampler=sampler,
    #                                            drop_last=True,
    #                                            collate_fn=collate,
    #                                            )
    vqvae_path = os.path.join(args.save_vqvae, 'best_vqvae.pth')
    gen_path = os.path.join(args.save_gen_model, 'best_gen.pth')
    # vqvae.load_state_dict(torch.load(vqvae_path))
    # train_vqvae(vqvae, train_loader, ckpt_path=args.save_vqvae, n_epochs=args.vqvae_epoch)
    vqvae.load_state_dict(torch.load(vqvae_path))
    # train_generative_model(vqvae, pixelcnn, train_loader, ckpt_path=args.save_gen_model, n_epochs=args.gen_model_epoch)
    # vqvae.load_state_dict(torch.load('./checkpoints/VQVAE/vqvae.pth'))
    pixelcnn.load_state_dict(torch.load(gen_path))
    net = ImageDepthNet(args, vqvae, pixelcnn)
    net.train()
    net.cuda()

    base_params = [params for name, params in net.named_parameters() if ("backbone" in name)]
    other_params = [params for name, params in net.named_parameters() if ("backbone" not in name)]

    optimizer = optim.Adam([{'params': base_params, 'lr': args.lr * 0.1},
                            {'params': other_params, 'lr': args.lr}])

    print('''
        Starting training:
            Train steps: {}
            Batch size: {}
            Learning rate: {}
            Training size: {}
        '''.format(args.train_steps, args.batch_size, args.lr, len(train_loader.dataset)))

    N_train = len(train_loader) * args.batch_size

    loss_weights = [1, 0.9, 0.8, 0.5, 0.5, 0.5]
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    criterion = nn.BCEWithLogitsLoss()
    criterion1 = IOU()
    iou = IoU_loss()
    FL = PTL.BinaryFocalLoss()
    whole_iter_num = 0
    iter_num = math.ceil(len(train_loader.dataset) / args.batch_size)


    best_p, best_j =0, 0
    for epoch in range(args.epochs):

        print('Starting epoch {}/{}.'.format(epoch + 1, args.epochs))
        print('epoch:{0}-------lr:{1}'.format(epoch + 1, args.lr))

        epoch_total_loss = 0
        epoch_loss = 0

        for i, data_batch in enumerate(train_loader):
            if (i + 1) > iter_num: break

            images, label_224, label_14, label_28, label_56, label_112, \
            contour_224, contour_14, contour_28, contour_56, contour_112 = data_batch

            #
            # images, label_224, contour_224 = Variable(images.cuda(local_rank, non_blocking=True)), \
            #                             Variable(label_224.cuda(local_rank, non_blocking=True)),  \
            #                             Variable(contour_224.cuda(local_rank, non_blocking=True))

            images, label_224, contour_224 = Variable(images.cuda(0, non_blocking=True)), \
                                        Variable(label_224.cuda(0, non_blocking=True)),  \
                                        Variable(contour_224.cuda(0, non_blocking=True))

            label_14, label_28, label_56, label_112 = Variable(label_14.cuda()), Variable(label_28.cuda()),\
                                                      Variable(label_56.cuda()), Variable(label_112.cuda())

            contour_14, contour_28, contour_56, contour_112 = Variable(contour_14.cuda()), \
                                                                                      Variable(contour_28.cuda()), \
                                                      Variable(contour_56.cuda()), Variable(contour_112.cuda())

            outputs = net(images, label_224)

            outputs_saliency, outputs_contour = outputs
            mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency
            cont_1_16, cont_1_8, cont_1_4, cont_1_1 = outputs_contour
            # loss
            loss5 = criterion(mask_1_16, label_14)
            loss4 = criterion(mask_1_8, label_28)
            loss3 = criterion(mask_1_4, label_56)
            loss1 = criterion(mask_1_1, label_224)
            loss11= criterion1(mask_1_1, label_224)

            # contour loss
            c_loss5 = criterion(cont_1_16, contour_14)
            c_loss4 = criterion(cont_1_8, contour_28)
            c_loss3 = criterion(cont_1_4, contour_56)
            c_loss1 = criterion(cont_1_1, contour_224)

            img_total_loss = loss_weights[0] * loss1 + loss_weights[2] * loss11 + loss_weights[2] * loss3 + loss_weights[3] * loss4 + loss_weights[4] * loss5
            contour_total_loss = loss_weights[1] * c_loss1 + loss_weights[2] * c_loss3 + loss_weights[3] * c_loss4 + loss_weights[4] * c_loss5

            total_loss = img_total_loss + contour_total_loss

            epoch_total_loss += total_loss.cpu().data.item()
            epoch_loss += loss1.cpu().data.item()

            optimizer.zero_grad()

            total_loss.backward()

            optimizer.step()
            whole_iter_num += 1

            if whole_iter_num == args.train_steps:
                torch.save(net.state_dict(),
                           args.save_model_dir + 'baseline.pth')
                           
            if (whole_iter_num+1)%args.show_iter == 0:
                save_loss(whole_iter_num+1, epoch_total_loss /args.show_iter, epoch_loss /args.show_iter, log_txt_file)
                epoch_total_loss = 0
                epoch_loss = 0

            if (whole_iter_num+1)%args.val_iter == 0:
                print('testing!')
                custom_print(datetime.datetime.now().strftime('%F %T'), log_txt_file, 'a+')
                ave_p, ave_j = validation(net, val_datapath, group_size=args.group_size)
                import numpy as np
                if(np.mean(ave_p) >= best_p):
                    best_p = np.mean(ave_p)
                    torch.save(net.state_dict(), models_root + 'baseline_best.pth')
                torch.save(net.state_dict(), models_root + 'baseline_last.pth')

                custom_print('-' * 100, log_txt_file, 'a+')
                custom_print(datetime.datetime.now().strftime('%F %T') + ' CoCA     p: [%.4f], j: [%.4f]' %
                             (ave_p[0], ave_j[0]), log_txt_file, 'a+')
                # custom_print(datetime.datetime.now().strftime('%F %T') + ' Cosal15  p: [%.4f], j: [%.4f]' %
                #              (ave_p[1], ave_j[1]), log_txt_file, 'a+')
                # custom_print(datetime.datetime.now().strftime('%F %T') + ' CoSOD3k  p: [%.4f], j: [%.4f]' %
                #              (ave_p[2], ave_j[2]), log_txt_file, 'a+')
                custom_print('-' * 100, log_txt_file, 'a+')
                net.train()

            if whole_iter_num == args.train_steps:
                return 0

            if whole_iter_num == args.stepvalue1 or whole_iter_num == args.stepvalue2:
                optimizer = adjust_learning_rate(optimizer, decay_rate=args.lr_decay_gamma)
                save_lr(log_txt_file, optimizer)










