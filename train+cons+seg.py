import os
import sys
import cv2
import argparse
import numpy as np
import logging
import time
import torch
import random
from torch import nn
from torch.nn import MSELoss
from torch.nn import L1Loss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms
from defs import *
import network



def get_args():
    parser = argparse.ArgumentParser(description="train JESS")
    parser.add_argument("--is_resume", type=bool, default=False,
                        help="whether to augment data")
    parser.add_argument("--resume_checkpoint", default='',
                        help='the checkpoint to start training from')
    parser.add_argument("--num_workers", type=int, default=8,
                        help="numworks in dataloader")
    parser.add_argument("--aug_data", type=bool, default=True,
                        help="whether to augment data")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--net", type=str, default="HDC_cons_seg",
                        help="Network architecture: Baseline, HDC_cons, HDC_seg, HDC_cons_seg is expected")
    parser.add_argument("--opt", type=str, default="Adam",
                        help="Optimizer for updating the network parameters")
    parser.add_argument("--patch_size", type=int, default=128,
                        help="patch size or random crop")
    parser.add_argument("--batch_size", type=int, default = 8,
                        help="batch size")
    parser.add_argument("--epochs", type=int, default=150,
                        help="number of epochs")
    parser.add_argument("--milestones", default=[40, 80, 120, 150, 190],
                        help="steps for multistep LR")
    parser.add_argument("--sessname", type=str, default="",
                        help="different session names for parameter modification")
    parser.add_argument("--train_dir", type=str, default="",
                        help="train dir")
    parser.add_argument("--val_dir", type=str, default="",
                        help="validation dir")
    parser.add_argument("--edge_dir", type=str, default="",
                        help="edge label dir")
    ######################################
    
    parser.add_argument("--seg_dir", type=str, default="",
                        help="seg label dir")
    parser.add_argument("--over_dir", type=str, default="",
                        help="the oversmmoothed image dir, note that, the index must be correspongding")
    parser.add_argument("--pathN", type=str, default="",
                        help="random negetive sample dir")
    parser.add_argument("--pathP", type=str, default="",
                        help="random positive sample dir")
    parser.add_argument("--ckpt", type=str, default='./checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth',
                        help="restore from ckpt")
    parser.add_argument("--picknumber", type=int, default = 8,
                        help="the number of images for contrast module")
    parser.add_argument("--rate", type=int, default = 1,
                        help="the ratio of pos:neg")
    parser.add_argument("--k1", type=float, default=0.3, 
                        help="the parameter alpha of triplet loss")
    parser.add_argument("--k2", type=float, default=0.04,
                        help="the parameter beta of triplet loss")
    parser.add_argument("--triplet", type=bool, default=True,
                        help="whether or not use the triplet loss")
    parser.add_argument("--bothneg", type=bool, default=True,
                        help="true means both unsmoothed and oversmmoothed image are denoted as negtive samples")
    parser.add_argument("--useover", type=bool, default=True,
                        help="whetherr or not use the oversmmoothed as negtive sample")
    parser.add_argument("--segmodel", type=str, default="deeplabv3plus_mobilenet",
                        help="the pretrained segmentation model")
    parser.add_argument("--num_classes", type=int, default=21,
                        help="the class of seg (voc dataset)")
    parser.add_argument("--output_stride", type=int, default=8,
                        help="[8,16], i dont know,just copy,attention here")
    parser.add_argument("--fix_seg", type=bool, default=True,
                        help="if freeze the segmantation part")
    parser.add_argument("--use_pretrained_seg", type=bool, default=True,
                        help="if use the pretrained model, yes:need to load")
    parser.add_argument("--lambda_s", type=float, default=1,
                        help="the coefficient of the segloss")
    parser.add_argument("--lambda_c", type=float, default=1,
                        help="the coefficient of the contrastloss")
    
    args = parser.parse_args()
    return args


class Session:
    def __init__(self, args):
        self.output_dir = os.path.join('./VOC_outfiles', args.sessname)
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info('set output dir as %s' % self.output_dir)


        if args.net == "Baseline" or args.net == "HDC_cons" or args.net == "HDC_cons_seg" or args.net == "HDC_seg":
            self.net = HDC_edge_refine(in_c=3, out_c=3, dim=64, num_block=12).cuda()
        elif args.net == "PDP_edge_refine":
            self.net = PDP_edge_refine(in_c=3, out_c=3, dim=64, num_block=20).cuda()
        elif args.net == "RDB_edge_refine":
            self.net = RDB_edge_refine(in_c=3, out_c=3, dim=64, num_block=12).cuda()
        elif args.net == "Edge_guided":
            self.net = Edge_guided(in_c=3, out_c=3, dim=64, num_block=12).cuda()    
        else:
            logger.warning("NET NAME NOT EXIST!!! --net should be one of : Baseline, HDC, HDC_edge, HDC_edge_refine")

        if args.segmodel == "deeplabv3plus_mobilenet":
            self.model = network.deeplabv3plus_mobilenet(num_classes=args.num_classes, output_stride=args.output_stride).cuda()
        if args.use_pretrained_seg:
            checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint["model_state"])
            print('====>load parameters for seg module<====')
        if args.fix_seg:
            for para in self.model.parameters():
                para.requires_grad = False
            print("====>the parameters in seg has been fixed<====")
        #self.model = nn.DataParallel(self.model)
        #self.model.to(device)

        self.ssim = SSIM().cuda()
        self.patch_size = args.patch_size
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.lambda_s = args.lambda_s
        self.lambda_c = args.lambda_c

        self.step = 0
        self.epoch = args.epochs
        self.now_epoch = 0
        self.start_epoch = 0
        self.writers = {}
        self.total_step = 0

        self.sessname = args.sessname
        self.mse = MSELoss().cuda()
        self.l2Loss = MSELoss().cuda()
        self.tvLoss = TVLoss().cuda()
        self.GDLoss = GDLoss().cuda()
        self.l1Loss = L1Loss().cuda()
        self.edgeLoss = edgeLoss().cuda()
        self.mssimLoss = mSSIMLoss().cuda()
        self.constLoss1 = const_loss().cuda()
        self.constLoss2 = const_loss2().cuda()
        self.contrastloss = ContrastLoss().cuda() #CR loss
        self.segLoss = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

        if args.opt == "SGD":
            self.opt = SGD(self.net.parameters(), lr=args.lr)
        else:
            self.opt = Adam(self.net.parameters(), lr=args.lr)

        self.sche = MultiStepLR(self.opt, milestones=args.milestones, gamma=0.5)

    def tensorboard(self, name):
        path = self.output_dir
        self.writers[name] = SummaryWriter(os.path.join(path, name+'.events'))
        return self.writers[name]

    def write(self, name, total_loss, loss_components, ssim, epoch):
        lr = self.opt.param_groups[0]['lr']
        self.writers[name].add_scalar("lr", lr, epoch)
        self.writers[name].add_scalars("loss", total_loss, epoch)
        self.writers[name].add_scalars("loss_components", loss_components, epoch)
        self.writers[name].add_scalars("ssim", {"train": ssim[0], "val": ssim[1]}, epoch)


    def write_close(self,name):
        self.writers[name].close()

    def get_dataloader(self, dir, name):
        if name == "train":
            dataset = TrainDataset(dir, self.patch_size, aug_data=args.aug_data)
            train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                              num_workers=self.num_workers, drop_last=True)
            self.total_step = len(train_loader)
            return train_loader
        elif name == "val":
            dataset = TestDataset(dir, self.patch_size)
            return DataLoader(dataset, batch_size=1, shuffle=False,
                              num_workers=self.num_workers, drop_last=True)
        else:
            logger.warning("Incorrect Name for Dataloader!!!")


    def get_dataloader(self, dir, dir2, name):
        if name == "train":
            dataset = TrainDatasetE(dir, dir2, self.patch_size, aug_data=args.aug_data)
            train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                              num_workers=self.num_workers, drop_last=True)
            self.total_step = len(train_loader)
            return train_loader
        elif name == "val":
            dataset = TestDatasetE(dir, dir2, self.patch_size)
            return DataLoader(dataset, batch_size=1, shuffle=False,
                              num_workers=self.num_workers, drop_last=True)
        else:
            logger.warning("Incorrect Name for Dataloader!!!")

    def get_dataloader(self, dir, dir2,dir3, name):
        if name == "train":
            dataset = TrainDatasetS(dir, dir2, dir3, self.patch_size, aug_data=args.aug_data)
            train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                              num_workers=self.num_workers, drop_last=True)
            self.total_step = len(train_loader)
            return train_loader
        elif name == "val":
            dataset = TestDatasetS(dir, dir2, dir3, self.patch_size)
            return DataLoader(dataset, batch_size=1, shuffle=False,
                              num_workers=self.num_workers, drop_last=True)
        else:
            logger.warning("Incorrect Name for Dataloader!!!")

    def get_dataloader(self, dir, dir2, dir3, dir4, name):
        if name == "train":
            dataset = TrainDatasetF(dir, dir2, dir3, dir4, self.patch_size, aug_data=args.aug_data)
            train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                              num_workers=self.num_workers, drop_last=True)
            self.total_step = len(train_loader)
            return train_loader
        elif name == "val":
            dataset = TestDatasetF(dir, dir2, dir3, dir4, self.patch_size, aug_data=args.aug_data)
            return DataLoader(dataset, batch_size=2, shuffle=False,
                              num_workers=self.num_workers, drop_last=True)
        else:
            logger.warning("Incorrect Name for Dataloader!!!")

    def save_checkpoints(self, name):
        dir = self.output_dir
        ckp_path = os.path.join(dir, name)
        obj = {
            'net': self.net.state_dict(),
            'now_epoch': self.now_epoch+1,
            'opt': self.opt.state_dict(),
        }
        torch.save(obj, ckp_path)

    def load_checkpoints(self, dir):
        ckp_path = dir
        try:
            obj = torch.load(ckp_path)
            logger.info('Load checkpoint %s' % ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!!' % ckp_path)
            return
        self.net.load_state_dict(obj['net'])
        self.opt.load_state_dict(obj['opt'])
        self.start_epoch = obj['now_epoch']

    def get_con(self, path, picknumber):
        transff = transforms.ToTensor()
        pathDir = os.listdir(path)

        sample = random.sample(pathDir, picknumber)
        #print(path+ '/'+sample[0])
        
        #assert 1==0
        X = cv2.imread(path + '/' + sample[0]).astype(np.float32) / 255
        X = cv2.resize(X, (self.patch_size, self.patch_size))
        X = transff(X)
        X = torch.unsqueeze(X, 0)#(1,c,h,w)
        for i in range(1, picknumber):
            X_ = cv2.imread(path + '/' + sample[i]).astype(np.float32) / 255
            X_ = cv2.resize(X_, (self.patch_size, self.patch_size))
            X_ = transff(X_)
            X_ = torch.unsqueeze(X_, 0)
            X = torch.cat((X, X_), 0)
        
        X = X.cuda()    
        return X
            

    def inf_batch(self, name, batch):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.net == "HDC_cons":
            if args.useover:
                OB, GT, EG, OV = batch['OB'].cuda(), batch['GT'].cuda(), batch['EG'].cuda(), batch['SE'].cuda()
            else:
                OB, GT, EG = batch['OB'].cuda(), batch['GT'].cuda(), batch['EG'].cuda()

            edge, pre, res = self.net(OB)
            pred = pre + res
            lossL = self.l1Loss(pre, GT) + self.l1Loss(pred, GT)
            lossS = 0.1 * (4 - self.mssimLoss(pre, GT) + 4 - self.mssimLoss(pred, GT))
            lossD = 10 * self.GDLoss(pre, mask=1 - EG) + self.GDLoss(pred, mask=1 - EG)
            lossE = 0.001 * self.edgeLoss(edge, EG)
            
            if args.useover:
                if args.bothneg:
                    rand_ind = random.random()
                    neg = OV if rand_ind<0.5 else OB
                else :
                    neg=OV
            else:
                neg = OB
            
            pos=GT

            if args.triplet:
                lossC = self.constLoss1(pred, neg, pos, args.k1) + self.constLoss2(pred, neg, pos, args.k2)
            else:
                lossC = self.contrastloss(pred, pos, neg)
            
            loss = lossL + lossS + lossE + lossC

            loss_dict = {
                'total': loss.item(),
                'L1': lossL.item(),
                'mSSIM': lossS.item(),
                'DTV': lossD.item(),
                'edge': lossE.item(),
                'contrast': lossC.item
            }
                          


        elif args.net == "HDC_seg":
            OB, GT, EG, SE, OV = batch['OB'].cuda(), batch['GT'].cuda(), batch['EG'].cuda(), batch['SE'].cuda(), batch['OV'].cuda()
            SE = SE.to(device, dtype=torch.long)

            edge, pre, res = self.net(OB)
            seg_pred = self.model(OB)
            
            pred = pre + res
            
            lossL = self.l1Loss(pre, GT) + self.l1Loss(pred, GT)
            lossS = 0.1 * (4 - self.mssimLoss(pre, GT) + 4 - self.mssimLoss(pred, GT))
            lossD = 10 * self.GDLoss(pre, mask=1 - EG) + self.GDLoss(pred, mask=1 - EG)
            lossE = 0.001 * self.edgeLoss(edge, EG)
            lossseg =  self.segLoss(seg_pred, SE)
            

            loss = lossL + lossS + lossE + self.lambda_s * lossseg 
            loss_dict = {
                'total': loss.item(),
                'L1': lossL.item(),
                'mSSIM': lossS.item(),
                'DTV': lossD.item(),
                'edge': lossE.item(),
                'segmantation':lossseg.item()
            }

        elif args.net == "Baseline":
            OB, GT, EG = batch['OB'].cuda(), batch['GT'].cuda(), batch['EG'].cuda()
            edge, pre, res = self.net(OB)
            pred = pre + res
            lossL = self.l1Loss(pre, GT) + self.l1Loss(pred, GT)
            lossS = 0.1*(4 - self.mssimLoss(pre, GT) + 4 - self.mssimLoss(pred, GT))
            lossD = 10*self.GDLoss(pre, mask=1-EG) + self.GDLoss(pred, mask=1-EG)
            lossE = 0.001*self.edgeLoss(edge, EG)
            loss = lossL+lossS+lossE
            loss_dict = {
                'total': loss.item(),
                'L1': lossL.item(),
                'mSSIM': lossS.item(),
                'DTV': lossD.item(),
                'edge': lossE.item()
            }

        #elif args.net == "HDC_cons_seg" or args.net == "PDP_edge_refine" or args.net=="RDB_edge_refine": Edge_guided
        else:
            #OB: origin GT: smooth GT EG: edge GT SE: seg GT OV:correspongding oversmooth
            OB, GT, EG, SE, OV = batch['OB'].cuda(), batch['GT'].cuda(), batch['EG'].cuda(), batch['SE'].cuda(), batch['OV'].cuda()
            SE = SE.to(device, dtype=torch.long)
            edge, pre, res = self.net(OB)
            seg_pred = self.model(OB)
            
            pred = pre + res
            
            lossL = self.l1Loss(pre, GT) + self.l1Loss(pred, GT)
            lossS = 0.1 * (4 - self.mssimLoss(pre, GT) + 4 - self.mssimLoss(pred, GT))
            lossD = 10 * self.GDLoss(pre, mask=1 - EG) + self.GDLoss(pred, mask=1 - EG)
            lossE = 0.001 * self.edgeLoss(edge, EG)
            lossseg =  self.segLoss(seg_pred, SE)
            
            if args.useover:
                if args.bothneg:
                    rand_ind = random.random()
                    neg = OV if rand_ind<0.5 else OB
                else :
                    neg = OV
            else:
                neg = OB
            
            pos=GT

            #========================>ablation<=====================(too slow, abandon)
            '''
            lossC = 0
            if args.rate == 1:
                weight = [1]
            else:
                weight = [0.5/(args.rate-1) for x in range(args.rate-1)]
                weight.insert(0, 0.5)
            for i in range(args.rate):
                if args.triplet:
                    lossC += weight[i] * (self.constLoss1(pred, neg, pos, args.k1) + self.constLoss2(pred, neg, pos, args.k2))
                else:#CR loss
                    lossC += weight[i] * self.contrastloss(pred, pos, neg)
                neg = self.get_con(args.pathN, args.picknumber)
'''
            if args.triplet:
                lossC = self.constLoss1(pred, neg, pos, args.k1) + self.constLoss2(pred, neg, pos, args.k2)
            else:#CR loss
                lossC = self.contrastloss(pred, pos, neg)
            
            loss = lossL + lossS + lossE + self.lambda_c * lossC + self.lambda_s * lossseg 
            loss_dict = {
                'total': loss.item(),
                'L1': lossL.item(),
                'mSSIM': lossS.item(),
                'DTV': lossD.item(),
                'edge': lossE.item(),
                'contrast': lossC.item(),
                'segmantation':lossseg.item()
            }
            

        
        ssim = self.ssim(pred, GT)
        psnr = 10*torch.log10((1.0/self.mse(pred, GT)))
        if name == 'train':
            #net update parameters
            self.net.zero_grad()
            loss.backward()
            self.opt.step()
            lr_now = self.opt.param_groups[0]["lr"]
            logger.info("epoch %d/%d: step %d/%d: loss is %f ssim is %f psnr is %f lr is %f"
                        % (self.now_epoch, self.epoch, self.step, self.total_step, loss, ssim, psnr,  lr_now))
            self.step += 1

        return pred, loss_dict, ssim.item(), psnr.item()

    def save_image(self, name, img_lists):
        data, pred, label = img_lists
        pred = pred.cpu().data

        data, label, pred = data * 255, label * 255, pred * 255
        pred = np.clip(pred, 0, 255)

        h, w = pred.shape[-2:]

        gen_num = (6, 2)
        img = np.zeros((gen_num[0] * h, gen_num[1] * 3 * w, 3))
        for img_list in img_lists:
            for i in range(gen_num[0]):
                row = i * h
                for j in range(gen_num[1]):
                    idx = i * gen_num[1] + j
                    tmp_list = [data[idx], pred[idx], label[idx]]
                    for k in range(3):
                        col = (j * 3 + k) * w
                        tmp = np.transpose(tmp_list[k], (1, 2, 0))
                        img[row: row+h, col: col+w] = tmp 

        img_file = os.path.join(self.output_dir, '%d_%s.jpg' % (self.step, name))
        cv2.imwrite(img_file, img)

    def epoch_out(self):
        self.step = 0


def run_train_val(args):
    sess = Session(args)
    if args.is_resume:
        sess.load_checkpoints(args.resume_checkpoint)
    sess.tensorboard(args.sessname)
    ssim_m = 0.0
    sess.now_epoch = sess.start_epoch
    for epoch in range(int(sess.epoch-sess.start_epoch)):
        start_time = time.time()
        epoch = epoch + sess.start_epoch
        if args.net == "Baseline" :
            dt_train = sess.get_dataloader(dir=args.train_dir, dir2=args.edge_dir, name='train')
            dt_val = sess.get_dataloader(dir=args.val_dir, dir2=args.edge_dir, name='val')

        #==========================>   <================================
        elif args.net == "HDC_cons":
            dt_train = sess.get_dataloader(dir=args.train_dir, dir2=args.edge_dir, dir3=args.over_dir, name='train')
            dt_val = sess.get_dataloader(dir=args.val_dir, dir2=args.edge_dir, dir3=args.over_dir, name='val')        
        else:
        #args.net == "HDC_cons_seg" or args.net == "PDP_edge_refine" or args.net == "HDC_seg" or args.net=="RDB_edge_refine" or args.net=="Edge_guided":
            dt_train = sess.get_dataloader(dir=args.train_dir, dir2=args.edge_dir, dir3=args.seg_dir, dir4=args.over_dir, name='train')
            dt_val =sess.get_dataloader(dir=args.train_dir, dir2=args.edge_dir, dir3=args.seg_dir, dir4=args.over_dir, name='val')
        

        sess.net.train()
        losses_train = {}
        ssim_train = []
        psnr_train = []
        for i, batch in enumerate(dt_train):
            result_train, loss_dict, ssim, psnr = sess.inf_batch("train", batch)
            if i == 0:
                for key in loss_dict:
                    losses_train[key] = []
            else:
                for key, value in loss_dict.items():
                    losses_train[key].append(value)
            ssim_train.append(ssim)
            psnr_train.append(psnr)
        sess.epoch_out()
        losses_val = {}
        ssim_val = []
        psnr_val = []
        sess.net.eval()
        with torch.no_grad():
             for i, batch in enumerate(dt_val):
                result_val, loss_dict, ssim, psnr = sess.inf_batch("val", batch)
                if i == 0:
                    for key in loss_dict:
                        losses_val[key] = []
                else:
                    for key, value in loss_dict.items():
                        losses_val[key].append(value)
                ssim_val.append(ssim)
                psnr_val.append(psnr)
        total_loss_dict = {'loss_train': np.mean(losses_train['total']), 'loss_val': np.mean(losses_val['total'])}
        loss_components_dict = {}
        for k,v in losses_train.items():
            loss_components_dict[k+'_train']= np.mean(v)
        for k,v in losses_val.items():
            loss_components_dict[k+'_val'] = np.mean(v)
        sess.write(name=args.sessname, total_loss=total_loss_dict, loss_components=loss_components_dict,
                   ssim=[np.mean(ssim_train), np.mean(ssim_val)], epoch=epoch)
        ssim_now = np.mean(ssim_val)
        psnr_now = np.mean(psnr_val)
        if ssim_now > ssim_m:
            logger.info('ssim increase from %f to %f now, psnr %f' % (ssim_m, ssim_now, psnr_now))
            ssim_m = ssim_now
            sess.save_checkpoints("epoch %d_ssim %f_psnr %f " % (epoch, ssim_m,psnr_now))
            logger.info('save model as epoch_%d_ssim %f_psnr %f' % (epoch, ssim_m, psnr_now))
        elif epoch % 30 == 0 :
            sess.save_checkpoints("epoch %d_ssim %f_psnr %f " % (epoch, ssim_now, psnr_now))
            logger.info('save model as epoch_%d_ssim %f_psnr %f' % (epoch, ssim_now, psnr_now))
        else:
            logger.info("ssim now is %f, not increase from %f" % (ssim_now, ssim_m))
        sess.now_epoch += 1
        sess.sche.step(epoch=epoch)
        end_time = time.time()
        logger.info("this epoch costs time: %f" % (end_time - start_time))
    sess.write_close(args.sessname)





if __name__ == '__main__':
    log_level = 'info'
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    args = get_args()
    
    run_train_val(args=args)

