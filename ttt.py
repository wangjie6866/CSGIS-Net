import math
import os
from torch.nn import MSELoss
import cv2
import argparse
from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--sessname", default="SPS1")
    parser.add_argument("--test_dir", default="./test_files")
    parser.add_argument("--pred_dir", default="", help="the smoothing results dir")
    parser.add_argument("--gt_dir", default="", help="the gt dir")
    parser.add_argument("--title", default="")
    parser.add_argument("--dataset", default="NKS", help="NKS or VOC")
    return parser.parse_args()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):

    window = create_window(window_size, 3)

    # if img1.is_cuda:
    #     window = window.cuda(img1.get_device())
    # window = window.type_as(img1)
    #
    return _ssim(img1, img2, window, window_size, 3, size_average)


def cal_psnr(img1, img2):
   mse = np.mean((img1/1.0 - img2/1.0) ** 2)
   if mse < 1.0e-10:
      return 100
   return 10 * math.log10(1.0**2/mse)

def get_image(image):
    image = image*[255]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image



if __name__ == '__main__':
    args = get_args()
    mse = MSELoss()
    SSim = SSIM()
    psnr=[]
    ssim=[]
    pred_path = args.pred_dir
    gt_path = args.gt_dir
    ind=0
    mapping={}
    for img in os.listdir(pred_path):
        mapping[img]=ind
        ind = ind+1
        image_file = os.path.join(pred_path,img)
        #nks
        if args.dataset == "NKS":
            gt_name = img.split('_')[0]
            gt_file = os.path.join(gt_path,gt_name)+'.png'
        #voc
        elif args.dataset == "VOC":
            gt_name = int(img.split('.')[0]) % 150
            gt_file = os.path.join(gt_path,str(gt_name))+'.jpg'

        image = (cv2.imread(str(image_file))/255.0).astype(np.float32)

        gt = (cv2.imread(str(gt_file))/255.0).astype(np.float32)

        image = np.transpose(image,(2,0,1))
        gt = np.transpose(gt,(2,0,1))

        image = torch.from_numpy(np.expand_dims(image, axis=0)).type(torch.FloatTensor)
        gt = torch.from_numpy(np.expand_dims(gt, axis=0)).type(torch.FloatTensor)
        p=10*torch.log10((1.0/mse(image, gt)))
        psnr.append(p)
        ssim.append(SSim(image, gt))
        print('1')

    print(sum(psnr) / len(psnr))
    print(sum(ssim) / len(ssim))

    filename = 'result.txt'

    with open(filename,'a') as f:

        f.write(args.title + "   psnr:")
        f.write(str(sum(psnr) / len(psnr)))
        f.write("    ssim:")
        f.write(str(sum(ssim) / len(ssim))+'\n')
        





