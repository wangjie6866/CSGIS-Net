import os
import cv2
import torch
import os
import numpy as np
import argparse
import math
from torch.nn import MSELoss
from defs import *


def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--modelPath", default="/1T/WJ/Easy2Hard-master/outfiles/SPS1/epoch 144_ssim 0.886298_psnr 28.479137 ")
    parser.add_argument("--modelPath", default="/1T/WJ/Easy2Hard-master/outfiles/SPS_triplet_over/epoch 149_ssim 0.890724_psnr 28.756027 ")
    parser.add_argument("--net", default="HDC_edge_refine")
    parser.add_argument("--sessname", default="SPSsub_cons(over)")
    parser.add_argument("--test_dir", default="./dataset_utils/test_dataset/NKS/data")
    parser.add_argument("--output_dir",default="./test_results/NKS")
    parser.add_argument("--if_crop",default = False)
    return parser.parse_args()

def cal_psnr(img1, img2):
   mse = np.mean((img1/1.0 - img2/1.0) ** 2)
   if mse < 1.0e-10:
      return 100
   return 10 * math.log10(1.0**2/mse)

def get_image(image):
    image = image*[255]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def load_checkpoints(dir):
    ckp_path = dir
    try:
        obj = torch.load(ckp_path)
        print('Load checkpoint %s' % ckp_path)
        return obj
    except FileNotFoundError:
        print('No checkpoint %s!!' % ckp_path)
        return

def run_test(input_dir, outout_dir, args):
    with torch.no_grad():
        ssim = SSIM().cuda()
        mse = MSELoss().cuda()
        if args.net == "Baseline":
            net = Baseline(in_c=3, out_c=3, dim=64, num_block=12).cuda()
        elif args.net == "HDC":
            net = HDC(in_c=3, out_c=3, dim=64, num_block=20).cuda()
        elif args.net == "HDC_edge":
            net = HDC_edge(in_c=3, out_c=3, dim=64, num_block=20).cuda()
        elif args.net == "HDC_edge_refine":
            net = HDC_edge_refine(in_c=3, out_c=3, dim=64, num_block=20).cuda()
        #elif args.net == "HDC_edge_seg" or args.net == "HDC_edge_seg_cons":
            #net = HDC_edge_seg(in_c=3, out_c=3, dim=64, num_block=20).cuda()
        elif args.net == "PDP_edge_refine":
            net = PDP_edge_refine(in_c=3, out_c=3, dim=64, num_block=12).cuda()
        elif args.net == "Edge_guided":
            net = Edge_guided(in_c=3, out_c=3, dim=64, num_block=12).cuda()
        else:
            print("NET name error!!")
        net.eval()
        obj = load_checkpoints(args.modelPath)
        net.load_state_dict(obj['net'])


        for image_name in os.listdir(input_dir):
            image_file = os.path.join(input_dir, image_name)

            image_o = (cv2.imread(str(image_file))/255.0).astype(np.float32)
            h, w, c = image_o.shape
            #
            re_w = int(w/2)
            #print(re_w) NKS no need
            if args.if_crop:
                image_o = image_o[0:h, re_w:w]
            #
            #print(image_o.shape)
            #

            image_o = np.transpose(image_o, (2, 0, 1))


            image_o = torch.from_numpy(np.expand_dims(image_o, axis=0)).type(torch.FloatTensor).cuda()
            #image_seg
            #seg_file = os.path.join(seg_dir, image_name)
            #image_s = (cv2.imread(str(seg_file))/255.0).astype(np.float32)
            #image_s = np.transpose(image_s, (2, 0, 1))
            #image_s = torch.from_numpy(np.expand_dims(image_s, axis=0)).type(torch.FloatTensor).cuda()



            if args.net == "HDC_edge":
                edge, result = net(image_o)
            elif args.net == "HDC_edge_refine" or args.net == "PDP_edge_refine" or args.net =="Edge_guided":
                edge, result, res = net(image_o)
                result = result+res
            #elif args.net == "HDC_edge_seg" or args.net == "HDC_edge_seg_cons":
                #edge, result, res = net(image_o, image_s)
                #result = result+res 

            else:
                result = net(image_o)
            result = result.cpu().detach().numpy()
            result = np.transpose(result[0], (1, 2, 0))
            result = get_image(result)
            cv2.imwrite(outout_dir + "/%s" % image_name, result)
            #psnr = cal_psnr(result,)

if __name__ == '__main__':
    args = get_args()

    input_dir = args.test_dir
    outout_dir = os.path.join(args.output_dir, args.sessname)
    os.makedirs(outout_dir, exist_ok=True)
    run_test(input_dir, outout_dir, args)
