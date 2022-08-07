'''
generate dataset by blending given groundtruth domain and texture patterns in './tx' to './train'
simultaneously yield edge map of GTs to './edge'
'''
import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

tx_dir = "./tx"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bg_dir", type=str, default="./VOC_GT",
                        help="(Your) Ground Truth Domain.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    output_dir = '/1T/datasets/VOC_SPS/train'
    edge_dir = './vocedge'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(edge_dir, exist_ok=True)
    ind = 0
    f = open('map.txt','w+')
    for ii, t in enumerate(os.listdir(tx_dir)):  #t=brick11.jpg
        input_tx = os.path.join(tx_dir, t)       #input_tx = full(brick11.jpg)
        tx = cv2.imread(input_tx).astype(np.float32)  #tx:image of texture
        tx = cv2.cvtColor(tx, cv2.COLOR_RGB2GRAY)
        ht, wt = tx.shape
        for b in os.listdir(args.bg_dir):  # b: the location of original image
            input_bg = os.path.join(args.bg_dir, b)  #input_bg: the location of original image
            bg = cv2.imread(input_bg).astype(np.float32)  #bg: image of gt
            '''adjust the size of texture layer'''
            h, w, c = bg.shape
            tt = np.zeros((h, w, c))
            hh, hhh = h // ht, h % ht
            ww, www = w // wt, w % wt
            txn = np.zeros((h, w))
            for i in range(hh):
                for j in range(ww):
                    txn[i * ht:i * ht + ht, j * wt:j * wt + wt] = tx
                txn[i * ht:i * ht + ht, ww * wt:] = tx[:, :www]
            for i in range(ww):
                txn[(h - hhh):, i * wt:i * wt + wt] = tx[:hhh, :]
            txn[(h - hhh):, (w - www):] = tx[:hhh, :www]
            '''adjust color shift'''
            mean_txn = np.mean(txn)
            txn = txn - mean_txn
            '''texture blending'''
            for i in range(3):
                tt[:, :, i] = bg[:, :, i] + txn
            imc = np.concatenate([bg, tt], axis=1)  # tt: the mixture of two images
            output_img = os.path.join(output_dir, str(ind)+".png")
            #print(ind)
            #assert 1==0
            cv2.imwrite(output_img, imc)
            if ii == 0 :
                key = b.split('.')[0]
                f.write(key+','+str(ind)+'\n')
                edge = cv2.addWeighted(cv2.convertScaleAbs(cv2.Sobel(bg[:,:,0], cv2.CV_16S,1,0)),0.5,cv2.convertScaleAbs(cv2.Sobel(bg[:,:,0], cv2.CV_16S,0,1)),0.5,0)
                edge[edge<50] = 0
                edge[edge!=0] = 255
                edge_img = os.path.join(edge_dir, str(ind)+".png")
                cv2.imwrite(edge_img, edge)
            ind += 1
        if ii==0 :
            f.close()
        print("finish %s, No. %d" % (t, ind))



