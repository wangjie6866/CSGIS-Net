# PNLS — Matlab Implementation 
This repository contains the Matlab implementation of the following paper:
> **Pixel-level Non-local Image Smoothing with Objective Evaluation**
>
> **Abstract:** With the rapid development of image processing techniques, image smoothing has gained increasing attention due to its important role in other image processing tasks, e.g., image editing and enhancement.
However, the evaluation of image smoothing methods is subjectively performed on datasets without proper ground truth images. Therefore, an image smoothing benchmark with reasonable ground-truths is essential to prosper the image smoothing community.
In this paper, we construct a new Nankai Smoothing (NKS) dataset containing $200$ versatile images blended by natural textures and structure images.
The structure images are inherent smooth and can be safely taken as ground truths. On our NKS dataset, we comprehensively evaluate 14 popular image smoothing algorithms.
Moreover, we propose a novel Pixel-level Non-Local Smoothing (PNLS) method, exploiting better the non-local self-similarity of natural images to well preserve the structure of the smoothed images. Extensive experiments on several benchmark datasets demonstrate that our PNLS is very effective on the image smoothing task.
Comprehensive ablation studies also reveal the work mechanism of our PNLS on image smoothing.To further show its effectiveness, we apply the proposed PNLS on semantic region smoothing, detail/edge enhancement, and image abstraction.  

This code is written by Yingkun Hou (https://github.com/njusthyk1972).

## Contents
- [Resources](https://github.com/zal0302/PNLS#resources)
- [System requirements](https://github.com/zal0302/PNLS#system-requirements)
- [NKS dataset](https://github.com/zal0302/PNLS#nks-dataset)
- [Benchmarking Image Smoothing on our NKS dataset](https://github.com/zal0302/PNLS#benchmarking-image-smoothing-on-our-nks-dataset)
- [Visual Results](https://github.com/zal0302/PNLS#visual-results)
- [Citation](https://github.com/zal0302/PNLS#citation)

## Resources
Material related to our paper is available via the following links:
- Code: https://github.com/zal0302/PNLS
- NKS dataset (Baidu Netdisk): https://pan.baidu.com/s/1bd0TlpZV1Nl71ziSYwG2Gg (**Extracted Code: ect6**)
- NKS dataset (Google Drive): https://drive.google.com/drive/folders/1rsWLc7kpyM2VfGwY_Gu94TH2bnt3ywfV?usp=sharing

## System requirements
Both Linux and Windows are supported.

## NKS dataset
We observe that vector images can be safely regarded as smooth structure images, and construct our NKS dataset by blending vector images and texture images. To generate mixed structure and texture images, we blend each of the 20 structure images and each of the 10 natural textures in a reasonable manner. Each structure image can be safely taken as the ground truth for the corresponding images blended by that structure image and the 10 natural textures. We mix the structure vector images and natural textures. 
<div align=center><img src="https://github.com/zal0302/PNLS/blob/master/figs/data.png" width="1000"  /></div>
Examples of our NKS dataset.
<div align=center><img src="https://github.com/zal0302/PNLS/blob/master/figs/gt.png" width="1000"  /></div>
The 20 structure images we used in NKS dataset.
<div align=center><img src="https://github.com/zal0302/PNLS/blob/master/figs/texture.png" width="1000"  /></div>
The 10 natural texture images we used in NKS dataset.

## Benchmarking Image Smoothing on our NKS dataset
<div align=center><img src="https://github.com/zal0302/PNLS/blob/master/figs/benchmark.jpg" width="650"  /></div>

## Visual Results
<div align=center><img src="https://github.com/zal0302/PNLS/blob/master/figs/nks.png" width="1000"  /></div>
Comparison of smoothed images and PSNR(dB)/SSIM/FSIM results by different methods on the image S15T1 from our NKS dataset.
<div align=center><img src="https://github.com/zal0302/PNLS/blob/master/figs/div.png" width="1000"  /></div>
Comparison of smoothed images by different methods on the image 0117 from DIV2K dataset.
<div align=center><img src="https://github.com/zal0302/PNLS/blob/master/figs/RTV.jpg" width="1000"  /></div>
Comparison of smoothed images by different methods on the image 11_11 from RTV dataset.
<div align=center><img src="https://github.com/zal0302/PNLS/blob/master/figs/500.png" width="1000"  /></div>
Comparison of smoothed images by different methods on the image 0334 from 500images dataset.

## Citation
If you find the code helpful in your resarch or work, please cite the following paper.
```
@ARTICLE{9257021,
  author={J. {Xu} and Z. -A. {Liu} and Y. {Hou} and X. {Zhen} and L. {Shao} and M. -M. {Cheng}},
  journal={IEEE Transactions on Multimedia}, 
  title={Pixel-level Non-local Image Smoothing with Objective Evaluation}, 
  year={2020},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMM.2020.3037535}}
```
