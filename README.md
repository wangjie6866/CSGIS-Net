### CSGIS-Net
Contrastive Semantic-Guided Image Smoothing Network

This is the PyTorch impplementation for our paper:

Here we provide sourse code, the setablished dataset VOC-smooth and pre-models will be released after published!

### Abstract
Image smoothing is a fundamental low-level vision task that aims to preserve salient structures of an image while removing insignificant details. However, existing smoothing efforts usually fail to achieve the trade-off between texture removal and semantic structure preservation, leading to problems of over-smoothing or under-smoothing. To address this issue, we propose a novel Contrastive Semantic-guided Image Smoothing Network (CSGIS-Net) that combines two different prior knowledge from contrastive learning and semantic information. We exploit ground-truths as positive samples, while negative samples involving unexplored over/under-smoothed images would introduce additional information to prevent the result from being trapped in negative domains. Beyond existing image smoothing wisdom, our network is capable to distinguish the over/under-smoothed domain by leveraging negative representations. Moreover, high-level semantic information can identify the texture features of images at the semantic level, allowing for the retention of weak structures, which is often ignored by existing efforts. Hence, CSGIS-Net bridges the high-level semantic information and contrastive learning paradigm together for better reconstructing the salient structures. In order to realize the proposed network, we construct a VOC-smooth dataset blended with versatile smoothing ground-truths and natural textures. Extensive experiments demonstrate that the proposed CSGIS-Net outperforms state-of-the-art algorithms by a significant margin.

### Requirements
- Python 3.7
- Pytorch >= 0.4.0
- opencv
- MATLAB for traditional algorithms


### Sources

The following sources can be downloaded fron Google drive:
- dataset : 
- trained models for ablation study:
- trained model of our method : 

#### Test
Download the trained model and put the model file in your model path.
Put your own test files in your test path.
```bash
python  show.py --modelPath MODEL_PATH --test_dir TEST_PATH --sessname YOUR_SESSNAME --net HDC_edge_refine 
````
#### Train from sratch:
##### First generate the SPS dataset
Download the ground-truth images and textures from the above links.
Put the texture pattern into 'tx' directory, and put GTs into 'VOC_GT' directory. Both directories should be under the 'dataset utils'.
```bash
cd dataset_utils
python blend&conc.py
````
then wait for the dataset generation process to complete.
Next, randomly select a subset from the generated files in 'train' for cross validation.
```bash
python get_val.py
````
Put the 'train', 'val' and 'edge' directories into datasets/YOUR_DATASET_NAME/

Then, according to the 'map.txt', rename the correspongding segmantation GT and oversmoothed image.
```bash
python rename.py
````

##### Train
Download the pre-trained segmantation network.
```bash
python train.py --sessname YOUR_SESSNAME --net HDC_cons_seg --train_dir './datasets/YOUR_DATASET_NAME/train' --val_dir './datasets/YOUR_DATASET_NAME/val' --edge_dir './datasets/YOUR_DATASET_NAME/edge' --seg_dir './datasets/YOUR_DATASET_NAME/val' --over_dir './datasets/YOUR_DATASET_NAME/edge'
````

