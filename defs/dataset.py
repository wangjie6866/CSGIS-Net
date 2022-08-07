import os
import cv2
import numpy as np
import defs.ext_transforms as et
from numpy.random import RandomState
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from matplotlib.pyplot import *

__all__ = [
    "TrainDataset",
    "TestDataset",
    "TrainDatasetE",
    "TestDatasetE",
    "TrainDatasetS",
    "TestDatasetS",
    "TrainDatasetF",
    "TestDatasetF"
]


'''
Dataset consisting of concatenated image pairs (Ground Truth in the left and Observation in the right)
'''


class TrainDataset(Dataset):
    def __init__(self, dir, patch_size, aug_data):
        super().__init__()
        self.rand_state = RandomState()
        self.root_dir = dir
        self.mat_files = os.listdir(self.root_dir)
        self.patch_size = patch_size
        self.aug_data = aug_data
        self.file_num = len(self.mat_files)


    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        img_pair = cv2.imread(img_file).astype(np.float32) / 255
        h, w, _ = img_pair.shape

        if self.aug_data:
            O, B = self.crop(img_pair, aug=True)
            O, B = self.flip(O, B)
            O, B = self.rotate(O, B)
            O, B = self.ToPIL(O), self.ToPIL(B)
            O, B = self.hue(O, B)
            O, B = self.contrast(O, B)
            O, B = self.bright(O, B)
            O, B = self.ToCvArray(O), self.ToCvArray(B)
        else:
            O, B = self.crop(img_pair, aug=False)

        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))
        sample = {'OB': O, 'GT': B}

        return sample

    def ToPIL(self, img):
        img = (img*255).astype(np.uint8)
        img = Image.fromarray(img).convert('RGB')
        return img

    def ToCvArray(self, img):
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        return (img/255.0).astype(np.float32)

    def crop(self, img_pair, aug):
        patch_size = self.patch_size
        h, ww, c = img_pair.shape
        w = int(ww / 2)

        if aug:
            mini = - 1 / 4 * self.patch_size
            maxi =   1 / 4 * self.patch_size + 1
            p_h = patch_size + self.rand_state.randint(mini, maxi)
            p_w = patch_size + self.rand_state.randint(mini, maxi)
        else:
            p_h, p_w = patch_size, patch_size

        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)
        O = img_pair[r: r+p_h, c+w: c+p_w+w]
        B = img_pair[r: r+p_h, c: c+p_w]

        if aug:
            O = cv2.resize(O, (patch_size, patch_size))
            B = cv2.resize(B, (patch_size, patch_size))

        return O, B

    def flip(self, O, B):
        if self.rand_state.rand() > 0.5:
            O = np.flip(O, axis=1)
            B = np.flip(B, axis=1)
        return O, B

    def rotate(self, O, B):
        angle = self.rand_state.randint(-45, 45)
        patch_size = self.patch_size
        center = (int(patch_size / 2), int(patch_size / 2))
        M = cv2.getRotationMatrix2D(center, angle, 1)
        O = cv2.warpAffine(O, M, (patch_size, patch_size))
        B = cv2.warpAffine(B, M, (patch_size, patch_size))
        return O, B

    def bright(self, O, B):
        brightness_factor = self.rand_state.uniform(0.3, 1.7)
        O = TF.adjust_brightness(img=O, brightness_factor=brightness_factor)
        B = TF.adjust_brightness(img=B, brightness_factor=brightness_factor)
        return O, B

    def contrast(self, O, B):
        contrast_factor = self.rand_state.uniform(0.3, 1.7)
        O = TF.adjust_contrast(img=O, contrast_factor=contrast_factor)
        B = TF.adjust_contrast(img=B, contrast_factor=contrast_factor)
        return O, B

    def hue(self, O, B):
        hue_factor = self.rand_state.uniform(-0.3, 0.3)
        O = TF.adjust_hue(img=O, hue_factor=hue_factor)
        B = TF.adjust_hue(img=B, hue_factor=hue_factor)
        return O, B

class TrainDatasetE(Dataset):
    def __init__(self, dir, dir2, patch_size, aug_data): #dir2 for edge maps
        super().__init__()
        self.rand_state = RandomState()
        self.root_dir = dir
        self.edge_dir = dir2
        self.mat_files = os.listdir(self.root_dir)
        self.patch_size = patch_size
        self.aug_data = aug_data
        self.file_num = len(self.mat_files)
        # self.transforms = transforms.Compose([
        #     transforms.functional.adjust_contrast(),
        #     transforms.
        # ])

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        key = file_name.split('.')[0]
        key = int(key) % len(os.listdir(self.edge_dir))
        edgefile_name = str(key) + '.png'
        img_file = os.path.join(self.root_dir, file_name)
        edge_file = os.path.join(self.edge_dir, edgefile_name)
        img_pair = cv2.imread(img_file).astype(np.float32) / 255
        edge_label = cv2.imread(edge_file,0)

        edge_label[edge_label < 100] = 0
        edge_label[edge_label > 100] = 1

        h, w, _ = img_pair.shape

        if self.aug_data:
            O, B, E = self.crop(img_pair,edge_label, aug=True)
            O, B, E = self.flip(O, B, E)
            O, B, E = self.rotate(O, B, E)
            O, B = self.ToPIL(O), self.ToPIL(B)
            O, B = self.hue(O, B)
            O, B = self.contrast(O, B)
            O, B = self.bright(O, B)
            O, B = self.ToCvArray(O), self.ToCvArray(B)
        else:
            O, B, E = self.crop(img_pair,edge_label, aug=False)

        E = np.expand_dims(E, axis=2)
        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))
        E = np.transpose(E, (2, 0, 1))
        sample = {'OB': O, 'GT': B, 'EG': E}

        return sample

    def crop(self, img_pair, edge, aug):
        patch_size = self.patch_size
        h, ww, c = img_pair.shape
        w = int(ww / 2)

        if aug:
            mini = - 1 / 4 * self.patch_size
            maxi =   1 / 4 * self.patch_size + 1
            p_h = patch_size + self.rand_state.randint(mini, maxi)
            p_w = patch_size + self.rand_state.randint(mini, maxi)
        else:
            p_h, p_w = patch_size, patch_size

        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)
        O = img_pair[r: r+p_h, c+w: c+p_w+w]
        B = img_pair[r: r+p_h, c: c+p_w]
        E = edge[r: r+p_h, c: c+p_w]

        if aug:
            O = cv2.resize(O, (patch_size, patch_size))
            B = cv2.resize(B, (patch_size, patch_size))
            E = cv2.resize(E, (patch_size, patch_size))

        return O, B, E

    def bright(self, O, B):
        brightness_factor = self.rand_state.uniform(0.3, 1.7)
        O = TF.adjust_brightness(img=O, brightness_factor=brightness_factor)
        B = TF.adjust_brightness(img=B, brightness_factor=brightness_factor)
        return O, B

    def contrast(self, O, B):
        contrast_factor = self.rand_state.uniform(0.3, 1.7)
        O = TF.adjust_contrast(img=O, contrast_factor=contrast_factor)
        B = TF.adjust_contrast(img=B, contrast_factor=contrast_factor)
        return O, B

    def hue(self, O, B):
        hue_factor = self.rand_state.uniform(-0.3, 0.3)
        O = TF.adjust_hue(img=O, hue_factor=hue_factor)
        B = TF.adjust_hue(img=B, hue_factor=hue_factor)
        return O, B

    def ToPIL(self, img):
        img = (img*255).astype(np.uint8)
        img = Image.fromarray(img)
        return img

    def ToCvArray(self, img):
        img = np.asarray(img)
        return (img/255.0).astype(np.float32)
    def flip(self, O, B, E):
        if self.rand_state.rand() > 0.5:
            O = np.flip(O, axis=1)
            B = np.flip(B, axis=1)
            E = np.flip(E, axis=1)
        return O, B, E

    def rotate(self, O, B, E):
        angle = self.rand_state.randint(-45, 45)
        patch_size = self.patch_size
        center = (int(patch_size / 2), int(patch_size / 2))
        M = cv2.getRotationMatrix2D(center, angle, 1)
        O = cv2.warpAffine(O, M, (patch_size, patch_size))
        B = cv2.warpAffine(B, M, (patch_size, patch_size))
        E = cv2.warpAffine(E, M, (patch_size, patch_size))
        return O, B, E


class TestDataset(Dataset):
    def __init__(self, dir, patch_size):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = dir
        self.mat_files = os.listdir(self.root_dir)
        self.patch_size = patch_size
        self.file_num = len(self.mat_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        img_pair = cv2.imread(img_file).astype(np.float32) / 255
        h, ww, c = img_pair.shape
        O, B = self.crop(img_pair)
        B = np.transpose(B, (2, 0, 1))
        O = np.transpose(O, (2, 0, 1))
        sample = {'OB': O, 'GT': B}
        return sample

    def crop(self, img_pair):
        patch_size = self.patch_size
        h, ww, c = img_pair.shape
        w = int(ww / 2)
        p_h, p_w = patch_size, patch_size
        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)
        O = img_pair[r: r + p_h, c + w: c + p_w + w]
        B = img_pair[r: r + p_h, c: c + p_w]
        return O, B

class TestDatasetE(Dataset):
    def __init__(self, dir, dir2, patch_size):
        super().__init__()
        self.rand_state = RandomState(66)
        self.edge_dir = dir2
        self.root_dir = dir
        self.mat_files = os.listdir(self.root_dir)
        self.patch_size = patch_size
        self.file_num = len(self.mat_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        key = file_name.split('.')[0]
        key = int(key) % len(os.listdir(self.edge_dir))
        edgefile_name = str(key) + '.png'
        img_file = os.path.join(self.root_dir, file_name)
        img_pair = cv2.imread(img_file).astype(np.float32) / 255
        edge_file = os.path.join(self.edge_dir, edgefile_name)
        edge_label = cv2.imread(edge_file,0)
        #print(edge_file)

        edge_label[edge_label < 100] = 0
        edge_label[edge_label > 100] = 1
        O, B, E = self.crop(img_pair, edge_label)
        E = np.expand_dims(E, axis=2)
        B = np.transpose(B, (2, 0, 1))
        O = np.transpose(O, (2, 0, 1))
        E = np.transpose(E, (2, 0, 1))
        sample = {'OB': O, 'GT': B, 'EG': E}
        return sample

    def crop(self, img_pair, edge):
        patch_size = self.patch_size
        h, ww, c = img_pair.shape
        w = int(ww / 2)
        p_h, p_w = patch_size, patch_size
        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)
        O = img_pair[r: r + p_h, c + w: c + p_w + w]
        B = img_pair[r: r + p_h, c: c + p_w]
        E = edge[r: r + p_h, c: c + p_w]
        return O, B, E



class TrainDatasetS(Dataset):
    def __init__(self, dir, dir2, dir3, patch_size, aug_data): #dir2 for edge maps
        super().__init__()
        self.rand_state = RandomState()
        self.root_dir = dir
        self.edge_dir = dir2
        self.seg_dir = dir3
        self.mat_files = os.listdir(self.root_dir)
        self.patch_size = patch_size
        self.aug_data = aug_data
        self.file_num = len(self.mat_files)



    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        ##################
        ##################
        key = file_name.split('.')[0]                      #key: name of image
        key = int(key) % len(os.listdir(self.edge_dir))    #key: corresponding edge map (name)
        edgefile_name = str(key) + '.png'
        segfile_name = str(key) + '.png'
        img_file = os.path.join(self.root_dir, file_name)
        edge_file = os.path.join(self.edge_dir, edgefile_name)
        seg_file = os.path.join(self.seg_dir, segfile_name)
        
        img_pair = cv2.imread(img_file).astype(np.float32) / 255

        seg_label = cv2.imread(seg_file).astype(np.float32) / 255

        edge_label = cv2.imread(edge_file,0)               # read an image in grayscale mode

        edge_label[edge_label < 100] = 0
        edge_label[edge_label > 100] = 1      #binary

        h, w, _ = img_pair.shape

        if self.aug_data:
            O, B, E, S = self.crop(img_pair,edge_label,seg_label, aug=True)
            O, B, E, S = self.flip(O, B, E, S)
            O, B, E, S = self.rotate(O, B, E, S)
            O, B = self.ToPIL(O), self.ToPIL(B)
            O, B = self.hue(O, B)
            O, B = self.contrast(O, B)
            O, B = self.bright(O, B)
            O, B = self.ToCvArray(O), self.ToCvArray(B)
        else:
            O, B, E, S = self.crop(img_pair,edge_label,seg_label, aug=False)

        E = np.expand_dims(E, axis=2)   
        O = np.transpose(O, (2, 0, 1))  
        B = np.transpose(B, (2, 0, 1))
        E = np.transpose(E, (2, 0, 1))  
        S = np.transpose(S, (2, 0, 1))
        sample = {'OB': O, 'GT': B, 'EG': E, 'SE': S}

        return sample

    def crop(self, img_pair, edge, seg, aug):
        patch_size = self.patch_size
        h, ww, c = img_pair.shape
        w = int(ww / 2)

        if aug:
            mini = - 1 / 4 * self.patch_size
            maxi =   1 / 4 * self.patch_size + 1
            p_h = patch_size + self.rand_state.randint(mini, maxi)
            p_w = patch_size + self.rand_state.randint(mini, maxi)
        else:
            p_h, p_w = patch_size, patch_size

        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)
        O = img_pair[r: r+p_h, c+w: c+p_w+w]  #with texture
        B = img_pair[r: r+p_h, c: c+p_w]      #gt 
        E = edge[r: r+p_h, c: c+p_w]
        S = seg[r: r+p_h, c: c+p_w]

        if aug:
            O = cv2.resize(O, (patch_size, patch_size))
            B = cv2.resize(B, (patch_size, patch_size))
            E = cv2.resize(E, (patch_size, patch_size))
            S = cv2.resize(S, (patch_size, patch_size))

        return O, B, E, S

    def bright(self, O, B):
        brightness_factor = self.rand_state.uniform(0.3, 1.7)
        O = TF.adjust_brightness(img=O, brightness_factor=brightness_factor)
        B = TF.adjust_brightness(img=B, brightness_factor=brightness_factor)
        return O, B

    def contrast(self, O, B):
        contrast_factor = self.rand_state.uniform(0.3, 1.7)
        O = TF.adjust_contrast(img=O, contrast_factor=contrast_factor)
        B = TF.adjust_contrast(img=B, contrast_factor=contrast_factor)
        return O, B

    def hue(self, O, B):
        hue_factor = self.rand_state.uniform(-0.3, 0.3)
        O = TF.adjust_hue(img=O, hue_factor=hue_factor)
        B = TF.adjust_hue(img=B, hue_factor=hue_factor)
        return O, B

    def ToPIL(self, img):
        img = (img*255).astype(np.uint8)
        img = Image.fromarray(img)
        return img

    def ToCvArray(self, img):
        img = np.asarray(img)
        return (img/255.0).astype(np.float32)
    def flip(self, O, B, E, S):
        if self.rand_state.rand() > 0.5:
            O = np.flip(O, axis=1)
            B = np.flip(B, axis=1)
            E = np.flip(E, axis=1)
            S = np.flip(S, axis=1)
        return O, B, E, S

    def rotate(self, O, B, E, S):
        angle = self.rand_state.randint(-45, 45)
        patch_size = self.patch_size
        center = (int(patch_size / 2), int(patch_size / 2))
        M = cv2.getRotationMatrix2D(center, angle, 1)
        O = cv2.warpAffine(O, M, (patch_size, patch_size))
        B = cv2.warpAffine(B, M, (patch_size, patch_size))
        E = cv2.warpAffine(E, M, (patch_size, patch_size))
        S = cv2.warpAffine(S, M, (patch_size, patch_size))
        return O, B, E, S



class TestDatasetS(Dataset):
    def __init__(self, dir, dir2, dir3, patch_size):
        super().__init__()
        self.rand_state = RandomState(66)
        self.edge_dir = dir2
        self.root_dir = dir
        self.seg_dir = dir3
        self.mat_files = os.listdir(self.root_dir)
        self.patch_size = patch_size
        self.file_num = len(self.mat_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        key = file_name.split('.')[0]
        key = int(key) % len(os.listdir(self.edge_dir))
        edgefile_name = str(key) + '.png'
        segfile_name = str(key) + '.png'
        img_file = os.path.join(self.root_dir, file_name)
        img_pair = cv2.imread(img_file).astype(np.float32) / 255
        seg_file = os.path.join(self.seg_dir, segfile_name)
        seg_label = cv2.imread(seg_file).astype(np.float32) / 255
        edge_file = os.path.join(self.edge_dir, edgefile_name)

        edge_label = cv2.imread(edge_file,0)

        edge_label[edge_label < 100] = 0
        edge_label[edge_label > 100] = 1
        O, B, E, S = self.crop(img_pair, edge_label,seg_label)
        E = np.expand_dims(E, axis=2)
        B = np.transpose(B, (2, 0, 1))
        O = np.transpose(O, (2, 0, 1))
        E = np.transpose(E, (2, 0, 1))
        S = np.transpose(S, (2, 0, 1))
        sample = {'OB': O, 'GT': B, 'EG': E, 'SE': S}
        return sample

    def crop(self, img_pair, edge, seg):
        patch_size = self.patch_size
        h, ww, c = img_pair.shape
        w = int(ww / 2)
        p_h, p_w = patch_size, patch_size
        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)
        O = img_pair[r: r + p_h, c + w: c + p_w + w]
        B = img_pair[r: r + p_h, c: c + p_w]
        E = edge[r: r + p_h, c: c + p_w]
        S = seg[r: r + p_h, c: c + p_w]
        return O, B, E, S


class TrainDatasetF(Dataset):
    def __init__(self, dir, dir2, dir3, dir4,patch_size, aug_data): #dir2 for edge maps
        super().__init__()
        self.rand_state = RandomState()
        self.root_dir = dir
        self.edge_dir = dir2
        self.seg_dir = dir3
        self.over_dir = dir4
        self.mat_files = os.listdir(self.root_dir)
        self.patch_size = patch_size
        self.aug_data = aug_data
        self.file_num = len(self.mat_files)
        self.crop = et.ExtRandomCrop(size=self.patch_size)
        self.transf = et.ExtCompose([
            et.ExtRandomHorizontalFlip(),
            et.ExtRandomRotation(degrees=45),
            et.ExtColorJitter(brightness=0.7, contrast=0.7, saturation=0, hue=0.3),
            et.ExtToTensor(),
            #et.ExtNormalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        ##################
        ##################
        key = file_name.split('.')[0]                      #key: name of image
        key = int(key) % len(os.listdir(self.edge_dir))    #key: corresponding edge map (name)
        #name
        edgefile_name = str(key) + '.png'
        segfile_name = str(key) + '.png'
        overfile_name = str(key) + '.png'
        #path
        img_file = os.path.join(self.root_dir, file_name)
        edge_file = os.path.join(self.edge_dir, edgefile_name)
        seg_file = os.path.join(self.seg_dir, segfile_name)
        over_file = os.path.join(self.over_dir, overfile_name)
        #data
        img_pair = Image.open(img_file).convert('RGB')
        seg_label = Image.open(seg_file).convert('P')
        over_image = Image.open(over_file).convert('RGB')
        edge_label = Image.open(edge_file)  # "L gray  np.array() =>[0,255]

        O, B, E, S, V = self.crop(img_pair,edge_label,seg_label,over_image)
        O, B, E, S, V = self.transf(O, B, E, S, V)
        

        sample = {'OB': O, 'GT': B, 'EG': E, 'SE': S, 'OV': V}

        return sample



class TestDatasetF(Dataset):
    def __init__(self, dir, dir2, dir3, dir4,patch_size, aug_data): #dir2 for edge maps
        super().__init__()
        self.rand_state = RandomState()
        self.root_dir = dir
        self.edge_dir = dir2
        self.seg_dir = dir3
        self.over_dir = dir4
        self.mat_files = os.listdir(self.root_dir)
        self.patch_size = patch_size
        self.aug_data = aug_data
        self.file_num = len(self.mat_files)
        self.crop = et.ExtRandomCrop(size=self.patch_size)
        self.transf = et.ExtCompose([
            et.ExtToTensor(),
            #et.ExtNormalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        ##################
        ##################
        key = file_name.split('.')[0]                      #key: name of image
        key = int(key) % len(os.listdir(self.edge_dir))    #key: corresponding edge map (name)
        #name
        edgefile_name = str(key) + '.png'
        segfile_name = str(key) + '.png'
        overfile_name = str(key) + '.png'
        #path
        img_file = os.path.join(self.root_dir, file_name)
        edge_file = os.path.join(self.edge_dir, edgefile_name)
        seg_file = os.path.join(self.seg_dir, segfile_name)
        over_file = os.path.join(self.over_dir, overfile_name)
        #data
        img_pair = Image.open(img_file).convert('RGB')
        seg_label = Image.open(seg_file)
        over_image = Image.open(over_file).convert('RGB')
        edge_label = Image.open(edge_file)  # "L gray  np.array() =>[0,255]

        O, B, E, S, V = self.crop(img_pair,edge_label,seg_label,over_image)
        O, B, E, S, V = self.transf(O, B, E, S, V)

        
        sample = {'OB': O, 'GT': B, 'EG': E, 'SE': S, 'OV': V}

        return sample

    