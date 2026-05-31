import os
import csv
import glob
import random
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import matplotlib.pyplot as plt

# ===================================================================
# 音频相关依赖
# soundfile  : 读 wav/flac/ogg, 不依赖 TorchCodec / FFmpeg (torchaudio 2.11+ 的 load 需要它们)
# torchaudio   : 重采样 (Resample) 与 μ-law 编解码 (functional)
# ===================================================================
import soundfile as sf
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as AT

def get_CIFAR(bs):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    train_dataset=datasets.CIFAR10(
        root=r"..\data",
        train=True,
        transform=transform,
        download=True
    )
    test_dataset=datasets.CIFAR10(
        root=r"..\data",
        train=False,
        transform=transform,
        download=True
    )

    train_loader=DataLoader(train_dataset,bs,shuffle=True)
    test_loader=DataLoader(test_dataset,bs)
    return train_loader,test_loader


class MiniImageNetDataset(Dataset):
    """
    读取 Ravi split 版 miniImageNet 的简单数据集
    CSV 每行: filename,label  (例如 n0153282900000005.jpg, n01532829)
    所有图片都直接位于 images_dir 下
    """
    def __init__(self,csv_file,images_dir,transform=None,class_to_idx=None):
        self.images_dir=images_dir
        self.transform=transform
        self.samples=[]
        with open(csv_file,"r",newline="") as f:
            reader=csv.reader(f)
            next(reader,None)
            for row in reader:
                if len(row)<2:
                    continue
                filename,label=row[0],row[1]
                self.samples.append((filename,label))

        if class_to_idx is None:
            classes=sorted({lbl for _,lbl in self.samples})
            self.class_to_idx={c:i for i,c in enumerate(classes)}
        else:
            self.class_to_idx=class_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        filename,class_name=self.samples[idx]
        img_path=os.path.join(self.images_dir,filename)
        with open(img_path,"rb") as f:
            with Image.open(f) as img:
                img=img.convert("RGB")
        if self.transform is not None:
            img=self.transform(img)
        label=self.class_to_idx[class_name]
        return img,label


def get_MiniImageNet(bs,image_size=84,num_workers=0):
    transform=transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_root=r"..\data\miniImageNet--ravi"
    images_dir=os.path.join(data_root,"images")
    train_csv=os.path.join(data_root,"train.csv")
    val_csv=os.path.join(data_root,"val.csv")
    test_csv=os.path.join(data_root,"test.csv")

    train_dataset=MiniImageNetDataset(train_csv,images_dir,transform=transform)
    val_dataset=MiniImageNetDataset(val_csv,images_dir,transform=transform)
    test_dataset=MiniImageNetDataset(test_csv,images_dir,transform=transform)

    train_loader=DataLoader(train_dataset,bs,shuffle=True,num_workers=num_workers)
    val_loader=DataLoader(val_dataset,bs,num_workers=num_workers)
    test_loader=DataLoader(test_dataset,bs,num_workers=num_workers)
    return train_loader,val_loader,test_loader


class IndicesDataset(Dataset):
    def __init__(self, indices_tensor):
        self.indices = indices_tensor
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # 直接返回一个indices样本，形状：(H, W)
        return self.indices[idx]


