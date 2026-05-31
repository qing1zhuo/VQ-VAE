import os
import os.path as path
import json
import torch
import torch.utils.data as data
import numpy as np
import random
from PIL import Image
import pdb
import csv
#torch.multiprocessing.set_sharing_strategy('file_system')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def gray_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('P')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)





class SiameseminiImageNet(object):
    """
       Dataloader for miniImageNet_ravi dataset in Siamese Network.
       Total classes: 100
       Train classes: 64
       Val classes:   16
       Test classes:  20
    """

    def __init__(self, data_dir="./", mode="train", image_size=84,
                 transform=None, loader=default_loader, gray_loader=gray_loader):
        
        super(SiameseminiImageNet, self).__init__()

        # set the paths of the csv files
        train_csv = os.path.join(data_dir, 'train.csv')
        val_csv = os.path.join(data_dir, 'val.csv')
        test_csv = os.path.join(data_dir, 'test.csv')


        # construct the positive and negative pairs
        if mode == 'train':

            # store all the classes and images into a dict
            class_img_dict = {}
            with open(train_csv) as f_csv:
                f_train = csv.reader(f_csv, delimiter=',')
                for row in f_train:
                    if f_train.line_num == 1:
                        continue
                    img_name, img_class = row

                    if img_class in class_img_dict:
                        class_img_dict[img_class].append(img_name)
                    else:
                        class_img_dict[img_class]=[]
                        class_img_dict[img_class].append(img_name)

            class_list = class_img_dict.keys()


            # construct pairs
            train_pair_list = []
            for index1, class_item1 in enumerate(class_list):
                temp_imgs = class_img_dict[class_item1]

                # positive pairs
                pos_list = []
                for i in range(len(temp_imgs)):
                    img1 = temp_imgs[i]

                    for j in range(i+1,len(temp_imgs)):
                        img2 = temp_imgs[j]

                        data_file = {
                            "img1": class_item1+'/'+img1,
                            "img2": class_item1+'/'+img2,
                            "target": 1
                        }

                        pos_list.append(data_file)

                # negative pairs
                neg_list = []
                pos_pair_num = len(pos_list)
                per_num = round(pos_pair_num/(len(class_list)-1))

                for index2, class_item2 in enumerate(class_list, index1+1):
                    temp_imgs2 = class_img_dict[class_item2]
                    count = 0
                    for img1 in temp_imgs:
                        for img2 in temp_imgs2:
                            count = count+1
                            data_file = {
                                "img1": img1,
                                "img2": img2,
                                "target": 0
                            }

                            neg_list.append(data_file)
                    if count>per_num:
                        break

                # combine the positive and negnative pairs
                train_pair_list.extend(pos_list)
                train_pair_list.extend(neg_list)


            self.data_list = train_pair_list
            self.image_size = image_size
            self.transform = transform
            self.loader = loader
            self.gray_loader = gray_loader
            self.data_dir = data_dir   

        elif mode == 'val':

            # store all the classes and images into a dict
            class_img_dict = {}
            with open(val_csv) as f_csv:
                f_val = csv.reader(f_csv, delimiter=',')
                for row in f_val:
                    if f_val.line_num == 1:
                        continue
                    img_name, img_class = row

                    if img_class in class_img_dict:
                        class_img_dict[img_class].append(img_name)
                    else:
                        class_img_dict[img_class]=[]
                        class_img_dict[img_class].append(img_name)

            class_list = class_img_dict.keys()


            # construct pairs
            val_pair_list = []
            for index1, class_item1 in enumerate(class_list):
                temp_imgs = class_img_dict[class_item1]

                # positive pairs
                pos_list = []
                for i in range(len(temp_imgs)):
                    img1 = temp_imgs[i]

                    for j in range(i+1,len(temp_imgs)):
                        img2 = temp_imgs[j]

                        data_file = {
                            "img1": class_item1+'/'+img1,
                            "img2": class_item1+'/'+img2,
                            "target": 1
                        }

                        pos_list.append(data_file)

                # negative pairs
                neg_list = []
                pos_pair_num = len(pos_list)
                per_num = round(pos_pair_num/(len(class_list)-1))

                for index2, class_item2 in enumerate(class_list, index1+1):
                    temp_imgs2 = class_img_dict[class_item2]
                    count = 0
                    for img1 in temp_imgs:
                        for img2 in temp_imgs2:
                            count = count+1
                            data_file = {
                                "img1": img1,
                                "img2": img2,
                                "target": 0
                            }

                            neg_list.append(data_file)
                    if count>per_num:
                        break

                # combine the positive and negnative pairs
                val_pair_list.extend(pos_list)
                val_pair_list.extend(neg_list)


            self.data_list = val_pair_list
            self.image_size = image_size
            self.transform = transform
            self.loader = loader
            self.gray_loader = gray_loader
            self.data_dir = data_dir

        elif mode == 'test'
            # store all the classes and images into a dict
            class_img_dict = {}
            with open(test_csv) as f_csv:
                f_test = csv.reader(f_csv, delimiter=',')
                for row in f_test:
                    if f_test.line_num == 1:
                        continue
                    img_name, img_class = row

                    if img_class in class_img_dict:
                        class_img_dict[img_class].append(img_name)
                    else:
                        class_img_dict[img_class]=[]
                        class_img_dict[img_class].append(img_name)

            class_list = class_img_dict.keys()


            # construct pairs
            test_pair_list = []
            for index1, class_item1 in enumerate(class_list):
                temp_imgs = class_img_dict[class_item1]

                # positive pairs
                pos_list = []
                for i in range(len(temp_imgs)):
                    img1 = temp_imgs[i]

                    for j in range(i+1,len(temp_imgs)):
                        img2 = temp_imgs[j]

                        data_file = {
                            "img1": img1,
                            "img2": img2,
                            "target": 1
                        }

                        pos_list.append(data_file)

                # negative pairs
                neg_list = []
                pos_pair_num = len(pos_list)
                per_num = round(pos_pair_num/(len(class_list)-1))

                for index2, class_item2 in enumerate(class_list, index1+1):
                    temp_imgs2 = class_img_dict[class_item2]
                    count = 0
                    for img1 in temp_imgs:
                        for img2 in temp_imgs2:
                            count = count+1
                            data_file = {
                                "img1": img1,
                                "img2": img2,
                                "target": 0
                            }

                            neg_list.append(data_file)
                    if count>per_num:
                        break

                # combine the positive and negnative pairs
                test_pair_list.extend(pos_list)
                test_pair_list.extend(neg_list)


            self.data_list = test_pair_list
            self.image_size = image_size
            self.transform = transform
            self.loader = loader
            self.gray_loader = gray_loader
            self.data_dir = data_dir

        else: 
            print('Wrong mode!')


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        '''
            Load an episode each time, including C-way K-shot and Q-query           
        '''
        image_size = self.image_size
        data_file = self.data_list[index]
        data_dir = self.data_dir

        # load img1 & img2
        img1_path = os.path.join(data_dir, 'images', data_file['img1'])
        img2_path = os.path.join(data_dir, 'images', data_file['img2'])

        img1 = self.loader(img1_path)
        img2 = self.loader(img2_path)

        # Normalization
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        target = data_file['target']

        return (img1, img2, target)