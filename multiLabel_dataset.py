#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image


# In[6]:


class multiLabel_dataset(Dataset):
    """Dataset class for the multiLabel dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs = None, crop_size = 128, mode = 'train'):
        
        """
        Initialize and preprocess the dataset.
        
        args
        image_dir: (str) path for inputs
        attr_path: (str) path of file containing attributes label
            format of attr path :- 
            first line: number of images
            ex. 202599
            second line: names of all atrributes seperated by commas
            ex. 5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs
            remaining lines: name of image and then int (1 : present, -1 : not present) seperated by commas 
            ex. 000001.jpg -1  1  1 -1 -1 -1
        selected_attrs: (list, optional) list of atttributes for processing labels. Default, all attributes
                        in attributes file will be use to process labels.
        
        crop_size: (int, optional)
        mode: ({'train','test'}, optional)
        
        returns: 
        one image and corresponding one hot vector
        
        """
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        
        transform = []
        # to be extended for data augmentation
        
#         if mode == 'train':
#             transform.append(T.RandomHorizontalFlip())
        
        transform.append(T.CenterCrop(crop_size))
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)
            
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()
        
        self.transform = transform

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        
        if self.selected_attrs == None: 
            self.selected_attrs = all_attr_names
        
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images

