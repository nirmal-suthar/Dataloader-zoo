#!/usr/bin/env python
# coding: utf-8

# In[3]:


import glob, os
from PIL import Image
from torch.utils.data import Dataset


# In[4]:


class ImageDataset(Dataset):
    def __init__(self, root, transform=None, mode='train'):
        self.transform = transform
        self.files_A = sorted(glob.glob(os.path.join(root, '{}A'.format(mode)) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '{}B'.format(mode)) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

