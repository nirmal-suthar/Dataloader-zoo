#!/usr/bin/env python
# coding: utf-8

# In[2]:


from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


# In[3]:


class Celeb(Dataset):

    def __init__(self, root, transform=None):
        
        self.root = root
        self.transform = transform
        self.dataset = self.preprocess()
        self.num_images = len(self.dataset)

    def preprocess(self):
        """Can be extended for further processing dataset"""

        return ImageFolder(self, self.transform)


    def __getitem__(self, index):
        """Return input for its corresponding index."""
        return self.dataset[index]


    def __len__(self):
        """Return the number of images."""
        return self.num_images

