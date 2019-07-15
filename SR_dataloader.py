#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T


# In[2]:


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


# In[3]:


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


# In[4]:


def train_HR_transform(crop_size):
    return T.Compose([ T.RandomCrop(crop_size, pad_if_needed=True), T.ToTensor() ])


# In[5]:


def train_LR_transform(crop_size, upscale_factor):
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(crop_size // upscale_factor, interpolation = Image.BICUBIC),
        T.ToTensor(),
    ])
    
    return transform

# def display_transform():
#     return T.Compose([
#         T.ToPILImage(),
#         T.Resize(400),
#         T.CenterCrop(400),
#         T.ToTensor()
#     ])


# In[6]:


class dataset_train_from_folder(Dataset):
    
    """
    args: 
    dataset_dir: (str) directory of datset
    crop_size: (int) image size of HR
    upscale_factor: (int) upscaling factor for processing LR from HR
    
    returns: 
    LR_image: tensor of size (crop_size // upscale_factor, crop_size // upscale_factor)
    HR_restore_image:  tensor of size (crop_size, crop_size)
    HR_image: tensor of size (crop_size, crop_size)
    """
    
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        
        super().__init__()
        
        self.image_filename = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.HR_transform = train_HR_transform(crop_size) 
        self.LR_transform = train_LR_transform(crop_size, upscale_factor)
        
    
    def __getitem__(self, idx):
        
        HR_image = self.HR_transform(Image.open(self.image_filename[idx]))
        LR_image = self.LR_transform(HR_image)
        
        return LR_image, HR_image
    
    def __len__(self):
        return len(self.image_filename)


# In[7]:


class dataset_val_from_folder(Dataset):
    
    """
    args: 
    dataset_dir: (str) directory of datset
    crop_size: (int) image size of HR
    upscale_factor: (int) upscaling factor for processing LR from HR
    
    returns: 
    LR_image: tensor of size (crop_size // upscale_factor, crop_size // upscale_factor)
    HR_restore_image:  tensor of size (crop_size, crop_size)
    HR_image: tensor of size (crop_size, crop_size)
    """
    
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        
        super().__init__()
        
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.upscale_factor = upscale_factor
        
    def __getitem__(self, idx):
        
        image = Image.open(self.image_filenames[idx])
        w, h = image.size
        crop_size = calculate_valid_crop_size(min(w,h) , self.upscale_factor)
        LR_scale = T.Resize(crop_size // self.upscale_factor, interpolation = Image.BICUBIC)
        HR_scale = T.Resize(crop_size, interpolation = Image.BICUBIC)
        HR_image = T.CenterCrop(crop_size)(image) 
        LR_image = LR_scale(HR_image)
        HR_restore_image = HR_scale(LR_image)
        
        return T.ToTensor()(LR_image), T.ToTensor()(HR_restore_image), T.ToTensor()(HR_image)
    
    def __len__(self):
        return len(self.image_filenames)
   


# In[8]:


class dataset_test_from_folder(Dataset):
    
    """
    args: 
    dataset_dir: (str) directory of datset
    upscale_factor: (int) upscaling factor for processing LR from HR
    
    returns: 
    LR_image: tensor of size (h, w)
    HR_restore_image:  tensor of size (h * upscale_factor, w * upscale_factor)
    HR_image: tensor of size (h * upscale_factor, w * upscale_factor)
    """
    
    def __init__(self, dataset_dir, upscale_factor):
        
        super().__init__()
        
        self.LR_path = dataset_dir + '/data/'
        self.HR_path = dataset_dir + '/target/'
        self.upscale_factor = upscale_factor
        self.LR_filenames = [join(self.LR_path, x) for x in listdir(self.LR_path) if is_image_file(x)]
        self.HR_filenames = [join(self.HR_path, x) for x in listdir(self.HR_path) if is_image_file(x)]

    def __getitem__(self, index):
        
        image_name = self.lr_filenames[index].split('/')[-1]
        LR_image = Image.open(self.lr_filenames[index])
        w, h = LR_image.size
        HR_image = Image.open(self.HR_filenames[index])
        HR_scale = T.Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        HR_restore_img = hr_scale(LR_image)
        return image_name, ToTensor()(LR_image), ToTensor()(HR_restore_img), ToTensor()(HR_image)

    def __len__(self):
        return len(self.LR_filenames)

