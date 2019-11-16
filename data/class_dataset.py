"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, ImageFolder
from torchvision import datasets
import os
from PIL import Image
import random


class ClassDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        # parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        self.data_dir = opt.dataroot
        self.dir_train = os.path.join(opt.dataroot, 'train')
        self.dir_val = os.path.join(opt.dataroot, 'val')
        self.train_path_A = sorted(make_dataset(self.dir_train+'/ants', opt.max_dataset_size))
        self.train_path_B = sorted(make_dataset(self.dir_train+'/bees', opt.max_dataset_size))
        self.val_path_A = sorted(make_dataset(self.dir_val+'/ants', opt.max_dataset_size))
        self.val_path_B = sorted(make_dataset(self.dir_val+'/bees', opt.max_dataset_size))
        self.train_size = len(self.train_path_A)

        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        if opt.model_name == 'inception':
            opt.crop_size = 299
        else:
            opt.crop_size = 224
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        # self.dataset_dic = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), self.transform[x]) for x in
        #                   ['train', 'val']}
        #
        # return self.dataset_dic

        train_path_A = self.train_path_A[index]
        train_path_B = self.train_path_B[index]
        val_path_A = self.val_path_A[index]
        val_path_B = self.val_path_B[index]

        train_A_img = Image.open(train_path_A).convert('RGB')
        train_B_img = Image.open(train_path_B).convert('RGB')
        val_A_img = Image.open(val_path_A).convert('RGB')
        val_B_img = Image.open(val_path_B).convert('RGB')

        train_A = self.transform(train_A_img)
        train_B = self.transform(train_B_img)
        val_A = self.transform(val_A_img)
        val_B = self.transform(val_B_img)

        return {'train_A': train_A, 'train_B': train_B, 'val_A': val_A, 'val_B': val_B}

    def __len__(self):
        """Return the total number of images."""
        return self.train_size
