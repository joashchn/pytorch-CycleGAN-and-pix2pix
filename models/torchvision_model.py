import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn as nn
from torchvision import models
import torch.optim as optim


class TorchVisionModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['train']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        self.opt.model_names = ['Model_FT']

        # # define networks
        # self.model_ft = None

        if opt.model_name == "resnet":
            """ Resnet18
     """
            self.model_ft = models.resnet18(pretrained=opt.use_pretrained)
            self.set_requires_grad(self.model_ft, opt.feature_extract)
            num_ftrs = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(num_ftrs, opt.num_classes)

        elif opt.model_name == "alexnet":
            """ Alexnet
     """
            self.model_ft = models.alexnet(pretrained=opt.use_pretrained)
            self.set_requires_grad(self.model_ft, opt.feature_extract)
            num_ftrs = self.model_ft.classifier[6].in_features
            self.model_ft.classifier[6] = nn.Linear(num_ftrs, opt.num_classes)

        elif opt.model_name == "vgg":
            """ VGG11_bn
     """
            self.model_ft = models.vgg11_bn(pretrained=opt.use_pretrained)
            self.set_requires_grad(self.model_ft, opt.feature_extract)
            num_ftrs = self.model_ft.classifier[6].in_features
            self.model_ft.classifier[6] = nn.Linear(num_ftrs, opt.num_classes)

        elif opt.model_name == "squeezenet":
            """ Squeezenet
     """
            self.model_ft = models.squeezenet1_0(pretrained=opt.use_pretrained)
            self.set_requires_grad(self.model_ft, opt.feature_extract)
            self.model_ft.classifier[1] = nn.Conv2d(512, opt.num_classes, kernel_size=(1, 1), stride=(1, 1))
            self.model_ft.num_classes = opt.num_classes

        elif opt.model_name == "densenet":
            """ Densenet
     """
            self.model_ft = models.densenet121(pretrained=opt.use_pretrained)
            self.set_requires_grad(self.model_ft, opt.feature_extract)
            num_ftrs = self.model_ft.classifier.in_features
            self.model_ft.classifier = nn.Linear(num_ftrs, opt.num_classes)

        elif opt.model_name == "inception":
            """ Inception v3
     Be careful, expects (299,299) sized images and has auxiliary output
     """
            self.model_ft = models.inception_v3(pretrained=opt.use_pretrained)
            self.set_requires_grad(self.model_ft, opt.feature_extract)
            # 处理辅助网络
            num_ftrs = self.model_ft.AuxLogits.fc.in_features
            self.model_ft.AuxLogits.fc = nn.Linear(num_ftrs, opt.num_classes)
            # 处理主要网络
            num_ftrs = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(num_ftrs, opt.num_classes)

        else:
            print("Invalid model name, exiting...")
            exit()

        # 观察所有参数都在优化
        self.optimizer_ft = optim.SGD(self.model_ft.parameters(), lr=0.001, momentum=0.9)

        self.criterion = nn.CrossEntropyLoss()

    def set_input(self, inputs):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.train_A = inputs['train_A'].to(self.device)
        self.train_B = inputs['train_B'].to(self.device)
        self.val_A = inputs['val_A'].to(self.device)
        self.val_B = inputs['val_B'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.pred_train_A = self.model_ft(self.train_A)
        self.pred_train_B = self.model_ft(self.train_B)
        self.pred_val_A = self.model_ft(self.val_A)
        self.pred_val_B = self.model_ft(self.val_B)

    def backward(self):
        loss_A = self.criterion(self.pred_train_A, torch.ones([1], dtype=torch.long))
        loss_B = self.criterion(self.pred_train_B, torch.zeros([1], dtype=torch.long))
        self.loss_train = (loss_A + loss_B) * 0.5
        self.loss_train.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # backward
        self.optimizer_ft.zero_grad()
        self.backward()
        self.optimizer_ft.step()
