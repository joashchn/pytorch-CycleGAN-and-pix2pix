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
        self.loss_names = ['loss_model_ft']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        self.opt.model_names = ['Model_FT']

        # define networks
        self.model_ft = None
        self.input_size = 0

        if opt.model_name == "resnet":
            """ Resnet18
     """
            self.model_ft = models.resnet18(pretrained=opt.use_pretrained)
            self.set_requires_grad(self.model_ft, opt.feature_extract)
            num_ftrs = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(num_ftrs, opt.num_classes)
            self.input_size = 224

        elif opt.model_name == "alexnet":
            """ Alexnet
     """
            self.model_ft = models.alexnet(pretrained=opt.use_pretrained)
            self.set_requires_grad(self.model_ft, opt.feature_extract)
            num_ftrs = self.model_ft.classifier[6].in_features
            self.model_ft.classifier[6] = nn.Linear(num_ftrs, opt.num_classes)
            self.input_size = 224

        elif opt.model_name == "vgg":
            """ VGG11_bn
     """
            self.model_ft = models.vgg11_bn(pretrained=opt.use_pretrained)
            self.set_requires_grad(self.model_ft, opt.feature_extract)
            num_ftrs = self.model_ft.classifier[6].in_features
            self.model_ft.classifier[6] = nn.Linear(num_ftrs, opt.num_classes)
            self.input_size = 224

        elif opt.model_name == "squeezenet":
            """ Squeezenet
     """
            self.model_ft = models.squeezenet1_0(pretrained=opt.use_pretrained)
            self.set_requires_grad(self.model_ft, opt.feature_extract)
            self.model_ft.classifier[1] = nn.Conv2d(512, opt.num_classes, kernel_size=(1, 1), stride=(1, 1))
            self.model_ft.num_classes = opt.num_classes
            self.input_size = 224

        elif opt.model_name == "densenet":
            """ Densenet
     """
            self.model_ft = models.densenet121(pretrained=opt.use_pretrained)
            self.set_requires_grad(self.model_ft, opt.feature_extract)
            num_ftrs = self.model_ft.classifier.in_features
            self.model_ft.classifier = nn.Linear(num_ftrs, opt.num_classes)
            self.input_size = 224

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
            self.input_size = 299

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
        # AtoB = self.opt.direction == 'AtoB'
        # self.real_A = input['A' if AtoB else 'B'].to(self.device)
        # self.real_B = input['B' if AtoB else 'A'].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.input = inputs.to(self.device)
        # self.train = input['train'].to(self.device)
        # self.val = input['val'].to(self.device)


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        # self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
        # self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        # self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))
        self.pred = self.model_ft(self.input)

    def backward(self):
        loss = self.criterion(self.pred, )

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()


        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # #G_A and G_B
        # self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        # self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        # self.backward_G()  # calculate gradients for G_A and G_B
        # self.optimizer_G.step()  # update G_A and G_B's weights
        # # D_A and D_B
        # self.set_requires_grad([self.netD_A, self.netD_B], True)
        # self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        # self.backward_D_A()  # calculate gradients for D_A
        # self.backward_D_B()  # calculate graidents for D_B
        # self.optimizer_D.step()  # update D_A and D_B's weights

        self.optimizer_ft.zero_grad()
        self.backward()
        self.optimizer_ft.step()