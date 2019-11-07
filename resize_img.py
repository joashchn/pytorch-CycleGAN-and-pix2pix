import os

file_dir = '/data/object_detection/joash/pytorch-CycleGAN-and-pix2pix/datasets/oyo2none'
# file_dir = '/Users/joash/PycharmProjects/pytorch-CycleGAN-and-pix2pix/datasets/oyo2none/'

from PIL import Image

for fi in os.listdir(file_dir + 'train'):
    # print(fi)
    resize_z = (512, 256)
    print(file_dir + 'train/' + fi)
    if '.DS_' in (file_dir + 'train/' + fi):
        continue
    img1 = Image.open(file_dir + 'train/' + fi)
    img = img1.resize(resize_z, Image.ANTIALIAS)
    img.save('/data/object_detection/joash/pytorch-CycleGAN-and-pix2pix/datasets/oyo2none/train12/' + fi, quality=100)
