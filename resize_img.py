import os

# file_dir = '/data/object_detection/joash/pytorch-CycleGAN-and-pix2pix/datasets/oyo2none'
file_dir = '/Users/joash/PycharmProjects/pytorch-CycleGAN-and-pix2pix/datasets/oyo2none/'

from PIL import Image

for fi in os.listdir(file_dir + 'trainA'):

    resize_z = (256, 256)
    new_image = Image.new('RGB', (512, 256))
    print(file_dir + 'trainA/' + fi)
    if '.DS_' in (file_dir + 'trainA/' + fi) or '.DS_' in (file_dir + 'trainB/' + fi):
        continue
    img1 = Image.open(file_dir + 'trainA/' + fi)
    img1 = img1.resize(resize_z, Image.ANTIALIAS)
    try:
        img2 = Image.open(file_dir + 'trainB/' + fi)
        img2 = img2.resize(resize_z, Image.ANTIALIAS)
        new_image.paste(img1, (0, 0))
        new_image.paste(img2, (256, 0))
        new_image.save('/Users/joash/PycharmProjects/pytorch-CycleGAN-and-pix2pix/datasets/oyo2none/train13/' + fi,
                       quality=100)
    except Exception as e:
        print(e)
