import cv2
import os
import numpy as np

file_dir = '/data/object_detection/joash/pytorch-CycleGAN-and-pix2pix/datasets/oyo2none'

for fi in os.listdir(file_dir + '/' + 'trainA'):
    # resize_z = (256, 256)

    try:
        img1 = cv2.imread(file_dir + 'trainA/' + fi)
        img2 = cv2.imread(file_dir + 'trainB/' + fi)
        img = np.hstack([img1, img2])
        cv2.imwrite('/data/object_detection/joash/pytorch-CycleGAN-and-pix2pix/datasets/oyo2none/train/' + fi, img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    except Exception as e:
        print(str(e))

# from PIL import Image
# for fi in os.listdir(file_dir + '/' + 'trainA'):
#     # print(fi)
#     resize_z = (256, 256)
#     img1 = Image.open(file_dir + 'trainA/' + fi)
#     img2 = Image.open(file_dir + 'trainB/' + fi)
#     img1.resize(resize_z, Image.ANTIALIAS).save('/Users/joash/PycharmProjects/pytorch-CycleGAN-and-pix2pix/datasets/train12/1/' + fi)
#     img2.resize(resize_z, Image.ANTIALIAS).save('/Users/joash/PycharmProjects/pytorch-CycleGAN-and-pix2pix/datasets/train12/2/' + fi)
    # img = np.hstack([img1, img2])
    # cv2.imwrite('/Users/joash/PycharmProjects/pytorch-CycleGAN-and-pix2pix/datasets/train12/' + fi, img,
    #             [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    # img.save(tmppath)
