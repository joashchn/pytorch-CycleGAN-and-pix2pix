import os
from PIL import Image


def file_name(file_dir, file_type=''):  # 默认为文件夹下的所有文件
    lst = []
    tp_count = 0
    for dirs in os.listdir(file_dir):
        print('===========================')
        pre_1 = dirs.split('_')[0]
        print(pre_1)
        for tp in os.listdir(file_dir + '/' + dirs):
            if not os.path.isdir(file_dir + '/' + dirs + '/' + tp):
                continue
            if '前台' not in tp and '外观' not in tp:
                tp_count += 1
                print(pre_1 + '_' + str(tp_count))
                pre_fix = pre_1 + '_' + str(tp_count) + '_'
                for px in os.listdir(file_dir + '/' + dirs + '/' + tp):
                    if px == '1':
                        for f in os.listdir(file_dir + '/' + dirs + '/' + tp + '/' + px):
                            if 'jpg' in f:
                                oldname = file_dir + '/' + dirs + '/' + tp + '/' + px + '/' + f
                                newname = '/data/object_detection/joash/pix2pixHD/datasets/oyo2none' + '/trainA/' + pre_fix + f
                                print(oldname)
                                print(newname)
                                os.rename(oldname, newname)
                                img1 = Image.open(newname)
                                img1 = img1.resize((2048, 1024), Image.ANTIALIAS)
                                img1.save(
                                    '/data/object_detection/joash/pix2pixHD/datasets/oyo2none' + '/train_A/' + pre_fix + f,
                                    quality=100)
                    if px == '2':
                        for f in os.listdir(file_dir + '/' + dirs + '/' + tp + '/' + px):
                            if 'jpg' in f:
                                oldname = file_dir + '/' + dirs + '/' + tp + '/' + px + '/' + f
                                newname = '/data/object_detection/joash/pix2pixHD/datasets/oyo2none' + '/trainB/' + pre_fix + f
                                print(oldname)
                                print(newname)
                                os.rename(oldname, newname)
                                img2 = Image.open(newname)
                                img2 = img2.resize((2048, 1024), Image.ANTIALIAS)
                                img2.save(
                                    '/data/object_detection/joash/pix2pixHD/datasets/oyo2none' + '/train_B/' + pre_fix + f,
                                    quality=100)


file_dir = '/data/object_detection/joash/pic_a'
files = file_name(file_dir, 'file_type')

