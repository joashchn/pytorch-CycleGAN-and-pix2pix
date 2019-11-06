import os


def file_name(file_dir, file_type=''):  # 默认为文件夹下的所有文件
    lst = []
    tp_count = 0
    for dirs in os.listdir(file_dir):
        print('===========================')
        pre_1 = dirs.split('_')[0]
        print(pre_1)
        for tp in os.listdir(file_dir + '/' + dirs):
            if '前台' not in tp and '外观' not in tp:
                tp_count += 1
                print(pre_1 + '_' + str(tp_count))
                pre_fix = pre_1 + '_' + str(tp_count) + '_'
                for px in os.listdir(file_dir + '/' + dirs + '/' + tp):
                    if px == '1':
                        for f in os.listdir(file_dir + '/' + dirs + '/' + tp + '/' + px):
                            oldname = file_dir + '/' + dirs + '/' + tp + '/' + px + '/' + f
                            newname = '/data/object_detection/joash/pytorch-CycleGAN-and-pix2pix/datasets/oyo2none' + '/trainA/' + pre_fix + f
                            print(oldname)
                            print(newname)
                            os.rename(oldname, newname)
                    if px == '2':
                        for f in os.listdir(file_dir + '/' + dirs + '/' + tp + '/' + px):
                            oldname = file_dir + '/' + dirs + '/' + tp + '/' + px + '/' + f
                            newname = '/data/object_detection/joash/pytorch-CycleGAN-and-pix2pix/datasets/oyo2none' + '/trainB/' + pre_fix + f
                            print(oldname)
                            print(newname)
                            os.rename(oldname, newname)

        tp_count = 0

file_dir = '/data/object_detection/joash/pytorch-CycleGAN-and-pix2pix/tmpfile'
files = file_name(file_dir, 'file_type')

