# Some tags in final_train.txt may not have corresponding pictures.
# This short codes are used to check whether there are corresponding synthetic pictures in final_train.txt tags.
# If some tags do not exist, it will delete them and generate a new txt named synthetic_train.txt.

import os

# the path of final_train.txt
checked_txt_path = '/home/juling/PycharmProjects/deep_underwater_localization/data/my_data/final_train.txt'
# the path of synthetic images
syn_imgs_path = '/home/juling/deep_localization/synthetic/images/'
# save path for new txt file after checked images and labels
save_path = '/home/juling/PycharmProjects/deep_underwater_localization/data/my_data/'

txt = open(os.path.join(save_path, 'synthetic_train.txt'), 'w')

cnt_all = 0
cnt_exist = 0
cnt_no_exist = 0

syn_imgs_list = os.listdir(syn_imgs_path)
print('There are %s images in synthetic dataset.' % len(syn_imgs_list))

with open(checked_txt_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        cnt_all += 1
        img_index = line.split()[0]
        source_path = line.split()[1]
        img_name = source_path.split('/')[-1]
        remain = line.split()[2:]

        if img_name in syn_imgs_list:
            cnt_exist += 1
            txt.write(img_index + ' ' + syn_imgs_path + img_name + ' ' + ' '.join(str(i) for i in remain) + '\n')
        else:
            cnt_no_exist += 1
            # print('Don't find %s in the synthetic folder!' % img_name)

    # print('final_train.txt points to %s tagged pictures in total.' % cnt_all)
    print('final_train.txt has %s tags with corresponding images in the synthetic dataset.' % cnt_exist)
    # print('cnt_no_exist: %s' % cnt_no_exist)
    print('Finish writing synthetic_train.txt！')





