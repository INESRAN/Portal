import numpy as np
from glob import glob
from PIL import Image
import os
import cv2 as cv
from os.path import join as pjoin


os.system("rm -rf /home/yangzhihui/f2nerf1/outputs/scenemp4/*")
os.system("rm -rf /home/yangzhihui/f2nerf1/exp/catIM1/*")

data_dir1 = '/home/yangzhihui/f2nerf1/exp/sanchangjing/test/novel_images'
img_list1 = sorted(glob(pjoin(data_dir1, '*.png')))
imgs1 = []
for img_path in img_list1:
    imgs1.append(cv.cvtColor(cv.imread(img_path, cv.IMREAD_UNCHANGED), cv.COLOR_BGRA2RGBA))

data_dir2 = '/home/yangzhihui/f2nerf1/exp/sanchangjingnomen'
img_list2 = sorted(glob(pjoin(data_dir2, '*.png')))
imgs2 = []
for img_path in img_list2:
    imgs2.append(cv.cvtColor(cv.imread(img_path, cv.IMREAD_UNCHANGED), cv.COLOR_BGRA2RGBA))

data_dir3 = '/home/yangzhihui/f2nerf1/exp/sanchangjing_copy/test/novel_images'
img_list3 = sorted(glob(pjoin(data_dir3, '*.png')))
imgs3 = []
for img_path in img_list3:
    imgs3.append(cv.cvtColor(cv.imread(img_path, cv.IMREAD_UNCHANGED), cv.COLOR_BGRA2RGBA))

data_dir4 = '/home/yangzhihui/f2nerf1/exp/sanchangjingmask'
img_list4 = sorted(glob(pjoin(data_dir4, '*.png')))
imgs4 = []
for img_path in img_list4:
    imgs4.append(cv.cvtColor(cv.imread(img_path, cv.IMREAD_UNCHANGED), cv.COLOR_BGRA2RGBA))

for n in range(len(imgs1)):
    imgs_in1 = imgs1[n]
    imgs_in2 = imgs2[n]
    imgs_in3 = imgs3[n]
    imgs_in4 = imgs4[n]
    
    new_img = Image.new('RGBA', (imgs_in3.shape[1], imgs_in3.shape[0]))
    new_img.paste(Image.fromarray(imgs_in3,'RGBA'))
    new_img.paste(Image.fromarray(imgs_in4,'RGBA'), (0,0), mask=Image.fromarray(imgs_in4,'RGBA'))
    new_img.paste(Image.fromarray(imgs_in2,'RGBA'), (0,0), mask=Image.fromarray(imgs_in2,'RGBA'))
    new_img.paste(Image.fromarray(imgs_in1,'RGBA'), (0,0), mask=Image.fromarray(imgs_in1,'RGBA'))
    new_img.save(f"/home/yangzhihui/f2nerf1/exp/catIM1/{n:04}.png")

os.system("python image2video.py --data_dir /home/yangzhihui/f2nerf1/exp/catIM1/ --output /home/yangzhihui/f2nerf1/outputs/scenemp4/addscene.mp4")

