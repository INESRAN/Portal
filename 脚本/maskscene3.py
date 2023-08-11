import numpy as np
from glob import glob
from PIL import Image
import os
import cv2 as cv
from os.path import join as pjoin

os.system("rm -rf /home/yangzhihui/f2nerf1/exp/sanchangjingmask/*")

data_dir1 = '/home/yangzhihui/f2nerf1/exp/sanchangjinglanmen'
img_list1 = sorted(glob(pjoin(data_dir1, '*.png')))
imgs1 = []
for img_path in img_list1:
    imgs1.append(cv.cvtColor(cv.imread(img_path),cv.COLOR_BGR2RGB))


for n in range(len(imgs1)):
    imgs_in1 = imgs1[n]
    masks = 255 - np.logical_and(imgs_in1[:, :, 0] < 50, imgs_in1[:, :, 1] < 50, imgs_in1[:, :, 2] > 50)*255
    masks = masks.astype(np.uint8)
    masks = masks[:,:, np.newaxis]
    imgs_in1 = np.concatenate((imgs_in1,masks),axis=2)
    new_img = Image.new('RGBA', (imgs_in1.shape[1], imgs_in1.shape[0]))
    new_img.paste(Image.fromarray(imgs_in1,'RGBA'), (0,0), mask=Image.fromarray(imgs_in1,'RGBA'))
    new_img.save(f"/home/yangzhihui/f2nerf1/exp/sanchangjingmask/{n:05}.png")

