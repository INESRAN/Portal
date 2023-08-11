import numpy as np
from glob import glob
from PIL import Image
import os
import cv2 as cv
from os.path import join as pjoin


os.system("rm -rf /home/yangzhihui/f2nerf1/outputs/scenemp4/*")

data_dir1 = '/home/yangzhihui/f2nerf1/exp/scene'
img_list1 = sorted(glob(pjoin(data_dir1, '*.png')))
imgs1 = []
for img_path in img_list1:
    imgs1.append(cv.cvtColor(cv.imread(img_path),cv.COLOR_BGR2RGB))

data_dir2 = '/home/yangzhihui/f2nerf1/exp/scene2'
img_list2 = sorted(glob(pjoin(data_dir2, '*.png')))
imgs2 = []
for img_path in img_list2:
    imgs2.append(cv.cvtColor(cv.imread(img_path),cv.COLOR_BGR2RGB))

for n in range(len(imgs1)):
    imgs_in1 = imgs1[n]
    masks = 255 - np.logical_and(imgs_in1[:, :, 0] > 100, imgs_in1[:, :, 1] < 50, imgs_in1[:, :, 2] < 50)*255
    masks = masks.astype(np.uint8)
    masks = masks[:,:, np.newaxis]
    imgs_in1 = np.concatenate((imgs_in1,masks),axis=2)

    imgs_in2 = imgs2[n]
    mask2 = np.ones((imgs_in1.shape[0], imgs_in1.shape[1], 1),np.uint8)*255
    imgs_in2 = np.concatenate((imgs_in2,mask2),axis=2)

    # print(imgs_in1[:,:,0].shape)
    # new_img = Image.new('RGBA', imgs_in1[:,:,0].shape)
    new_img = Image.new('RGBA', (imgs_in1.shape[1], imgs_in1.shape[0]))
    new_img.paste(Image.fromarray(imgs_in2,'RGBA'))
    new_img.paste(Image.fromarray(imgs_in1,'RGBA'), (0,0), mask=Image.fromarray(imgs_in1,'RGBA'))
    new_img.save(f"/home/yangzhihui/f2nerf1/exp/catIMG/{n:05}.png")
    # outimg = Image.fromarray(imgs_in1,'RGBA')
    # outimg.save(f"/home/yangzhihui/f2nerf1/exp/catIMG/{n:04}.png")
os.system("python image2video.py --data_dir /home/yangzhihui/f2nerf1/exp/catIMG/ --output /home/yangzhihui/f2nerf1/outputs/scenemp4/addscene.mp4")

