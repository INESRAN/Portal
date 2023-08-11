import numpy as np
from glob import glob
from PIL import Image
import os
import cv2 as cv
from os.path import join as pjoin


os.system("rm -rf /home/yangzhihui/f2nerf1/outputs/scenemp4/*")
os.system("rm -rf /home/yangzhihui/f2nerf1/exp/catIMG/*")
os.system("rm -rf /home/yangzhihui/f2nerf1/exp/catIMG2/*")

data_dir1 = '/home/yangzhihui/f2nerf1/exp/scene1'
img_list1 = sorted(glob(pjoin(data_dir1, '*.png')))
imgs1 = []
for img_path in img_list1:
    imgs1.append(cv.cvtColor(cv.imread(img_path),cv.COLOR_BGR2RGB))

data_dir2 = '/home/yangzhihui/f2nerf1/exp/scene2'
img_list2 = sorted(glob(pjoin(data_dir2, '*.png')))
imgs2 = []
for img_path in img_list2:
    imgs2.append(cv.cvtColor(cv.imread(img_path),cv.COLOR_BGR2RGB))

data_dir3 = '/home/yangzhihui/f2nerf1/exp/scene3'
img_list3 = sorted(glob(pjoin(data_dir3, '*.png')))
imgs3 = []
for img_path in img_list3:
    imgs3.append(cv.cvtColor(cv.imread(img_path),cv.COLOR_BGR2RGB))

for n in range(len(imgs1)):
    imgs_in1 = imgs1[n]
    masks = np.logical_and(imgs_in1[:, :, 0] > 1, imgs_in1[:, :, 1] > 1, imgs_in1[:, :, 2] > 1)*255
    masks = masks.astype(np.uint8)
    masks = masks[:,:, np.newaxis]
    imgs_in1 = np.concatenate((imgs_in1,masks),axis=2)


    imgs_in2 = imgs2[n]
    masks2 = np.logical_and(imgs_in2[:, :, 0] > 10, imgs_in2[:, :, 1] > 10, imgs_in2[:, :, 2] > 10)*255
    masks2 = masks2.astype(np.uint8)
    masks2 = masks2[:,:, np.newaxis]
    imgs_in2 = np.concatenate((imgs_in2,masks2),axis=2)

    imgs_in3 = imgs3[n]
    mask3 = np.ones((imgs_in3.shape[0], imgs_in3.shape[1], 1),np.uint8)*255
    imgs_in3 = np.concatenate((imgs_in3,mask3),axis=2)


    new_img = Image.new('RGBA', (imgs_in1.shape[1], imgs_in1.shape[0]))
    new_img.paste(Image.fromarray(imgs_in3,'RGBA'))
    new_img.paste(Image.fromarray(imgs_in2,'RGBA'), (0,0), mask=Image.fromarray(imgs_in2,'RGBA'))
    new_img.save(f"/home/yangzhihui/f2nerf1/exp/catIMG2/{n:05}.png")
    new_img.paste(Image.fromarray(imgs_in1,'RGBA'), (0,0), mask=Image.fromarray(imgs_in1,'RGBA'))
    new_img.save(f"/home/yangzhihui/f2nerf1/exp/catIMG/{n:04}.png")

os.system("python image2video.py --data_dir /home/yangzhihui/f2nerf1/exp/catIMG/ --output /home/yangzhihui/f2nerf1/outputs/scenemp4/addscene.mp4")

