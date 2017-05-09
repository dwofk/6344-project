import numpy as np
import os
import cv2
import PIL
from PIL import Image

# generate and save wxw image patches from the input image
# save each of these to savepath
def savePatches(image, w, savepath):
    im_h, im_w = image.size
    for x in range(int(im_w/w)):
        for y in range(int(im_h/w)):
            patch = image.crop((y*w, x*w, (y+1)*w, (x+1)*w))
            patch.save(savepath + '{}x{}.bmp'.format(x,y))

# create patches from the phos image at the given scene and exposure
def genPhosPatches(scene, exposure):
    phos_path = 'C:\\Users\\vysarge\\Documents\\hdr_dataset\\Phos2_3MP\\Phos2_scene{}\\'.format(scene)
    phos_name = 'Phos2_uni_sc{}_{}.png'.format(scene, exposure)
    im = Image.open(phos_path + phos_name)
    print(im.size)

    savepath = 'phos\\{}\\{}\\'.format(exposure,scene)
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    savePatches(im, 32, savepath)

def genEmpaPatches(scene, exposure):
    empa_path = 'C:\\Users\\vysarge\\Documents\\hdr_dataset\\empa\\training\\'
    empa_file = empa_path + scene + '\\{}.JPG'.format(exposure)
    im = Image.open(empa_file)
    print(im.size)
    h, w = im.size
    im = im.resize((int(h/2), int(w/2)), PIL.Image.ANTIALIAS)
    print(im.size)

    #savepath = 'empa\\{}\\{}\\'.format(exposure,scene)
    savepath = 'C:\\Users\\vysarge\\Documents\\hdr_dataset\\empapatches\\{}\\{}\\'.format(exposure, scene)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    savePatches(im, 32, savepath)
    
    print(savepath + ' filled')

# generate patches for all parts of the phos dataset
'''
for scene in range(2,16):
    genPhosPatches(scene, '0')
    genPhosPatches(scene, 'plus_4')
    genPhosPatches(scene, 'minus_')
'''


empa_path = 'C:\\Users\\vysarge\\Documents\\hdr_dataset\\empa\\training\\'
for directory in os.listdir(empa_path):
    for exposure in ['0', 'minus_4', 'plus_4']:
    #empa_file = empa_path + directory + '\\{}.JPG'.format(exposure)
        genEmpaPatches(directory, exposure)
