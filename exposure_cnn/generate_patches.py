import numpy as np
import os
from PIL import Image

def savePatches(image, w, savepath):
    im_h, im_w = image.size
    for x in range(int(im_w/w)):
        for y in range(int(im_h/w)):
            patch = image.crop((y*w, x*w, (y+1)*w, (x+1)*w))
            patch.save(savepath + '{}x{}.bmp'.format(x,y))

def genPhosPatches(scene, exposure):
    phos_path = 'C:\\Users\\vysarge\\Documents\\hdr_dataset\\Phos2_3MP\\Phos2_scene{}\\'.format(scene)
    phos_name = 'Phos2_uni_sc{}_{}.png'.format(scene, exposure)
    im = Image.open(phos_path + phos_name)
    print(im.size)

    savepath = 'phos\\{}\\{}\\'.format(exposure,scene)
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    savePatches(im, 32, savepath)


for scene in range(2,16):
    genPhosPatches(scene, '0')
    genPhosPatches(scene, 'plus_2')
    genPhosPatches(scene, 'minus_2')

