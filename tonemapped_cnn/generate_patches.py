import numpy as np
import os
from PIL import Image
import imageio

# generate and save wxw image patches from the input image
# save each of these to savepath
def saveImagePatches(image, w, savepath):
    im_w,im_h= image.size
    for x in range(int(im_w/w)):
        for y in range(int(im_h/w)):
            patch = image.crop((x*w, y*w, (x+1)*w,(y+1)*w))
            patch.save(savepath + '_{}x{}.jpg'.format(x,y))

# create patches from the phos image at the given scene and exposure
def readImages(image_name, tonemapped_name):
    img = Image.open(image_name)
    img_w,img_h=img.size
    tonemapped = Image.open(tonemapped_name)
    tonemapped_w,tonemapped_h,=tonemapped.size
    print(img_w,img_h)
    print(tonemapped_w, tonemapped_h)
    cropx = min(img_w, tonemapped_w)
    cropy = min(img_h, tonemapped_h)
    img_start_x = img_w//2-cropx//2
    img_start_y = img_h//2-cropy//2
    tonemapped_start_x = tonemapped_w//2-cropx//2
    tonemapped_start_y = tonemapped_h//2-cropy//2
    img = img.crop((img_start_x,img_start_y,img_start_x+cropx,img_start_y+cropy))
    tonemapped = tonemapped.crop((tonemapped_start_x,tonemapped_start_y,tonemapped_start_x+cropx,tonemapped_start_y+cropy))
    print(img.size)
    print(tonemapped.size)
    return img, tonemapped


# generate patches for all parts of the phos dataset
image_name_list=['AgiaGalini', 'Cafe', 'Colorcheckers', 'CreteSeashore1', 'CreteSunset1', 'CreteSunset2', 'Flowers', 'FORTH1', 'FORTH2', 'FORTH3', 'FORTH4', 'Garden', 'HorseshoeLake', 'Knossos1', 'Knossos2', 'Knossos3', 'Knossos4', 'Knossos5', 'Knossos6', 'Knossos7', 'Knossos8', 'Lake1', 'LowerLewisFalls', 'MarketMires2', 'MontSaintMichel', 'Museum1', 'RevelStoke', 'StoneTower1', 'Stream', 'SwissSunset', 'TestChart1', 'Zurich2', 'Zurich']
original_image_dir = '/mnt/6344-project-data/_Images/'
original_tonemapped_dir = '/mnt/6344-project-data/_ToneMapped/'
output_image_dir = '/mnt/6344-project-data/_Image_patches/'
output_tonemapped_dir = '/mnt/6344-project-data/_ToneMapped_patches/'

if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
if not os.path.exists(output_tonemapped_dir):
        os.makedirs(output_tonemapped_dir)

for i in range(len(image_name_list)):
	print(image_name_list[i])
	image_name = original_image_dir+image_name_list[i]+'.jpg'
	tonemapped_name = original_tonemapped_dir+image_name_list[i]+'_tonemapped.jpg'
	img, tonemapped = readImages(image_name, tonemapped_name)
	saveImagePatches(img,64,output_image_dir+image_name_list[i])
	saveImagePatches(tonemapped,64,output_tonemapped_dir+image_name_list[i])

