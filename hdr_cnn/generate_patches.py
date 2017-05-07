import numpy as np
import os
from PIL import Image
import imageio

# generate and save wxw image patches from the input image
# save each of these to savepath
def saveImagePatches(image, w, savepath):
    im_h,im_w,img_c= image.shape
    for x in range(int(im_w/w)):
        for y in range(int(im_h/w)):
            patch = image[y*w:(y+1)*w, x*w:(x+1)*w,:]
            imageio.imwrite(savepath + '_{}x{}.jpg'.format(x,y),patch)

def saveHDRPatches(image, w, savepath):
    im_h,im_w,img_c= image.shape
    for x in range(int(im_w/w)):
        for y in range(int(im_h/w)):
            patch = image[y*w:(y+1)*w, x*w:(x+1)*w,:]
            imageio.imwrite(savepath + '_{}x{}.hdr'.format(x,y),patch)

# create patches from the phos image at the given scene and exposure
def readImages(image_name, hdr_name):
    img = imageio.imread(image_name)
    img_h,img_w,img_c=img.shape
    hdr = imageio.imread(hdr_name)
    hdr_h,hdr_w,hdr_c=hdr.shape
    cropx = min(img_h, hdr_h)
    cropy = min(img_w, hdr_w)
    img_start_x = img_h//2-cropx//2
    img_start_y = img_w//2-cropy//2
    hdr_start_x = hdr_h//2-cropx//2
    hdr_start_y = hdr_w//2-cropy//2
    img = img[img_start_x:img_start_x+cropx, img_start_y:img_start_y+cropy, :]
    hdr = hdr[hdr_start_x:hdr_start_x+cropx, hdr_start_y:hdr_start_y+cropy, :]
    return img, hdr

    #savepath = 'phos\\{}\\{}\\'.format(exposure,scene)


    #savePatches(im, 32, savepath)


# generate patches for all parts of the phos dataset
image_name_list=['AgiaGalini', 'Cafe', 'Colorcheckers', 'CreteSeashore1', 'CreteSunset1', 'CreteSunset2', 'Flowers', 'FORTH1', 'FORTH2', 'FORTH3', 'FORTH4', 'Garden', 'HorseshoeLake', 'Knossos1', 'Knossos2', 'Knossos3', 'Knossos4', 'Knossos5', 'Knossos6', 'Knossos7', 'Knossos8', 'Lake1', 'LowerLewisFalls', 'MarketMires2', 'MontSaintMichel', 'Museum1', 'RevelStoke', 'StoneTower1', 'Stream', 'SwissSunset', 'TestChart1', 'Zurich2', 'Zurich']
original_image_dir = '/mnt/6344-project-data/_Images/'
original_hdr_dir = '/mnt/6344-project-data/_HDR/'
output_image_dir = '/mnt/6344-project-data/_Image_patches/'
output_hdr_dir = '/mnt/6344-project-data/_HDR_patches/'

if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
if not os.path.exists(output_hdr_dir):
        os.makedirs(output_hdr_dir)

for i in range(len(image_name_list)):
	print(image_name_list[i])
	image_name = original_image_dir+image_name_list[i]+'.jpg'
	hdr_name = original_hdr_dir+image_name_list[i]+'.hdr'
	img, hdr = readImages(image_name, hdr_name)
	saveImagePatches(img,64,output_image_dir+image_name_list[i])
	saveHDRPatches(hdr,64,output_hdr_dir+image_name_list[i])

