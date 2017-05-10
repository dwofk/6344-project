import cv2
import os

os_slash = '/'
source_dir = 'inputs_large'+os_slash
dest_dir = 'inputs'+os_slash
height = 2000
width = 3000

for directory in os.listdir(source_dir):
    for imfile in os.listdir(source_dir + directory):
        im_in = cv2.imread(source_dir+directory+os_slash+imfile)
        im_out = cv2.resize(im_in, (width, height))
        cv2.imwrite(dest_dir+directory+os_slash+imfile,im_out)
        print("{} written".format(dest_dir+directory+os_slash+imfile))
