from PIL import Image
import os
import numpy as np


def center_crop(image, crop_h=108, crop_w=108):
    if crop_w is None:
        crop_w = crop_h
    w, h = image.size
    top = int(round((h - crop_h)/2.))
    left = int(round((w - crop_w)/2.))
    right = left + crop_w
    bottom = top + crop_h
    
    image = image.crop((left, top, right, bottom))
    image = image.resize([64, 64], Image.ANTIALIAS)

    return image  


def prepro_image(img_folder, resized_folder):

    if not os.path.exists(resized_folder):
        os.makedirs(resized_folder)

    print ('Start resizing images.')
    image_files = os.listdir(img_folder)
    num_images = len(image_files)
    for i, image_file in enumerate(image_files):
        with open(os.path.join(img_folder, image_file), 'r+b') as f:
            with Image.open(f) as image:
                image = center_crop(image)
                image.save(os.path.join(resized_folder, image_file), image.format)
        if i % 100 == 0:
            print ('Resized images: %d/%d' %(i, num_images))

            
def main():
    img_folder = 'data/img_align_celeba/'
    resized_folder = 'data/celeb_resized/'
    prepro_image(img_folder, resized_folder)
   

if __name__ == '__main__':
    main()