import cv2
from PIL import Image, ImageOps

border = (100, 100, 100, 100)
color = 'white'


def add_bord(image):
    img = Image.open(image)
    if isinstance(border, int) or isinstance(border, tuple):
        b_img = ImageOps.expand(img, border=border, fill=color)
    else:
        print(f'Error bordering file : {image}')
    return b_img