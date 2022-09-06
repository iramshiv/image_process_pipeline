import argparse
import os
import pickle

from add_border import add_bord
from file_ops import create, delete, read_border
from file_ops import read
from file_ops import save

from crop import crop
from prediction import predict_noise
from upscale import upscale
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input", required=True)  # dataset path
ap.add_argument("-t1", "--mint", default=240, type=int)  # cropping min threshold
ap.add_argument("-t2", "--maxt", default=255, type=int)  # cropping max threshold
ap.add_argument("-p", "--percentage", default=10, type=int)  # add y-axis border after cropping
ap.add_argument("-x", "--xborder", default=1, type=int)  # add x-axis border after cropping
ap.add_argument("-y", "--yborder", default=1, type=int)  # add y-axis border after cropping
ap.add_argument("-s", "--size", default=1024, type=int)  # output size for scaling
ap.add_argument("-m", "--model_predict", default="./model/model_multinoise_1.h5",
                type=str)  # model path for threshold prediction

args = vars(ap.parse_args())

dir_path = args['input']

mint = args['mint']
maxt = args['maxt']
percent_data = args['percentage']
d_mint = mint
xb = args['xborder']
yb = args['yborder']

size = args['size']

model_path = args['model_predict']

crop_fo_name = 'cropped'
border_fo_name = 'bordered'
upscale_fo_name = 'upscale'

folder_names = ['cropped', 'bordered', 'upscale']
for fol in folder_names:
    create(dir_path, fol)


def crop_images(d_path, d1_path, min_t, max_t, x_b, y_b):
    # crop objects from images
    f_list = read(d_path, d1_path)
    for file in f_list:
        try:
            file_path = os.path.join(d1_path, file)
            cropped_image = crop(file_path, min_t, max_t, x_b, y_b)
            save(cropped_image, file, folder_names[0], 1, d1_path)
        except:
            print(f'Error cropping {file_path} ')
            continue


crop_folder_path = dir_path + '/cropped'


def prediction(prediction_url, prediction_model_path, di_path):
    return predict_noise(prediction_url, prediction_model_path, di_path)


# calculate number of total images
file_total = len([name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name))])


def re_prediction(mint):
    noise_folder_path = crop_folder_path + '/noise'
    noise_total = len(
        [name for name in os.listdir(noise_folder_path) if os.path.isfile(os.path.join(noise_folder_path, name))])
    # print('noise total: ' + str(noise_total), 'file total: ' + str(file_total))
    percentage_noise = int((noise_total / file_total) * 100)

    while percentage_noise > percent_data:
        mint = mint - 5
        # print('mint: ' + str(mint))
        crop_images(noise_folder_path, dir_path, mint, maxt, xb, yb)
        no_list = prediction(crop_folder_path, model_path, crop_folder_path)
        delete(noise_folder_path, no_list)
        noise_total = len(
            [name for name in os.listdir(noise_folder_path) if os.path.isfile(os.path.join(noise_folder_path, name))])
        # print('noise total: ' + str(noise_total), 'file total: ' + str(file_total))
        percentage_noise = int((noise_total / file_total) * 100)


border_folder_path = dir_path + '/bordered'


def add_border():
    fo = '/bordered'
    crop_file_list = read_border(crop_folder_path, dir_path, fo)
    for fil in crop_file_list:
        try:
            border_image = add_bord(fil)
            head, tail = os.path.split(fil)
            fo_name = os.path.basename(head)
            save(border_image, tail, fo_name, 2, border_folder_path)
        except:
            print('File not found')
            continue

upscale_folder_path = dir_path + '/upscale'


def upscale_1():
    f01 = '/upscale'
    border_file_list = read_border(border_folder_path, dir_path, f01)
    for fol1 in border_file_list:
        try:
            head1, tail1 = os.path.split(fol1)
            upscale_image = upscale(size, fol1)
            upscal_fo_name = os.path.basename(head1)
            save(upscale_image, tail1, upscal_fo_name, 1, upscale_folder_path)
        except:
            print('err')
            continue


crop_images(dir_path, dir_path, mint, maxt, xb, yb)
prediction(crop_folder_path, model_path, crop_folder_path)
re_prediction(mint)
add_border()
upscale_1()