import os
import cv2


def create(d_path, fo_name):
    if not os.path.isdir(d_path + fo_name):
        try:
            os.makedirs(os.path.join(d_path, fo_name))
        except:
            print('Folder name:' + fo_name + ' already exists')


files = []


# print('Start size' + str(len(files)))


def read(path, d_path):
    files.clear()
    for f_name in os.listdir(path):
        if os.path.isfile(os.path.join(d_path, f_name)):
            files.append(f_name)
    # print('End Size:  ' + str(len(files)))
    return files


def delete(path, flist):
    try:
        for f in flist:
            fpath = os.path.join(path, f)
            if os.path.isfile(fpath):
                os.remove(fpath)
                # print("File removed")
    except:
        print('File not exists')


def save(image, f_name, fo_name, opt, d_path):
    try:
        fo = os.path.join(d_path, fo_name)
        #print(fo)
        path_save = os.path.join(fo, f_name)
        if opt == 1:
            cv2.imwrite(path_save, image)
        else:
            image.save(path_save)
    except:
        print('save error')

    # print(os.path.join(fo, fname))


def read_border(crop_folder_path, d_path, fo):
    files.clear()
    sub_folders = [name for name in os.listdir(crop_folder_path) if os.path.isdir(os.path.join(crop_folder_path, name))]
    folder = [x for x in sub_folders if 'noise' not in x]

    for x1 in folder:
        create(d_path + fo, x1)
        path = crop_folder_path + '/' + x1
        for f_name in os.listdir(path):
            if os.path.isfile(os.path.join(path, f_name)):
                files.append(os.path.join(path, f_name))
    return files
