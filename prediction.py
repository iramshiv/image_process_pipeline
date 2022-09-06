import concurrent.futures
import pickle
import time
import os
import cv2
import tensorflow as tf

from file_ops import create, read


img_height = 180
img_width = 180
n_list = []
n_list.clear()


def predict_noise(pred_url, model_path, dir_path):
    folder_names = []
    # read class folders
    try:
        with open("./model/folder_name.txt", 'rb') as handle:
            data = handle.read()

        d = pickle.loads(data)
        l = len(d) - 1

        for x in d:
            if l != 0:
                folder = d.get(x)
                folder_names.append(folder)
                l = l - 1

            # create class folders
            for fo in folder_names:
                create(dir_path, fo)
    except:
        print('folder name : read error')

    # image prediction
    file_list1 = read(pred_url, pred_url)

    for files in file_list1:
        pred_path = pred_url + '/' + files
        img = tf.keras.utils.load_img(pred_path, target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch
        model = tf.keras.models.load_model(model_path)
        predictions = model.predict(img_array)
        class_names = predictions.argmax(axis=-1)
        # print(int(class_names))
        score = tf.nn.softmax(predictions[0])
        # Confusion Matrix
        # matrix = tf.metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
        try:
            if class_names == 0:
                img1 = cv2.imread(pred_path)
                os.replace(pred_path, dir_path + '/' + folder_names[0] + '/' + files)
                n_list.append(files)
            elif class_names == 1:
                os.replace(pred_path, dir_path + '/' + folder_names[1] + '/' + files)
            else:
                os.replace(pred_path, dir_path + '/' + folder_names[2] + '/' + files)
                n_list.append(files)
        except:
            print('Replace Error')
            # i = i + 1 print(f)''' img1 = cv2.imread(pred_path) cv2.putText(img1, "{}: {:.2f}".format(class_names,
            # 100 * np.max(score)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3) cv2.imshow("Image",
            # img1) cv2.waitKey(0)

    return n_list

#def main_run():

#        pred(f)

#if __name__ == "__main__":
#    with concurrent.futures.ProcessPoolExecutor(10) as executor:
#        start_time = time.perf_counter()
#        result = list(executor.map(main_run(), range(-3)))
#        finish_time = time.perf_counter()
#    print(f"Program finished in {finish_time - start_time} seconds")
#    print(result)
#    print('Hopefully, I will keep improving the classifier, I am just learning.')
