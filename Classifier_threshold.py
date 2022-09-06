import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pathlib
from keras import layers
from keras.models import Sequential
from sklearn.metrics import ConfusionMatrixDisplay
import pickle
# Use to unzip
# data_dir = tf.keras.utils.get_file('Dresses', origin=dataset_url, untar=True)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True)  # dataset path
ap.add_argument("-batch", "--batch", default=64, type=int)
ap.add_argument("-h", "--height", default=180, type=int)
ap.add_argument("-w", "--width", default=180, type=int)
ap.add_argument("-e", "--epoch", default=25, type=int)
args = vars(ap.parse_args())

# Dataset directory and parameters
dataset_url = args['input']
batch_size = args['batch']
img_height = args['height']
img_width = args['width']
iteration = 0

# Check for files
data_dir = pathlib.Path(dataset_url)
image_count = len(list(data_dir.glob('*/*.jpg')))
if image_count > 0:
    print(image_count)
else:
    print('Error reading files')
    exit()


# Split the dataset train and test
def split(percent, subset):
    ds = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=percent, subset=subset, seed=123,
                                                     image_size=(img_height, img_width), batch_size=batch_size)
    return ds


train_ds = split(0.2, 'training')
val_ds = split(0.2, 'validation')

# Class Names and batch shapes
class_names = train_ds.class_names
print(class_names)
v_class_names = val_ds.class_names
print(v_class_names)

try:
    iteration = ['iteration', '1']
    my_dict = {}
    i = 0

    for class_name in class_names:
        i = i + 1
        my_dict['folder_' + str(i)] = class_name

    my_dict['iteration'] = '0'

    file = open("./model/folder_name.txt", "wb")
    # print(my_dict)
    pickle.dump(my_dict, file)
    file.close()

    with open("./model/folder_name.txt", 'rb') as handle:
        data = handle.read()

    d = pickle.loads(data)
    iteration = int(d.get("iteration"))

except:
    print('Folder file creation error')

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

# Autotune
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Standardize
normalization_layer = layers.Rescaling(1. / 255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

num_classes = len(class_names)

# Model
model = Sequential([
    layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

# Compile
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

# Fit
epochs = args['epoch']
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

iteration = str(iteration + 1)
file = open("./model/folder_name.txt", "wb")
my_dict["iteration"] = iteration
pickle.dump(my_dict, file)
file.close()

save_path = "./model/model_multinoise_"+iteration+".h5"
model.save(save_path)

# Analysis
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
model.evaluate(val_ds)

# Confusion Matrix
y_pred = model.predict(val_ds)
pred_cat = tf.argmax(y_pred, axis=1)
true_cat = tf.concat([y for x, y in val_ds], axis=0)
display = ConfusionMatrixDisplay.from_predictions(pred_cat, true_cat, display_labels=v_class_names)
plt.show()


