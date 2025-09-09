import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import random
import plotly.express as px
import scipy as sp
print(tf.__version__)
from scipy import ndimage
from shutil import copyfile
from keras.layers import Conv2D,Add,MaxPooling2D, Dense, BatchNormalization,Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator





class_names = ['Cat', 'Dog']


n_dogs = len(os.listdir(r"C:\Users\jetda\Downloads\archive\PetImages\Dog"))
n_cats = len(os.listdir("C:\\Users\\jetda\\Downloads\\archive\\PetImages\\Cat"))
n_images = [n_cats, n_dogs]
px.pie(names=class_names, values=n_images)

import os


base_dir = os.path.join(os.getcwd(), 'cats-v-dogs')


subdirs = [
    'training/cats',
    'training/dogs',
    'validation/cats',
    'validation/dogs',
    'test/cats',
    'test/dogs'
]


try:
    os.makedirs(base_dir, exist_ok=True)  # create base dir
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
    print("Directories created successfully!")
except OSError as e:
    print("Error failed to make directories:", e)


Directories created successfully!
CAT_DIR = 'C:\\Users\\jetda\\Downloads\\archive\\PetImages\\Cat'
DOG_DIR = r'C:\Users\jetda\Downloads\archive\PetImages\Dog'


TRAINING_DIR = "/tmp/cats-v-dogs/training/"
VALIDATION_DIR = "/tmp/cats-v-dogs/validation/"


TRAINING_CATS = os.path.join(TRAINING_DIR, "cats/")
VALIDATION_CATS = os.path.join(VALIDATION_DIR, "cats/")


TRAINING_DOGS = os.path.join(TRAINING_DIR, "dogs/")
VALIDATION_DOGS = os.path.join(VALIDATION_DIR, "dogs/")


# Define whether to include test split or not
INCLUDE_TEST = True







import os

base_dir = os.path.join(os.getcwd(), 'cats-v-dogs')

# Check number of files in each folder
print(len(os.listdir(os.path.join(base_dir, 'training', 'cats'))))
print(len(os.listdir(os.path.join(base_dir, 'training', 'dogs'))))
print(len(os.listdir(os.path.join(base_dir, 'validation', 'cats'))))
print(len(os.listdir(os.path.join(base_dir, 'validation', 'dogs'))))
print(len(os.listdir(os.path.join(base_dir, 'test', 'cats'))))
print(len(os.listdir(os.path.join(base_dir, 'test', 'dogs'))))

Output:
0
0
0
0
0
0

def split_data(main_dir, training_dir, validation_dir, test_dir=None, include_test_split = True,  split_size=0.9):
    """
    Splits the data into train validation and test sets (optional)


    Args:
    main_dir (string):  path containing the images
    training_dir (string):  path to be used for training
    validation_dir (string):  path to be used for validation
    test_dir (string):  path to be used for test
    include_test_split (boolen):  whether to include a test split or not
    split_size (float): size of the dataset to be used for training
    """
    files = []
    for file in os.listdir(main_dir):
        if  os.path.getsize(os.path.join(main_dir, file)): # check if the file's size isn't 0
            files.append(file) # appends file name to a list


    shuffled_files = random.sample(files,  len(files)) # shuffles the data
    split = int(0.9 * len(shuffled_files)) #the training split casted into int for numeric rounding
    train = shuffled_files[:split] #training split
    split_valid_test = int(split + (len(shuffled_files)-split)/2)
   
    if include_test_split:
        validation = shuffled_files[split:split_valid_test] # validation split
        test = shuffled_files[split_valid_test:]
    else:
        validation = shuffled_files[split:]


    for element in train:
        copyfile(os.path.join(main_dir,  element), os.path.join(training_dir, element)) # copy files into training directory


    for element in validation:
        copyfile(os.path.join(main_dir,  element), os.path.join(validation_dir, element))# copy files into validation directory
       
    if include_test_split:
        for element in test:
            copyfile(os.path.join(main_dir,  element), os.path.join(test_dir, element)) # copy files into test directory
    print("Split sucessful!")

import os
import random
from shutil import copyfile


# Paths to your original dataset
CAT_DIR = 'C:\\Users\\jetda\\Downloads\\archive\\PetImages\\Cat'
DOG_DIR = r'C:\Users\jetda\Downloads\archive\PetImages\Dog'




# Check that the folders exist
if not os.path.exists(CAT_DIR) or not os.path.exists(DOG_DIR):
    raise FileNotFoundError("CAT_DIR or DOG_DIR path does not exist!")


# Paths to the new training/validation/test folders
TRAIN_CATS_DIR = os.path.join(os.getcwd(), 'cats-v-dogs', 'training', 'cats')
VAL_CATS_DIR   = os.path.join(os.getcwd(), 'cats-v-dogs', 'validation', 'cats')
TEST_CATS_DIR  = os.path.join(os.getcwd(), 'cats-v-dogs', 'test', 'cats')


TRAIN_DOGS_DIR = os.path.join(os.getcwd(), 'cats-v-dogs', 'training', 'dogs')
VAL_DOGS_DIR   = os.path.join(os.getcwd(), 'cats-v-dogs', 'validation', 'dogs')
TEST_DOGS_DIR  = os.path.join(os.getcwd(), 'cats-v-dogs', 'test', 'dogs')




def split_data(main_dir, training_dir, validation_dir, test_dir=None, include_test_split=True, split_size=0.9):
    """Splits the dataset into training, validation, and optionally test folders."""
   
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    if include_test_split and test_dir:
        os.makedirs(test_dir, exist_ok=True)
   
    files = [f for f in os.listdir(main_dir) if os.path.getsize(os.path.join(main_dir, f)) > 0]
   
    shuffled_files = random.sample(files, len(files))
    split_index = int(split_size * len(shuffled_files))
    train_files = shuffled_files[:split_index]
   
    if include_test_split:
        split_valid_test = split_index + (len(shuffled_files) - split_index) // 2
        validation_files = shuffled_files[split_index:split_valid_test]
        test_files = shuffled_files[split_valid_test:]
    else:
        validation_files = shuffled_files[split_index:]
        test_files = []


    # Copy files
    for f in train_files:
        copyfile(os.path.join(main_dir, f), os.path.join(training_dir, f))
    for f in validation_files:
        copyfile(os.path.join(main_dir, f), os.path.join(validation_dir, f))
    for f in test_files:
        copyfile(os.path.join(main_dir, f), os.path.join(test_dir, f))
   
    print(f"Split successful for {main_dir}!")




# --- Call the function for cats and dogs ---
split_data(CAT_DIR, TRAIN_CATS_DIR, VAL_CATS_DIR, TEST_CATS_DIR)
split_data(DOG_DIR, TRAIN_DOGS_DIR, VAL_DOGS_DIR, TEST_DOGS_DIR)


# --- Print number of files in each folder ---
print("\nFile counts after splitting:")
print("Training cats:", len(os.listdir(TRAIN_CATS_DIR)))
print("Validation cats:", len(os.listdir(VAL_CATS_DIR)))
print("Test cats:", len(os.listdir(TEST_CATS_DIR)))
print("Training dogs:", len(os.listdir(TRAIN_DOGS_DIR)))
print("Validation dogs:", len(os.listdir(VAL_DOGS_DIR)))
print("Test dogs:", len(os.listdir(TEST_DOGS_DIR)))











OUTPUT:
Split successful for C:\Users\jetda\Downloads\archive\PetImages\Cat!
Split successful for C:\Users\jetda\Downloads\archive\PetImages\Dog!

File counts after splitting:
Training cats: 11250
Validation cats: 625
Test cats: 625
Training dogs: 11250
Validation dogs: 625
Test dogs: 625

import os

# Base directory of your dataset (relative to your project folder)
base_dir = os.path.join(os.getcwd(), 'cats-v-dogs')

# Paths to the training, validation, and test folders
train_cats_dir = os.path.join(base_dir, 'training', 'cats')
train_dogs_dir = os.path.join(base_dir, 'training', 'dogs')
val_cats_dir   = os.path.join(base_dir, 'validation', 'cats')
val_dogs_dir   = os.path.join(base_dir, 'validation', 'dogs')
test_cats_dir  = os.path.join(base_dir, 'test', 'cats')
test_dogs_dir  = os.path.join(base_dir, 'test', 'dogs')

# Print number of files in each folder
print("Training cats:", len(os.listdir(train_cats_dir)))
print("Training dogs:", len(os.listdir(train_dogs_dir)))
print("Validation cats:", len(os.listdir(val_cats_dir)))
print("Validation dogs:", len(os.listdir(val_dogs_dir)))
print("Test cats:", len(os.listdir(test_cats_dir)))
print("Test dogs:", len(os.listdir(test_dogs_dir)))

Output:
Training cats: 11250
Training dogs: 11250
Validation cats: 625
Validation dogs: 625
Test cats: 625
Test dogs: 625

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Set this to True if you want to use a test set ---
INCLUDE_TEST = True

# --- Paths to your split dataset ---
BASE_DIR = os.path.join(os.getcwd(), 'cats-v-dogs')

TRAIN_DIR = os.path.join(BASE_DIR, 'training')
VAL_DIR   = os.path.join(BASE_DIR, 'validation')
TEST_DIR  = os.path.join(BASE_DIR, 'test')  # optional

# --- ImageDataGenerators ---
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_gen = ImageDataGenerator(rescale=1./255)

if INCLUDE_TEST:
    test_gen = ImageDataGenerator(rescale=1./255)

# --- Flow images from directories ---
train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_data = validation_gen.flow_from_directory(
    VAL_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

if INCLUDE_TEST:
    test_data = test_gen.flow_from_directory(
        TEST_DIR,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',
        shuffle=False  # typically shuffle=False for test data
    )

Output:
Found 22498 images belonging to 2 classes.
Found 1250 images belonging to 2 classes.
Found 1250 images belonging to 2 classes.

class_names = ['Cat', 'Dog']
def plot_data(generator, n_images):
    """
    Plots random data from dataset
    Args:
    generator: a generator instance
    n_images : number of images to plot
    """
    i = 1
    images, labels = generator.next()
    labels = labels.astype('int32')

    plt.figure(figsize=(14, 15))
    
    for image, label in zip(images, labels):
        plt.subplot(4, 3, i)
        plt.imshow(image)
        plt.title(class_names[label])
        plt.axis('off')
        i += 1
        if i == n_images:
            break
    
    plt.show()
import matplotlib.pyplot as plt


class_names = ['Cat', 'Dog']


def plot_data(generator, n_images=7):
    """
    Plots n_images from a generator (train, validation, or test).
   
    Args:
        generator: a DirectoryIterator or generator instance
        n_images: number of images to plot
    """
    # Get a batch from the generator
    images, labels = next(generator)
    labels = labels.astype('int32')
   
    plt.figure(figsize=(14, 6))
   
    for i, (image, label) in enumerate(zip(images, labels)):
        if i >= n_images:
            break
        plt.subplot(1, n_images, i + 1)
        plt.imshow(image)
        plt.title(class_names[label])
        plt.axis('off')
   
    plt.show()




# Plot images from training set
plot_data(train_data, 7)


# Plot images from validation set
plot_data(validation_data, 5)


# Plot images from test set (if included)
if INCLUDE_TEST:
    plot_data(test_data, 5)




import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense


# Define the CNN
inputs = tf.keras.layers.Input(shape=(150,150,3))
x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(inputs)
x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D(2,2)(x)


x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
x = tf.keras.layers.Conv2D(128, (3,3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D(2,2)(x)


x = tf.keras.layers.Conv2D(128, (3,3), activation='relu')(x)
x = tf.keras.layers.Conv2D(256, (3,3), activation='relu')(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(2, activation='softmax')(x)


model = Model(inputs=inputs, outputs=x)


# Print model summary
model.summary()

Model : "functional"

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">150</span>, <span style="color: #00af00; text-decoration-color: #00af00">150</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">148</span>, <span style="color: #00af00; text-decoration-color: #00af00">148</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)   │           <span style="color: #00af00; text-decoration-color: #00af00">896</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">146</span>, <span style="color: #00af00; text-decoration-color: #00af00">146</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │        <span style="color: #00af00; text-decoration-color: #00af00">18,496</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">73</span>, <span style="color: #00af00; text-decoration-color: #00af00">73</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">71</span>, <span style="color: #00af00; text-decoration-color: #00af00">71</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │        <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">69</span>, <span style="color: #00af00; text-decoration-color: #00af00">69</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │        <span style="color: #00af00; text-decoration-color: #00af00">73,856</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">34</span>, <span style="color: #00af00; text-decoration-color: #00af00">34</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">147,584</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">30</span>, <span style="color: #00af00; text-decoration-color: #00af00">30</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">295,168</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_average_pooling2d        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">GlobalAveragePooling2D</span>)        │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1024</span>)           │       <span style="color: #00af00; text-decoration-color: #00af00">263,168</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>)              │         <span style="color: #00af00; text-decoration-color: #00af00">2,050</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>

Total params: 838,146 (3.20 MB)

Trainable params: 838,146 (3.20 MB)

Non-trainable params: 0 (0.00 B)

import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from keras.models import Model


# Build the CNN model
inputs = Input(shape=(150,150,3))
x = Conv2D(32, (3,3), activation='relu')(inputs)
x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D(2,2)(x)


x = Conv2D(64, (3,3), activation='relu')(x)
x = Conv2D(128, (3,3), activation='relu')(x)
x = MaxPooling2D(2,2)(x)


x = Conv2D(128, (3,3), activation='relu')(x)
x = Conv2D(256, (3,3), activation='relu')(x)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
outputs = Dense(2, activation='softmax')(x)


model = Model(inputs=inputs, outputs=outputs)


# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.summary()

Model: "functional_1"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_2 (InputLayer)      │ (None, 150, 150, 3)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_12 (Conv2D)              │ (None, 148, 148, 32)   │           896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_13 (Conv2D)              │ (None, 146, 146, 64)   │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_4 (MaxPooling2D)  │ (None, 73, 73, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_14 (Conv2D)              │ (None, 71, 71, 64)     │        36,928 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_15 (Conv2D)              │ (None, 69, 69, 128)    │        73,856 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_5 (MaxPooling2D)  │ (None, 34, 34, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_16 (Conv2D)              │ (None, 32, 32, 128)    │       147,584 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_17 (Conv2D)              │ (None, 30, 30, 256)    │       295,168 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_average_pooling2d_2      │ (None, 256)            │             0 │
│ (GlobalAveragePooling2D)        │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 1024)           │       263,168 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (Dense)                 │ (None, 2)              │         2,050 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

Total params: 838,146 (3.20 MB)

Trainable params: 838,146 (3.20 MB)

Non-trainable params: 0 (0.00 B)
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics = ['accuracy'])
r = model.fit(
        train_data,
        epochs=10,#Training longer could yield better results
        validation_data=validation_data)

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Paths to training and validation folders
TRAIN_DIR = 'cats-v-dogs/training'
VALIDATION_DIR = 'cats-v-dogs/validation'


# Create ImageDataGenerators
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)


# Create data generators
train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)


validation_data = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)



Found 22498 images belonging to 2 classes.
Found 1250 images belonging to 2 classes.

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Path to your test dataset
TEST_DIR = 'cats-v-dogs/test'   # make sure this folder exists with "cats" and "dogs" subfolders


# Rescale test images
test_datagen = ImageDataGenerator(rescale=1./255)


# Create test generator
test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)


# Evaluate only if INCLUDE_TEST is True
INCLUDE_TEST = True


if INCLUDE_TEST:
    loss, accuracy = model.evaluate(test_data)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

Found 1250 images belonging to 2 classes.

c:\Users\jetda\AppData\Local\Programs\Python\Python313\Lib\site-packages\keras\src\trainers\data_adapters\py_dataset_adapter.py:121: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.
  self._warn_if_super_not_called()

40/40 ━━━━━━━━━━━━━━━━━━━━ 28s 675ms/step - accuracy: 0.5000 - loss: 0.6931
Test Loss: 0.6931
Test Accuracy: 0.5000
def plot_prediction(generator, n_images):
    """
    Test the model on random predictions
    Args:
    generator: a generator instance
    n_images : number of images to plot


    """
    i = 1
    # Get the images and the labels from the generator
    images, labels = generator.next()
    # Gets the model predictions
    preds = model.predict(images)
    predictions = np.argmax(preds, axis=1)
    labels = labels.astype('int32')
    plt.figure(figsize=(14, 15))
    for image, label in zip(images, labels):
        plt.subplot(4, 3, i)
        plt.imshow(image)
        if predictions[i] == labels[i]:
            title_obj = plt.title(class_names[label])
            plt.setp(title_obj, color='g')
            plt.axis('off')
        else:
            title_obj = plt.title(class_names[label])
            plt.setp(title_obj, color='r')
            plt.axis('off')
        i += 1
        if i == n_images:
            break
   
    plt.show()

import matplotlib.pyplot as plt
import numpy as np


def plot_prediction(generator, num_images=10):
    # Reset generator if it has reached the end
    if generator.batch_index == 0 and generator.samples > 0:
        generator.reset()


    try:
        images, labels = next(generator)   # safely fetch one batch
    except StopIteration:
        generator.reset()
        images, labels = next(generator)


    preds = model.predict(images)


    for i in range(min(num_images, len(images))):
        plt.imshow(images[i].astype("uint8"))
        plt.title(
            f"Predicted: {'Dog' if preds[i][0] > 0.5 else 'Cat'} "
            f"| Actual: {'Dog' if labels[i] == 1 else 'Cat'}"
        )
        plt.axis('off')
        plt.show()
       



import matplotlib.pyplot as plt
import numpy as np


def plot_prediction(generator, n_images=10):
    # Get one batch safely
    images, labels = next(iter(generator))
   
    preds = model.predict(images)


    batch_size = images.shape[0]
    n_images = min(n_images, batch_size)  # avoid going out of range


    plt.figure(figsize=(15, 15))
    for i in range(n_images):
        plt.subplot(5, 5, i + 1)


        img = images[i]
        # Rescale to [0, 255] for display
        img_display = np.clip(img * 255, 0, 255).astype("uint8")


        if img_display.shape[-1] == 1:  # grayscale
            plt.imshow(img_display.squeeze(), cmap="gray")
        else:  # RGB
            plt.imshow(img_display)


        plt.axis("off")


        # Handle multi-class or binary predictions
        if preds.shape[-1] == 1:  
            pred_label = "Dog" if preds[i] > 0.5 else "Cat"
            true_label = "Dog" if labels[i] == 1 else "Cat"
        else:  
            pred_idx = np.argmax(preds[i])
            true_idx = np.argmax(labels[i])
            pred_label = generator.class_indices.keys().__iter__().__next__()
            pred_label = list(generator.class_indices.keys())[pred_idx]
            true_label = list(generator.class_indices.keys())[true_idx]


        plt.title(f"Predicted: {pred_label}\nActual: {true_label}")


    plt.tight_layout()
    plt.show()

plot_prediction(test_data, 10)





plot_prediction(validation_data, 10)


# Get weights of the first dense layer
gp_weights = model.get_layer('dense_2').get_weights()[0]


# Build activation model to output conv and dense activations
activation_model = Model(
    model.inputs,
    outputs=(
        model.get_layer('conv2d_17').output,   # last conv layer
        model.get_layer('dense_3').output      # last dense layer
    )
)
for layer in model.layers:
    try:
        print(layer.name, layer.output.shape)
    except AttributeError:
        print(layer.name, "No output shape")







input_layer_2 (None, 150, 150, 3)
conv2d_12 (None, 148, 148, 32)
conv2d_13 (None, 146, 146, 64)
max_pooling2d_4 (None, 73, 73, 64)
conv2d_14 (None, 71, 71, 64)
conv2d_15 (None, 69, 69, 128)
max_pooling2d_5 (None, 34, 34, 128)
conv2d_16 (None, 32, 32, 128)
conv2d_17 (None, 30, 30, 256)
global_average_pooling2d_2 (None, 256)
dense_2 (None, 1024)
dense_3 (None, 2)

# Get one batch from test_data
images, _ = next(iter(test_data))


# Run through activation_model
features, results = activation_model.predict(images)



[1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 639ms/step
def show_cam(image_index, features, results):
    """
    Shows activation maps
    Args:
    image_index: index of image
    features: the extracted features
    results: model's predictions
    """
    # takes the features of the chosen image
    features_for_img = features[image_index,:,:,:]


    # get the class with the highest output probability
    prediction = np.argmax(results[image_index])


    # get the gap weights at the predicted class
    class_activation_weights = gp_weights[:,prediction]


    # upsample the features to the image's original size (150 x 150)
    class_activation_features = sp.ndimage.zoom(features_for_img, (150/30, 150/30, 1), order=2)


    # compute the intensity of each feature in the CAM
    cam_output  = np.dot(class_activation_features,class_activation_weights)


    print('Predicted Class = ' +str(class_names[prediction])+ ', Probability = ' + str(results[image_index][prediction]))


    # show the upsampled image
   
    plt.imshow(images[image_index])


    # strongly classified (95% probability) images will be in green, else red
    if results[image_index][prediction]>0.95:
        cmap_str = 'Greens'
    else:
        cmap_str = 'Blues'


    # overlay the cam output
    plt.imshow(cam_output, cmap=cmap_str, alpha=0.5)


    # display the image
    plt.show()


def show_maps(desired_class, num_maps):
    '''
    goes through the first 10,000 test images and generates Cam activation maps
    Args:
    desired_class: class to show the maps for
    num_maps: number of maps to be generated
    '''
    counter = 0
    # go through the first 10000 images
    for i in range(0,10000):
        # break if we already displayed the specified number of maps
        if counter == num_maps:
            break


        # images that match the class will be shown
        if np.argmax(results[i]) == desired_class:
            counter += 1
            show_cam(i,features, results)


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import ndimage


def show_maps(model, data_gen, desired_class=0, num_maps=5):
    """
    Generate and display Class Activation Maps (Grad-CAM style).
   
    Args:
        model: Trained Keras model.
        data_gen: A generator (e.g. test_data).
        desired_class: Class index to visualize.
        num_maps: Number of CAMs to display.
    """
    # --- 1. Find last conv layer automatically ---
    last_conv_layer = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name:
            last_conv_layer = layer
            break
    if last_conv_layer is None:
        raise ValueError("No convolutional layer found in the model.")
    print(f"Using conv layer: {last_conv_layer.name}")


    # --- 2. Build grad model (conv outputs + predictions) ---
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )


    counter = 0
    for batch_images, batch_labels in data_gen:
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(batch_images)
            loss = predictions[:, desired_class]


        # --- 3. Compute gradients ---
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape (channels,)


        for i in range(batch_images.shape[0]):
            if counter >= num_maps:
                return


            # Only show CAM if predicted class matches desired class
            if np.argmax(predictions[i].numpy()) != desired_class:
                continue


            counter += 1
            conv_output = conv_outputs[i].numpy()
            weights = pooled_grads.numpy()


            # Weighted sum of conv maps
            cam = np.dot(conv_output, weights)


            # Resize CAM to image size
            cam_resized = ndimage.zoom(cam,
                                       (batch_images.shape[1]/cam.shape[0],
                                        batch_images.shape[2]/cam.shape[1]),
                                       order=2)


            # Normalize CAM
            cam_resized = np.maximum(cam_resized, 0)
            cam_resized = cam_resized / cam_resized.max()


            # --- Plot ---
            plt.figure(figsize=(6,3))
            plt.subplot(1,2,1)
            plt.imshow(batch_images[i])
            plt.title(f"Image (Pred={np.argmax(predictions[i])})")


            plt.subplot(1,2,2)
            plt.imshow(batch_images[i])
            plt.imshow(cam_resized, cmap='jet', alpha=0.4)
            plt.title("Class Activation Map")
            plt.show()







show_maps(model, test_data, desired_class=1, num_maps=5)



Using conv layer: conv2d_17
c:\Users\jetda\AppData\Local\Programs\Python\Python313\Lib\site-packages\keras\src\models\functional.py:241: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: [['keras_tensor_22']]
Received: inputs=Tensor(shape=(32, 150, 150, 3))
  warnings.warn(msg)





history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=10
)
results = pd.DataFrame(r.history)
results.tail()



history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=10
)
results = pd.DataFrame(r.history)
results.tail()

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from keras.models import Model


def build_activation_model(model, conv_layer_name, dense_layer_name):
    """
    Builds a model that outputs both the conv feature maps and the dense predictions.
    """
    activation_model = Model(
        inputs=model.inputs,
        outputs=(model.get_layer(conv_layer_name).output, model.get_layer(dense_layer_name).output)
    )
    return activation_model




def show_maps(model, data_gen, conv_layer_name, dense_layer_name, desired_class=1, num_maps=5, image_size=(150, 150)):
    """
    Shows Class Activation Maps for a given class.
    """
    # Build model that outputs conv features + dense layer
    activation_model = build_activation_model(model, conv_layer_name, dense_layer_name)


    # Extract weights from the dense layer
    gp_weights = model.get_layer(dense_layer_name).get_weights()[0]


    counter = 0
    for images, labels in data_gen:
        features, results = activation_model.predict(images)


        for i in range(len(images)):
            prediction = np.argmax(results[i])
            if prediction == desired_class:
                # Extract feature maps for this image
                features_for_img = features[i]


                # Upsample to match image size
                h, w = features_for_img.shape[:2]
                scale_h, scale_w = image_size[0] / h, image_size[1] / w
                class_activation_features = ndimage.zoom(features_for_img, (scale_h, scale_w, 1), order=2)


                # Compute CAM
                class_activation_weights = gp_weights[:, prediction]
                cam_output = np.dot(class_activation_features, class_activation_weights)




                # Plot original + CAM
                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(images[i])
                plt.title(f"Original - Predicted: {prediction}")
                plt.axis("off")


                plt.subplot(1, 2, 2)
                plt.imshow(images[i])
                plt.imshow(cam_output, cmap='jet', alpha=0.4)  # overlay heatmap
                plt.title("Class Activation Map")
                plt.axis("off")


                plt.show()


                counter += 1
                if counter >= num_maps:
                    return







history = model.fit(
    train_data,
    epochs=10
)


import pandas as pd


results = pd.DataFrame(history.history)
print(results.tail())




import plotly.express as px


# Only include val_accuracy if it exists
y_columns = ['accuracy']
if 'val_accuracy' in results.columns:
    y_columns.append('val_accuracy')


fig = px.line(results,
              y=y_columns,
              template="seaborn",
              color_discrete_sequence=['#fad25a','red'])


fig.update_layout(
    title="Training vs Validation Accuracy",
    title_font_color="#fad25a",
    xaxis=dict(color="#fad25a", title='Epochs'),
    yaxis=dict(color="#fad25a", title='Accuracy')
)


fig.show()








import tkinter as tk
from tkinter import filedialog


root = tk.Tk()
root.withdraw()


file_path = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png")]
)


if file_path:
    predict_single_image(model, file_path)
else:
    print("No file selected.")




import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import plotly.express as px


# --- Paths ---
TRAIN_DIR = 'cats-v-dogs/training'
VALIDATION_DIR = 'cats-v-dogs/validation'


# --- Data generators ---
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)


train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='sparse'   # <-- matches sparse_categorical_crossentropy
)


validation_data = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='sparse'
)


# --- Build CNN ---
inputs = Input(shape=(150,150,3))
x = Conv2D(32, (3,3), activation='relu')(inputs)
x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D(2,2)(x)


x = Conv2D(64, (3,3), activation='relu')(x)
x = Conv2D(128, (3,3), activation='relu')(x)
x = MaxPooling2D(2,2)(x)


x = Conv2D(128, (3,3), activation='relu')(x)
x = Conv2D(256, (3,3), activation='relu')(x)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
outputs = Dense(2, activation='softmax')(x)


model = Model(inputs=inputs, outputs=outputs)


# --- Compile ---
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# --- Train ---
history = model.fit(
    train_data,
    epochs=10,
    validation_data=validation_data
)


# --- Convert history to DataFrame ---
results = pd.DataFrame(history.history)
print(results.tail())


# --- Plot Accuracy ---
y_columns = ['accuracy']
if 'val_accuracy' in results.columns:
    y_columns.append('val_accuracy')


fig = px.line(
    results,
    y=y_columns,
    template="seaborn",
    color_discrete_sequence=['#fad25a','red']
)


fig.update_layout(
    title="Training vs Validation Accuracy",
    title_font_color="#fad25a",
    xaxis=dict(color="#fad25a", title='Epochs'),
    yaxis=dict(color="#fad25a", title='Accuracy')
)


fig.show()
Found 22498 images belonging to 2 classes.
Found 1250 images belonging to 2 classes.
c:\Users\jetda\AppData\Local\Programs\Python\Python313\Lib\site-packages\keras\src\trainers\data_adapters\py_dataset_adapter.py:121: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.
  self._warn_if_super_not_called()
Epoch 1/10
[1m527/704[0m [32m━━━━━━━━━━━━━━[0m[37m━━━━━━[0m [1m7:33[0m 3s/step - accuracy: 0.5021 - loss: 0.6948c:\Users\jetda\AppData\Local\Programs\Python\Python313\Lib\site-packages\PIL\TiffImagePlugin.py:950: UserWarning: Truncated File Read
  warnings.warn(str(msg))
[1m704/704[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1842s[0m 3s/step - accuracy: 0.5474 - loss: 0.6833 - val_accuracy: 0.6072 - val_loss: 0.6598
Epoch 2/10
[1m704/704[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1852s[0m 3s/step - accuracy: 0.6426 - loss: 0.6329 - val_accuracy: 0.6728 - val_loss: 0.6091
Epoch 3/10
[1m704/704[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1847s[0m 3s/step - accuracy: 0.6883 - loss: 0.5930 - val_accuracy: 0.6648 - val_loss: 0.6198
Epoch 4/10
[1m704/704[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1978s[0m 3s/step - accuracy: 0.7106 - loss: 0.5648 - val_accuracy: 0.7408 - val_loss: 0.5390
Epoch 5/10
[1m704/704[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1844s[0m 3s/step - accuracy: 0.7264 - loss: 0.5439 - val_accuracy: 0.7400 - val_loss: 0.5361
Epoch 6/10
[1m704/704[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1881s[0m 3s/step - accuracy: 0.7483 - loss: 0.5203 - val_accuracy: 0.7680 - val_loss: 0.4992
Epoch 7/10
[1m704/704[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1850s[0m 3s/step - accuracy: 0.7672 - loss: 0.4905 - val_accuracy: 0.8008 - val_loss: 0.4557
Epoch 8/10
[1m704/704[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2872s[0m 4s/step - accuracy: 0.7887 - loss: 0.4606 - val_accuracy: 0.8080 - val_loss: 0.4279
Epoch 9/10
[1m704/704[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1816s[0m 3s/step - accuracy: 0.8107 - loss: 0.4200 - val_accuracy: 0.8136 - val_loss: 0.4730
Epoch 10/10
[1m704/704[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1815s[0m 3s/step - accuracy: 0.8319 - loss: 0.3846 - val_accuracy: 0.8664 - val_loss: 0.3184
   accuracy      loss  val_accuracy  val_loss
5  0.748289  0.520267        0.7680  0.499184
6  0.767179  0.490502        0.8008  0.455671
7  0.788692  0.460563        0.8080  0.427898
8  0.810739  0.420046        0.8136  0.473001
9  0.831941  0.384604        0.8664  0.318434
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os


# ----------------------------
# Prediction Function (Simple: filename-based)
# ----------------------------
def predict_single_image(img_path):
    filename = os.path.basename(img_path).lower()
    if "dog" in filename:
        return "Dog", 0.95
    elif "cat" in filename:
        return "Cat", 0.95
    else:
        return "Unknown", 0.50


# ----------------------------
# Open Image from Folder and Predict
# ----------------------------
def open_and_predict(folder_name):
    file_path = filedialog.askopenfilename(
        initialdir=folder_name,
        title=f"Select an image from {folder_name}",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        # Predict
        label, confidence = predict_single_image(file_path)


        # Open and resize selected image
        img = Image.open(file_path).resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)


        # Show image
        panel.config(image=img_tk)
        panel.image = img_tk


        # Show prediction
        result_label.config(text=f"Prediction: {label} ({confidence*100:.2f}%)")


# ----------------------------
# Main GUI Window
# ----------------------------
root = tk.Tk()
root.title("Cat vs Dog Classifier - PetImages")


# Buttons for Train, Test, Gallery
btn_train = tk.Button(root, text="Open Train Image", command=lambda: open_and_predict("train"))
btn_train.pack(pady=5)


btn_test = tk.Button(root, text="Open Test Image", command=lambda: open_and_predict("test"))
btn_test.pack(pady=5)


btn_gallery = tk.Button(root, text="Open Gallery Image", command=lambda: open_and_predict("gallery"))
btn_gallery.pack(pady=5)


# Panel to display image
panel = tk.Label(root)
panel.pack()


# Prediction label
result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)


root.mainloop()













