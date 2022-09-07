import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from tensorflow import keras
import keras

from keras.models import Sequential
from keras.layers import Dense,Flatten,InputLayer, Dropout, Activation

from PIL import Image

from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder

root_path = "../UTKFace"
# root_path = "../part1"
# os.listdir is a out-of-order list of function
files = os.listdir(root_path)
size = len(files)

train = pd.read_csv('age.csv')
test = pd.read_csv('age_test.csv')

# print(train.head())
# print(test.head())

temp = []
images = []
train_ages = []
test_ages = []
train_group_num = []
test_group_num = []


for img_name in train.photo_id:
    img_path = os.path.join(root_path, img_name)
    img = Image.open(img_path)
    img = img.resize((32, 32))
    # img = img.resize((128, 128))
    # img = img.resize((64, 64))
    array = np.array(img)
    temp.append(array.astype('float32'))
train_x = np.stack(temp)

temp = []
for img_name in test.photo_id:
    img_path=os.path.join(root_path, img_name)
    img=Image.open(img_path)
    # img=img.resize((64,64))
    # img=img.resize((128,128))
    img=img.resize((32,32))
    array=np.array(img)
    temp.append(array.astype('float32'))
test_x=np.stack(temp)


train_x = train_x / 255
test_x = test_x / 255

print(train.group.value_counts(normalize=True))

lb = LabelEncoder()
train_y = lb.fit_transform(train.group)
# print(train_y)
train_y=keras.utils.np_utils.to_categorical(train_y)
# print(train_y)
# print(train_y.shape)



filters1=50
filters2=100
filters3=100
filtersize1=(5,5)
filtersize2=(3,3)
# filtersize3=(11,11)
pool_size1 = (2,2)
pool_size2 = (3,3)

epochs =30
# batchsize=128
batchsize=256

filtersize3=(7,7)
# input_shape=(64,64,3)
# input_shape=(128,128,3)
input_shape=(32,32,3)

model = Sequential()
model.add(keras.layers.InputLayer(input_shape=input_shape))

model.add(keras.layers.convolutional.Conv2D(64, filtersize1, strides=(1, 1), padding='valid', data_format="channels_last", activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=pool_size1))

model.add(keras.layers.convolutional.Conv2D(128, filtersize1, strides=(1, 1), padding='same', data_format="channels_last", activation='relu'))
# model.add(keras.layers.convolutional.Conv2D(128, filtersize2, strides=(1, 1), padding='same', data_format="channels_last", activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=pool_size1))
#
model.add(keras.layers.convolutional.Conv2D(256, filtersize1, strides=(1, 1), padding='same', data_format="channels_last", activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=pool_size1))
model.add(keras.layers.convolutional.Conv2D(512, filtersize1, strides=(1, 1), padding='same',  activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=pool_size1))
model.add(keras.layers.convolutional.Conv2D(1024, filtersize1, strides=(1, 1), padding='same',  activation='relu'))
# model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256,activation='relu'))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dropout(0.18))


# model.add(keras.layers.Dense(128, input_dim=50,activation='softmax'))
# model.add(keras.layers.Dense(32, input_dim=50,activation='softmax'))
model.add(keras.layers.Dense(8, input_dim=50,activation='softmax'))


model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
h = model.fit(train_x,train_y,batch_size=batchsize,epochs=epochs, validation_split=0.3, shuffle=True)

model.summary()

model.save('save_data_model.h5')

history = h
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


