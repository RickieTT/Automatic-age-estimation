import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.layers import *
from keras.models import load_model
from sklearn.model_selection import KFold
from keras.models import *
import tensorflow
from keras import backend as K

path = "../UTKFace"
# os.listdir is a out-of-order list of function
files = os.listdir(path)
size = len(files)
# print("Total samples:", size)
# print(files[0])

images = []
ages = []
genders = []
# function group by age
def age_group(age):
    if age >= 0 and age <= 6:
        real_age = {age,'0-6'}
        return 1
    if age >= 7 and age <= 11:
        real_age = {age, '7-11'}
        return 2
    elif age >= 12 and age <= 17:
        real_age = {age, '12-17'}
        return 3
    elif age >= 18 and age <= 23:
        return 4
    elif age >= 24 and age <= 28:
        return 5
    elif age >= 29 and age <= 34:
        return 6
    elif age >= 35 and age <= 42:
        return 7
    elif age >= 43 and age <= 50:
        return 8
    elif age >= 51 and age <= 59:
        return 9
    else:
        return 10


for file in files:
    image = cv2.imread(path + '/' + file, 0)
    # print(file)
    try:
        image = cv2.resize(image, dsize=(64, 64))
    except:
        continue
    # 64 * 64 * 1 tuple(元组) can not be changed
    image = image.reshape((image.shape[0], image.shape[1], 1))
    images.append(image)
    split_var = file.split('_')
    # correspond age and gender
    ages.append(split_var[0])
    genders.append(int(split_var[1]))

x_ages = list(set(ages))
print(x_ages)
print(set(ages))
y_ages = [ages.count(i) for i in x_ages]

plt.bar(x_ages, y_ages)
plt.show()
print("Max value:", max(ages))

#
#
def display(img):
    plt.imshow(img[:, :, 0])
    plt.set_cmap('gray')
    plt.show()

# test whether read image successfully or not
idx = 500
sample = images[idx]
# print("Gender:", genders[idx], "Age:", ages[idx])
print("Age:", ages[idx])
display(sample)

# pre-processing
# generate size(20856) rows and 2 columns zero's array
target = np.zeros((size, 2), dtype='float32')
# target = np.zeros((size, 1), dtype='float32')

# generate size(20856) * 64 * 64 3-D zero's array
# features = np.zeros((size, sample.shape[0], sample.shape[1], 1), dtype='float32')
features = np.zeros((size, sample.shape[0], sample.shape[1], 1), dtype='float32')

for i in range(size - 1):
    # print(i)
    target[i, 0] = age_group(int(ages[i])) % 10
    # print(target[i,0])
    # target[i, 0] = int(ages[i])
    target[i, 1] = int(genders[i])
    features[i] = images[i]
features = features / 255
display(features[550])

# divide data set to train set and test set
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=True)



# kf = KFold(n_splits=10)
#
# for train_index, test_index in kf.split(features):
#       print("Train:", train_index, "Validation:",test_index)
#       x_train, x_test = features[train_index], features[test_index]
#       y_train, y_test = target[train_index], target[test_index]

# print the number of training picture
print("Samples in Training:", x_train.shape[0])
# print the number of testing picture
print("Samples in Testing:", x_test.shape[0])
#
#
inputs = Input(shape=(64, 64, 1))
# out_size = [ ( in_size + padding - filter_size ) / stride ] + 1
conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(128, kernel_size=(3, 3), activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv3)
x = Dropout(0.25)(pool2)
flat = Flatten()(x)
#
#
dropout = Dropout(0.5)
age_model = Dense(128, activation='relu')(flat)
age_model = dropout(age_model)
age_model = Dense(64, activation='relu')(age_model)
age_model = dropout(age_model)
age_model = Dense(32, activation='relu')(age_model)
age_model = dropout(age_model)
age_model = Dense(16, activation='relu')(age_model)
age_model = dropout(age_model)
age_model = Dense(1, activation='relu')(age_model)
#
#
dropout = Dropout(0.5)
gender_model = Dense(128, activation='relu')(flat)
gender_model = dropout(gender_model)
gender_model = Dense(64, activation='relu')(gender_model)
gender_model = dropout(gender_model)
gender_model = Dense(32, activation='relu')(gender_model)
gender_model = dropout(gender_model)
gender_model = Dense(16, activation='relu')(gender_model)
gender_model = dropout(gender_model)
gender_model = Dense(8, activation='relu')(gender_model)
gender_model = dropout(gender_model)
gender_model = Dense(1, activation='sigmoid')(gender_model)

model = Model(inputs=inputs, outputs=[age_model, gender_model])
# model = Model(inputs=inputs, outputs=[age_model])
model.compile(optimizer='sgd', loss=['mse', 'binary_crossentropy'], metrics=['accuracy'])
# model.compile(optimizer='adam', loss=['mse'], metrics=['accuracy'])
# model.compile(optimizer='adam', loss=['mse'], metrics=['mae'])
model.summary()
#
h = model.fit(x_train, [y_train[:, 0], y_train[:, 1]], validation_data=(x_test, [y_test[:, 0], y_test[:, 1]]), epochs=10, batch_size=128, shuffle=True)
# h = model.fit(x_train, y_train[:, 0],
#               validation_data=(x_test, y_test[:, 0]), epochs=10, batch_size=128, shuffle=True)
# save data
model.save('save_data_3.h5')

# draw a plot about model accuracy
history = h
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


load_model("save_data_2.h5")
def display(img):
    plt.imshow(img[:, :, 0])
    plt.set_cmap('gray')
    plt.show()


def get_age(distr):
    distr = distr
    if distr >= 0 and distr < 3.5: return "0-10"
    if distr >= 3.5 and distr < 4.9: return "11-17"
    if distr >= 4.9 and distr < 6.1: return "18-28"
    # if distr >= 5.5 and distr < 6.1: return "24-28"
    if distr >= 6.10 and distr < 6.7: return "29-34"
    if distr >= 6.70 and distr < 7.2: return "35-42"
    if distr >= 7.2 and distr < 7.8: return "43-50"
    if distr >= 7.8 and distr < 8.4: return "51-58"
    if distr >= 8.4 and distr < 9.1: return "59-66"
    if distr >= 9.1 and distr < 9.8: return "67-74"
    if distr >= 9.8 and distr < 10.6: return "75-82"
    if distr >= 10.6 and distr < 11.3: return "83-90"
    if distr >= 11.30  : return "over90"
    return "Unknown"


def get_result(sample):
    sample = sample / 255
    print(sample)
    val = model.predict(np.array([sample]))
    age = get_age(val[0])
    print(val)
    print("Values:", val[0], "Predicted Age:", age)


indexes = [10, 400, 1200, 1700, 2200, 2700, 3200, 3700, 4200]
# indexes = [10, 300, 1300, 1800, 2300, 2800, 3300, 3700, 4200]
for idx in indexes:
    if idx % 100 == 0 :
        sample = images[idx]
        display(sample)
        # print("Actual Gender:", get_gender(genders[idx]), "Age:", ages[idx])
        print("Actual Age:", ages[idx])
        res = get_result(sample)
