import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization,GlobalAveragePooling2D,AveragePooling2D,ReLU,Softmax
import os
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from PIL import Image
from keras.models import Sequential
import keras
import matplotlib.pyplot as plt

from tensorflow.keras import models,layers,datasets,losses,utils,optimizers,activations
from tensorflow.keras.datasets import mnist,cifar10,cifar100
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,BatchNormalization,Activation,ReLU,GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop,SGD,Adagrad,Adadelta,Adam
from tensorflow.keras.losses import mean_squared_error,sparse_categorical_crossentropy,categorical_crossentropy
import os
import tensorflow as tf
import matplotlib.pyplot as plt

# def conv(x,filters=64,kernel_size=(3,3),strides=(1,1),padding="same",):
#     x=Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding)(x)
#     x=BatchNormalization()(x)
#     return x
#
# def convblock1(inputs,filter=64,kernel_size=(3,3),strides=(1,1)):
#     x=conv(inputs,filters=filter,kernel_size=(1,1),strides=strides)
#     x=conv(x,filters=filter,kernel_size=kernel_size,strides=(1,1))
#     print("1:",x.shape)
#     if strides==(2,2):
#         resdual=conv(inputs,filters=filter,kernel_size=(3,3),strides=strides)
#         print("2:",resdual.shape)
#     else:
#         resdual = conv(inputs, filters=filter, kernel_size=(3, 3), strides=(1,1))
#         print("2:", resdual.shape)
#     # x=tf.keras.layers.add([x,resdual])
#     x=x+resdual
#     x=ReLU()(x)
#     print("3:",resdual.shape)
#     return x
#
# def resnet18(input):
#     x=conv(input,filters=64,kernel_size=(1,1),padding="same")#1
#     x=ReLU()(x)
#     print(x.shape)
#     # block1
#     x=convblock1(x, filter=64, strides=(1, 1))#2
#     x = convblock1(x, filter=64, strides=(1, 1))#2
#     print(x.shape)
#     # block2
#     x = convblock1(x, filter=128,strides=(2, 2))#2
#     x = convblock1(x, filter=128, strides=(1,1))#2
#     print(x.shape)
#     # block3
#     x = convblock1(x, filter=256, strides=(2,2))#2
#     x = convblock1(x, filter=256, strides=(1,1))#2
#     print(x.shape)
#     # block4
#     x = convblock1(x, filter=512, strides=(2, 2))#2
#     x = convblock1(x, filter=512, strides=(1, 1))#2
#     print(x.shape)
#
#     x=GlobalAveragePooling2D(name='avg_pool')(x)
#     x=Dense(6,activation="softmax",)(x)
#     return x

class cellblock(models.Model):
    def __init__(self,filter_num,strides=1):
        super(cellblock, self).__init__()
        self.conv1=Conv2D(filter_num,(3,3),strides=strides,padding='same')
        self.bn1=BatchNormalization()
        self.relu=Activation('relu')

        self.conv2 = Conv2D(filter_num,(3,3), strides=1,padding='same')
        self.bn2 = BatchNormalization()

        if strides!=1:
            self.residual=Conv2D(filter_num,(1,1), strides=strides)
        else:
            self.residual=lambda x:x


    def call(self, inputs, training=None, mask=None):
        x=self.conv1(inputs)
        x=self.bn1(x)
        x=self.relu(x)

        x=self.conv2(x)
        x=self.bn2(x)
        print(x.shape)
        r=self.residual(inputs)
        x=layers.add([x,r])
        output=tf.nn.relu(x)
        return output

class ResNet(models.Model):
    def __init__(self,layers_dims,n_classes=10):
        super(ResNet, self).__init__()
        self.model=Sequential([
            Conv2D(64,(7,7),strides=(2,2),padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D((3,3),strides=(2,2),padding='same')
        ])

        self.layer1=self.build_cellblock(64,layers_dims[0])
        self.layer2=self.build_cellblock(128,layers_dims[1],strides=2)
        self.layer3=self.build_cellblock(256,layers_dims[2],strides=2)
        self.layer4=self.build_cellblock(512,layers_dims[3],strides=2)

        self.avgpool=GlobalAveragePooling2D()
        self.fc=Dense(n_classes,activation='softmax')


    def build_cellblock(self,filter_num,blocks,strides=1):
        res_cellblock=Sequential()
        res_cellblock.add(cellblock(filter_num,strides=strides))
        for _ in range(1,blocks):
            res_cellblock.add(cellblock(filter_num,strides=1))
        return res_cellblock


    def call(self, inputs, training=None, mask=None):
        x=self.model(inputs)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        x=self.avgpool(x)
        x=self.fc(x)
        return x

def bulid_resnet(resname,n_classes):
    res_configer={
            'ResNet34':[3,4,6,3],
                  'ResNet18':[2,2,2,2]}
    return ResNet(res_configer[resname],n_classes)

if __name__ == '__main__':
    root_path = "../UTKFace"
    # os.listdir is a out-of-order list of function
    files = os.listdir(root_path)
    size = len(files)

    train = pd.read_csv('age.csv')
    test = pd.read_csv('age_test.csv')

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
        array = np.array(img)
        temp.append(array.astype('float32'))
    train_x = np.stack(temp)

    temp = []
    for img_name in test.photo_id:
        img_path = os.path.join(root_path, img_name)
        img = Image.open(img_path)
        img = img.resize((32, 32))
        array = np.array(img)
        temp.append(array.astype('float32'))
    test_x = np.stack(temp)
    print(test_x.shape)

    train_x = train_x / 255
    test_x = test_x / 255

    lb = LabelEncoder()
    train_y = lb.fit_transform(train.group)
    train_y = keras.utils.np_utils.to_categorical(train_y)
    # print(train_y)
    # print(train_y.shape)

    model = bulid_resnet('ResNet34',6)
    model.build(input_shape=(None,32,32,3))
    # input = tf.keras.Input([32, 32, 3])
    # output = resnet18(input)
    # model = tf.keras.Model(input, output)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy',
                  metrics=["accuracy"])
    h = model.fit(train_x, train_y, epochs=50, batch_size=512, validation_split=0.2)
    # loss, acc = model.evaluate(test_x, test_y)
    # print("acc:", acc*100,"%")

    model.save('save_data_model_3.h5')

    history = h
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()