#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Activation, Conv2D, Dense, Dropout, BatchNormalization, ReLU, DepthwiseConv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Add
from keras.optimizers import RMSprop


mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# MNISTデータを加工する
x_train  = x_train.reshape(60000, 28*28)
x_test   = x_test.reshape(10000, 28*28)

x_train  = x_train.astype('float32')
x_test   = x_test.astype('float32')

x_train, x_test = x_train / 255.0, x_test / 255.0

y_train  = keras.utils.to_categorical(y_train, 10)
y_test   = keras.utils.to_categorical(y_test, 10)


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)



def BaseModel():
    inputs = Input(shape=(784,), name='input')

    x = Dense(512, name='layer1')(inputs)
    x = Activation('relu', name='activation1')(x)
    # x = Dropout(0.25, name='dropout1')(x)

    x = Dense(256, name='layer2')(x)
    x = Activation('relu', name='activation2')(x)
    # x = Dropout(0.20, name='dropout2')(x)

    x = Dense(128, name='layer3')(x)
    x = Activation('relu', name='activation3')(x)     
    # x = Dropout(0.20, name='dropout3')(x)

    x = Dense(64, name='layer4')(x)
    x = Activation('relu', name='activation4')(x)    
    # x = Dropout(0.20, name='dropout4')(x)

    x = Dense(10, name='layer5')(x)
    x = Activation('softmax', name='activation5')(x)        

    model = Model(inputs=inputs, outputs=x, name="base_mnist")
    return model


# In[5]:


model = BaseModel()
model.summary()


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=30)




test_loss, test_acc = model.evaluate(x_test, y_test)
print("loss (test) :", test_loss)
print("acc  (test) :", test_acc)


model.layers


# ### Kerasで中間層の出力
# 
# ```python
# 
# from keras.models import Model
# 
# model = ...  # create the original model
# 
# layer_name = 'my_layer'
# intermediate_layer_model = Model(inputs=model.input,
#                                  outputs=model.get_layer(layer_name).output)
# intermediate_output = intermediate_layer_model.predict(data)
# ```



for lay in model.layers: print(lay.name)

# --------------------------
# intermediate layer output
# --------------------------

in_data = x_train
print("in_data", x_train.shape)

layer_name = "layer1"
m = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
layer1_output = m.predict(in_data)
print(layer_name, layer1_output.shape)

layer_name = "activation1"
m = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
activation1_output = m.predict(in_data)
print(layer_name, activation1_output.shape)

layer_name = "layer2"
m = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
layer2_output = m.predict(in_data)
print(layer_name, layer2_output.shape)

layer_name = "activation2"
m = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
activation2_output = m.predict(in_data)
print(layer_name, activation2_output.shape)

layer_name = "layer3"
m = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
layer3_output = m.predict(in_data)
print(layer_name, layer3_output.shape)

layer_name = "activation3"
m = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
activation3_output = m.predict(in_data)
print(layer_name, activation3_output.shape)

layer_name = "layer4"
m = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
layer4_output = m.predict(in_data)
print(layer_name, layer4_output.shape)

layer_name = "activation4"
m = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
activation4_output = m.predict(in_data)
print(layer_name, activation4_output.shape)

layer_name = "layer5"
m = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
layer5_output = m.predict(in_data)
print(layer_name, layer5_output.shape)

layer_name = "activation5"
m = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
activation5_output = m.predict(in_data)
print(layer_name, activation5_output.shape)


np.save('data/dnn_base/in_data', in_data)

np.save('data/dnn_base/layer1_output', layer1_output)
np.save('data/dnn_base/layer2_output', layer2_output)
np.save('data/dnn_base/layer3_output', layer3_output)
np.save('data/dnn_base/layer4_output', layer4_output)
np.save('data/dnn_base/layer5_output', layer5_output)

np.save('data/dnn_base/activation1_output', activation1_output)
np.save('data/dnn_base/activation2_output', activation2_output)
np.save('data/dnn_base/activation3_output', activation3_output)
np.save('data/dnn_base/activation4_output', activation4_output)
np.save('data/dnn_base/activation5_output', activation5_output)

