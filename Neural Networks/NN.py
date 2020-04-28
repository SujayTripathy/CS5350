import numpy as np
import tensorflow as tf
import pandas as pd



def main():
    training=np.loadtxt('train.csv',delimiter=',')
    x=training[:,0:4]
    y=training[:,4]
    testing=np.loadtxt('test.csv',delimiter=',')
    testx=testing[:,0:4]
    testy=testing[:,4]

    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=25,input_dim=4,activation='tanh',kernel_initializer="glorot_normal",use_bias=True,bias_initializer='ones'))
    model.add(tf.keras.layers.Dense(units=25,activation='relu',kernel_initializer="he_normal",use_bias=True,bias_initializer='ones'))
    model.add(tf.keras.layers.Dense(units=25,activation='tanh',kernel_initializer="glorot_normal",use_bias=True,bias_initializer='ones'))
    model.add(tf.keras.layers.Dense(units=25,activation='relu',kernel_initializer="he_normal",use_bias=True,bias_initializer='ones'))
    model.add(tf.keras.layers.Dense(units=25,activation='relu',kernel_initializer="he_normal",use_bias=True,bias_initializer='ones'))
    model.add(tf.keras.layers.Dense(units=25,activation='tanh',kernel_initializer="glorot_normal",use_bias=True,bias_initializer='ones'))
    model.add(tf.keras.layers.Dense(units=25,activation='relu',kernel_initializer="he_normal",use_bias=True,bias_initializer='ones'))
    model.add(tf.keras.layers.Dense(units=25,activation='tanh',kernel_initializer="glorot_normal",use_bias=True,bias_initializer='ones'))
    model.add(tf.keras.layers.Dense(units=1))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    model.fit(x,y,epochs=100)
    _,accuracy=model.evaluate(x,y)
    print("Training Accuracy is: "+str(accuracy*100))
    _,accuracy=model.evaluate(testx,testy)
    print("Testing Accuracy is "+str(accuracy*100))

if __name__ =="__main__":
    main()