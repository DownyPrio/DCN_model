import tensorflow as tf
import numpy as np

import tensorflow as tf
import numpy as np
import random
import time

from tensorflow.python.client import device_lib


list_p=[1]*1000
list_1=list(map(lambda x:x*random.randrange(1,100),list_p))

list_2=list(map(lambda x:x*random.randrange(1,100),list_p),"33333333333334")


trainSet=np.array([list_1,list_2]).T.reshape((1000,1,2)).astype("float64")

print(trainSet.shape)
#
# print(trainFinalSet)
w=np.array([[3],[4]])
b=np.array([[100]])
n=np.array([[1,2]]).astype('float64')
labelSet=((np.matmul(trainSet,w)+b)).astype("float64").reshape((-1,1,1)).astype("float64")


trainData=np.array([[[1,2,3,4,5]],
                    [[2,3,4,5,6]],
                    [[4,5,6,7,8]]]).astype("float64")
labelData=np.array([[[1]],[[1]],[[1]]])

class dnn:


    def build(self,input):
        self.m=input.shape[0]
        self.x1=input.shape[1]
        self.x2=input.shape[2]
        self.x=tf.placeholder(shape=(None,self.x1,self.x2),dtype=tf.float64)
        self.y=tf.placeholder(shape=(None,1,1),dtype=tf.float64)
        self.w=tf.Variable(tf.zeros((self.x2,self.x1),dtype=tf.float64))
        self.b=tf.Variable(tf.zeros((1,1),dtype=tf.float64))
        self.com()

    def com(self):
        self.res=tf.matmul(self.x,self.w)+self.b

    def predict(self):
        return tf.matmul(self.x,self.w)+self.b

    def fit(self,train,label,epochs):
        y_pred=self.res
        losses=tf.losses.mean_squared_error(predictions=y_pred,labels=self.y)
        opt=tf.train.AdamOptimizer(0.001)#GradientDescentOptimizer(0.000001)
        train_fit=opt.minimize(losses)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epochs):
                print("epochs:{}/{}".format(i,epochs))
                sess.run(train_fit,feed_dict={self.x:train,self.y:label})
                # print("pred:")
                # print(sess.run(y_pred,feed_dict={self.x:train,self.y:label}))
                print("losses is:")
                print(sess.run(losses,feed_dict={self.x:train,self.y:label}))

model=dnn()
model.build(trainSet)
model.com()
model.fit(trainSet,labelSet,10000)


