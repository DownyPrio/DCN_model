import tensorflow as tf
import numpy as np

class CN_model:
    def __init__(self,degree):
        self.degree=degree
        self.built=False

    def __build(self,input):
        #转化列向量
        self.samples=input.shape[0]
        self.d=input.shape[1]
        self.dimension=input.shape[2]

        self.x=tf.placeholder(dtype=tf.float64,shape=(None,self.dimension,self.d))
        self.y=tf.placeholder(dtype=tf.float64,shape=(None,1,1))
        self.x0=self.x
        print("..........")
        self.wList=[]
        self.bList=[]
        for each in range(self.degree):
            tmp_w=tf.Variable(tf.zeros((self.dimension,self.d),dtype=tf.float64))
            tmp_b=tf.Variable(tf.zeros((self.dimension,self.d),dtype=tf.float64))
            self.wList.append(tmp_w)
            self.bList.append(tmp_b)
        self.built=True
        print("build sucess")

    def forward(self,w,b,tensor_tmp):
        return tf.matmul(tf.matmul(self.x,tf.transpose(tensor_tmp,perm=(0,2,1))),w)+b+tensor_tmp


    def predict(self,input):
        if not self.built:
            self.__build(input)
        #input=input.reshape((-1,input.shape[2],input.shape[1]))
        tmp=self.x
        for index in range(self.degree):
            tmp=self.forward(self.wList[index],self.bList[index],tmp)
            print("index:")
            print(index)
        print("predict com")
        res=tf.transpose(tmp,perm=(0,2,1))
        return res




