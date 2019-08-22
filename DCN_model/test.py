import tensorflow as tf
import numpy as np
import tensorflow as tf
import random
import time



list_p=[1]*200
list_1=list(map(lambda x:x*random.randrange(1,100),list_p))

list_2=list(map(lambda x:x*random.randrange(1,100),list_p))


trainSet=np.array([list_1,list_2]).T.reshape((200,1,2))

print(trainSet.shape)
#
# print(trainFinalSet)
w=np.array([[3],[4]])
b=np.array([[100]])
n=np.array([[1,2]]).astype('float64')
labelSet=((np.matmul(trainSet,w)+b)/800).astype("float64").reshape((-1,1,1))
print(labelSet.shape)


from DCN_model import DNN_Layer
from DCN_model import DNN_model
from DCN_model import CN_model as CN
from DCN_model import Combination as CMB


list_p=[1]*10000
list_1=list(map(lambda x:x*random.randrange(1,100),list_p))

list_2=list(map(lambda x:x*random.randrange(1,100),list_p))


trainSet=np.array([list_1,list_2]).T.reshape((10000,1,2)).astype("float64")

print(trainSet.shape)
#
# print(trainFinalSet)
w=np.array([[3],[4]])
b=np.array([[100]])
n=np.array([[1,2]]).astype('float64')
labelSet=((np.matmul(trainSet,w)+b)/800).astype("float64").reshape((-1,1,1)).astype("float64")




trainData=np.array([[[1,2,3,4,5]],
                    [[2,3,4,5,6]],
                    [[4,5,6,7,8]]]).astype("float64")
labelData=np.array([[[1]],[[1]],[[1]]])
L1=DNN_Layer.DNN_Layer(5,"relu")
L2=DNN_Layer.DNN_Layer(10,"relu")
model_list=[CN.CN_model(3),DNN_model.DNN_model([L1,L2])]
cmb=CMB.Combination(model_list)
# res=cmb.predict(trainSet)
cmb.fit(trainSet,labelSet,10000)

test_data=np.array([[[100,500]]]).astype("float64")
cmb.predict(test_data)