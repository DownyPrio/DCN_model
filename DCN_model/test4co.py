import numpy as np
from DCN_model import DNN_Layer
from DCN_model import DNN_model
from DCN_model import CN_model as CN
from DCN_model import Combination as CMB
import tensorflow as tf

trainData=np.array([[[1,2,3,4,5]],
                    [[2,3,4,5,6]],
                    [[4,5,6,7,8]]]).astype("float64")
L1=DNN_Layer.DNN_Layer(5,"relu")
L2=DNN_Layer.DNN_Layer(10,"relu")
model_list=[CN.CN_model(3),DNN_model.DNN_model([L1,L2])]
cmb=CMB.Combination(model_list)
res=cmb.predict(trainData)


