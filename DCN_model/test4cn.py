from DCN_model import CN_model as CN
import numpy as np

trainData=np.array([[[1,2,3,4,5]],
                    [[2,3,4,5,6]],
                    [[4,5,6,7,8]]]).astype("float64")
model=CN.CN_model(3)

RES=model.predict(trainData)

import tensorflow as tf

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(RES,feed_dict={model.x:trainData.reshape((-1,5,1))}))