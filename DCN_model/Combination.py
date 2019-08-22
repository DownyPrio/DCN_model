import tensorflow as tf

class Combination:
    def __init__(self,modelList):
        self.cn_model=modelList[0]
        self.dnn_model=modelList[1]
    def predict(self,input):
        input_col=input.reshape((-1,input.shape[2],input.shape[1]))
        cn_res=self.cn_model.predict(input)
        dnn_res=self.dnn_model.predict(input)
        print(cn_res)
        print(dnn_res)
        flatten_res=tf.concat([cn_res,dnn_res],axis=2)
        out_Layer=tf.layers.Dense(units=1,activation=tf.nn.sigmoid)
        out_res=out_Layer(flatten_res)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            print("result=",sess.run(out_res,feed_dict={self.cn_model.x:input_col}))
        return out_res

    def fit(self,train,label,epochs):
        input_col=train.reshape((-1,train.shape[2],train.shape[1]))
        y_pred=self.predict(train)
        losses=tf.losses.log_loss(predictions=y_pred,labels=self.cn_model.y)
        opt=tf.train.AdamOptimizer(0.0001)
        train_fit=opt.minimize(losses)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epochs):
                print("epochs:{}/{}".format(i,epochs))
                sess.run(train_fit,feed_dict={self.cn_model.x:input_col,self.cn_model.y:label})
                print("losses is:")
                print(sess.run(losses,feed_dict={self.cn_model.x:input_col,self.cn_model.y:label}))

