import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
from keras.models import Model
from keras.layers import *


def return_model(shape=(480,640,1)):
    I1 = Input(shape=shape)

    I2 = Input(shape=shape)

    I=Concatenate(axis=-1)([I1,I2])
    z = Conv2D(8,(7,7),activation="tanh")(I)
    z=MaxPooling2D((2,2))(z)
    z = Conv2D(8,(5,5),activation="tanh")(z)
    z=MaxPooling2D((2,2))(z)
    z = Conv2D(8,(3,3),activation="tanh")(z)
    z = Conv2D(8,(3,3),activation="tanh")(z)
    z=Flatten()(z)
    z=Dense(500,activation="tanh")(z)
    z=Dense(90,activation="tanh")(z)
    z=Dense(9,activation="linear")(z)

    model = Model(inputs=[I1,I2], outputs=[z])
    model.compile(loss="mse",optimizer="Adam")
    return model
##----------------------------------------------
def return_new(model):
    I1=model.inputs[0]
    I2=model.inputs[1]
    o1=model.outputs[0]   
    wt_err = .0001
    e=(K.abs(1-K.sum(o1*o1,axis=-1)))*(wt_err)


    model.add_loss(e)
    new_model=Model(inputs=[I1,I2],outputs=[o1])
    model.compile(loss="mse",optimizer="Adam")
    return model
##----------------------------------------------



model_temp=return_model()
model=return_new(model_temp)



#-------------------DATA-------------------
data = np.load('../training_data.npz')
X1 = data['X1']/255.0
X2 = data['X2']/255.0

y = data['e'].reshape((36,9))

#-------------------------Training-----------

model.fit([X1,X2],y,batch_size=3,epochs=100)

# model.load_weights('data/model_1')

y1=model.predict([X1,X2])

np.savez('prediction_model1', e =y1)

# modX1el.predict([X1[0:1],X2[:1]])



















# ssh -X  ranjan@192.168.59.103


