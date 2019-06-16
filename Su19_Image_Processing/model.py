import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
from keras.models import Model
from keras.layers import *


I1 = Input(shape=(480,640,1))

I2 = Input(shape=(480,640,1))

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



#-------------------DATA-------------------
data = np.load('../training_data.npz')
X1 = data['X1']/255.0
X2 = data['X2']/255.0

y = data['e'].reshape((36,9))




#-------------------------Training-----------
model.fit([X1,X2],y,batch_size=,epochs=100)

y2=model.predict([X1,X2])

np.savez('prediction_model', e =y2)

# RT=model.predict([X1[0:1],X2[:1]])
# model.predict([X1[0:1],X2[:1]])



















# ssh -X  ranjan@192.168.59.103
# ssh -X  ecsuiplab@192.168.59.241

