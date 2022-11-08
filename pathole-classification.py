import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
import pickle
from tensorflow.keras.callbacks import TensorBoard

DIRECTORY=r'D:\jupyter\pathole'
CATEGORIES=['normal','pothole']

IMG_SIZE=100;
data=[]

for category in CATEGORIES:
    folder=os.path.join(DIRECTORY,category)
    label=CATEGORIES.index(category)
    for img in os.listdir(folder):
        img_path=os.path.join(folder,img)
        img_arr=cv2.imread(img_path)
        img_arr=cv2.resize(img_arr,(IMG_SIZE,IMG_SIZE))
        data.append([img_arr,label])
        print(img_path)
        break

print(len(data))

random.shuffle(data)
X=[]
Y=[]

for features,labels in data:
    X.append(features)
    Y.append(labels)

X=np.array(X)
Y=np.array(Y)
X=X/255

print(X.shape)

pickle.dump(data,open("dataset",'wb'))

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense

name="pothole classification"
tensorboard= TensorBoard(log_dir='logs/{}'.format(name))


model=Sequential()
model.add( Conv2D(64,(3,3), activation='relu' ) )
model.add( MaxPooling2D( (2,2) ) )

model.add( Conv2D(64,(3,3), activation='relu' ) )
model.add( MaxPooling2D( (2,2) ) )

model.add( Flatten() )

model.add( Dense(128,input_shape=X.shape[1:],activation='relu') )
model.add(Dense(2,activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X,Y,epochs=5,validation_split=0.1,callbacks=[tensorboard])


#for normal road < 0.5
model.evaluate(X,Y)[0]

#for pothole roads value > 0.5
model.evaluate(X,Y)[1]

model.save("my_model")

predict_data=data[10]
x=predict_data[0]/255
print(x)

plt.imshow(x)

predictions=model.predict(np.array([x]))
print(predictions)

if(predictions[0][0]==1):
    print("The road is normal")
if(predictions[0][1]==1):
    print("The road has potholes")

predict_data=data[99]
x=predict_data[0]

plt.imshow(x)

predictions=model.predict(np.array([x]))
print(predictions)

if(predictions[0][0]==1):
    print("The road is normal")
if(predictions[0][1]==1):
    print("The road has potholes")

img= r'D:\jupyter\pathole\potholes\327.jpg'
img=cv2.imread(img)
img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
plt.imshow(img)

predictions=model.predict(np.array([img]))
print(predictions)

if(predictions[0][0]==1):
    print("The road is normal")
if(predictions[0][1]==1):
    print("The road has potholes")
