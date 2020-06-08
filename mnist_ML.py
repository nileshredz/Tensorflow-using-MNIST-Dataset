# Import and Load the MNIST Dataset

# Import Tensorflow and Time Library:


import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import time


#For Tensorboard Look at the Current working Directory and see the Location in tensorboard 
NAME = "MNIST_NN_{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir = 'MNIST\logs\{}'.format(NAME))

mnist = tf.keras.datasets.mnist    # Handwritten 0-9 Images.

## Import matplotlib for Image show 


import matplotlib.pyplot as plt

plt.imshow(x_train[0])
plt.show()

#Plot Image as Binary Image 
plt.imshow(x_train[0] ,cmap = plt.cm.binary)




print(x_train[0])


# Load the MNIST Dataset from Tensorflow Build a Model, Choose the Optimizers, Loss Function and Metrics and Train the Model:

(x_train,y_train), (x_test,y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis =1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

model.compile(optimizer = 'adam' ,loss= 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 5, callbacks = [tensorboard])


# The Model has been trained which should have low loss function and High Accuracy. But this is not an actionable model yet. 
# We have to Test the Model using test Data and Evaluate.



val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)



#Save a Model
model.save('mnist_nnmodel') # Save the Model

#Load the Saved Model
LoadModel = tf.keras.models.load_model('mnist_nnmodel') #Load the Saved Model



predictions = LoadModel.predict(x_test) #Make Predictions using the Model
print(predictions)

#Find the Maximum Probabibility of the Target Number
print(np.argmax(predictions[0])) #Argmax is most commonly used in machine learning for finding the class with the largest predicted probability.

#to plot the Image Being Predicted
plt.imshow(x_test[0])
plt.show()







# In[ ]:




