import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import matplotlib.pyplot as plt
(X_train,y_train),(X_test,y_test)=tf.keras.datasets.mnist.load_data()
X_train,X_test=X_train/255.0,X_test/255.0
model=keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),keras.layers.Dense(128,activation='relu'),keras.layers.Dense(10,activation='softmax')])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=5)
test_loss,test_acc=model.evaluate(X_test,y_test)
print(f'the accuracy of the model is {test_acc*100:.2f}%')
model.save('mnist_model.h5')