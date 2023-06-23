import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


def create_model():
    """
    Create a sequential model for handwritten digit classification.
    
    Returns:
        The created sequential model.
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # Flatten the input images
    
    # Add two dense layers with ReLU activation
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    
    # Add the output layer with softmax activation for classification
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Fit the model to the training data
    model.fit(x_train, y_train, epochs=3)
    
    # Save the trained model
    model.save('handwritten.model')


def check_loss_accuracy():
    """
    Check the loss and accuracy of a trained model on test data.

    Returns:
        None
    """
    model = tf.keras.models.load_model('handwritten.model')
    loss, accuracy = model.evaluate(x_test, y_test)

    print(loss)

    print(accuracy)


def tests():
    """
    Perform tests on handwritten digit images using a trained model.

    Returns:
        None
    """
    model = tf.keras.models.load_model('handwritten.model')
    image_number = 1
    while os.path.isfile(f'digits/{image_number}.png'):
        try:
            img = cv2.imread(f'digits/{image_number}.png')[:,:,0]
            img = np.invert(np.array([img]))
            prediction = model.predict(img)
            print(f"This digith is probably a {np.argmax(prediction)}")
            plt.imshow(img[0],cmap=plt.cm.binary)
            plt.show()
        except:
            print("Error")
        finally:
            image_number += 1
        
        
if __name__ == '__main__':
    create_model()
    check_loss_accuracy()
    tests()