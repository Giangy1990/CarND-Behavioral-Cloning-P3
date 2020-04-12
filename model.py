# import section
import os
import math
import csv
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Conv2D, Cropping2D, Lambda
from keras.optimizers import Adam

# load data function
def load_data(data_dir, csv_name='driving_log.csv', imgs_dir='IMG'):
    csv_file = os.path.join(data_dir, csv_name)
    img_path = os.path.join(data_dir, imgs_dir)
    
    dataset = list()
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            steering_center = float(row[3])
            # create adjusted steering measurements for the side camera images
            steering_left = steering_center + .2
            steering_right = steering_center - .2
            
            img_center = os.path.join(img_path, row[0].split('\\')[-1])
            img_left = os.path.join(img_path, row[1].split('\\')[-1])
            img_right = os.path.join(img_path, row[2].split('\\')[-1])
            
            dataset.append([img_center, steering_center])
            dataset.append([img_left, steering_left])
            dataset.append([img_right, steering_right])
    shuffle(dataset)
    train_dataset, validation_dataset = train_test_split(dataset, test_size=0.3)
    
    return train_dataset, validation_dataset

# create the keras model to train
def create_model(img_input_shape, rate=0.001):
    model = Sequential()
    
    #Nvidia architecture + Dropout
    # Normalization
    model.add(Lambda(lambda x: (x / 127.5) - 1, input_shape=img_input_shape))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Lambda(lambda x: tf.image.resize_images(x, (66, 200))))
    # convolution + dropuot
    model.add(Conv2D(24, (5, 5), activation='relu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='relu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='relu', strides=(2, 2)))
    model.add(Dropout(0.6))
    # convolution + flatten + dropuot
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    # fully connected
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    
    model.compile(loss='mse', optimizer=Adam(lr=rate))
    return model

# data generator
def dataGenerator(samples, batch_size=64):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
        
            imgs = list()
            angles = list()
            for batch_sample in batch_samples:
                img = mpimg.imread(batch_sample[0])
                #img = self.process_image(img)
                imgs.append(img)

                angle = float(batch_sample[1])
                angles.append(angle)

                # randomly select images to flip
                X_train = np.array(imgs)
                y_train = np.array(angles)
                flip_indices = np.random.choice(len(X_train), len(X_train)//2)
                X_train[flip_indices] = np.fliplr(X_train[flip_indices])
                y_train[flip_indices] = -y_train[flip_indices]

                yield shuffle(X_train, y_train)

# mail procedure
def main():
    # data
    dataset_dir = '../simulator_data'
    img_input_shape = (160, 320, 3)
    learning_rate = 0.0001
    batch_size = 512
    model_name = 'model.h5'
    
    ## Train steps
    # 1 load dataset
    train_dataset, validation_dataset = load_data(dataset_dir)
    
    print("train_dataset samples ", len(train_dataset))
    print("validation_dataset samples ", len(validation_dataset)) 
    
    # 3 generators definition
    training_generator = dataGenerator(shuffle(train_dataset), batch_size)
    validation_generator = dataGenerator(shuffle(validation_dataset), batch_size)
    
    # 3 create model
    model = create_model(img_input_shape, learning_rate)
    
    # 4 Checkpoints definition: EarlyStopping and ModelCheckpoint
    checkpoint = ModelCheckpoint(filepath="weights.{epoch:02d}-{val_loss:.4f}.hdf5", monitor='val_loss', save_best_only=True)
    stopper = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience=5)
    
    # 5 Train model
    history_object = model.fit_generator(generator=training_generator, validation_data=validation_generator,
                                         steps_per_epoch=math.ceil(len(train_dataset)/batch_size),
                                         validation_steps=math.ceil(len(validation_dataset)/batch_size),
                                         use_multiprocessing=True,
                                         workers=8,
                                         verbose=1,
                                         callbacks=[checkpoint, stopper],
                                         epochs=10)
    
    print(history_object.history.keys())
    #  Plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    # Saving model
    model.save(model_name)

# call main
if __name__ == '__main__':
    main()