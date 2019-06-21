import numpy as np
import os
import glob
from skimage import transform, io
import keras
from keras.models import Model
from keras.layers import Input, Activation, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam, Adagrad, SGD, RMSprop
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def load_dataset(path, size, random_index):
    train_x = []
    test_x = []
    train_y = []
    test_y = []
    cat = {'AF' : 0, 'AN' : 1, 'DI' : 2, 'HA' : 3, 'NE' : 4, 'SA' : 5, 'SU' : 6}
    
    r1 = []
    test_path = os.path.join(path, '*g')
    test_files = glob.glob(test_path)
    for tmp in test_files:
        image = io.imread(tmp, as_gray=True)
        image = transform.resize(image, (size,size), mode='symmetric', preserve_range=True)
        split = tmp.split("\\")
        fileName = split[len(split) - 1].split(".")[0]
        r1.append([image[:,:], cat[fileName[4:6]]]);
        
    for i in range(2450):
        train_x.append(r1[i][0])
        train_y.append(r1[i][1])
        
    for i in range(len(random_index)):
        if i < 1450:  
            train_x.append(r1[random_index[i]][0])
            train_y.append(r1[random_index[i]][1])
        else:
            test_x.append(r1[random_index[i]][0])
            test_y.append(r1[random_index[i]][1]) 
            
    train_x = reshape(train_x, size)
    test_x = reshape(test_x, size)
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
         
    return train_x,train_y,test_x,test_y

def reshape(data, size):    
    data = np.array(data)
    data = data.astype('float32') / 255.
    data = np.reshape(data, (len(data), size, size, 1))
    return data

def wave_convolution_layer(inputs, size):
    conv_layer = Conv2D(32, (5,5), strides=(1,1), activation='relu', kernel_initializer='he_normal')(inputs)
    conv_layer = BatchNormalization()(conv_layer)
    conv_layer = MaxPooling2D((2, 2))(conv_layer)
    
    conv_layer = Conv2D(32, (5,5), strides=(1,1), activation='relu', kernel_initializer='he_normal')(conv_layer)
    conv_layer = BatchNormalization()(conv_layer)
    conv_layer = MaxPooling2D((2, 2))(conv_layer)
    
    conv_layer = Conv2D(32, (5,5), strides=(1,1), activation='relu', kernel_initializer='he_normal')(conv_layer)
    conv_layer = BatchNormalization()(conv_layer)
    conv_layer = MaxPooling2D((2, 2))(conv_layer)
#    print(conv_layer._keras_shape)
    return conv_layer

def model_build():
    inputs1 = Input(shape=(127, 127, 1))
    conv_layer1 = wave_convolution_layer(inputs1, 127)
    inputs2 = Input(shape=(128, 128, 1))
    conv_layer2 = wave_convolution_layer(inputs2, 128)
    inputs3 = Input(shape=(129, 129, 1))
    conv_layer3 = wave_convolution_layer(inputs3, 129)
    inputs4 = Input(shape=(130, 130, 1))
    conv_layer4 = wave_convolution_layer(inputs4, 130)
    
    print(conv_layer4._keras_shape)
    
    concat = keras.layers.Concatenate()([conv_layer1, conv_layer2, conv_layer3, conv_layer4])
    print(concat._keras_shape)
        
    conv_layer = Dropout(0.4)(concat)
    conv_layer = Conv2D(64, (5,5), strides=(1,1), activation='relu', kernel_initializer='he_normal')(conv_layer)
    print(conv_layer._keras_shape)
    conv_layer = BatchNormalization()(conv_layer)
    conv_layer = Activation('relu')(conv_layer)
    conv_layer = MaxPooling2D((2,2))(conv_layer)
    print(conv_layer._keras_shape)
    
    conv_layer = Dropout(0.2)(conv_layer)
    conv_layer = Conv2D(7, (4,4), strides=(1,1), activation='relu', kernel_initializer='he_normal')(conv_layer)
    print(conv_layer._keras_shape)
    
    flatten = Flatten()(conv_layer)
    outputs = Dense(7, activation='softmax')(flatten)
    
    model = Model(inputs=[inputs1, inputs2, inputs3, inputs4], outputs=outputs)
    return model

def read_random_index(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [int(x.strip()) for x in content] 
    return content

def print_plot(history, filename):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
     
def evaluate_model(model, scenario_name, test_x, test_y):
    print(scenario_name)
    model.save_weights(scenario_name+'.h5')
    print(model.evaluate(test_x, test_y))
    print(model.metrics_names)
    
def return_to_label(y):
    label = []
    for i in range(len(y)):
        label.append(np.argmax(y[i]))
    return label

test_random_index = read_random_index('sesi_index4.txt')

data_path = "D:\\Tugas Akhir Hendry 05111540000102\\skenario\\wavelet\\"
train_HH_x, train_HH_y, test_HH_x, test_HH_y = load_dataset(data_path+"HH", 127, test_random_index) 
train_HL_x, train_HL_y, test_HL_x, test_HL_y = load_dataset(data_path+"HL", 128, test_random_index) 
train_LH_x, train_LH_y, test_LH_x, test_LH_y = load_dataset(data_path+"LH", 129, test_random_index) 
train_LL_x, train_LL_y, test_LL_x, test_LL_y = load_dataset(data_path+"LL", 130, test_random_index)

model = model_build()
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
sgd = SGD(lr=0.0001)
rmsprop = RMSprop(lr=0.001)
adagrad=Adagrad(lr=0.0001)

model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
#print(model.summary())

train_x = [train_HH_x, train_HL_x, train_LH_x, train_LL_x] 
train_y = [train_HH_y] 

test_x = [test_HH_x, test_HL_x, test_LH_x, test_LL_x]
test_y = [test_HH_y]

time_callback = TimeHistory()
history = model.fit(train_x, train_y, batch_size=39, epochs=500, verbose=1, callbacks=[time_callback], validation_data=(test_x, test_y))

print("Time:")
print(sum(time_callback.times))

scenario_name = '500_rmsprop_0,001'
evaluate_model(model, scenario_name, test_x, test_y)
print_plot(history, scenario_name)

test = return_to_label(test_HH_y)
pred_y = model.predict(test_x)
pred = return_to_label(pred_y)
print(classification_report(test, pred))