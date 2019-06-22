# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 12:54:23 2019

@author: fenezema
"""

###IMPORT###
from core import *
###IMPORT###


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def convolution_layer(inputs, size):
    conv_layer = Conv2D(32, (3,3), strides=(1,1), kernel_initializer='he_normal')(inputs)
    conv_layer = BatchNormalization()(conv_layer)
    
    conv_layer = Conv2D(32, (3,3), strides=(1,1), kernel_initializer='he_normal')(conv_layer)
    conv_layer = LeakyReLU()(conv_layer)
    conv_layer = BatchNormalization()(conv_layer)
    conv_layer = MaxPooling2D((2, 2))(conv_layer)
    
    conv_layer = Conv2D(32, (2,2), strides=(1,1), kernel_initializer='he_normal')(conv_layer)
    conv_layer = LeakyReLU()(conv_layer)
    conv_layer = BatchNormalization()(conv_layer)
    conv_layer = MaxPooling2D((2, 2))(conv_layer)
#    print(conv_layer._keras_shape)
    return conv_layer

def modelBuild():
    inputs = Input(shape=(32, 32, 1))
    conv_layer = convolution_layer(inputs, 32)
    
    conv_layer = Dropout(0.5)(conv_layer)
    #print(conv_layer._keras_shape)
    
    flatten = Flatten()(conv_layer)
    outputs = Dense(36, activation='softmax')(flatten)
    
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(lr=0.0001,beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)#beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False
    # optimizer = SGD(lr=0.0001)
    # optimizer = Adagrad(lr=0.0001)
    # optimizer = RMSprop(lr=0.0001)
    return model,optimizer

def train_test_dataSplitting(path):
    train_x = []
    train_y= []
    test_x = []
    test_y = []
    dir_list = os.listdir(path)
    dir_list_char = dir_list[10:len(dir_list)]
    lab = {dir_list_char[i]:i+10 for i in range(len(dir_list_char))}
    
    for directory in dir_list:
        try:
            directory_check = lab[directory]
        except:
            directory_check = directory
        current_dir_path = path+directory+"/"
        current_dir_files = os.listdir(current_dir_path)

        random_index = []            
        n = len(current_dir_files)
        n_limit = int(30/100*n)
        i=0
        while i<n_limit:
            x = randint(0,n-1)
            if x not in random_index and x != 0 and x != 264 and x != 527:
                random_index.append(x)
                i+=1
        for i in range(n):
            temp = cv2.imread(current_dir_path+current_dir_files[i],0)
            if i in random_index:
                test_x.append(temp)
                test_y.append(directory_check)
            else:
                train_x.append(temp)
                train_y.append(directory_check)
                
    train_x = np.reshape(train_x, (len(train_x), 32, 32, 1))
    test_x = np.reshape(test_x, (len(test_x), 32, 32, 1))
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    
    return train_x,train_y,test_x,test_y

def getDatas(path):
    X = []
    Y = []
    dir_list = os.listdir(path)
    dir_list_char = dir_list[10:len(dir_list)]
    lab = {dir_list_char[i]:i+10 for i in range(len(dir_list_char))}
    
    for directory in dir_list:
        try:
            directory_check = lab[directory]
        except:
            directory_check = directory
        current_dir_path = path+directory+"/"
        current_dir_files = os.listdir(current_dir_path)
           
        n = len(current_dir_files)

        for i in range(n):
            temp = cv2.imread(current_dir_path+current_dir_files[i],0)
            X.append(temp)
            Y.append(directory_check)
                
    X = np.reshape(X, (len(X), 32, 32, 1))
    Y = np.array(Y)

    return X,Y

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
    