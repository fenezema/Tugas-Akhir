# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 11:22:06 2019

@author: fenezema
"""

###IMPORT###
from Preprocessing import *
from ModelBuild import *
###IMPORT###

def doPreprocessing(img,dir_path,toggle_flag):
    if toggle_flag == True:
        dir_list = os.listdir(dir_path) #get directory list in Datasets directory
        #print(type(dir_list))
        #print(dir_list)
        
        for element in dir_list:
            if len(element)==1:
                img.setCurrentPath(element)
                img.augmentImages(img.getCurrentPath(),False)
                saved_in_path = img.directory_path_processed+element
                file_list = os.listdir(img.getCurrentPath())
                print(file_list)
                
                for currentDir_element in file_list:
                    #print(type(currentDir_element))
                    #print(currentDir_element)
                    img.setFilename(currentDir_element)
                    try:
                        os.makedirs(saved_in_path)
                        os.chdir(saved_in_path)
                        img.toBinary(resizeImg=True,sizeImg=32)
                    except:    
                        os.chdir(saved_in_path)
                        img.toBinary(resizeImg=True,sizeImg=32)
                    os.chdir("D:\\05111540000055_PBaskara\\src")
    elif toggle_flag == False:
        print("Preprocessing Data process skipped")

def doMakeModel(train_x,train_y,test_x,test_y,toggle=True):
    if toggle==True:
        model,optimizer = modelBuild()
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        time_callback = TimeHistory()
        history = model.fit(train_x, train_y, batch_size=50, epochs=200, verbose=1, callbacks=[time_callback], validation_data=(test_x, test_y))
        
        print("Time:")
        print(sum(time_callback.times))
        
        scenario_name = 'ADAM_0,0001_200epochs'
        evaluate_model(model, scenario_name, test_x, test_y)
        print_plot(history, scenario_name)
        
        test = return_to_label(test_y)
        pred_y = model.predict(test_x)
        pred = return_to_label(pred_y)
        print(classification_report(test, pred))
    elif toggle==False:
        print("Make Model skipped. No model has been made")


def main():
    dataset_path = "resources/Datasets/" # "F:\\Kuliah\\Tugas Akhir\\05111540000055_PBaskara\\src\\resources\\Datasets\\"
    saved_path = "resources/Processed/" #D:\05111540000055_PBaskara\src "F:\\Kuliah\\Tugas Akhir\\05111540000055_PBaskara\\src\\resources\\Processed\\"
    resources_path = "resources/" #D:\05111540000055_PBaskara\src
    
    img = ImagePreprocessing(dataset_path,saved_path) #ends directory name with \\(Windows) or /(UNIX)
    dir_path = img.getDirPath()
    
    #do preprocessing for training
    doPreprocessing(img,dir_path,False)
    #do preprocessing for training
    
    #split data to train and test
    train_x,train_y,test_x,test_y = train_test_dataSplitting(saved_path)
    print(len(train_x),len(train_y),len(test_x),len(test_y))
    
    doMakeModel(train_x,train_y,test_x,test_y,toggle=True)
    
if __name__=="__main__":
    main()