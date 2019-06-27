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

def doMakeModel(train_x,train_y,test_x,test_y,toggle=False, kFolds_checker=False, k=10, X=None, Y=None):
    if kFolds_checker == True:
        print("Number of batches : "+str(k))
        seed = 7
        np.random.seed(seed)
        kfold = StratifiedKFold(n_splits=k,shuffle=True,random_state=seed)
        cvscores = []
        for index,(train, test) in enumerate(kfold.split(X,Y)):
            xtrain, xval = X[train], X[test]
            ytrain, yval = to_categorical(Y[train]), to_categorical(Y[test])
            model,optimizer = modelBuild()
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(xtrain, ytrain, batch_size=50, epochs=200, verbose=1,validation_data=(xval,yval))
            scores = model.evaluate(xval,yval)
            print("%s : %.2f%%" % (model.metrics_names[1],scores[1]*100))
            cvscores.append(scores[1]*100)
        print("All batches accuracy : ")
        for element in cvscores:
            print("Accuracy : "+str(element))
        print("After all batches accuracy : %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    elif kFolds_checker == False:
        if toggle==True:
            model,optimizer = modelBuild()
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            print(model.summary())
            time_callback = TimeHistory()
            history = model.fit(train_x, train_y, batch_size=50, epochs=200, verbose=1, callbacks=[time_callback], validation_data=(test_x, test_y))
            
            print("Time:")
            print(sum(time_callback.times))
            
            scenario_name = 'Adam_Kernel3_3x3_0,0001_200epochs'
            evaluate_model(model, scenario_name, test_x, test_y)
            print_plot(history, scenario_name)
            
            test = return_to_label(test_y)
            pred_y = model.predict(test_x)
            pred = return_to_label(pred_y)
            print(classification_report(test, pred))
        elif toggle==False:
            print("Make Model skipped. No model has been made")


def main():
    args = sys.argv
    kfolds_index = args.index('--kfolds_flag') + 1
    if args[kfolds_index] == 'True':
        kfolds_flag = True
        k_ind = args.index('--k') + 1
        try:
            k_nya = int(args[k_ind])
        except:
            return 0
    elif args[kfolds_index] == 'False':
        kfolds_flag = False
        toggle_ind = args.index('--toggle') + 1
        if args[toggle_ind] == 'True':
            toggle_nya = True
        elif args[toggle_ind] == 'False':
            toggle_nya = False
    dataset_path = "resources/Datasets/" # "F:\\Kuliah\\Tugas Akhir\\05111540000055_PBaskara\\src\\resources\\Datasets\\"
    saved_path = "resources/Processed/" #D:\05111540000055_PBaskara\src "F:\\Kuliah\\Tugas Akhir\\05111540000055_PBaskara\\src\\resources\\Processed\\"
    resources_path = "resources/" #D:\05111540000055_PBaskara\src
    
    img = ImagePreprocessing(dataset_path,saved_path) #ends directory name with \\(Windows) or /(UNIX)
    dir_path = img.getDirPath()
    
    #do preprocessing for training
    doPreprocessing(img,dir_path,False)
    #do preprocessing for training
    
    if kfolds_flag == False:
        #split data to train and test
        train_x,train_y,test_x,test_y = train_test_dataSplitting(saved_path)
        # print(len(train_x),len(train_y),len(test_x),len(test_y))
        doMakeModel(train_x,train_y,test_x,test_y,toggle=toggle_nya)
    elif kfolds_flag == True:
        #if using k-Folds Cross Validation
        X, Y = getDatas(saved_path)
        #if using k-Folds, fill X and Y from returned value from getDatas and fill 4 first parameter with None
        doMakeModel(None,None,None,None, kFolds_checker=True, k=k_nya, X=X, Y=Y)
    
    #if using k-Folds, fill X and Y from returned value from getDatas and fill 4 first parameter with None
    # doMakeModel(train_x,train_y,test_x,test_y,toggle=True)
    # doMakeModel(None,None,None,None,toggle=True, kFolds_checker=True, k=5, X=X, Y=Y)
    
if __name__=="__main__":
    main()