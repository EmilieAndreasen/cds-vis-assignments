#####
# Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks
# Author: Emilie Munch Andreasen
# Date: 10-05-2024
#####

######
# Classification benchmark for neural network
######

#importing packages 
import cv2
import numpy as np
import sys
import os
import pandas as pd
from tensorflow.keras.datasets import cifar10
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn import metrics
import matplotlib.pyplot as plt


##### FUNCTIONS

#function for greyscaling imanges 
def greyscaling(img_array):
    grey_list = []

    for inx,i in enumerate(range(0,len(img_array))):
        grey = cv2.cvtColor(img_array[i], cv2.COLOR_BGR2GRAY)
        grey_list.append(grey)

    return grey_list

#function for normalizing images 
def norm_img(grey_images):
    norm_list = []

    for inx,i in enumerate(range(0,len(grey_images))):
        norm = cv2.normalize(grey_images[i], grey_images[i], 0, 1.0, cv2.NORM_MINMAX)
        norm_list.append(norm)

    return norm_list


#function for flattening greyscaled images
def flat_img(grey_images):
    flat_list = []
    flat_nr = grey_images[0].shape[0]*grey_images[0].shape[1]

    for inx,i in enumerate(range(0,len(grey_images))):
        flat = grey_images[i].reshape(-1, flat_nr)
        flat_list.append(flat)

    return flat_list

#function for renaming labels
def label_names(train, test):

    #list of labels corrisponing to numerical value
    label_map = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }

    #flattening array
    list_y_train = train.flatten()
    list_y_test = test.flatten()
    
    #chaning the numerical values to label name
    y_train = np.array([label_map[label] for label in list_y_train])
    y_test = np.array([label_map[label] for label in list_y_test])

    return y_train, y_test


#function for preprocessing images 
def img_prep(train, test):
    #greyscaling
    grey_train = greyscaling(train)
    grey_test = greyscaling(test)
    #normalizing
    norm_train = norm_img(grey_train)
    norm_test = norm_img(grey_test)
    #flattening
    flat_train = flat_img(norm_train)
    flat_test = flat_img(norm_test)

    #concatenating the list of arrays into single 2D arrays (so they can be used in the classifier function)
    flat_train_array = np.vstack(flat_train)
    flat_test_array = np.vstack(flat_test)


    return flat_train_array, flat_test_array


#function for neural network classifier
def nnw_classifier(x_train, y_train):
    #defining the model
    classifier = MLPClassifier(activation= "logistic",
                            hidden_layer_sizes = (64,),
                            max_iter = 1000,
                            random_state = 42)
    
    #fitting the model (training)
    nnw_model = classifier.fit(x_train, y_train)

    return nnw_model


#function for classification report (testing)
def classification_testing(classifier, x_test, y_test):
    #predicting on test data
    y_pred = classifier.predict(x_test)
    
    #making a report
    classifier_metrics = metrics.classification_report(y_test, y_pred)

    return classifier_metrics, y_pred


#function for saving reports
def save_report(report, report_name):
    path = os.path.join("..", "out", f"{report_name}.txt")
    
    with open(path, "w") as report_file:
        report_file.write(report)


#function for saving loss curve
def loss_curve(classifier, plot_name):
    #making the loss curve during training
    plt.plot(classifier.loss_curve_)
    plt.title("Loss curve during training on Cifar10 data", fontsize=14)
    plt.xlabel('Iterations')
    plt.ylabel('Loss score')

    #saving plot in out folder
    path = os.path.join("..", "out", f"{plot_name}.png")
    plt.savefig(path)





##### MAIN CODE

def main():
    #load in data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    #preproces image data
    X_train_flat, X_test_flat = img_prep(X_train,X_test)
    
    #rename the labels
    y_train_lab, y_test_lab = label_names(y_train, y_test)

    #classification training
    print("classification might take some time (estimate time with u1-standard-64: 15 min)")
    classifier = nnw_classifier(X_train_flat, y_train_lab)

    #testing
    report, y_pred = classification_testing(classifier, X_test_flat, y_test_lab)

    #saving into "out" folder
    save_report(report, "neural_network_report")
    print("report saved!")

    #saving loss curve plot in "out" folder
    loss_curve(classifier, "loss_curve_nnw")
    print("plot saved!")
    
if __name__=="__main__":
    main()

