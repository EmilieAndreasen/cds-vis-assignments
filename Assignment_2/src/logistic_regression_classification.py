######
# Classification benchmark for logistic regression
######

#importing packages 
import cv2
import numpy as np
import sys
import os
import pandas as pd
from tensorflow.keras.datasets import cifar10
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
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
    flat_nr = grey_images[0].shape[0]*grey_images[0].shape[1] #32 * 32 (but can take images in other dimensions)

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
    
    #changing the numerical values to label name
    y_train = np.array([label_map[label] for label in list_y_train])
    y_test = np.array([label_map[label] for label in list_y_test])

    return y_train, y_test


#function for preprocessing images (with previously defined functions)
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


#function for logistic regression classifier (training)
def lgr_classifier(x_train, y_train):
    #Specifying the model and fitting it
    classifier = LogisticRegression(random_state=42).fit(x_train, y_train)

    return classifier 


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


##### MAIN CODE

def main():
    #load in data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    #preproces image data
    X_train_flat, X_test_flat = img_prep(X_train,X_test)
    
    #rename the labels
    y_train_lab, y_test_lab = label_names(y_train, y_test)

    #classification training
    classifier = lgr_classifier(X_train_flat, y_train_lab)

    #testing
    print("classification takes a few minutes")
    report, y_pred = classification_testing(classifier, X_test_flat, y_test_lab)

    #saving into "out" folder
    save_report(report, "logistic_regression_report")
    print("report saved!")

    
if __name__=="__main__":
    main()

