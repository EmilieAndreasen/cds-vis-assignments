#####
# Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks
# Author: Emilie Munch Andreasen
# Date: 12-05-2024
#####

# Importing libraries
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import argparse

# Defining argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description='Run classification benchmarks using Neural Networks')
    parser.add_argument('--output_dir', type=str, default='../out', help='Directory to save the output report and plot')
    return parser.parse_args()

##### 
# Defining Functions
#####

def greyscaling(img_array):
    """
    Converts an array of images to grayscale.
    
    Parameters:
        img_array (np.array): Array of images.
    
    Returns:
        list: List of grayscale images.
    """
    grey_list = []
    for inx, i in enumerate(img_array):
        grey = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        grey_list.append(grey)
    return grey_list

def norm_img(grey_images):
    """
    Normalises grayscale images.
    
    Parameters:
        grey_images (list): List of grayscale images.
    
    Returns:
        list: List of normalized images.
    """
    norm_list = []
    for grey_image in grey_images:
        norm = cv2.normalize(grey_image, None, 0, 1.0, cv2.NORM_MINMAX)
        norm_list.append(norm)
    return norm_list

def flat_img(grey_images):
    """
    Flattens a list of grayscale images.
    
    Parameters:
        grey_images (list): List of normalised grayscale images.
    
    Returns:
        list: List of flattened images.
    """
    flat_list = []
    flat_nr = grey_images[0].shape[0] * grey_images[0].shape[1]
    for grey_image in grey_images:
        flat = grey_image.reshape(-1, flat_nr)
        flat_list.append(flat)
    return flat_list

def label_names(train, test):
    """
    Converts numeric labels to categorical names using the CIFAR-10 label set.
    
    Parameters:
        train (np.array): Training labels.
        test (np.array): Testing labels.
    
    Returns:
        tuple: Tuple containing arrays of training and testing labels with names.
    """
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
    y_train = np.array([label_map[label] for label in train.flatten()])
    y_test = np.array([label_map[label] for label in test.flatten()])
    return y_train, y_test

def img_prep(train, test):
    """
    Preprocesses image data for neural network model.
    
    Parameters:
        train (np.array): Training images.
        test (np.array): Testing images.
    
    Returns:
        tuple: Tuple containing flattened arrays of training and testing images.
    """
    grey_train = greyscaling(train)
    grey_test = greyscaling(test)

    norm_train = norm_img(grey_train)
    norm_test = norm_img(grey_test)

    flat_train = flat_img(norm_train)
    flat_test = flat_img(norm_test)

    flat_train_array = np.vstack(flat_train)
    flat_test_array = np.vstack(flat_test)
    return flat_train_array, flat_test_array

def nnw_classifier(x_train, y_train):
    """
    Trains a neural network classifier.
    
    Parameters:
        x_train (np.array): Flattened training images.
        y_train (np.array): Training labels.
    
    Returns:
        MLPClassifier: Trained neural network model.
    """
    classifier = MLPClassifier(activation="logistic", 
                            hidden_layer_sizes=(64,), 
                            max_iter=1000, 
                            random_state=42)
    nnw_model = classifier.fit(x_train, y_train)
    return nnw_model

def classification_testing(classifier, x_test, y_test):
    """
    Tests a trained neural network classifier and generates a classification report.
    
    Parameters:
        classifier (MLPClassifier): Trained neural network model.
        x_test (np.array): Flattened testing images.
        y_test (np.array): Testing labels.
    
    Returns:
        tuple: Tuple containing the classification report and predictions.
    """
    y_pred = classifier.predict(x_test)
    classifier_metrics = metrics.classification_report(y_test, y_pred)
    return classifier_metrics, y_pred

def save_report(report, report_name, output_dir):
    """
    Saves classification report to a file.
    
    Parameters:
        report (str): Text of the classification report.
        report_name (str): Name of the report file.
        output_dir (str): Directory to save the report.
    """
    path = os.path.join(output_dir, f"{report_name}.txt")

    os.makedirs(output_dir, exist_ok=True)

    with open(path, "w") as report_file:
        report_file.write(report)

def loss_curve(classifier, plot_name, output_dir):
    """
    Generates and saves the loss curve of the neural network training process.
    
    Parameters:
        classifier (MLPClassifier): Trained neural network model.
        plot_name (str): Name of the plot file.
        output_dir (str): Directory to save the plot.
    """
    plt.plot(classifier.loss_curve_)
    plt.title("Loss curve during training on CIFAR-10 data", fontsize=14)
    plt.xlabel('Iterations')
    plt.ylabel('Loss score')

    path = os.path.join(output_dir, f"{plot_name}.png")
    plt.savefig(path)

#####
# Main Function
#####

def main():
    args = parse_arguments()

    print("Loading the CIFAR-10 dataset...")
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    print("Preprocessing images...")
    X_train_flat, X_test_flat = img_prep(X_train, X_test)
    y_train_lab, y_test_lab = label_names(y_train, y_test)

    print("Training neural network classifier... This can take a while (estimate time with u1-standard-64: 15 min)")
    classifier = nnw_classifier(X_train_flat, y_train_lab)

    print("Classification underway...")
    report, y_pred = classification_testing(classifier, X_test_flat, y_test_lab)

    print("Saving classification report...")
    save_report(report, "neural_network_report", args.output_dir)
    print(f"Results saved to {args.output_dir}")

    print("Saving loss curve plot...")
    loss_curve(classifier, "loss_curve_nnw", args.output_dir)
    print(f"Plot saved to {args.output_dir}")
    
if __name__ == "__main__":
    main()
