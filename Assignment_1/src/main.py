#####
# Assignment 1 - Building a Simple Image Search Algorithm
# Author: Emilie Munch Andreasen
# Date: 28-04-2024
#####

# Importing libraries
import os
import sys
import cv2
import numpy as np
import pandas as pd
import argparse
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.neighbors import NearestNeighbors

# Defining argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description='Image Search using either Colour Histograms or a pretrained CNN')
    parser.add_argument('--method', choices=['histogram', 'cnn'], required=True, help='Choose "histogram" for Color Histograms or "cnn" for CNN-based search')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset directory containing images')
    parser.add_argument('--output_dir', type=str, default='../out', help='Output directory for the resulting CSV file')
    parser.add_argument('--reference_image', type=str, required=True, help='Path to the reference image')
    return parser.parse_args()

##### 
# Defining Functions
#####

# Functions for CNN-based search
def preprocess_image(image_path):
    """
    Loads and preprocesses an image for model input.
    
    Parameters:
        image_path (str): Path to the image file.
    
    Returns:
        np.array: Preprocessed image for VGG16 input.
    """
    input_shape = (224, 224, 3)
    img = load_img(image_path, target_size=input_shape)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def setup_model():
    """
    Initialises and returns a pre-trained VGG16 model.

    Returns:
        VGG16 model loaded with the pre-trained ImageNet weights.
    """
    return VGG16(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_array, model):
    """
    Extracts features from an image using a pre-trained CNN.
    
    Parameters:
        img_array (np.array): Preprocessed image array.
        model (keras.Model): Pre-trained model (i.e., VGG16).

    Returns:
        np.array: Flattened feature array.
    """
    features = model.predict(img_array, verbose=False)
    return features.flatten()

def find_similar_images_cnn(reference_features, feature_list):
    """
    Finds similar images using CNN extracted features and K-Nearest Neighbors.
    
    Parameters:
        reference_features (np.array): Features of the reference image.
        feature_list (list): List of features for all dataset images.
    
    Returns:
        tuple: Indices and distances of similar images found.
    """
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='cosine')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([reference_features])
    return indices[0][1:], distances[0][1:]

# Functions for histogram-based search
def load_target_image(image_path):
    """
    Loads an image from a file.

    Parameters:
        image_path (str): Path to the image file.
    
    Returns:
        np.array: Loaded image.
    """
    return cv2.imread(image_path)

def calc_hist(image):
    """
    Calculates colour histogram for an image split by its RGB channels.
    
    Parameters:
        image (np.array): Image array.
    
    Returns:
        np.array: Normalised histogram concatenated for all channels.
    """
    hist = []
    channels = cv2.split(image)
    for channel in channels:
        hist_channel = cv2.calcHist([channel], [0], None, [255], [0,256])
        hist.append(hist_channel)
    hist = np.concatenate(hist)
    return cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)

def compare_hists(target_hist, image_hist):
    """
    Compares histograms of two images.
    
    Parameters:
        target_hist (np.array): Histogram of the target image.
        image_hist (np.array): Histogram of other image.
    
    Returns:
        float: Chi-squared distance between histograms.
    """
    return round(cv2.compareHist(target_hist, image_hist, cv2.HISTCMP_CHISQR), 2)

def find_similar_images_histogram(dataset_path, target_hist):
    """
    Finds images similar to the reference image based on histogram comparison.
    
    Parameters:
        dataset_path (str): Path to the dataset directory.
        target_hist (np.array): Histogram of the reference image.
    
    Returns:
        pd.DataFrame: DataFrame containing filenames and distances of similar images.
    """
    filenames = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path) if file.endswith('.jpg')]
    results = []
    for filename in filenames:
        image = cv2.imread(filename)
        image_hist = calc_hist(image)
        distance = compare_hists(target_hist, image_hist)
        results.append((filename, distance))
    return pd.DataFrame(sorted(results, key=lambda x: x[1])[:6], columns=["Filename", "Distance"])

#####
# Main Function
#####

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.method == 'cnn':
        print("Processing with CNN method.")
        model = setup_model()
        ref_img_array = preprocess_image(args.reference_image)
        ref_features = extract_features(ref_img_array, model)
        all_images = [args.reference_image] + [os.path.join(args.dataset_path, x) for x in os.listdir(args.dataset_path) if x.endswith('.jpg')]
        feature_list = [extract_features(preprocess_image(f), model) for f in all_images]
        indices, distances = find_similar_images_cnn(ref_features, feature_list)
        results = pd.DataFrame({
            "Filename": [os.path.basename(all_images[i]) for i in indices],
            "Distance": distances
        })
        results = pd.concat([pd.DataFrame([{"Filename": os.path.basename(args.reference_image), "Distance": 0.0}]), results]).reset_index(drop=True)
    else: # i.e., args.method == 'histogram'
        print("Processing with histogram comparison method.")
        target_image = load_target_image(args.reference_image)
        target_hist = calc_hist(target_image)
        results = find_similar_images_histogram(args.dataset_path, target_hist)
        results['Filename'] = results['Filename'].apply(lambda x: os.path.basename(x))

    output_file_path = os.path.join(args.output_dir, f"most_similar_images_{args.method}.csv")
    results.head(6).to_csv(output_file_path, index=False)
    print(f"Results saved to {output_file_path}")

if __name__ == "__main__":
    main()
