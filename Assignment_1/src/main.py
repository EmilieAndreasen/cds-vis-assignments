######
# Assignment 1 - Building a simple image search algorithm
######

# Importing packages
import os
import sys
import cv2
import numpy as np
import pandas as pd
import argparse

# Defining argument parsing with argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='Image Search using Color Histograms with OpenCV')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset directory containing images')
    parser.add_argument('--reference_image', type=str, required=True, help='Path to the reference image')
    parser.add_argument('--output_dir', type=str, default='../out', help='Output directory for the results CSV file')
    return parser.parse_args()

# Defining functions
# Loading and return the target image
def load_target_image(image_path):
    return cv2.imread(image_path)

## Calculating histograms for inputted image
def calc_hist(image):
    hist = []
    channels = cv2.split(image)
    
    # Calculating hist for each channel
    for channel in channels:
        hist_channel = cv2.calcHist([channel], [0], None, [255], [0,256])
        hist.append(hist_channel)
    
    # Concatenating hists into 1 array --> normalise
    hist = np.concatenate(hist)
    hist_normalised = cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)

    return hist_normalised    

## Comparing histograms of target img with other image
def compare_hists(target_hist, image_hist):
    distance = round(cv2.compareHist(target_hist, image_hist, cv2.HISTCMP_CHISQR),2)

    return distance

## Finding and returning similar images to the target image
def find_similar_images(dataset_path, target_hist):
    filenames = [file for file in os.listdir(dataset_path) if file.endswith('.jpg')]
    results = []

    for filename in filenames:
        img_path = os.path.join(dataset_path, filename)
        image = cv2.imread(img_path)
        image_hist = calc_hist(image)
        distance = compare_hists(target_hist, image_hist)
        results.append((filename, distance))
    sorted_dist = sorted(results, key=lambda x: x[1])[:6]

    return pd.DataFrame(sorted_dist, columns=["Filename", "Distance"])

def main():
    args = parse_arguments()
    
    # Ensuring the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Loading target image and calculate its histogram
    target_image = load_target_image(args.reference_image)
    target_hist = calc_hist(target_image)
    
    # Finding similar images
    similar_images = find_similar_images(args.dataset_path, target_hist)
    
    # Uncomment the below code for some images, as they do not "find" themselves
    # Adds the reference image with a distance of 0.0 (other attempts cause errors)
    #reference_image_entry = pd.DataFrame([{"Filename": os.path.basename(args.reference_image), "Distance": 0.0}])
    #final_results = pd.concat([reference_image_entry, similar_images], ignore_index=True)

    # Saving results to CSV
    output_file_path = os.path.join(args.output_dir, "most_similar_images.csv")
    similar_images.to_csv(output_file_path, index=False)
    #final_results.to_csv(output_file_path, index=False)
    
    print(f"Results saved to {output_file_path}")

if __name__ == "__main__":
    main()