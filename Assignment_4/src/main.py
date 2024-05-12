######
# Assignment 4 - Detecting Faces in Historical Newspapers
######

# Importing packages
import argparse
import os
from PIL import Image
from facenet_pytorch import MTCNN
import torch
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import re
from tqdm import tqdm

# Parsing command-line arguments
def parse_arguments():
    """
    Parse command line arguments for specifying input and output directories.

    Returns:
        argparse.Namespace: The namespace containing the arguments.
    """
    parser = argparse.ArgumentParser(description="Detect faces in historical newspaper archives.")
    parser.add_argument('-d', '--data_dir', type=str, required=True, help='Directory containing the archive images.')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Directory to save outputs.')
    return parser.parse_args()

# Defining functions
## Initialising MTCNN model for face detection
def init_mtcnn():
    """
    Initialize and return the MTCNN model for face detection.

    Returns:
        MTCNN: The face detection model.
    """
    return MTCNN(keep_all=True)

## Detecting faces in an image
def detect_faces(image_path, mtcnn):
    """
    Detect and count faces in a given image using MTCNN.

    Args:
        image_path (str): Path to the image file.
        mtcnn (MTCNN): The MTCNN model for face detection.

    Returns:
        int: Number of faces detected in the image.
    """
    try:
        img = Image.open(image_path)
        img.load()  # Force loading the image to catch truncation errors
        boxes, _ = mtcnn.detect(img)
        return len(boxes) if boxes is not None else 0
    except (IOError, OSError) as e:
        print(f"Error processing image {image_path}: {e}")
        return 0  # Return zero faces detected if there is an error loading the image


## Processing images in the dataset
def process_images(data_dir, mtcnn):
    """
    Process all images in the dataset to detect faces and compile results by decade.

    Args:
        data_dir (str): Directory containing images.
        mtcnn (MTCNN): The MTCNN model for face detection.

    Returns:
        dict: Dictionary with results by newspaper and decade.
    """
    results = defaultdict(lambda: defaultdict(int))
    page_counts = defaultdict(lambda: defaultdict(int))

    for root, _, files in os.walk(data_dir):
        for file in tqdm(files, desc="Processing images"):
            if file.endswith(".jpg"):
                path = os.path.join(root, file)
                num_faces = detect_faces(path, mtcnn)
                newspaper, year = re.match(r'(\w+)-(\d{4})-\d{2}-\d{2}-a-p\d{4}.jpg', file).groups()
                decade = (int(year) // 10) * 10
                results[newspaper][decade] += num_faces
                page_counts[newspaper][decade] += 1

    return results, page_counts

## Generating outputs (both csvs and plots)
def generate_outputs(results, page_counts, output_dir):
    """
    Generate CSV files and plots for the face detection results.

    Args:
        results (dict): The face detection results organized by newspaper and decade.
        page_counts (dict): Page count for each newspaper and decade.
        output_dir (str): Output directory to save results.
    """
    os.makedirs(output_dir, exist_ok=True)

    for newspaper, decades in results.items():
        df = pd.DataFrame({
            "Decade": list(decades.keys()),
            "Total Faces": list(decades.values()),
            "Percentage Pages with Faces": [(faces / page_counts[newspaper][decade]) * 100 for decade, faces in decades.items()]
        })
        df.sort_values("Decade", inplace=True)

        # Saving results to csv file
        df.to_csv(os.path.join(output_dir, f"{newspaper}_face_counts.csv"), index=False)

        # Plotting results
        plt.figure()
        plt.plot(df["Decade"], df["Percentage Pages with Faces"], marker='o', linestyle='-')
        plt.title(f"Percentage of Pages with Faces per Decade in {newspaper}")
        plt.xlabel("Decade")
        plt.ylabel("Percentage of Pages with Faces")
        plt.savefig(os.path.join(output_dir, f"{newspaper}_faces_plot.png"))
        plt.close()

# Main function
def main():
    """
    Main function to execute the processing of historical newspaper images for face detection.
    """
    args = parse_arguments()
    mtcnn = init_mtcnn()
    results, page_counts = process_images(args.data_dir, mtcnn)
    generate_outputs(results, page_counts, args.output_dir)
    print(f"Face detection completed. Results are saved in {args.output_dir}")

if __name__ == "__main__":
    main()
