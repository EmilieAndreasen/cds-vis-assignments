# Assignment 1: Building a Simple Image Search Algorithm
Course: CDS Visual Analytics  
Author: Emilie Munch Andreasen  

## Description
This repository contains the code for a simple image search algorithm. The objective for Assignment 1 was to develop a Python script that could both compare colour histograms of images or use a pre-trained CNN to identify the most similar images to a chosen reference image. The script supports two methods: colour histogram comparison and a CNN-based approach using the VGG16 model.

### Folder structure

The folders in this repository are structured as follows:

| Column | Description|
|--------|:-----------|
| ```in```  | Folder for data to be processed |
| ```src```  | Folder for the Python script |
| ```out```| Folder for the outputted csv files |


## Data Source
The dataset used for this assignment is a collection of over 1000 images of flowers, sampled from 17 different species. The dataset comes from the Visual Geometry Group at the University of Oxford, and it can be accessed [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/). 

To use the data ... (EXPLAIN HOW TO SAVE IN THE *IN* FOLDER).

## Steps for Re-running the Analysis
### Setting Up and Running the Code
(How to set up virtual environments, install requirements, run the code, etc)  

**1. Set Up the Virtual Environment:**
```
source setup.sh 
```
**2. Activate the Virtual Environment and Run the Code:**
```
source run.sh 
```

### Command Line Arguments
These are the four different args that can be passed:  
--method: Choose 'histogram' for colour histogram comparison or 'cnn' for CNN-based search.  
--dataset_path: Path to the directory containing images.  
--output_dir: Optional. Directory where the results CSV will be saved, defaults to ../out.  
--reference_image: Path to the reference image.  

## Summary of Key Points from Outputs
...

## Discussion
(limitations and possible steps to improvement)

