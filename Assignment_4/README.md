# Assignment 4: Detecting Faces in Historical Newspapers
This repository contains a Python script designed to detect human faces in historical Swiss newspapers and analyse the prevalence of images with human faces over time. 

More specifically, the repository contains the main Python script, output CSV files of the newspapers face counts (among other details) and plots of pertentage of pages with faces per decade for each newspaper, and other files for setting up and running the script (for further details, see *Repository structure*).

### Task Overview
For this assignment, the main objective was detect faces in pages of three historical Swiss newspapers: the Journal de Genève (JDG, 1826-1994), the Gazette de Lausanne (GDL, 1804-1991), and the Impartial (IMP, 1881-2017) (for further details, see *Data Source*).  
The code had to be able to do the following:
1. 

### Repository Structure
Below is the directory structure of the repository. Make sure to have a similar layout for easy navigation and reproducibility purposes.
```
.
Assignment_1/
│
├── in/
│   └── flowers/
│       ├── image_0001.jpg
│       ├── ...    
│
├── out/
│   ├── most_similar_images_cnn.csv
│   ├── most_similar_images_histogram.csv
│
├── src/
│   └── main.py
│
├── README.md
├── requirements.txt
├── run.sh
└── setup.sh
```

## Data Source
The dataset used for this assignment is a collection of over 1000 images (.jpg) of flowers, sampled from 17 different species. The dataset comes from the Visual Geometry Group at the University of Oxford, and it can be accessed [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/).  

To use the data, follow the link above, download the dataset for the images, and save it to the `in` folder.

## Steps for Re-running the Analysis
### Setting Up and Running the Code
To re-run the analysis, follow the steps outlined below:

**1. Download and Prepare the Repository:**  
Start by downloading the zip file and unzip it in your desired location. When done, navigate to the `Assignment_1` folder.  
(Ensure that the dataset of images is downloaded and placed in the `in` folder, as specified above.)

**2. Set Up the Virtual Environment:**  
Execute the following command in your terminal to set up the Python virtual environment and install the needed dependencies.
```
bash setup.sh 
```
**3. Activate the Virtual Environment and Run the Code:**  
Run the script by executing the following command in your terminal. It will activate the virtual environment, run the Python script with the command line arguments that you provide, and then deactivate the environment when finished.
```
bash run.sh --method=<method> --dataset_path=./in --output_dir=./out --reference_image=<path_to_reference_image>
```

### Command Line Arguments
These are the four different args that can be passed:  
**--method:** Choose 'histogram' for colour histogram comparison or 'cnn' for CNN-based search.  
**--dataset_path:** Path to the directory containing images.  
**--output_dir:** Optional. Directory where the results CSV will be saved, defaults to ../out.  
**--reference_image:** Path to the reference image. 


## Summary of Key Points from Outputs
The outputs for both ... are presented below.  

**XXX:**  

**XXX:**
  
<br>
The results from ...

## Discussion of Limitations and Possible Steps for Improvement
This script offers insights into ... However, there are also certain limitations present which should be taken into consideration.  

First of all, ...

...

In short, ...

