# Assignment 4: Detecting Faces in Historical Newspapers
This repository contains a Python script designed to detect human faces in historical Swiss newspapers and analyse the prevalence of images with human faces over time. 

More specifically, the repository contains the main Python script, output CSV files of the newspapers face counts (among other details) and plots of pertentage of pages with faces per decade for each newspaper, and other files for setting up and running the script (for further details, see *Repository structure*).

### Task Overview
For this assignment, the main objective was detect faces in pages of three historical Swiss newspapers: the Journal de Genève (JDG, 1826-1994), the Gazette de Lausanne (GDL, 1804-1991), and the Impartial (IMP, 1881-2017) (for further details, see *Data Source*).  
The code had to be able to do the following:
1. For each of the three newspapers, detect if faces are present in the pages using a pre-trained CNN model.  
2. Aggregate the results by decade.  
3. For each newspaper, generate a CSV file showing the total number of faces per decade and the percentage of pages with faces for that decade.  
4. For each newspaper, create a plot showing the percentage of pages with faces per decade.

### Repository Structure
Below is the directory structure of the repository. Make sure to have a similar layout for easy navigation and reproducibility purposes.  
```
.
Assignment_4/
│
├── in/
│   └── newspapers/
│       ├── GDL/
│       ├── IMP/
│       ├── JDG/
│           ├── JDG-1826-02-16-a-p0001.jpg
│           ├── ...
│
├── out/
│   ├── GDL_face_counts.csv
│   ├── GDL_faces_plot.png
│   ├── IMP_face_counts.csv
│   ├── IMP_faces_plot.png
│   ├── JDG_face_counts.csv
│   ├── JDG_faces_plot.png
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
The dataset used for this assignment is a collection historic Swiss newspapers, namely the the Journal de Genève (JDG, 1826-1994); the Gazette de Lausanne (GDL, 1804-1991); and the Impartial (IMP, 1881-2017). Images (.jpg) of the newspaper pages are contained within and are sorted by which newspaper they belong to along with the year-month-date of publication. There are therefore three sub-folders in the dataset:
- GDL
- IMP
- JDG

For more details about the data, visit the following [website](https://zenodo.org/records/3706863). To use the data, simply follow the link, download the dataset for the images, and save it to the `in` folder.

## Steps for Re-running the Analysis
### Setting Up and Running the Code
To re-run the analysis, follow the steps outlined below:

**1. Download and Prepare the Repository:**  
If the attachment has not already been downloaded and unzipped, then start by downloading the zip file and unzip it in your desired location. When done, navigate to the `Assignment_4` folder.  
(Ensure that the dataset of images is downloaded and placed in the `in` folder, as specified above.)

**2. Set Up the Virtual Environment:**  
Execute the following command in your terminal to set up the Python virtual environment and install the needed dependencies.
```
bash setup.sh 
```

**3. Activate the Virtual Environment and Run the Code:**  
Run the script by executing the following command in your terminal. It will activate the virtual environment, run the Python script with the command line arguments that you provide, and then deactivate the environment when finished.
```
bash run.sh --dataset_path=./in/newspapers --output_dir=./out
```

### Command Line Arguments
These are the two args that can be passed:  
**--dataset_path:** Path to the directory containing the sub-folder with images.  
**--output_dir:** Optional. Directory where the results CSV and plots will be saved, defaults to ../out.  

## Summary of Key Points from Outputs
The outputs for the face detection analysis are presented below.  

**Journal de Genève (JDG):**  

**Gazette de Lausanne (GDL):**  
  
**Impartial (IMP):**  

## Discussion of Limitations and Possible Steps for Improvement
This script offers insights into ... However, there are also certain limitations present which should be taken into consideration.  

First of all, ...

...

In short, ...

