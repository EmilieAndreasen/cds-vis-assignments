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
| Decade | Total Faces | Percentage Pages with Faces |
|--------|-------------|-----------------------------|
| 1820   | 0           | 0.0%                        |
| 1830   | 1           | 1.85%                       |
| 1840   | 1           | 2.0%                        |
| 1850   | 2           | 4.17%                       |
| 1860   | 4           | 6.25%                       |
| 1870   | 5           | 4.35%                       |
| 1880   | 6           | 10.0%                       |
| 1890   | 1           | 1.92%                       |
| 1900   | 11          | 16.67%                      |
| 1910   | 6           | 8.0%                        |
| 1920   | 21          | 10.0%                       |
| 1930   | 32          | 19.23%                      |
| 1940   | 13          | 10.87%                      |
| 1950   | 49          | 25.0%                       |
| 1960   | 44          | 18.67%                      |
| 1970   | 48          | 22.04%                      |
| 1980   | 143         | 34.64%                      |
| 1990   | 190         | 29.01%                      |

The JDG newspaper displays a gradual increase in the percentage of pages with faces from the 1820s to the 1980s. Notably, there are low percentages of pages with faces in the early decades (i.e., 0% in the 1820s, 1.85% in the 1830s), but then from the 1900s onward there is a steady increase. This  The 1950s and the 1980s stand out in the later years with 

with significant jumps in the 1950s (25.0%) and the 1980s (34.64%), and a slight decrease in the 1990s with 29.01% of pages containing faces.

**Gazette de Lausanne (GDL):**  
| Decade | Total Faces | Percentage Pages with Faces |
|--------|-------------|-----------------------------|
| 1790   | 3           | 15.0%                       |
| 1800   | 10          | 27.78%                      |
| 1810   | 2           | 7.69%                       |
| 1820   | 1           | 3.13%                       |
| 1830   | 2           | 5.56%                       |
| 1840   | 0           | 0.0%                        |
| 1850   | 2           | 8.33%                       |
| 1860   | 1           | 4.17%                       |
| 1870   | 1           | 4.17%                       |
| 1880   | 1           | 4.17%                       |
| 1890   | 1           | 3.85%                       |
| 1900   | 4           | 14.29%                      |
| 1910   | 4           | 13.33%                      |
| 1920   | 8           | 28.57%                      |
| 1930   | 8           | 28.57%                      |
| 1940   | 10          | 26.32%                      |
| 1950   | 8           | 16.67%                      |
| 1960   | 25          | 27.78%                      |
| 1970   | 14          | 17.95%                      |
| 1980   | 60          | 44.12%                      |
| 1990   | 115         | 55.29%                      |  


**Impartial (IMP):**  
| Decade | Total Faces | Percentage Pages with Faces |
|--------|-------------|-----------------------------|
| 1880   | 1           | 2.94%                       |
| 1890   | 17          | 32.69%                      |
| 1900   | 25          | 36.76%                      |
| 1910   | 27          | 51.92%                      |
| 1920   | 46          | 71.88%                      |
| 1930   | 31          | 57.41%                      |
| 1940   | 24          | 42.86%                      |
| 1950   | 105         | 111.70%                     |
| 1960   | 175         | 121.53%                     |
| 1970   | 202         | 101.0%                      |
| 1980   | 343         | 163.33%                     |
| 1990   | 210         | 103.96%                     |
| 2000   | 657         | 304.17%                     |
| 2010   | 699         | 371.81%                     |  


## Discussion of Limitations and Possible Steps for Improvement
This script offers insights into ... However, there are also certain limitations present which should be taken into consideration.  

First of all, ...

...

In short, ...

