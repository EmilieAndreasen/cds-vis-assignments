# Assignment 1: Building a Simple Image Search Algorithm
This repository contains a Python-based simple image search algorithm that identifies images similar to a given reference image, using two distinct methods:
- **Colour Histogram Comparison**: Employs OpenCV to extract and compare colour histograms of images.  
- **CNN & KNN-based Approach**: Employs a pre-trained Convolutional Neural Network (i.e., VGG16) and K-Nearest Neighbours for extracting and finding images with similar features.  

More specifically, the repository contains the main Python script, output CSV files listing the top five most similar images for each method, and other relevant files for setting up and running the script (for further details, see *Repository structure*).

### Task Overview
For this assignment, the main objective was to code an image search algorithm using a dataset of over 1000 flower images (see *Data Source*). The code had to be able to do the following:
1. Select an image as the reference image.
2. Extract the colour histogram of the reference image using OpenCV.
3. Compute colour histograms for all other images in the dataset.
4. Use the `cv2.compareHist()` function with the `cv2.HISTCMP_CHISQR` metric to compare histograms.
5. Include functionality for another method using a pre-trained CNN and KNN.
6. For either method, identify and list the five images most similar to the reference image based on the comparison.
7. Output the results along with the given reference image in a CSV file in the `out` folder with columns for Filename and Distance.

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
The dataset used for this assignment is a collection of over 1000 images (.jpg) of flowers, sampled from 17 different species. The dataset comes from the Visual Geometry Group at the University of Oxford, and for more details about the data, visit the following [website](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/).

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
The outputs for both image search methods are presented below.  

**Colour Histogram Comparison:**  
| Filename       | Distance |
|----------------|----------|
| image_0556.jpg | 0.0      |
| image_0684.jpg | 35.56    |
| image_1117.jpg | 43.06    |
| image_0322.jpg | 43.41    |
| image_0598.jpg | 44.34    |
| image_0333.jpg | 44.55    |  

The histogram comparison display rather high distance values (ranging from 35.56 to 44.55) for the top five most similar images to the reference image (i.e., image_0556.jpg). These high values imply that simply comparing the colour histograms of images may not yield a very precise measure of similarity.
  
**CNN & KNN-based Approach:**  
| Filename       | Distance |
|----------------|----------|
| image_0556.jpg | 0.0      |
| image_0552.jpg | 0.37     |
| image_0549.jpg | 0.39     |
| image_0545.jpg | 0.4      |
| image_0536.jpg | 0.41     |
| image_0550.jpg | 0.42     |
  
The CNN and KNN-based approach show much lower distance values for all of the found images (ranging from 0.37 to 0.42). Here it should be noted that unlike the first method which can have distances ranging from 0 to, theoretically, infinity, the CNN and KNN-based approach's distance metric can range between 0 and 2. However, these results still suggest that the CNN and KNN method can more accurately and consistently 'capture' important visual features from the images to use for an effective comparison with the reference image (i.e., image_0556.jpg).  

The distinct differences in the outputs seem appropriate given the two methods' underlying mechanisms. While the colour histogram comparison method is less computationally intensive, it lacks the capability to perform as effectively as the pre-trained VGG16 model, which handles extracting detailed visual features very well. Additionally, by utilising KNN for identifying similar images - which in this case uses cosine distance to measure similarity between the feature vectors extracted by the VGG16 model - the second method ends up producing the more robust and precise results out of the two.

## Discussion of Limitations and Possible Steps for Improvement
This script offers insights into image search and how two different methods provide very different results when attempting to identify similar images. However, there are also certain limitations present which should be taken into consideration.  

First of all, although the chosen pre-trained image model, VGG16, is trained on the 'ImageNet' dataset, which is very diverse, and that it produces very solid identification of similar images, it is currently the only model available in the script. Future iterations of the script could therefore work on expanding the model choices, allowing users to switch between a couple of pre-trained models depending on what the user wants and needs (e.g., if they want to use a different and more specialised image dataset). In such a case - much like comparing the colour histogram comparison method to the CNN and KNN-based approach - users could compare how the different models perform and make an informed decision on which model best suits their requirements.

Another limitation to consider, is that the CNN and KNN-based approach, while more accurate, is much more computationally intensive than the histogram method. That means that when processing larger datasets - like the flowers one - the system requires substantial computational resources to complete the task. For users who might lack robut computing resources, this can prove to be an issue. Consequently, an idea for improving the computational efficiency could be to implement more efficient algorithms in place of some of the current ones. For instance, instead of employing KNN, one could use ANN (i.e., approximate nearest neighbours). Another ideas could be to consider using feature reduction techniques (e.g., principal component analysis).  

Finally, it seems that in some few cases when employing the colour histogram comparison method, the script will produce outputs with negative distances. Such an occurance is suprising considering that the distance metric for this method should - theoretically - always produce non-negative values. Potential causes could be tied to the normalisation process and the later use of the `cv2.compareHist()` function with the `cv2.HISTCMP_CHISQR` metric to compare histograms, but as using that function with that specific metric was part of the specific task requirements, it has not been changed in the current script. Moreover, as these results do not occur for most of the reference images the author tested (e.g., image_0010.jpg, image_0011.jpg, image_0110.jpg, image_0111.jpg, image_0555.jpg, and image_0556.jpg), a solution is instead proposed for future iterations of this script. One such solution could be to change the method for comparing histograms and to include error checks. In regards to error checks, these could be implemented to specifically check for negative values and log details to then better identify specific cases or patterns when these values occur.

In short, while the current script provides a functional foundation for searching for similar images to a reference image using two distinct methods, addressing the above limitations would likely improve its robustness, effectiveness, and relevance in other applications.

