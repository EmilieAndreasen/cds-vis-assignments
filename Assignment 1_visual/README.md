# cds-vis-assignment 1

Assignment 1 for CDS Visual Analytics.

By: Emilie Munch Andreasen

## Objectives
### Building a simple image search algorithm

For this assignment, you'll be using ```OpenCV``` to design a simple image search algorithm.

The dataset is a collection of over 1000 images of flowers, sampled from 17 different species. The dataset comes from the Visual Geometry Group at the University of Oxford, and full details of the data can be found [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/).

For this exercise, you should write some code which does the following:

- Define a particular image that you want to work with
- For that image
  - Extract the colour histogram using ```OpenCV```
- Extract colour histograms for all of the **other* images in the data
- Compare the histogram of our chosen image to all of the other histograms 
  - For this, use the ```cv2.compareHist()``` function with the ```cv2.HISTCMP_CHISQR``` metric
- Find the five images which are most simlar to the target image
  - Save a CSV file to the folder called ```out```, showing the five most similar images and the distance metric:

|Filename|Distance|
|---|---|
|target|0.0|
|filename1|---|
|filename2|---|

## Technicalities

For this assignment two standard Python packages/modules are needed (i.e., 'os' and 'sys') along with 'pandas', 'cv2', and 'numpy'.

## Folder structure

The folders are structured as follows:

| Column | Description|
|--------|:-----------|
| ```src```  | Contains the Python script used for assignment 1 |
| ```out```| Contains all the outputted csv files |

## References
Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools.

Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2.

McKinney, W., & others. (2010). Data structures for statistical computing in python. In Proceedings of the 9th Python in Science Conference (Vol. 445, pp. 51–56).

Van Rossum, G., & Drake, F. L. (2009). Python 3 Reference Manual. Scotts Valley, CA: CreateSpace.
