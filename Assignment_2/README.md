# cds-vis-assignment 1

Assignment 2 for CDS Visual Analytics.

By: Emilie Munch Andreasen
Made in collaboration with Maria Mujemula Olsen.

## Objectives
### Classification benchmarks with Logistic Regression and Neural Networks
For this assignment, we'll be writing scripts which classify the Cifar10 dataset. You can read more about this dataset here

You should write code which does the following:

Load the Cifar10 dataset
Preprocess the data (e.g. greyscale, normalize, reshape)
Train a classifier on the data
A logistic regression classifier and a neural network classifier
Save a classification report
Save a plot of the loss curve during training
UPDATE: This is only possible for the MLP Classifier in scikit-learn
You should write two scripts for this assignment one script which does this for a logistic regression classifier and one which does it for a neural network classifier. In both cases, you should use the machine learning tools available via scikit-learn to evaluate model performance.

## Technicalities

For this assignment, the 'setup.sh' should be run to install the required modules.

## Folder structure

The folders are structured as follows:

| Column | Description|
|--------|:-----------|
| ```out```| Contains the outputted reports and loss curve png |
| ```src```  | Contains the Python scripts used for assignment 2 |

## References
Bradski, G. (2023). OpenCV-Python [Software]. OpenCV. Available from https://pypi.org/project/opencv-python/

Hunter, J. D. (2023). Matplotlib [Software]. Matplotlib Project. Available from https://matplotlib.org/

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2023). Scikit-learn: Machine Learning in Python [Software]. Scikit-learn Development Team. Available from http://scikit-learn.org/

McKinney, W. (2023). pandas [Software]. PyData. Available from https://pandas.pydata.org/

Waskom, M. (2023). Seaborn [Software]. Seaborn. Available from https://seaborn.pydata.org/

Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., ... & Zheng, X. (2023). TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems [Software]. TensorFlow. Available from https://www.tensorflow.org/
