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
Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools.

Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2.

McKinney, W., & others. (2010). Data structures for statistical computing in python. In Proceedings of the 9th Python in Science Conference (Vol. 445, pp. 51–56).

Van Rossum, G., & Drake, F. L. (2009). Python 3 Reference Manual. Scotts Valley, CA: CreateSpace.
