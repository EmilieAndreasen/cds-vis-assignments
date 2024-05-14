# Assignment 3: Document Classification Using Pretrained Image Embeddings
This repository contains a Python-based document classification script that predicts document types based on their visual features. The approach uses a pre-trained Convolutional Neural Network (CNN) model, namely the VGG16, to extract image embeddings for classification.  

More specifically, the repository contains the main Python script, an output classification report and a plot showing the learning curves for the trained classifier, and other files for setting up and running the script (for further details, see *Repository structure*).

### Task Overview
For this assignment, the main objective was to predict the type of document based solely on its visual appearance using the Tobacco3482 dataset (see *Data Source*).  
The code had to be able to do the following:
1. Load the Tobacco3482 dataset and generate labels for each image.
2. Train a CNN-based classifier to predict document type based on visual features.
3. Output a classification report and a plot of learning curves for the trained classifier.
4. Save the results in the `out` folder.

### Repository Structure
Below is the directory structure of the repository. Make sure to have a similar layout for easy navigation and reproducibility purposes.  
```
.
Assignment_3/
│
├── in/
│   └── Tobacco3482/
│       ├── ADVE/
│       ├── Email/
│       ├── ...
│
├── out/
│   ├── Classification_Report.txt
│   ├── Learning_Curves.png
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
The dataset used for this assignment is the Tobacco3482 dataset, a collection of over 3000 document images across 10 different classes. Specifically, the 10 classes in the dataset are:  
- ADVE
- Email
- Form
- Letter
- Memo
- News
- Note
- Report
- Resume
- Scientific  

For more details about the data, visit the following [website](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg?resource=download). To use the data, simply follow the link, download the dataset, and save it to the `in` folder.

## Steps for Re-running the Analysis
### Setting Up and Running the Code
To re-run the analysis, follow the steps outlined below:

**1. Download and Prepare the Repository:**  
If the attachment has not already been downloaded and unzipped, then start by downloading the zip file and unzip it in your desired location. When done, navigate to the `Assignment_3` folder.  
(Ensure that the Tobacco3482 dataset is downloaded and placed in the `in` folder, as specified above.)

**2. Set Up the Virtual Environment:**  
Execute the following command in your terminal to set up the Python virtual environment and install the needed dependencies.
```
bash setup.sh 
```

**3. Activate the Virtual Environment and Run the Code:**  
Run the script by executing the following command in your terminal. It will activate the virtual environment, run the Python script with the command line arguments that you provide, and then deactivate the environment when finished.
```
bash run.sh --dataset_path=./in/Tobacco3482 --output_dir=./out --batch_size=<batch_size> --epochs=<epochs>
```

### Command Line Arguments
These are the four different args that can be passed:  
**--dataset_path:** Path to the dataset directory containing the folder with images.  
**--output_dir:** Optional. Directory where the results (classification report and learning curves) will be saved, defaults to ../out.  
**--batch_size:** Optional. Batch size for training and evaluation, defaults to 128.  
**--epochs:** Optional. Number of epochs to train the model, defaults to 10.

## Summary of Key Points from Outputs
The outputs for both ... are presented below.  

**XXX:**  

**XXX:**
  
<br>
The results from ...
(OLD INTERPRETATION)
The classification report shows generally low precision scores across all classes. Additionally, both recall and F1-scores also display poor metrics. This is further reinforced by the accuracy score of only 15%, indicating that the model is only correct 15% of the time on the validation set. However, the support (i.e., the number of actual occurrences of the class in the dataset) is also generally low. Overall, it suggests that the model has not learned to classify the different documents that effictively. 

In regards to the two plots, the loss curve shows that both the training loss and the validation loss decrease over epochs - generally a good sign that points to the model learning from the training data - but the validation loss also shows notable volatility. This could suggest that there are issues with model generalisation. For the accuracy curve, the training accuracy increases over epochs, indicating improvements in the model's predictions over the training data. While the validation accuracy also increases, it does not do it as smoothly as the training one. Moreover, it is overall lower than the other. Taking all this together, it would seem that there might be problems with overfitting. Additionally, the notable volatility might suggest that the model's learning process is not very stable. 

Future reworks of this code therefore aims to work to improve these problems.

## Discussion of Limitations and Possible Steps for Improvement
This script offers insights into ... However, there are also certain limitations present which should be taken into consideration.  

First of all, ...

...

In short, ...
