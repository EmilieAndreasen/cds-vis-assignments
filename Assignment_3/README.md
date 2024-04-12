# cds-vis-assignment 3

Assignment 3 for CDS Visual Analytics.

By: Emilie Munch Andreasen

## Objectives
### Assignment 3 - Document classification using pretrained image embeddings
In Language Analytics so far, we've done lots of document classification based on linguistic features of documents. This week, we are working on a slightly different problem - can we predict what type of document we have based only on its appearance rather than its contents?

Think about how different documents look when written on a page. A poem appears differently to a newpaper article, both of which are different from a billboard advertisement. This assignment tries to leverage this knowledge to try to predict what type of document we have, based on its visual appearance.

For this assignment, we'll be working with the Tobacco3482 dataset. You can learn more about this dataset in the original paper which uses it here. The dataset we are working with is only a small subset of this larger corpus.

You should write code which does the following:

- Loads the Tobacco3482 data and generates labels for each image
- Train a classifier to predict document type based on visual features
- Present a classification report and learning curves for the trained classifier
- Your repository should also include a short description of what the classification report and learning curve show.

## Data availability
The data needed for the assignment can be found here: https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg?resource=download 

## Technicalities

For this assignment, the 'setup.sh' should be run to install the required packages. 

## Folder structure

The folders are structured as follows:

| Column | Description|
|--------|:-----------|
| ```out```| Contains the outputted report and learning curves for the trained classifier |
| ```src```  | Contains the Python script used for assignment 3 |

## Interpretation of outputs
The classification report shows generally low precision scores across all classes. Additionally, both recall and F1-scores also display poor metrics. This is further reinforced by the accuracy score of only 15%, indicating that the model is only correct 15% of the time on the validation set. However, the support (i.e., the number of actual occurrences of the class in the dataset) is also generally low. Overall, it suggests that the model has not learned to classify the different documents that effictively. 

In regards to the two plots, the loss curve shows that both the training loss and the validation loss decrease over epochs - generally a good sign that points to the model learning from the training data - but the validation loss also shows notable volatility. This could suggest that there are issues with model generalisation. For the accuracy curve, the training accuracy increases over epochs, indicating improvements in the model's predictions over the training data. While the validation accuracy also increases, it does not do it as smoothly as the training one. Moreover, it is overall lower than the other. Taking all this together, it would seem that there might be problems with overfitting. Additionally, the notable volatility might suggest that the model's learning process is not very stable. 

Future reworks of this code therefore aims to work to improve these problems.