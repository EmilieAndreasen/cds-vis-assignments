#####
# Assignment 3 - Document classification using pretrained image embeddings
# Author: Emilie Munch Andreasen
# Date: 13-05-2024
#####

# Importing packages
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import (load_img, img_to_array, ImageDataGenerator)
from tensorflow.keras.applications.vgg16 import (preprocess_input, decode_predictions, VGG16)
from tensorflow.keras.layers import (Flatten, Dense, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Defining argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Document classification using pretrained image embeddings.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset directory containing images')
    parser.add_argument('--output_dir', type=str, default='../out', help='Output directory for the resulting report and plot')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model')
    return parser.parse_args()

##### 
# Defining Functions
#####

def load_image_data(data_path):
    print("Loading image data from the directory...")
    directories = sorted(os.listdir(data_path))
    image_list = []

    for directory in directories:
        subdir_path = os.path.join(data_path, directory)
        image_files = sorted(os.listdir(subdir_path))

        for image_file in image_files:
            image_path = os.path.join(subdir_path, image_file)
            if image_path.endswith('.jpg'):
                img_data = load_img(image_path, target_size=(224, 224))
                image_list.append({"label": directory, "image": img_data})

    print(f"Loaded {len(image_list)} images.")
    return image_list

def prep_images(image_list):
    print("Preparing images...")
    processed_list = [img_to_array(image_data['image']) for image_data in image_list]
    return [preprocess_input(image) for image in processed_list]

def make_model(classes):
    print("Building the model...")
    vgg_base = VGG16(include_top=False, 
                    pooling='avg', 
                    input_shape=(224, 224, 3))
    for layer in vgg_base.layers:
        layer.trainable = False

    flat_layer = Flatten()(vgg_base.output)
    dense_layer = Dense(128, activation='relu')(flat_layer) #128 seems to perform the best
    final_output = Dense(classes, activation='softmax')(dense_layer)

    compiled_model = Model(inputs=vgg_base.inputs, outputs=final_output)
    compiled_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model Built!")
    return compiled_model

def encode_labels(image_data, label_encoder):
    return label_encoder.fit_transform([img["label"] for img in image_data])

def split_data(preprocessed_images, image_labels, test_size=0.2, seed=26):
    return train_test_split(preprocessed_images, image_labels, test_size=test_size, random_state=seed, stratify=image_labels)

def train_model(model, X_train, y_train, batch_size, epochs):
    print("Training the model...")
    return model.fit(np.array(X_train), np.array(y_train), validation_split=0.1, batch_size=batch_size, epochs=epochs)

def plot_history(H, epochs, output_dir):
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, 'Learning_Curves.png'))
    plt.close()

def evaluate_model(model, X_test, y_test, labels, batch_size):
    predictions = model.predict(np.array(X_test), batch_size=batch_size)
    return classification_report(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1), target_names=labels)

##### 
# Main Function
#####

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    image_data = load_image_data(args.dataset_path)
    preprocessed_images = prep_images(image_data)

    labels = [
        'ADVE', 
        'Email', 
        'Form', 
        'Letter', 
        'Memo', 
        'News', 
        'Note', 
        'Report', 
        'Resume', 
        'Scientific'
    ]
    label_encoder = LabelBinarizer()
    image_labels = encode_labels(image_data, label_encoder)

    X_train, X_test, y_train, y_test = split_data(preprocessed_images, image_labels)

    model = make_model(len(labels))
    training_history = train_model(model, X_train, y_train, args.batch_size, args.epochs)

    print("Plotting learning curves for classifier...")
    plot_history(training_history, args.epochs, args.output_dir)

    print("Evaluating model...")
    classification_report_text = evaluate_model(model, X_test, y_test, labels, args.batch_size)

    print("Saving classification report...")
    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
        f.write(classification_report_text)
        print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()