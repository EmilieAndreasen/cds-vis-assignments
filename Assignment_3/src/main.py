######
# Assignment 3 - Document Classification Using Pretrained Image Embeddings
######

# Importing packages
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf

from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping

# Parsing command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Document classification using pretrained image embeddings.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the dataset.')
    parser.add_argument('--output_dir', type=str, default='./out', help='Directory to save outputs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model')
    return parser.parse_args()

# Defining Functions
## Plotting training history
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
    
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
    plt.close()

## Preparing data
def prepare_data(data_dir, batch_size):
    data_gen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    train_data = data_gen.flow_from_directory(directory=data_dir,
                                              target_size=(224, 224),
                                              class_mode='categorical',
                                              subset='training',
                                              batch_size=batch_size,
                                              shuffle=True)
    
    val_data = data_gen.flow_from_directory(directory=data_dir,
                                            target_size=(224, 224),
                                            class_mode='categorical',
                                            subset='validation',
                                            batch_size=batch_size,
                                            shuffle=True)
    return train_data, val_data

## Building the model
def build_model(train_data):
    # Loading the VGG16 network
    baseModel = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    # Constructing the head of the model that will be placed on top of the base model
    headModel = baseModel.output
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(512, activation="relu")(headModel)
    headModel = BatchNormalization()(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(len(train_data.class_indices), activation="softmax")(headModel)
    
    # Placing that model on top of the base model
    model = Model(inputs=baseModel.input, outputs=headModel)

    for layer in baseModel.layers:
        layer.trainable = False
    
    return model

# Main function
def main():
    args = parse_args()

    # Checking for if output directory exists    
    os.makedirs(args.output_dir, exist_ok=True)

    train_data, val_data = prepare_data(args.data_dir, args.batch_size)

    model = build_model(train_data)
    
    # Compiling model
    print("Compiling model... This make take a while...")
    lr_schedule = ExponentialDecay(initial_learning_rate=0.01, decay_steps=10000, decay_rate=0.9)
    sgd = SGD(learning_rate=lr_schedule)
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

    # Training model
    print("Training...")
    H = model.fit(train_data, validation_data=val_data, epochs=args.epochs, verbose=1)

    # Making predictions on the testing set
    print("Evaluating network...")
    predictions = model.predict(val_data, batch_size=args.batch_size)
    
    # Saving model and writing report
    model.save(os.path.join(args.output_dir, 'doc_classifier_model.keras'))
    plot_history(H, 20, args.output_dir)
    report = classification_report(val_data.classes, predictions.argmax(axis=1), target_names=val_data.class_indices.keys(), output_dict=False)
    report_filename = os.path.join(args.output_dir, 'classification_report.txt')
    with open(report_filename, 'w') as f:
        f.write(report)
    print(f"Training completed. Outputs saved to {args.output_dir}")

if __name__ == "__main__":
    main()
