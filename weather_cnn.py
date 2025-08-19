import time
import math
import random
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

from sklearn.metrics import confusion_matrix, classification_report
from datetime import timedelta

# Import our custom weather dataset with clustering
from weather_dataset import (
    read_weather_sets_with_clustering, 
    get_weather_classes_from_clusters, 
    analyze_weather_distribution
)

# Configuration and Hyperparameters
# Convolutional Layer 1.
filter_size1 = 3 
num_filters1 = 32

# Convolutional Layer 2.
filter_size2 = 3
num_filters2 = 32

# Convolutional Layer 3.
filter_size3 = 3
num_filters3 = 64

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

# Weather features layer
weather_fc_size = 64      # Number of neurons for weather features

# Number of color channels for the images: 3 channels for RGB.
num_channels = 3

# image dimensions (only squares for now)
img_size = 128

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of weather features (cloud_coverage, sun_obscuration_percentage, irradiance, hour_sin, hour_cos, month_sin, month_cos)
num_weather_features = 7

# Number of clusters (same as ClusteringV1.py)
n_clusters = 5

# batch size
batch_size = 32

# validation split
validation_size = 0.2

# how long to wait after validation loss stops improving before terminating training
early_stopping = 10  # use None if you don't want to implement early stopping

# Data paths
json_dir = "D:\\Image\\test_result\\json"  # Directory containing JSON files with weather data
image_dir = "D:\\Image"  # Directory containing the actual images
checkpoint_dir = "models/"

def plot_images(images, cls_true, cls_pred=None, cluster_descriptions=None):
    """Plot 9 images in a 3x3 grid with true and predicted classes."""
    if len(images) == 0:
        print("no images to show")
        return 
    else:
        random_indices = random.sample(range(len(images)), min(len(images), 9))
        
    if cls_pred is not None:
        images, cls_true, cls_pred = zip(*[(images[i], cls_true[i], cls_pred[i]) for i in random_indices])
    else:
        images, cls_true = zip(*[(images[i], cls_true[i]) for i in random_indices])
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_size, img_size, num_channels))

        # Show true and predicted classes.
        if cls_pred is None:
            if cluster_descriptions and cls_true[i] in cluster_descriptions:
                true_label = cluster_descriptions[cls_true[i]]['description']
            else:
                true_label = f"Cluster {cls_true[i]}"
            xlabel = f"True: {true_label}"
        else:
            if cluster_descriptions and cls_true[i] in cluster_descriptions:
                true_label = cluster_descriptions[cls_true[i]]['description']
            else:
                true_label = f"Cluster {cls_true[i]}"
            
            if cluster_descriptions and cls_pred[i] in cluster_descriptions:
                pred_label = cluster_descriptions[cls_pred[i]]['description']
            else:
                pred_label = f"Cluster {cls_pred[i]}"
            
            xlabel = f"True: {true_label}, Pred: {pred_label}"

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()

def new_weights(shape):
    """Create new TensorFlow weights."""
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    """Create new TensorFlow biases."""
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
    """Create a new convolutional layer."""
    # Shape of the filter-weights for the convolution.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    layer = tf.nn.relu(layer)

    return layer, weights

def flatten_layer(layer):
    """Flatten a layer for fully-connected layers."""
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The number of features is: img_height * img_width * num_channels
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features

def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    """Create a new fully-connected layer."""
    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

def create_weather_cnn(num_classes):
    """Create the multi-modal weather classification CNN with both images and weather features."""
    # Placeholder variables for images
    x_images = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x_images')
    x_image = tf.reshape(x_images, [-1, img_size, img_size, num_channels])
    
    # Placeholder variables for weather features
    x_weather = tf.placeholder(tf.float32, shape=[None, num_weather_features], name='x_weather')
    
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, axis=1)

    # Convolutional layers for image processing
    layer_conv1, weights_conv1 = new_conv_layer(input=x_image,
                                               num_input_channels=num_channels,
                                               filter_size=filter_size1,
                                               num_filters=num_filters1,
                                               use_pooling=True)

    layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                                               num_input_channels=num_filters1,
                                               filter_size=filter_size2,
                                               num_filters=num_filters2,
                                               use_pooling=True)

    layer_conv3, weights_conv3 = new_conv_layer(input=layer_conv2,
                                               num_input_channels=num_filters2,
                                               filter_size=filter_size3,
                                               num_filters=num_filters3,
                                               use_pooling=True)

    # Flatten the convolutional output
    layer_flat, num_features = flatten_layer(layer_conv3)

    # Fully-connected layer for image features
    layer_fc1_images = new_fc_layer(input=layer_flat,
                                   num_inputs=num_features,
                                   num_outputs=fc_size,
                                   use_relu=True)

    # Fully-connected layer for weather features
    layer_fc1_weather = new_fc_layer(input=x_weather,
                                    num_inputs=num_weather_features,
                                    num_outputs=weather_fc_size,
                                    use_relu=True)

    # Concatenate image and weather features
    layer_combined = tf.concat([layer_fc1_images, layer_fc1_weather], axis=1)
    combined_size = fc_size + weather_fc_size

    # Final fully-connected layer for classification
    layer_fc2 = new_fc_layer(input=layer_combined,
                            num_inputs=combined_size,
                            num_outputs=num_classes,
                            use_relu=False)

    # Predicted Class
    y_pred = tf.nn.softmax(layer_fc2)
    y_pred_cls = tf.argmax(y_pred, axis=1)

    # Cost-function to be optimized
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)

    # Optimization Method
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    # Performance Measures
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return {
        'x_images': x_images,
        'x_weather': x_weather,
        'y_true': y_true,
        'y_true_cls': y_true_cls,
        'y_pred': y_pred,
        'y_pred_cls': y_pred_cls,
        'cost': cost,
        'optimizer': optimizer,
        'accuracy': accuracy,
        'layer_conv1': layer_conv1,
        'layer_conv2': layer_conv2,
        'layer_conv3': layer_conv3,
        'layer_fc1_images': layer_fc1_images,
        'layer_fc1_weather': layer_fc1_weather,
        'layer_combined': layer_combined,
        'weights_conv1': weights_conv1,
        'weights_conv2': weights_conv2,
        'weights_conv3': weights_conv3
    }

def print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss, session, accuracy):
    """Print training progress."""
    # Calculate the accuracy on the training-set.
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

def optimize(num_iterations, data, session, cnn, early_stopping=None):
    """Perform optimization iterations."""
    # Counter for total number of iterations performed so far.
    total_iterations = 0

    # Start-time used for printing time-usage below.
    start_time = time.time()
    
    best_val_loss = float("inf")
    patience = 0

    for i in range(total_iterations, total_iterations + num_iterations):
        # Get a batch of training examples.
        batch_data = data.train.next_batch(batch_size)
        if len(batch_data) == 5:  # With features
            x_batch, y_true_batch, _, cls_batch, weather_features_batch = batch_data
        else:  # Without features
            x_batch, y_true_batch, _, cls_batch = batch_data
            weather_features_batch = None
            
        # Get validation batch
        val_batch_data = data.valid.next_batch(batch_size)
        if len(val_batch_data) == 5:  # With features
            x_valid_batch, y_valid_batch, _, valid_cls_batch, weather_features_valid_batch = val_batch_data
        else:  # Without features
            x_valid_batch, y_valid_batch, _, valid_cls_batch = val_batch_data
            weather_features_valid_batch = None

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, flattened image shape]
        x_batch = x_batch.reshape(batch_size, img_size_flat)
        x_valid_batch = x_valid_batch.reshape(batch_size, img_size_flat)

        # Create feed dictionaries
        feed_dict_train = {cnn['x_images']: x_batch, cnn['y_true']: y_true_batch}
        feed_dict_validate = {cnn['x_images']: x_valid_batch, cnn['y_true']: y_valid_batch}
        
        # Add weather features if available
        if weather_features_batch is not None:
            feed_dict_train[cnn['x_weather']] = weather_features_batch
            feed_dict_validate[cnn['x_weather']] = weather_features_valid_batch

        # Run the optimizer using this batch of training data.
        session.run(cnn['optimizer'], feed_dict=feed_dict_train)

        # Print status at end of each epoch
        if i % int(data.train.num_examples/batch_size) == 0: 
            val_loss = session.run(cnn['cost'], feed_dict=feed_dict_validate)
            epoch = int(i / int(data.train.num_examples/batch_size))
            
            print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss, session, cnn['accuracy'])
            
            if early_stopping:    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                else:
                    patience += 1

                if patience == early_stopping:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))

def print_validation_accuracy(data, session, cnn, cluster_descriptions, show_example_errors=False, show_confusion_matrix=False):
    """Print the classification accuracy on the validation-set."""
    # Number of images in the validation-set.
    num_test = len(data.valid.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_test)

        # Get the images from the validation-set between index i and j.
        images = data.valid.images[i:j, :].reshape(j-i, img_size_flat)

        # Get the associated labels.
        labels = data.valid.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {cnn['x_images']: images, cnn['y_true']: labels}
        
        # Add weather features if available
        if data.valid.features is not None:
            weather_features = data.valid.features[i:j, :]
            feed_dict[cnn['x_weather']] = weather_features

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(cnn['y_pred_cls'], feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    cls_true = np.array(data.valid.cls)

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the validation-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Validation-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Get class names for classification report
    if cluster_descriptions:
        class_names = get_weather_classes_from_clusters(cluster_descriptions)
    else:
        class_names = [f"Cluster {i}" for i in range(n_clusters)]

    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(cls_true, cls_pred, target_names=class_names))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred, correct, data, cluster_descriptions)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred, cls_true, class_names)

def plot_example_errors(cls_pred, correct, data, cluster_descriptions):
    """Plot examples of images that have been mis-classified."""
    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the validation-set that have been
    # incorrectly classified.
    images = data.valid.images[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.valid.cls[incorrect]

    # Plot the first 9 images.
    plot_images(images=images, cls_true=cls_true, cls_pred=cls_pred, cluster_descriptions=cluster_descriptions)

def plot_confusion_matrix(cls_pred, cls_true, class_names):
    """Plot the confusion matrix."""
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.show()

def main():
    """Main function to run the multi-modal weather classification CNN with clustering."""
    print("Multi-Modal Weather Classification CNN (Images + Weather Features)")
    print("=" * 70)
    
    # Load data with clustering
    print("Loading weather data and performing K-means clustering...")
    try:
        data = read_weather_sets_with_clustering(
            json_dir, image_dir, img_size, validation_size, use_features=True, n_clusters=n_clusters
        )
        print("Data loaded and clustered successfully!")
        
        # Get cluster descriptions
        cluster_descriptions = data.cluster_descriptions
        print("\nCluster Characteristics:")
        for cluster_id, desc in cluster_descriptions.items():
            print(f"  {desc['description']}:")
            print(f"    Cloud Coverage: {desc['avg_cloud_coverage']:.3f}")
            print(f"    Irradiance: {desc['avg_irradiance']:.1f}")
            print(f"    Sun Obscuration %: {desc['avg_sun_obscuration_percentage']:.3f}")
            print(f"    Count: {desc['count']}")
        
        # Check if weather features are available
        if data.train.features is not None:
            print(f"\nWeather features enabled: {data.train.features.shape[1]} features per sample")
            print("Features: cloud_coverage, sun_obscuration_percentage, irradiance, hour_sin, hour_cos, month_sin, month_cos")
        else:
            print("\nWarning: No weather features available. Using images only.")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"\nSize of:")
    print(f"- Training-set:\t\t{len(data.train.labels)}")
    print(f"- Validation-set:\t{len(data.valid.labels)}")

    # Get number of classes from clustering
    num_classes = n_clusters

    # Create the multi-modal CNN
    print(f"\nCreating multi-modal CNN model for {num_classes} weather classes...")
    cnn = create_weather_cnn(num_classes)

    # Create TensorFlow session
    session = tf.Session()

    # Initialize variables
    session.run(tf.global_variables_initializer())

    # Train the model
    print("\nStarting training...")
    optimize(num_iterations=1000, data=data, session=session, cnn=cnn, early_stopping=early_stopping)

    # Print validation accuracy
    print("\nFinal validation accuracy:")
    print_validation_accuracy(
        data, session, cnn, cluster_descriptions, 
        show_example_errors=True, show_confusion_matrix=True
    )

    # Close TensorFlow session
    session.close()
    print("\nTraining completed!")

if __name__ == "__main__":
    main() 