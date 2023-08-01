# Classification-model-with-CNN
Convolutional neural network is used to build a model which classify plants into diseased and normal 

The code starts by loading the image dataset using tf.keras.preprocessing.image_dataset_from_directory function. The images are loaded from the "C:/Users/rajan/Downloads/archive/PlantVillage" directory. The data is shuffled, resized to a specific image size (Img_size), and batched into batches of size 32.

A function dataset_partition is defined to split the dataset into training, validation, and test sets. The function takes the data and three split ratios (train_split, val_split, test_split) as input and returns the three split datasets.

The dataset is partitioned into training, validation, and test sets using the dataset_partition function.

Data caching and prefetching are applied to the training, validation, and test sets to optimize data loading during model training.

A data augmentation pipeline is created using the tf.keras.models.Sequential class. It includes resizing, rescaling, random horizontal and vertical flipping, random rotation, random contrast adjustment, and random zooming. Data augmentation is beneficial to increase the diversity of the training data, which helps the model generalize better.

The CNN model is defined using the tf.keras.models.Sequential class. It consists of several convolutional layers with activation functions, max-pooling layers for down-sampling, a flatten layer to convert the 2D feature maps into a 1D vector, and two dense layers for classification. The model is compiled with the "adam" optimizer, "sparse_categorical_crossentropy" loss function (suitable for integer labels), and "accuracy" as the evaluation metric.

The model is trained on the training data for 5 epochs, and the training history is stored in the history variable.

Two plots are generated to visualize the training and validation loss, as well as the training and validation accuracy, over the 5 epochs.

A function predict is defined to make predictions using the trained model on individual images. The function takes an image and the trained model as input and returns the predicted class and the confidence score of the prediction.

Finally, the code displays a set of test images from the test dataset and their corresponding actual and predicted classes. The images are plotted with their labels and the predicted labels along with the confidence score.
