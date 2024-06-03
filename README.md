# **🚗 Automated License Plate Detection and Recognition System**

## **🌐 Domain**

Computer Vision and Deep Learning

## **📝 Problem Statement**

Develop an automated system capable of detecting and recognizing license plates from vehicle images.

The system should accurately locate the license plate in an image and identify the characters on the plate.

## **🛠️ Skills Used**

Python Programming

Computer Vision

Machine Learning and Deep Learning

Data Preprocessing and Augmentation

Model Evaluation and Optimization

## **🌟 Overview of the Project**

This project involves creating two primary models: one for detecting license plates in vehicle images and another for recognizing the characters on the plates.

The project is structured into multiple stages, including data loading, preprocessing, model building, training, and integration.

## **Project Approach**

**Data Loading:** Import vehicle images, license plate images, and annotations.

**Pre-processing:** Resize and normalize images.

**Model Building:**

**Detection Model:** CNN for locating license plates.

**Recognition Model:** CNN + LSTM for character recognition.

**Model Training:** Train both models using appropriate loss functions and optimizers.

**Integration:** Combine both models for end-to-end license plate recognition.

**Testing and Validation:** Validate the system using a test dataset, ensuring accuracy and robustness.

## **🚀 Detailed Project Approach**

### **Stage 1: Data Preprocessing and Exploration**

**Importing Required Libraries:** Import necessary libraries including OpenCV for image processing, NumPy for numerical operations, Pandas for data manipulation, Matplotlib for visualization,
                              Scikit-learn for data splitting and preprocessing, and TensorFlow/Keras for building and training neural networks.

**Define Paths for Data Directories:** Specify paths for directories containing vehicle images, license plate images, and annotations.

**Loading Data:** Load the annotations from CSV files for both vehicle detection and license plate recognition. 
              Additionally, load the vehicle images and license plate images from their respective directories.


**Pre-processing Data:** Resize the vehicle images and license plate images to a fixed size (e.g., 224x224) and normalize their pixel values to a range of [0, 1].

**Data Exploration:** Visualize samples of vehicle images with annotations (bounding boxes) and license plate images without annotations to gain insights 
                      into the dataset and verify the correctness of annotations.


### **Stage 2: Model Building**

#### **Model Building for License Plate Detection:**

**Prepare Data:** Extract bounding box coordinates and normalize them.

**Build Model:** Create a Convolutional Neural Network (CNN) for license plate detection.

**Compile and Train:** Compile the model using Mean Squared Error (MSE) loss and the Adam optimizer. Train the model and monitor the validation loss.

**Evaluate Model:** Plot training and validation loss curves to assess model performance.

#### **Model Building for License Plate Recognition:**

**Encode License Plate Characters:** Convert characters in license plates to numerical format using Label Encoding.

**Pad Sequences:** Ensure uniform sequence length by padding shorter sequences.

**Build Model:** Develop a model combining CNN for feature extraction and LSTM for sequence learning.

**Compile and Train:** Use categorical cross-entropy loss and the Adam optimizer to compile and train the model.

**Evaluate Model:** Assess the model's performance on the validation set using loss and accuracy metrics.

## **Stage 3: Integration and Testing**

**Bounding Box Validation:** Ensure bounding box coordinates are within image bounds and validate bounding box area before resizing.

**Handling Invalid Boxes:** Implement handling for invalid bounding boxes by setting predicted text to an empty string to prevent processing errors.

**Image Loading Check:** Verify correct image loading before processing to avoid errors from missing or corrupted files, ensuring smooth execution.

**Integration and Testing:** Integrate the detection and recognition models to create a complete system. Test the system with new images, process them, and generate results in a structured format.

## **About the Models Used**

**License Plate Detection Model:** A Convolutional Neural Network (CNN) designed to locate license plates in vehicle images.

**License Plate Recognition Model:** A combination of CNN and LSTM designed to recognize and decode characters from the detected license plate images.

## **Advantages of the Project**

**Automation:** Reduces manual effort in license plate recognition.

**Accuracy:** High precision in detecting and recognizing license plates.

**Scalability:** Can be scaled to work with a larger dataset or real-time systems.

## **Conclusion**

This project demonstrates an effective approach to license plate detection and recognition using deep learning techniques. 

The integration of CNN and LSTM models ensures robust and accurate performance, making it a valuable solution for automated license plate recognition systems.
