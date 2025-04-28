This project implements a machine learning-based cloud security anomaly detection system using the Random Forest Classifier. The model is trained on a dataset from the NSL-KDD (Network Security Lab - KDD Cup 1999) dataset to classify network traffic anomalies. The primary objective is to detect abnormal patterns, such as unauthorized access attempts, suspicious login activity, or unusual data transmission, which could indicate potential security breaches.

The system uses Label Encoding to process categorical features and applies Random Forest to detect anomalies. The model's performance is evaluated based on metrics like Accuracy, Precision, Recall, F1-Score, ROC-AUC, and the Confusion Matrix.

Features
Anomaly detection using Random Forest Classifier.

Missing data handling for both numeric and categorical features.

Multi-metric evaluation: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

Output includes confusion matrix and evaluation scores.

Requirements
Python 3.7+

pandas (for data handling)

scikit-learn (for machine learning model and metrics)

NumPy (for numeric calculations)

Install the required libraries using pip:

bash
Copy
Edit
pip install pandas scikit-learn numpy
File Structure
The project consists of the following main files:

AnomalyDetection.py: Main Python script for anomaly detection, model training, and evaluation.

KDDTest-21.txt: The dataset used for training and testing the model. Ensure you place the correct path to this file in the script.

How to Use
Prepare the Dataset:

Download the NSL-KDD dataset (KDDTest-21.txt) from the official repository or use your own dataset.

Place the dataset file in the correct directory as indicated in the code.

Run the Script:

Run the Python script AnomalyDetection.py.

The script will perform the following steps:

Load and preprocess the dataset.

Handle missing values for numeric and categorical columns.

Split the dataset into training and testing sets (70% training, 30% testing).

Train a Random Forest Classifier on the training data.

Evaluate the model using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix.

Results:

The script will output the following:

Accuracy: The proportion of correct predictions.

Precision (macro): The average precision across all classes.

Recall (macro): The average recall across all classes.

F1-Score (macro): The harmonic mean of precision and recall.

ROC-AUC (macro): The area under the ROC curve for multi-class classification.

Confusion Matrix: A matrix that shows true positives, false positives, true negatives, and false negatives.

Interpret the Results:

Accuracy shows how well the model performs overall.

Precision and Recall reflect how well the model identifies anomalies while avoiding false positives and negatives.

F1-Score gives a balance between precision and recall.

ROC-AUC indicates how well the model distinguishes between classes.

