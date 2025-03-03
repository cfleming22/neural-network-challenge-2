# Module 19 Challenge README

## Introduction
This repository contains the solution to the Module 19 Challenge, which involves creating a neural network to predict employee attrition and department.

## Dataset
The dataset used for this challenge is the Attrition Dataset, which can be found at [https://static.bc-edx.com/ai/ail-v-1-0/m19/lms/datasets/attrition.csv](https://static.bc-edx.com/ai/ail-v-1-0/m19/lms/datasets/attrition.csv).

## Preprocessing
The following preprocessing steps were taken:

* **Imported necessary libraries:** 
    We imported the necessary libraries, including `pandas` for data manipulation, `numpy` for numerical computations, `sklearn.model_selection` for splitting the data, `sklearn.preprocessing` for scaling and encoding, and `tensorflow` for building the neural network.
    
    ```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
Loaded the dataset into a Pandas DataFrame:
We loaded the Attrition Dataset into a Pandas DataFrame using pd.read_csv.

attrition_df = pd.read_csv('https://static.bc-edx.com/ai/ail-v-1-0/m19/lms/datasets/attrition.csv')


* **Split the data into training and testing sets (80% for training and 20% for testing):** 
    We split the data into training and testing sets using `train_test_split`.
    
    ```python
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)
Converted categorical features to numerical using One-Hot Encoding:
We converted categorical features to numerical using OneHotEncoder.

encoder = OneHotEncoder(sparse=False)
X_train_categorical = encoder.fit_transform(X_train[categorical_cols])
X_test_categorical = encoder.transform(X_test[categorical_cols])


* **Scaled numerical features using Standard Scaler:** 
    We scaled numerical features using `StandardScaler`.
    
    ```python
scaler = StandardScaler()
X_train_numerical = scaler.fit_transform(X_train[numerical_cols])
X_test_numerical = scaler.transform(X_test[numerical_cols])
Model Architecture
Shared Layers:
We designed two shared dense layers with 64 and 32 neurons, respectively, using ReLU activation.

shared_layer1 = layers.Dense(64, activation='relu')(input_layer)
shared_layer2 = layers.Dense(32, activation='relu')(shared_layer1)


* **Department Branch:** 
    **Target Column:** Department
    **Hidden Layer:** 1 dense layer with 16 neurons using ReLU activation
    **Output Layer:** 1 output layer with softmax activation (number of neurons equals the number of unique departments)
    
    ```python
department_branch = layers.Dense(16, activation='relu')(shared_layer2)
department_output = layers.Dense(len(np.unique(y_train[:, 1:]), activation='softmax')(department_branch)
Attrition Branch:
Target Column: Attrition
Hidden Layer: 1 dense layer with 16 neurons using ReLU activation
Output Layer: 1 output layer with softmax activation (number of neurons equals the number of unique attrition classes, which is 2 in this case)

attrition_branch = layers.Dense(16, activation='relu')(shared_layer2)
attrition_output = layers.Dense(len(np.unique(y_train[:, 0]), activation='softmax')(attrition_branch)


## Model Compilation and Training
* **Compiled the model with the Adam optimizer and categorical cross-entropy loss function:** 
    We compiled the model using `Model.compile`.
    
    ```python
model.compile(optimizer='adam', loss=['categorical_crossentropy', 'categorical_crossentropy'], metrics=['accuracy'])
Trained the model for 10 epochs with a batch size of 128:
We trained the model using Model.fit.

model.fit(X_train, [y_train[:, 1:], y_train[:, 0]], epochs=10, batch_size=128, validation_data=(X_test, [y_test[:, 1:], y_test[:, 0]]))


## Evaluation
* **Evaluated the model on the testing data:** 
    We evaluated the model using `Model.evaluate`.
    
    ```python
department_loss, department_accuracy, attrition_loss, attrition_accuracy = model.evaluate(X_test, [y_test[:, 1:], y_test[:, 0]])
Printed the accuracy for both Department and Attrition:
We printed the accuracy using print.

print("Department Accuracy:", department_accuracy)
print("Attrition Accuracy:", attrition_accuracy)


## Training Improvement Analysis

* **Epochs and Batch Size:** 
    The model was trained for 10 epochs with a batch size of 128, allowing it to learn patterns and relationships within the data without overfitting or underfitting.
* **Optimizer and Loss Function:** 
    The Adam optimizer and categorical cross-entropy loss function were used, enabling the model to converge efficiently and optimize its predictions for both Department and Attrition.
* **Accuracy Improvement:** 
    The training process likely improved the model's accuracy by reducing the loss function's value and increasing the model's ability to generalize to unseen data.

## Part 3 Summary Answers

### 1. Is accuracy the best metric to use on this data? Why or why not?
Accuracy might not be the best metric for this data, especially for the Department prediction, as it is a multi-class classification problem with an imbalanced dataset. In such cases, metrics like precision, recall, F1-score, or even AUC-ROC might provide a more comprehensive understanding of the model's performance. However, for the Attrition prediction, which is a binary classification problem, accuracy can be a suitable metric.

### 2. What activation functions did you choose for your output layers, and why?
For the output layers, I chose the softmax activation function. This is because both Department and Attrition predictions are classification problems, and softmax is suitable for multi-class classification. It ensures that the output probabilities are normalized and add up to 1, making it easier to interpret the results.

### 3. Can you name a few ways that this model might be improved?
* **Collect more data**: 
    Increasing the dataset size, especially for the minority classes in the Department prediction, could improve the model's performance.
* **Feature Engineering**: 
    Extracting more relevant features from the existing data or incorporating additional datasets (e.g., economic indicators, industry trends) might enhance the model's predictive power.
* **Hyperparameter Tuning**: 
    Systematically tuning hyperparameters (e.g., number of layers, neurons, epochs, batch size) using techniques like Grid Search or Random Search could lead to better model performance.
* **Experiment with Different Models**: 
    Trying out other machine learning models (e.g., Random Forest, Gradient Boosting, Support Vector Machines) or deep learning architectures (e.g., CNN, LSTM) might yield better results for this specific problem.

## Code
The code for this challenge can be found in the `attrition.ipynb` file.
