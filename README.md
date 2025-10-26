
Lending Club Loan Default Prediction (EDA + PyTorch MLP)

Project Overview

This project aims to predict the likelihood of a loan default using the Lending Club dataset (2007-2018). The notebook performs a comprehensive Exploratory Data Analysis (EDA) to clean and prepare the data, followed by the implementation and training of a Multi-Layer Perceptron (MLP) neural network using PyTorch to classify loans.

Dataset

The project uses the accepted_2007_to_2018Q4.csv.gz file from the Lending Club Loan Data dataset on Kaggle. This is a very large (over 2.2 million rows) and wide (over 150 columns) tabular dataset of peer-to-peer loans.

Workflow

The notebook follows these key steps:

Data Loading: The large, compressed Gzip CSV is loaded into a pandas DataFrame.

Target Variable Definition: The loan_status column is used to create the binary target variable.

Charged Off loans are mapped to 1 (default).

Fully Paid loans are mapped to 0 (non-default).

All other statuses (e.g., 'Current', 'Late') are filtered out as their final outcome is unknown.

Exploratory Data Analysis (EDA) & Feature Cleaning:

Missing Values: Columns with a high percentage of missing data are identified and dropped.

Irrelevant Features: Columns that are not useful for prediction (like unique IDs, free text) are removed.

Data Leakage: Features that "leak" information from the future are removed (e.g., total_pymnt, last_pymnt_d). These are columns that would not be available at the time of the loan application.

Feature Engineering:

Date columns (like earliest_cr_line) are converted into numerical features (e.g., length of credit history).

Categorical string columns (like emp_length) are mapped to numerical values.

Remaining categorical features (like purpose, home_ownership) are one-hot encoded.

Remaining missing values in key features are imputed (e.g., using the median).

Preprocessing for Deep Learning:

The final feature set is split into training and testing sets.

All features are scaled using StandardScaler to normalize the data, which is crucial for neural network performance.

Model Training (PyTorch):

A custom Dataset and DataLoader are created to feed the data to the model in batches.

A simple Multi-Layer Perceptron (MLP) architecture is defined.

The model is trained for 10 epochs using Binary Cross-Entropy (BCE) Loss and the Adam optimizer.

Evaluation:

The trained model is evaluated on the unseen test set.

The ROC AUC Score and F1-Score are calculated to measure the model's performance.

Model Architecture

The model is a feed-forward neural network (MLP) built with PyTorch. A typical architecture for this task, as implemented in the notebook, would look like this:

Input Layer (size = number of features)

Hidden Layer 1 (e.g., 128 neurons) + ReLU Activation

Dropout (for regularization)

Hidden Layer 2 (e.g., 64 neurons) + ReLU Activation

Dropout

Output Layer (1 neuron) + Sigmoid Activation (to output a probability between 0 and 1)

Results

The model was trained for 10 epochs, with the training loss steadily decreasing. The final performance is measured on the test set using two key metrics:

Test ROC AUC Score: Measures the model's ability to distinguish between default and non-default loans.

Test F1-Score: Provides a balanced measure of precision and recall, which is important for an imbalanced dataset.

How to Run

Environment: Ensure you have a Python environment with the following libraries installed:

pandas

numpy

scikit-learn

torch (PyTorch)

Data: Download the accepted_2007_to_2018Q4.csv.gz file from the Kaggle dataset linked above.

Path: Make sure the path to the data in the notebook matches its location on your system (e.g., /kaggle/input/lending-club/accepted_2007_to_2018Q4.csv.gz).

Execute: Run the cells in the shodh-ml-eda-dl (2).ipynb notebook sequentially. A GPU is recommended for faster training.
