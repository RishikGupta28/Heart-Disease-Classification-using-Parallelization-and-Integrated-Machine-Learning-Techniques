# Heart-Disease-Classification-using-Parallelization-and-Integrated-Machine-Learning-Techniques

Overview
This project focuses on the early detection of heart disease using advanced machine learning classification techniques. By leveraging multiple algorithms and parallel processing techniques, the goal is to enhance the accuracy and efficiency of heart disease prognosis.

Dataset
The datasets used in this project are sourced from the UCI Machine Learning Repository. Specifically, three distinct heart disease datasets were utilized:

Cleveland Heart Disease Dataset
Hungarian Heart Disease Dataset
Switzerland Heart Disease Dataset
These datasets include a variety of features related to patient medical records and diagnostic results. The datasets are preprocessed to handle missing values, normalize feature scales, and eliminate outliers.

Algorithms Used
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Artificial Neural Networks (ANN)
Tech Stack
Programming Language: Python
Libraries and Frameworks:
Pandas
NumPy
Scikit-learn
TensorFlow/Keras
Matplotlib
Approach
Data Preprocessing:

Handle missing values using imputation techniques.
Normalize feature scales using z-score normalization and robust scaling.
Remove outliers and smooth data to ensure high-quality input.
Feature Selection:

Utilize backward modeling and rigorous statistical tests to identify the most relevant and informative features.
Apply mutual information and Fisher score for feature selection.
Algorithm Selection:

Implement SVM, KNN, and ANN to leverage their unique advantages in classification tasks.
Use parallel processing techniques to reduce training time and improve efficiency.
Model Training and Evaluation:

Split the dataset into training and testing subsets.
Train each model using cross-validation to ensure generalizability.
Evaluate the models using metrics such as accuracy, precision, recall, F1-score, and ROC curves.
Perform hyperparameter tuning using grid search and other optimization techniques.
Parallelization:

Utilize parallel processing to handle large datasets efficiently.
Implement multiprocessing techniques to accelerate the prediction process and enhance computational efficiency.
Results
The models demonstrated improved accuracy and efficiency in predicting heart disease. KNN, combined with parallel processing techniques, achieved the highest performance with an accuracy of 90.25%.

Conclusion
This project showcases the potential of integrating multiple machine learning algorithms and parallel processing techniques for enhanced heart disease classification. The proposed methodology can be further extended and optimized for real-world clinical applications.
