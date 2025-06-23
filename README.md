# RFE-CatBoost
This repository contains the Python implementation of the RFE-CatBoost model for network anomaly detection.
Project Structure
. \
├── src/ 
│   └── main.py
├── 2018 dataset/ 
│   └── 02-28-2018.csv  (Placeholder for the actual dataset)
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE

• src/main.py: Contains the core Python code for data loading, preprocessing, feature selection, model training, and evaluation.

• 2018 dataset/: Directory to store the CICIDS2017 dataset files. A dummy file is created for demonstration purposes if the actual dataset is not present.

• README.md: This file, providing an overview of the project.

• requirements.txt: Lists all necessary Python dependencies.

• .gitignore: Specifies intentionally untracked files to ignore.

• LICENSE: Details the licensing for the project.

Note: The script expects the 02-28-2018.csv file from the CICIDS2017 dataset to be present in the 2018 dataset/ directory. If the dataset is not found, a small dummy dataset will be created for demonstration purposes, allowing the script to run without errors, but the results will not be representative of the thesis findings.

Dataset

The model is trained and evaluated using the CICIDS2017 dataset. This dataset is publicly available and widely used for network intrusion detection research. Due to its size, it is not included directly in this repository. Please download the relevant CSV files (specifically 02-28-2018.csv for this implementation) and place them in the 2018 dataset/ directory.


Model Architecture

The RFE-CatBoost model consists of two main stages:

1.
Recursive Feature Elimination (RFE): This technique is used to select the most relevant features from the high-dimensional network traffic data. As per the thesis, RFE is configured to select 59 salient features, significantly reducing noise and computational load.

2.
CatBoost Classifier: The selected features are then fed into a CatBoost classifier. CatBoost is a gradient boosting on decision trees library that is particularly effective with heterogeneous data and handles categorical features natively. It is chosen for its high accuracy, speed, and robustness against overfitting.

Evaluation Metrics

The model's performance is evaluated using the following metrics, consistent with the thesis:

•
Accuracy: Overall correctness of the model.

•
Confusion Matrix: A table showing the performance of a classification model on a set of test data for which the true values are known.

•
Classification Report: Provides precision, recall, and F1-score for each class.

•
Detection Rate (Recall per class): The proportion of actual positive cases that are correctly identified.

Reproducibility

To ensure reproducibility of the results, the following measures have been implemented:

•
Fixed Random Seeds: All stochastic processes (e.g., data splitting, model initialization) are initialized with a fixed random seed (random_state=42) to ensure consistent results across multiple runs.

•
Clear Dependencies: All required Python packages are listed in requirements.txt with their exact versions.

•
Modular Code: The codebase is structured into functions for clarity and reusability.

Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

License

This project is licensed under the MIT License - see the LICENSE file for details.






