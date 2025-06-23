
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

def load_data(file_path):
    """
    Loads data from a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    """
    Cleans the DataFrame by performing the following steps:
    1. Drops rows with any NaN values.
    2. Drops duplicate rows.
    3. Removes specific non-numeric or irrelevant columns identified from the dataset.
    4. Replaces infinite values with NaN and then drops rows containing them.
    5. Converts all remaining columns to numeric types, coercing errors to NaN and dropping those rows.

    Args:
        df (pd.DataFrame): The input DataFrame to clean.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Columns identified as non-numeric or irrelevant based on common CICIDS2017 issues
    columns_to_drop = [
        'Flow ID', ' Source IP', ' Destination IP', ' Timestamp', 'SimillarHTTP',
        ' Fwd PSH Flags', ' Bwd PSH Flags', 'Fwd URG Flags', ' Bwd URG Flags',
        'CWE Flag Count', ' Fwd Byts/b Avg', ' Fwd Pkts/b Avg', ' Fwd Blk Rate Avg',
        ' Bwd Byts/b Avg', ' Bwd Pkts/b Avg', ' Bwd Blk Rate Avg'
    ]
    
    for col in columns_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # Replace infinite values with NaN and then drop them
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Convert all columns to numeric, coercing errors to NaN and dropping those rows
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    return df

def encode_labels(df, target_column='Label'):
    """
    Encodes the target column using sklearn.preprocessing.LabelEncoder.

    Args:
        df (pd.DataFrame): The DataFrame containing the target column.
        target_column (str): The name of the target column to encode.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The DataFrame with the encoded target column.
            - LabelEncoder: The fitted LabelEncoder instance.
    """
    le = LabelEncoder()
    df[target_column] = le.fit_transform(df[target_column])
    return df, le

def perform_feature_selection(X, y, n_features=59):
    """
    Performs Recursive Feature Elimination (RFE) to select the most important features.
    The estimator used is CatBoostClassifier, as specified in the thesis.

    Args:
        X (pd.DataFrame): The feature DataFrame.
        y (pd.Series): The target Series.
        n_features (int): The number of features to select, as specified in the thesis.

    Returns:
        tuple: A tuple containing:
            - pd.Index: An Index object containing the names of the selected features.
            - RFE: The fitted RFE selector instance.
    """
    # Use a basic CatBoostClassifier for RFE with reduced iterations for speed
    estimator = CatBoostClassifier(iterations=100, 
                                   learning_rate=0.1,
                                   depth=6,
                                   random_seed=42,
                                   verbose=0, # Suppress verbose output during RFE
                                   allow_writing_files=False)
    
    # Ensure n_features_to_select does not exceed the number of available features
    n_features_to_select = min(n_features, X.shape[1])
    
    rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=0.5)
    rfe.fit(X, y)
    selected_features = X.columns[rfe.support_]
    return selected_features, rfe

def train_model(X_train, y_train, class_weights=None):
    """
    Trains the CatBoostClassifier model with specified hyperparameters and class weights.
    Includes early stopping for improved generalization.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        class_weights (dict, optional): A dictionary of class weights to handle imbalance.
                                        Defaults to None.

    Returns:
        CatBoostClassifier: The trained CatBoost model.
    """
    model = CatBoostClassifier(iterations=1000,  # Number of boosting iterations
                               learning_rate=0.05, # Step size shrinkage
                               depth=10,           # Depth of the tree
                               loss_function='MultiClass',
                               eval_metric='Accuracy',
                               random_seed=42,
                               verbose=200, # Print training progress every 200 iterations
                               early_stopping_rounds=50, # Stop if no improvement for 50 rounds
                               class_weights=class_weights,
                               allow_writing_files=False) # Prevent CatBoost from writing files
    
    model.fit(X_train, y_train, early_stopping_rounds=50)
    return model

def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluates the trained model on the test set and prints various performance metrics.
    Metrics include Accuracy, Confusion Matrix, Classification Report, and Detection Rate.

    Args:
        model (CatBoostClassifier): The trained CatBoost model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        label_encoder (LabelEncoder): The fitted LabelEncoder used for target encoding.
    """
    y_pred = model.predict(X_test)
    # CatBoost predict can return a list of lists for multiclass, flatten it
    y_pred_classes = [int(x[0]) for x in y_pred]

    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_classes):.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_classes)
    print(cm)

    print("\nClassification Report:")
    # Use target_names from label_encoder to show original class names
    report = classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_, output_dict=True)
    print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))

    # Calculate and print Detection Rate (Recall for each class)
    print("\nDetection Rate (Recall per class):")
    for label_idx, label_name in enumerate(label_encoder.classes_):
        if label_name in report:
            print(f"  {label_name}: {report[label_name]['recall']:.4f}")


if __name__ == "__main__":
    # Define file path for the dataset
    data_file = '2018 dataset/02-28-2018.csv'

    # --- Dummy Data Creation for Demonstration/Testing ---
    # In a real scenario, ensure your actual CICIDS2017 dataset is correctly placed.
    # This block creates a small dummy CSV if the directory/file doesn't exist,
    # allowing the script to run for testing purposes without the full dataset.
    if not os.path.exists('2018 dataset'):
        os.makedirs('2018 dataset')
        dummy_data = {
            f'col{i}': np.random.rand(100) for i in range(1, 79) # Create 78 feature columns
        }
        dummy_data['Label'] = np.random.choice(['Benign', 'Infilteration', 'Attack'], 100, p=[0.8, 0.19, 0.01])
        dummy_data[' Timestamp'] = pd.to_datetime(pd.Series(range(100)).astype(str) + '-01-01').astype(str) # Dummy timestamp
        dummy_data['Flow ID'] = [f'flow_{i}' for i in range(100)] # Dummy Flow ID
        dummy_data['SimillarHTTP'] = [f'http_{i}' for i in range(100)] # Dummy SimillarHTTP
        # Add other dummy columns that are dropped in clean_data if they are expected in raw data
        dummy_data[' Fwd PSH Flags'] = np.random.randint(0, 2, 100)
        dummy_data[' Bwd PSH Flags'] = np.random.randint(0, 2, 100)
        dummy_data['Fwd URG Flags'] = np.random.randint(0, 2, 100)
        dummy_data[' Bwd URG Flags'] = np.random.randint(0, 2, 100)
        dummy_data['CWE Flag Count'] = np.random.randint(0, 2, 100)
        dummy_data[' Fwd Byts/b Avg'] = np.random.rand(100)
        dummy_data[' Fwd Pkts/b Avg'] = np.random.rand(100)
        dummy_data[' Fwd Blk Rate Avg'] = np.random.rand(100)
        dummy_data[' Bwd Byts/b Avg'] = np.random.rand(100)
        dummy_data[' Bwd Pkts/b Avg'] = np.random.rand(100)
        dummy_data[' Bwd Blk Rate Avg'] = np.random.rand(100)
        dummy_data[' Source IP'] = [f'192.168.1.{i}' for i in range(100)]
        dummy_data[' Destination IP'] = [f'192.168.2.{i}' for i in range(100)]

        dummy_df = pd.DataFrame(dummy_data)
        dummy_df.to_csv(data_file, index=False)
        print(f"Created dummy dataset at {data_file} for demonstration/testing.")

    # --- Main Workflow ---
    print(f"\nStarting RFE-CatBoost Model Training and Evaluation...")

    # 1. Load Data
    print(f"Loading data from {data_file}...")
    df = load_data(data_file)
    print(f"Original data shape: {df.shape}")

    # 2. Clean Data
    print("Cleaning data...")
    df = clean_data(df)
    print(f"Cleaned data shape: {df.shape}")

    # Separate features and target
    X = df.drop('Label', axis=1)
    y = df['Label']

    # 3. Encode Labels
    print("Encoding labels...")
    y, label_encoder = encode_labels(pd.DataFrame(y), target_column='Label')
    y = y['Label'] # Convert back to Series for consistency
    print(f"Label distribution after encoding:\n{y.value_counts()}")

    # 4. Feature Selection using RFE
    # The thesis specifies 59 features. We will attempt to select this many.
    n_features_thesis = 59
    print(f"Performing Recursive Feature Elimination (RFE) to select {n_features_thesis} features...")
    selected_features, rfe_selector = perform_feature_selection(X, y, n_features=n_features_thesis)
    X_selected = X[selected_features]
    print(f"Selected {len(selected_features)} features: {list(selected_features)}")

    # 5. Split Data into Training and Testing Sets
    # Using stratified split to maintain class distribution, as per best practices for imbalanced data.
    print("Splitting data into training and testing sets (80/20 split, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # 6. Handle Class Imbalance on Training Data (using SMOTE)
    # SMOTE is applied only to the training data to prevent data leakage.
    print("Checking for and handling class imbalance in training data with SMOTE...")
    min_class_count = y_train.value_counts().min()
    # SMOTE requires at least 2 samples for the minority class to create synthetic samples.
    if min_class_count > 1:
        smote = SMOTE(random_state=42, k_neighbors=min(min_class_count - 1, 5)) 
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"Training data shape after SMOTE: {X_train_resampled.shape}")
        print(f"Label distribution after SMOTE:\n{y_train_resampled.value_counts()}")
    else:
        print(f"Skipping SMOTE: Minority class has {min_class_count} sample(s). SMOTE requires > 1 sample.")
        X_train_resampled, y_train_resampled = X_train, y_train

    # Calculate class weights for CatBoost to further address imbalance
    class_counts = y_train_resampled.value_counts()
    total_samples = len(y_train_resampled)
    # Ensure class_weights are floats for CatBoost
    class_weights = {int(cls): total_samples / count for cls, count in class_counts.items()}
    print(f"Calculated class weights for CatBoost: {class_weights}")

    # 7. Train CatBoost Model
    print("Training CatBoost model on resampled data...")
    model = train_model(X_train_resampled, y_train_resampled, class_weights=class_weights)
    print("CatBoost model training complete.")

    # 8. Evaluate Model
    evaluate_model(model, X_test, y_test, label_encoder)

    print("\nScript execution finished successfully.")


