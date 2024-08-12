import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_preprocess_data(file_path, target_column):
    dataset = pd.read_csv(file_path)

    # Handle missing values
    imputer = SimpleImputer(strategy="most_frequent")
    dataset = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)

    glucose_bins = [0, 140, float("inf")]
    glucose_labels = [0, 1]  # 0: Normal, 1: High
    dataset["Glucose_class"] = pd.cut(
        dataset["Glucose"], bins=glucose_bins, labels=glucose_labels, right=False
    )

    blood_pressure_bins = [0, 80, float("inf")]
    blood_pressure_labels = [0, 1]  # 0: Normal, 1: High
    dataset["BloodPressure_class"] = pd.cut(
        dataset["BloodPressure"],
        bins=blood_pressure_bins,
        labels=blood_pressure_labels,
        right=False,
    )

    skin_bins = [0, dataset["SkinThickness"].median(), float("inf")]
    skin_labels = [0, 1]
    dataset["SkinThickness_class"] = pd.cut(
        dataset["SkinThickness"], bins=skin_bins, labels=skin_labels, right=False
    )

    insulin_bins = [0, 125, float("inf")]
    insulin_labels = [0, 1]  # 0: Normal, 1: High
    dataset["Insulin_class"] = pd.cut(
        dataset["Insulin"], bins=insulin_bins, labels=insulin_labels, right=False
    )

    bmi_bins = [0, 25.0, 30.0, 35.0, 40.0, float("inf")]
    bmi_labels = [0, 1, 2, 3, 4]
    dataset["BMI_class"] = pd.cut(
        dataset["BMI"], bins=bmi_bins, labels=bmi_labels, right=False
    )

    dpf_bins = [0, dataset["DiabetesPedigreeFunction"].median(), float("inf")]
    dpf_labels = [0, 1]
    dataset["DiabetesPedigreeFunction_class"] = pd.cut(
        dataset["DiabetesPedigreeFunction"],
        bins=dpf_bins,
        labels=dpf_labels,
        right=False,
    )

    age_bins = [0, 30, 60, float("inf")]
    age_labels = [0, 1, 2]  # 0: muda, 1: dewasa, 2: lansia
    dataset["Age_class"] = pd.cut(
        dataset["Age"], bins=age_bins, labels=age_labels, right=False
    )

    # Drop original columns used for classification
    dataset.drop(
        columns=[
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ],
        inplace=True,
    )

    # Normalize the remaining data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(dataset.drop(columns=[target_column]))
    labels = dataset[target_column].values

    return scaled_features, labels


def split_data_kfold(scaled_features, labels, n_splits=10, test_size=0.1):
    # First split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_features, labels, test_size=test_size, random_state=42
    )

    # Now split the training data using KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = []

    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        folds.append((X_train_fold, X_val_fold, y_train_fold, y_val_fold))

    return folds, (X_test, y_test)


def save_split_data_kfold(folds, test_data, directory_path):
    for i, fold in enumerate(folds):
        X_train, X_val, y_train, y_val = fold
        fold_directory_path = os.path.join(directory_path, f"Fold_{i + 1}")
        os.makedirs(fold_directory_path, exist_ok=True)

        pd.DataFrame(X_train).to_csv(
            os.path.join(fold_directory_path, "X_train.csv"), index=False
        )
        pd.DataFrame(X_val).to_csv(
            os.path.join(fold_directory_path, "X_val.csv"), index=False
        )
        pd.DataFrame(y_train).to_csv(
            os.path.join(fold_directory_path, "y_train.csv"), index=False
        )
        pd.DataFrame(y_val).to_csv(
            os.path.join(fold_directory_path, "y_val.csv"), index=False
        )

    X_test, y_test = test_data
    pd.DataFrame(X_test).to_csv(os.path.join(directory_path, "X_test.csv"), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(directory_path, "y_test.csv"), index=False)


def visualize_preprocessing(original_data, scaled_features, target_column):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    sns.boxplot(data=original_data.drop(columns=[target_column]), ax=axes[0])
    axes[0].set_title("Original Data")

    sns.boxplot(
        data=pd.DataFrame(
            scaled_features, columns=original_data.drop(columns=[target_column]).columns
        ),
        ax=axes[1],
    )
    axes[1].set_title("Scaled Data")

    plt.tight_layout()
    return fig
