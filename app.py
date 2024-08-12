import streamlit as st
import pandas as pd

# import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.tree import plot_tree
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from preprocessing_data import (
    load_and_preprocess_data,
    split_data_kfold,
    save_split_data_kfold,
    visualize_preprocessing,
)
from feature_optimization import optimize_features
from model_training import (
    train_and_evaluate_model_c45,
    train_and_evaluate_model_c45_pso,
    plot_accuracies,
)


def save_model(model, model_path):
    with open(model_path, "wb") as file:
        pickle.dump(model, file)


def load_model(model_path):
    with open(model_path, "rb") as file:
        return pickle.load(file)


st.title("Diabetes Classification")

file = st.file_uploader("Choose a CSV file", type="csv")

if not file:
    st.warning("Please upload a CSV file before preprocessing and classification.")
    st.stop()

if file:
    try:
        file_path = os.path.join("dataset", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        st.success(f"File '{file.name}' successfully saved to 'dataset/' folder.")

        dataset = pd.read_csv(file_path)
        st.subheader("Preview of Uploaded Dataset")
        st.dataframe(dataset.head())

        dataset = pd.read_csv(file_path)
        if dataset.empty:
            st.error("The uploaded file is empty. Please upload a valid CSV file.")
        else:
            st.subheader("Data Information Before Preprocessing")
            st.write(f"Number of attributes: {dataset.shape[1]}")
            st.write(f"Number of data points: {dataset.shape[0]}")

            columns = dataset.columns.tolist()

        # Let the user select the target column
        target_column = st.selectbox("Select the target column", columns)
        if target_column:
            try:
                scaled_features, labels = load_and_preprocess_data(file, target_column)
                folds, (X_test, y_test) = split_data_kfold(
                    scaled_features, labels, n_splits=10
                )
                # Continue with your further processing or display
            except KeyError:
                st.warning(
                    f"Column '{target_column}' cannot be selected as the target column."
                )

            # Let the user select the balancing method
            balance_method = st.selectbox(
                "Select balancing method", ["None", "SMOTE", "Undersampling"]
            )

        if st.button("Preprocess Dataset"):
            if target_column:
                # Scale features and get labels
                print("Preprocessing data...")
                scaled_features, labels = load_and_preprocess_data(
                    file_path, target_column
                )
                print("Data preprocessing complete.")

                # Apply balancing method
                if balance_method == "SMOTE":
                    smote = SMOTE()
                    scaled_features, labels = smote.fit_resample(
                        scaled_features, labels
                    )
                elif balance_method == "Undersampling":
                    rus = RandomUnderSampler()
                    scaled_features, labels = rus.fit_resample(scaled_features, labels)

                st.subheader("Preview of Preprocessed Data")
                st.dataframe(
                    pd.DataFrame(
                        scaled_features,
                        columns=dataset.drop(columns=[target_column]).columns,
                    ).head()
                )

                st.subheader("Data Visualization")
                fig = visualize_preprocessing(dataset, scaled_features, target_column)
                st.pyplot(fig)

                st.subheader("Data Information After Preprocessing")
                st.write(f"Number of attributes: {scaled_features.shape[1]}")
                st.write(f"Number of data points: {scaled_features.shape[0]}")

                # Count values in the target column
                value_counts = pd.Series(labels).value_counts()
                st.write(f"Value counts in `{target_column}` column:")
                st.write(value_counts)

                # Split data using KFold
                folds, test_data = split_data_kfold(
                    scaled_features, labels, n_splits=10
                )

                save_split_data_kfold(folds, test_data, "dataset/folds")
                st.success("Data processed and split successfully.")
        # else:
        #     st.warning("Please preprocess data before starting classification.")
    except pd.errors.EmptyDataError:
        st.error("No columns to parse from file. Please upload a valid CSV file.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Check if preprocessed data exists
if os.path.exists("dataset/folds/X_test.csv"):
    # Load the split data
    folds = []
    for fold in range(10):
        fold_directory = f"dataset/folds/Fold_{fold + 1}"
        X_train_fold = pd.read_csv(f"{fold_directory}/X_train.csv").values
        X_val_fold = pd.read_csv(f"{fold_directory}/X_val.csv").values
        y_train_fold = pd.read_csv(f"{fold_directory}/y_train.csv").values.ravel()
        y_val_fold = pd.read_csv(f"{fold_directory}/y_val.csv").values.ravel()
        folds.append((X_train_fold, X_val_fold, y_train_fold, y_val_fold))

    # Load test data
    X_test = pd.read_csv("dataset/folds/X_test.csv").values
    y_test = pd.read_csv("dataset/folds/y_test.csv").values.ravel()

col1, col2, col3, col4 = st.columns(4)

with col1:
    swarmsize = st.number_input("Particle Size", min_value=1, step=1, value=10)
with col2:
    maxiter = st.number_input("Max Iterations", min_value=1, step=1, value=10)
with col3:
    c1 = st.number_input("C1", min_value=0.0, step=0.1, value=2.0)
with col4:
    c2 = st.number_input("C2", min_value=0.0, step=0.1, value=2.0)

if st.button("Start Classification"):
    for fold, (
        X_train_fold,
        X_val_fold,
        y_train_fold,
        y_val_fold,
    ) in enumerate(folds):
        fold_number = fold + 1

        c45_model_path = f"model/c45_model_fold_{fold_number}.pkl"
        c45_pso_model_path = f"model/c45_pso_model_fold_{fold_number}.pkl"

        # Train and evaluate C4.5 model (without PSO)
        if not os.path.exists(c45_model_path):
            c45_model, c45_accuracy = train_and_evaluate_model_c45(
                X_train_fold, X_val_fold, y_train_fold, y_val_fold
            )
            save_model(c45_model, c45_model_path)
        else:
            c45_model = load_model(c45_model_path)
            c45_accuracy = train_and_evaluate_model_c45(
                X_train_fold,
                X_val_fold,
                y_train_fold,
                y_val_fold,
                model=c45_model,
            )

        # init
        train_accuracies = []
        val_accuracies = []
        val_precisions = []
        val_recalls = []
        val_f1_scores = []
        iterations = []

        # Train and evaluate C4.5-PSO model (with PSO)
        if not os.path.exists(c45_pso_model_path):
            selected_features = optimize_features(
                scaled_features,
                labels,
                swarmsize=swarmsize,
                maxiter=maxiter,
                c1=c1,
                c2=c2,
                w=1.0,
                use_pso=True,
            )
            for i, (
                X_train_fold,
                X_val_fold,
                y_train_fold,
                y_val_fold,
            ) in enumerate(folds):
                (
                    c45_pso_model,
                    val_accuracy,
                    val_precision,
                    val_recall,
                    val_f1,
                ) = train_and_evaluate_model_c45_pso(
                    X_train_fold,
                    X_val_fold,
                    y_train_fold,
                    y_val_fold,
                    selected_features,
                )
                train_accuracies.append(
                    c45_pso_model.score(
                        X_train_fold[:, selected_features], y_train_fold
                    )
                )
                val_accuracies.append(val_accuracy)
                val_precisions.append(val_precision)
                val_recalls.append(val_recall)
                val_f1_scores.append(val_f1)
                iterations.append(i + 1)
                save_model(c45_pso_model, c45_pso_model_path)
        else:
            c45_pso_model = load_model(c45_pso_model_path)
            selected_features = optimize_features(
                scaled_features,
                labels,
                swarmsize=swarmsize,
                maxiter=maxiter,
                c1=c1,
                c2=c2,
                w=1.0,
                use_pso=True,
            )
            for i, (
                X_train_fold,
                X_val_fold,
                y_train_fold,
                y_val_fold,
            ) in enumerate(folds):
                (
                    c45_pso_model,
                    val_accuracy,
                    val_precision,
                    val_recall,
                    val_f1,
                ) = train_and_evaluate_model_c45_pso(
                    X_train_fold,
                    X_val_fold,
                    y_train_fold,
                    y_val_fold,
                    selected_features,
                )
                train_accuracies.append(
                    c45_pso_model.score(
                        X_train_fold[:, selected_features], y_train_fold
                    )
                )
                val_accuracies.append(val_accuracy)
                val_precisions.append(val_precision)
                val_recalls.append(val_recall)
                val_f1_scores.append(val_f1)
                iterations.append(i + 1)

    col1, col2 = st.columns(2)

    with col1:
        st.header("C4.5")
        # C4.5 Confusion Matrix
        c45_conf_matrix = confusion_matrix(y_test, c45_model.predict(X_test))

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(c45_conf_matrix, annot=True, cmap="Blues", fmt="d")
        plt.title("C4.5 Confusion Matrix")
        plt.xlabel("Classified")
        plt.ylabel("Actual")
        plt.savefig("result/c45_confusion_matrix.png")
        st.image("result/c45_confusion_matrix.png", use_column_width=True)

        # Calculate metrics c4.5
        c45_accuracy = accuracy_score(y_test, c45_model.predict(X_test))
        c45_precision = precision_score(y_test, c45_model.predict(X_test))
        c45_recall = recall_score(y_test, c45_model.predict(X_test))
        c45_f1 = f1_score(y_test, c45_model.predict(X_test))

        c45_accuracy = round(c45_accuracy * 100, 2)
        c45_precision = round(c45_precision * 100, 2)
        c45_recall = round(c45_recall * 100, 2)
        c45_f1 = round(c45_f1 * 100, 2)

        # Display metrics c4.5
        st.write("Accuracy:", c45_accuracy)
        st.write("Precision:", c45_precision)
        st.write("Recall:", c45_recall)
        st.write("F1-score:", c45_f1)

        # Plot decision tree
        st.subheader("C4.5 Decision Tree")
        feature_names = dataset.drop(columns=[target_column]).columns
        plt.figure(figsize=(50, 21))
        plot_tree(
            c45_model,
            filled=True,
            feature_names=feature_names,
            class_names=["No Diabetes", "Diabetes"],
        )
        plt.title("C4.5 Decision Tree")
        # st.pyplot(plt)
        plt.savefig("result/c45_dt.svg", dpi=200)
        st.image("result/c45_dt.svg")
        st.caption("Figure 1: C4.5 Decision Tree")

    with col2:
        st.header("C4.5-PSO")
        # C4.5-PSO Confusion Matrix
        c45_pso_conf_matrix = confusion_matrix(
            y_test, c45_pso_model.predict(X_test[:, selected_features])
        )

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(c45_pso_conf_matrix, annot=True, cmap="Blues", fmt="d")
        plt.title("C4.5-PSO Confusion Matrix")
        plt.xlabel("Classified")
        plt.ylabel("Actual")
        plt.savefig("result/c45_pso_confusion_matrix.png")
        st.image("result/c45_pso_confusion_matrix.png", use_column_width=True)

        # Plot accuracies
        best_accuracy, average_accuracy = plot_accuracies(
            train_accuracies, val_accuracies, iterations
        )
        # st.pyplot(plt)

        # Calculate and display the best precision, recall, and F1 score
        best_index = val_accuracies.index(best_accuracy)
        best_precision = val_precisions[best_index]
        best_recall = val_recalls[best_index]
        best_f1 = val_f1_scores[best_index]

        # Round metrics to two decimal places
        best_accuracy = round(best_accuracy * 100, 2)
        best_precision = round(best_precision * 100, 2)
        best_recall = round(best_recall * 100, 2)
        best_f1 = round(best_f1 * 100, 2)

        with st.expander("Show All Accuracies"):
            st.image("result/accuracy_graph.png")
            st.write("Accuracies:")
            rounded_accuracies = [round(acc * 100, 2) for acc in val_accuracies]
            st.write(rounded_accuracies)
            # st.write(val_accuracies)
        st.write("Average Accuracy:", round(average_accuracy * 100, 2))

        # Display metrics c4.5-pso
        # st.write(f"Accuracy: {best_accuracy:.4f}")
        st.write("Accuracy:", best_accuracy)
        st.write("Precision:", best_precision)
        st.write("Recall:", best_recall)
        st.write("F1 Score:", best_f1)
        st.write("Selected Features:", selected_features)

        # Ambil nama kolom dari dataset asli sebelum scaling
        # feature_names = dataset.drop(columns=[target_column]).columns
        # st.write("Selected features by PSO:")
        # for feature_index in selected_features_pso:
        #     st.write(feature_names[feature_index])

        # Plot decision tree
        st.subheader("C4.5-PSO Decision Tree")
        plt.figure(figsize=(40, 18))
        plot_tree(
            c45_pso_model,
            filled=True,
            feature_names=feature_names,
            class_names=["No Diabetes", "Diabetes"],
        )
        plt.title("C4.5-PSO Decision Tree")
        plt.savefig("result/c45_pso_dt.svg", dpi=200)
        st.image("result/c45_pso_dt.svg")
        st.caption("Figure 2: C4.5-PSO Decision Tree")

    st.success("Classification completed successfully.")
