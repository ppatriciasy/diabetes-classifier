from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import logging


def train_and_evaluate_model_c45(X_train, X_test, y_train, y_test, model=None):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    return model, test_accuracy


logging.basicConfig(level=logging.INFO)


def train_and_evaluate_model_c45_pso(
    X_train, X_val, y_train, y_val, selected_features, model=None
):
    # Ensure selected_features is not empty
    if len(selected_features) == 0:
        return "No features selected for training. Please try again."

    # Log selected features for debugging
    logging.info(f"Selected Features: {selected_features}")

    # Create a new model if not provided
    if model is None:
        model = DecisionTreeClassifier()

    # Train the model
    model.fit(X_train[:, selected_features], y_train)

    # Evaluate the model
    val_predictions = model.predict(X_val[:, selected_features])
    val_accuracy = accuracy_score(y_val, val_predictions)
    val_precision = precision_score(y_val, val_predictions)
    val_recall = recall_score(y_val, val_predictions)
    val_f1 = f1_score(y_val, val_predictions)

    return model, val_accuracy, val_precision, val_recall, val_f1


def plot_accuracies(
    train_accuracies, test_accuracies, iterations, save_path="result/accuracy_graph.png"
):
    plt.figure(figsize=(10, 6))
    # plt.plot(iterations, train_accuracies, label="train", marker="o")
    plt.plot(iterations, test_accuracies, label="val", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Graph")
    plt.savefig(save_path)
    plt.show()
    highest_accuracy = max(test_accuracies)
    average_accuracy = sum(test_accuracies) / len(test_accuracies)

    return highest_accuracy, average_accuracy
