from sklearn.model_selection import KFold
from pyswarm import pso
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np


def evaluate_feature_subset(feature_subset, scaled_features, labels, kf):
    accuracies = []
    feature_subset = [int(x) for x in feature_subset]
    selected_features = [
        i for i, val in enumerate(feature_subset) if val
    ]  # Membuat daftar indeks fitur yang dipilih (nilai 1 dalam feature_subset).
    if len(selected_features) == 0:
        return 0
    for train_index, test_index in kf.split(scaled_features):
        X_train, X_test = (
            scaled_features[train_index][:, selected_features],
            scaled_features[test_index][:, selected_features],
        )
        y_train, y_test = labels[train_index], labels[test_index]

        model = DecisionTreeClassifier(random_state=42, max_features=None)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))

    # return sum(accuracies) / len(accuracies)
    return -np.mean(accuracies)


def optimize_features(
    scaled_features, labels, swarmsize, maxiter, c1, c2, w, use_pso=False
):
    if use_pso:
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        n_features = scaled_features.shape[1]
        lb = [0] * n_features
        ub = [1] * n_features

        def objective_function(feature_subset):
            return -evaluate_feature_subset(
                [i for i, val in enumerate(feature_subset) if val],
                scaled_features,
                labels,
                kf,
            )

        best_subset, _ = pso(
            objective_function,
            lb,
            ub,
            swarmsize=swarmsize,
            maxiter=maxiter,
            debug=True,
            omega=w,
            phip=c1,
            phig=c2,
        )
        return [i for i, val in enumerate(best_subset) if val]
    else:
        # Jika PSO tidak digunakan, kembalikan seluruh fitur
        return list(range(scaled_features.shape[1]))
