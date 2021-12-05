import sys
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score

# ======================================
# AdaBoost Tuning Results on Full data
# ======================================
# n_estimators: 100 	 learning_rate: 0.001 	 Accuracy: 81.51% 	 F1 Score(Macro): 0.68
# n_estimators: 100 	 learning_rate: 0.01 	 Accuracy: 79.82% 	 F1 Score(Macro): 0.59
# n_estimators: 100 	 learning_rate: 0.1 	 Accuracy: 81.57% 	 F1 Score(Macro): 0.7
# n_estimators: 100 	 learning_rate: 0.2 	 Accuracy: 82.11% 	 F1 Score(Macro): 0.71
# n_estimators: 100 	 learning_rate: 0.5 	 Accuracy: 83.3% 	 F1 Score(Macro): 0.74
# n_estimators: 100 	 learning_rate: 1.0 	 Accuracy: 83.55% 	 F1 Score(Macro): 0.75
# n_estimators: 100 	 learning_rate: 2.0 	 Accuracy: 77.04% 	 F1 Score(Macro): 0.44
# n_estimators: 200 	 learning_rate: 0.001 	 Accuracy: 81.51% 	 F1 Score(Macro): 0.68
# n_estimators: 200 	 learning_rate: 0.01 	 Accuracy: 79.78% 	 F1 Score(Macro): 0.58
# n_estimators: 200 	 learning_rate: 0.1 	 Accuracy: 82.09% 	 F1 Score(Macro): 0.71
# n_estimators: 200 	 learning_rate: 0.2 	 Accuracy: 82.99% 	 F1 Score(Macro): 0.73
# n_estimators: 200 	 learning_rate: 0.5 	 Accuracy: 83.45% 	 F1 Score(Macro): 0.75
# n_estimators: 200 	 learning_rate: 1.0 	 Accuracy: 83.86% 	 F1 Score(Macro): 0.75
# n_estimators: 200 	 learning_rate: 2.0 	 Accuracy: 77.04% 	 F1 Score(Macro): 0.44
# n_estimators: 400 	 learning_rate: 0.001 	 Accuracy: 79.82% 	 F1 Score(Macro): 0.59
# n_estimators: 400 	 learning_rate: 0.01 	 Accuracy: 80.23% 	 F1 Score(Macro): 0.6
# n_estimators: 400 	 learning_rate: 0.1 	 Accuracy: 82.97% 	 F1 Score(Macro): 0.73
# n_estimators: 400 	 learning_rate: 0.2 	 Accuracy: 83.54% 	 F1 Score(Macro): 0.75
# n_estimators: 400 	 learning_rate: 0.5 	 Accuracy: 83.76% 	 F1 Score(Macro): 0.75
# n_estimators: 400 	 learning_rate: 1.0 	 Accuracy: 84.03% 	 F1 Score(Macro): 0.76
# n_estimators: 400 	 learning_rate: 2.0 	 Accuracy: 77.04% 	 F1 Score(Macro): 0.44
# n_estimators: 800 	 learning_rate: 0.001 	 Accuracy: 79.82% 	 F1 Score(Macro): 0.59
# n_estimators: 800 	 learning_rate: 0.01 	 Accuracy: 81.44% 	 F1 Score(Macro): 0.69
# n_estimators: 800 	 learning_rate: 0.1 	 Accuracy: 83.49% 	 F1 Score(Macro): 0.74
# n_estimators: 800 	 learning_rate: 0.2 	 Accuracy: 83.54% 	 F1 Score(Macro): 0.75
# n_estimators: 800 	 learning_rate: 0.5 	 Accuracy: 83.86% 	 F1 Score(Macro): 0.75
# --> n_estimators: 800 	 learning_rate: 1.0 	 Accuracy: 84.32% 	 F1 Score(Macro): 0.76 <--
# n_estimators: 800 	 learning_rate: 2.0 	 Accuracy: 77.04% 	 F1 Score(Macro): 0.44


# =========================================
#  AdaBoost Tuning Results on PCA Data
# =========================================
# n_estimators: 100 	 learning_rate: 0.001 	 Accuracy: 77.04% 	 F1 Score(Macro): 0.44
# n_estimators: 100 	 learning_rate: 0.01 	 Accuracy: 77.04% 	 F1 Score(Macro): 0.44
# n_estimators: 100 	 learning_rate: 0.1 	 Accuracy: 79.86% 	 F1 Score(Macro): 0.6
# n_estimators: 100 	 learning_rate: 0.2 	 Accuracy: 81.35% 	 F1 Score(Macro): 0.66
# n_estimators: 100 	 learning_rate: 0.5 	 Accuracy: 81.94% 	 F1 Score(Macro): 0.7
# n_estimators: 100 	 learning_rate: 1.0 	 Accuracy: 81.7% 	 F1 Score(Macro): 0.7
# n_estimators: 100 	 learning_rate: 2.0 	 Accuracy: 38.4% 	 F1 Score(Macro): 0.33
# n_estimators: 200 	 learning_rate: 0.001 	 Accuracy: 77.04% 	 F1 Score(Macro): 0.44
# n_estimators: 200 	 learning_rate: 0.01 	 Accuracy: 77.04% 	 F1 Score(Macro): 0.44
# n_estimators: 200 	 learning_rate: 0.1 	 Accuracy: 81.39% 	 F1 Score(Macro): 0.66
# n_estimators: 200 	 learning_rate: 0.2 	 Accuracy: 82.3% 	 F1 Score(Macro): 0.7
# n_estimators: 200 	 learning_rate: 0.5 	 Accuracy: 82.18% 	 F1 Score(Macro): 0.71
# n_estimators: 200 	 learning_rate: 1.0 	 Accuracy: 82.19% 	 F1 Score(Macro): 0.71
# n_estimators: 200 	 learning_rate: 2.0 	 Accuracy: 38.24% 	 F1 Score(Macro): 0.33
# n_estimators: 400 	 learning_rate: 0.001 	 Accuracy: 77.04% 	 F1 Score(Macro): 0.44
# n_estimators: 400 	 learning_rate: 0.01 	 Accuracy: 77.33% 	 F1 Score(Macro): 0.45
# n_estimators: 400 	 learning_rate: 0.1 	 Accuracy: 82.21% 	 F1 Score(Macro): 0.69
# n_estimators: 400 	 learning_rate: 0.2 	 Accuracy: 82.47% 	 F1 Score(Macro): 0.7
# n_estimators: 400 	 learning_rate: 0.5 	 Accuracy: 82.51% 	 F1 Score(Macro): 0.71
# n_estimators: 400 	 learning_rate: 1.0 	 Accuracy: 82.76% 	 F1 Score(Macro): 0.72
# n_estimators: 400 	 learning_rate: 2.0 	 Accuracy: 38.08% 	 F1 Score(Macro): 0.33
# n_estimators: 800 	 learning_rate: 0.001 	 Accuracy: 77.04% 	 F1 Score(Macro): 0.44
# n_estimators: 800 	 learning_rate: 0.01 	 Accuracy: 79.04% 	 F1 Score(Macro): 0.56
# n_estimators: 800 	 learning_rate: 0.1 	 Accuracy: 82.5% 	 F1 Score(Macro): 0.7
# n_estimators: 800 	 learning_rate: 0.2 	 Accuracy: 82.65% 	 F1 Score(Macro): 0.71
# --> n_estimators: 800 	 learning_rate: 0.5 	 Accuracy: 82.85% 	 F1 Score(Macro): 0.72 <--
# n_estimators: 800 	 learning_rate: 1.0 	 Accuracy: 82.46% 	 F1 Score(Macro): 0.72
# n_estimators: 800 	 learning_rate: 2.0 	 Accuracy: 37.38% 	 F1 Score(Macro): 0.33


def perform_adaboost_tuning(X, param):
    # Load the dataset
    X_train, X_val, X_test, X_train_pca, X_val_pca, \
                X_test_pca, y_train, y_val, _ = X

    parameters = {
                    'n_estimators': [100, 200, 400, 800], 
                    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 2.0]
                }

    if param == 'full':
        print("=========================================")
        print(f"\033[1m AdaBoost Tuning Results on Full Data\033[0m")
        print("=========================================")

        for n in parameters['n_estimators']:
            for lr in parameters['learning_rate']:
                    clf_ab = AdaBoostClassifier(n_estimators=n, random_state=0, learning_rate=lr)
                    clf_ab.fit(X_train, y_train)
                    y_pred = clf_ab.predict(X_val)
                    score = accuracy_score(y_val, y_pred)
                    f1_s = f1_score(y_val, y_pred, average='macro')
                    print(f"n_estimators: {n} \t learning_rate: {lr} \t Accuracy: {round(100*score, 2)}% \t F1 Score(Macro): {round(f1_s, 2)}") 

    elif param == 'pca':
        print("=========================================")
        print(f"\033[1m AdaBoost Tuning Results on PCA Data\033[0m")
        print("=========================================")

        for n in parameters['n_estimators']:
            for lr in parameters['learning_rate']:
                    clf_ab = AdaBoostClassifier(n_estimators=n, random_state=0, learning_rate=lr)
                    clf_ab.fit(X_train_pca, y_train)
                    y_pred = clf_ab.predict(X_val_pca)
                    score = accuracy_score(y_val, y_pred)
                    f1_s = f1_score(y_val, y_pred, average='macro')
                    print(f"n_estimators: {n} \t learning_rate: {lr} \t Accuracy: {round(100*score, 2)}% \t F1 Score(Macro): {round(f1_s, 2)}") 
    else:
        print("Incorrect argument was passed.")
