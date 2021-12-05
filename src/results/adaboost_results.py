import time
from sklearn.ensemble import AdaBoostClassifier
from src.results.reporting import classification_report_2

def get_adaboost_results(X):
    X_train, _, X_test, X_train_pca, _, X_test_pca, y_train, _, y_test = X

    start_time = time.time()
    clf_ab = AdaBoostClassifier(n_estimators=800, random_state=0, learning_rate=1)

    clf_ab.fit(X_train, y_train)
    train_time = time.time() - start_time
    print("======================================")
    print(f"\033[1m AdaBoost Train Results Full Data\033[0m")
    print("======================================")
    print(f"AdaBoost training time {round(train_time, 2)} secs")
    y_pred = clf_ab.predict(X_train)
    classification_report_2(y_train, y_pred)

    start_time = time.time()
    y_pred = clf_ab.predict(X_test)
    test_time = time.time() - start_time
    print("======================================")
    print(f"\033[1m AdaBoost Test Results Full Data\033[0m")
    print("======================================")
    print(f"AdaBoost prediction time {round(test_time, 2)} secs")
    classification_report_2(y_test, y_pred)
