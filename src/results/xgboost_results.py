import time
import xgboost as xgb
from src.results.reporting import classification_report_2

def get_xgboost_results(X):
    X_train, _, X_test, X_train_pca, _, X_test_pca, y_train, _, y_test = X

    start_time = time.time()
    clf_xg = xgb.XGBClassifier(use_label_encoder=False, verbosity = 0, \
                    random_state=42, n_estimator=100, scale_pos_weight=1, \
                    subsample=0.8, colsample_bytree=0.8, max_depth=20, \
                    learning_rate=0.3, reg_alpha=0.1, reg_lambda=10)
    clf_xg.fit(X_train, y_train)
    train_time = time.time() - start_time

    # print("======================================")
    # print(f"\033[1m XGBoost Train Results Full Data\033[0m")
    # print("======================================")
    # print(f"XGBoost training time {round(train_time, 2)} secs")
    # y_pred = clf_xg.predict(X_train)
    # classification_report_2(y_train, y_pred)

    start_time = time.time()
    y_pred = clf_xg.predict(X_test)
    test_time = time.time() - start_time
    print("======================================")
    print(f"\033[1m XGBoost Test Results Full Data\033[0m")
    print("======================================")
    print(f"XGBoost prediction time {round(test_time, 2)} secs")
    classification_report_2(y_test, y_pred)

    start_time = time.time()
    clf_xg = xgb.XGBClassifier(use_label_encoder=False, verbosity = 0, \
                    random_state=42, n_estimator=100, scale_pos_weight=3.35, \
                    subsample=0.8, colsample_bytree=0.8, max_depth=8, \
                    learning_rate=0.3)
    clf_xg.fit(X_train_pca, y_train)
    train_time = time.time() - start_time

    # print("======================================")
    # print(f"\033[1m XGBoost Train Results PCA Data\033[0m")
    # print("======================================")
    # print(f"XGBoost training time {round(train_time, 2)} secs")
    # y_pred = clf_xg.predict(X_train_pca)
    # classification_report_2(y_train, y_pred)

    start_time = time.time()
    y_pred = clf_xg.predict(X_test_pca)
    test_time = time.time() - start_time
    print("======================================")
    print(f"\033[1m XGBoost Test Results PCA Data\033[0m")
    print("======================================")
    print(f"XGBoost prediction time {round(test_time, 2)} secs")
    classification_report_2(y_test, y_pred)