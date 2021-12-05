import sys
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score

# Reference: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

# ============================================
#  XGBoost Tuning Results on Full Data
# ============================================
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 3 	 learning_rate: 1 	 Accuracy: 86.11% 	 F1 Score(Macro): 0.8
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 3 	 learning_rate: 0.3 	 Accuracy: 85.67% 	 F1 Score(Macro): 0.78
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 3 	 learning_rate: 0.01 	 Accuracy: 83.15% 	 F1 Score(Macro): 0.72
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 3 	 learning_rate: 0.05 	 Accuracy: 83.22% 	 F1 Score(Macro): 0.73
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 4 	 learning_rate: 1 	 Accuracy: 85.64% 	 F1 Score(Macro): 0.79
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 4 	 learning_rate: 0.3 	 Accuracy: 86.43% 	 F1 Score(Macro): 0.8
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 4 	 learning_rate: 0.01 	 Accuracy: 84.12% 	 F1 Score(Macro): 0.74
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 4 	 learning_rate: 0.05 	 Accuracy: 85.16% 	 F1 Score(Macro): 0.77
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 6 	 learning_rate: 1 	 Accuracy: 85.04% 	 F1 Score(Macro): 0.78
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 6 	 learning_rate: 0.3 	 Accuracy: 86.92% 	 F1 Score(Macro): 0.81
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 6 	 learning_rate: 0.01 	 Accuracy: 85.5% 	 F1 Score(Macro): 0.77
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 6 	 learning_rate: 0.05 	 Accuracy: 86.19% 	 F1 Score(Macro): 0.79
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 8 	 learning_rate: 1 	 Accuracy: 85.08% 	 F1 Score(Macro): 0.79
# --> n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 8 	 learning_rate: 0.3 	 Accuracy: 87.24% 	 F1 Score(Macro): 0.81 <--
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 8 	 learning_rate: 0.01 	 Accuracy: 86.03% 	 F1 Score(Macro): 0.78
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 8 	 learning_rate: 0.05 	 Accuracy: 86.93% 	 F1 Score(Macro): 0.8
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 3 	 learning_rate: 1 	 Accuracy: 82.33% 	 F1 Score(Macro): 0.78
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 3 	 learning_rate: 0.3 	 Accuracy: 80.58% 	 F1 Score(Macro): 0.77
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 3 	 learning_rate: 0.01 	 Accuracy: 78.84% 	 F1 Score(Macro): 0.75
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 3 	 learning_rate: 0.05 	 Accuracy: 78.79% 	 F1 Score(Macro): 0.75
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 4 	 learning_rate: 1 	 Accuracy: 82.58% 	 F1 Score(Macro): 0.78
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 4 	 learning_rate: 0.3 	 Accuracy: 81.71% 	 F1 Score(Macro): 0.78
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 4 	 learning_rate: 0.01 	 Accuracy: 79.14% 	 F1 Score(Macro): 0.75
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 4 	 learning_rate: 0.05 	 Accuracy: 79.03% 	 F1 Score(Macro): 0.75
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 6 	 learning_rate: 1 	 Accuracy: 83.72% 	 F1 Score(Macro): 0.79
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 6 	 learning_rate: 0.3 	 Accuracy: 83.45% 	 F1 Score(Macro): 0.79
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 6 	 learning_rate: 0.01 	 Accuracy: 78.98% 	 F1 Score(Macro): 0.76
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 6 	 learning_rate: 0.05 	 Accuracy: 81.19% 	 F1 Score(Macro): 0.78
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 8 	 learning_rate: 1 	 Accuracy: 85.28% 	 F1 Score(Macro): 0.8
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 8 	 learning_rate: 0.3 	 Accuracy: 85.45% 	 F1 Score(Macro): 0.81
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 8 	 learning_rate: 0.01 	 Accuracy: 80.87% 	 F1 Score(Macro): 0.77
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 8 	 learning_rate: 0.05 	 Accuracy: 82.86% 	 F1 Score(Macro): 0.79
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 3 	 learning_rate: 1 	 Accuracy: 79.98% 	 F1 Score(Macro): 0.76
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 3 	 learning_rate: 0.3 	 Accuracy: 78.19% 	 F1 Score(Macro): 0.75
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 3 	 learning_rate: 0.01 	 Accuracy: 74.2% 	 F1 Score(Macro): 0.71
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 3 	 learning_rate: 0.05 	 Accuracy: 75.5% 	 F1 Score(Macro): 0.73
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 4 	 learning_rate: 1 	 Accuracy: 81.07% 	 F1 Score(Macro): 0.77
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 4 	 learning_rate: 0.3 	 Accuracy: 78.85% 	 F1 Score(Macro): 0.76
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 4 	 learning_rate: 0.01 	 Accuracy: 76.48% 	 F1 Score(Macro): 0.73
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 4 	 learning_rate: 0.05 	 Accuracy: 76.39% 	 F1 Score(Macro): 0.73
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 6 	 learning_rate: 1 	 Accuracy: 82.64% 	 F1 Score(Macro): 0.78
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 6 	 learning_rate: 0.3 	 Accuracy: 81.64% 	 F1 Score(Macro): 0.78
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 6 	 learning_rate: 0.01 	 Accuracy: 77.35% 	 F1 Score(Macro): 0.74
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 6 	 learning_rate: 0.05 	 Accuracy: 78.06% 	 F1 Score(Macro): 0.75
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 8 	 learning_rate: 1 	 Accuracy: 84.72% 	 F1 Score(Macro): 0.79
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 8 	 learning_rate: 0.3 	 Accuracy: 83.72% 	 F1 Score(Macro): 0.8
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 8 	 learning_rate: 0.01 	 Accuracy: 78.88% 	 F1 Score(Macro): 0.76
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 8 	 learning_rate: 0.05 	 Accuracy: 80.34% 	 F1 Score(Macro): 0.77
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 3 	 learning_rate: 1 	 Accuracy: 86.11% 	 F1 Score(Macro): 0.8
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 3 	 learning_rate: 0.3 	 Accuracy: 85.67% 	 F1 Score(Macro): 0.78
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 3 	 learning_rate: 0.01 	 Accuracy: 83.15% 	 F1 Score(Macro): 0.72
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 3 	 learning_rate: 0.05 	 Accuracy: 83.22% 	 F1 Score(Macro): 0.73
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 4 	 learning_rate: 1 	 Accuracy: 85.64% 	 F1 Score(Macro): 0.79
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 4 	 learning_rate: 0.3 	 Accuracy: 86.43% 	 F1 Score(Macro): 0.8
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 4 	 learning_rate: 0.01 	 Accuracy: 84.12% 	 F1 Score(Macro): 0.74
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 4 	 learning_rate: 0.05 	 Accuracy: 85.16% 	 F1 Score(Macro): 0.77
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 6 	 learning_rate: 1 	 Accuracy: 85.04% 	 F1 Score(Macro): 0.78
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 6 	 learning_rate: 0.3 	 Accuracy: 86.92% 	 F1 Score(Macro): 0.81
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 6 	 learning_rate: 0.01 	 Accuracy: 85.5% 	 F1 Score(Macro): 0.77
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 6 	 learning_rate: 0.05 	 Accuracy: 86.19% 	 F1 Score(Macro): 0.79
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 8 	 learning_rate: 1 	 Accuracy: 85.08% 	 F1 Score(Macro): 0.79
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 8 	 learning_rate: 0.3 	 Accuracy: 87.24% 	 F1 Score(Macro): 0.81
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 8 	 learning_rate: 0.01 	 Accuracy: 86.03% 	 F1 Score(Macro): 0.78
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 8 	 learning_rate: 0.05 	 Accuracy: 86.93% 	 F1 Score(Macro): 0.8
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 3 	 learning_rate: 1 	 Accuracy: 82.33% 	 F1 Score(Macro): 0.78
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 3 	 learning_rate: 0.3 	 Accuracy: 80.58% 	 F1 Score(Macro): 0.77
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 3 	 learning_rate: 0.01 	 Accuracy: 78.84% 	 F1 Score(Macro): 0.75
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 3 	 learning_rate: 0.05 	 Accuracy: 78.79% 	 F1 Score(Macro): 0.75
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 4 	 learning_rate: 1 	 Accuracy: 82.58% 	 F1 Score(Macro): 0.78
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 4 	 learning_rate: 0.3 	 Accuracy: 81.71% 	 F1 Score(Macro): 0.78
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 4 	 learning_rate: 0.01 	 Accuracy: 79.14% 	 F1 Score(Macro): 0.75
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 4 	 learning_rate: 0.05 	 Accuracy: 79.03% 	 F1 Score(Macro): 0.75
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 6 	 learning_rate: 1 	 Accuracy: 83.72% 	 F1 Score(Macro): 0.79
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 6 	 learning_rate: 0.3 	 Accuracy: 83.45% 	 F1 Score(Macro): 0.79
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 6 	 learning_rate: 0.01 	 Accuracy: 78.98% 	 F1 Score(Macro): 0.76
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 6 	 learning_rate: 0.05 	 Accuracy: 81.19% 	 F1 Score(Macro): 0.78
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 8 	 learning_rate: 1 	 Accuracy: 85.28% 	 F1 Score(Macro): 0.8
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 8 	 learning_rate: 0.3 	 Accuracy: 85.45% 	 F1 Score(Macro): 0.81
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 8 	 learning_rate: 0.01 	 Accuracy: 80.87% 	 F1 Score(Macro): 0.77
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 8 	 learning_rate: 0.05 	 Accuracy: 82.86% 	 F1 Score(Macro): 0.79
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 3 	 learning_rate: 1 	 Accuracy: 79.98% 	 F1 Score(Macro): 0.76
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 3 	 learning_rate: 0.3 	 Accuracy: 78.19% 	 F1 Score(Macro): 0.75
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 3 	 learning_rate: 0.01 	 Accuracy: 74.2% 	 F1 Score(Macro): 0.71
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 3 	 learning_rate: 0.05 	 Accuracy: 75.5% 	 F1 Score(Macro): 0.73
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 4 	 learning_rate: 1 	 Accuracy: 81.07% 	 F1 Score(Macro): 0.77
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 4 	 learning_rate: 0.3 	 Accuracy: 78.85% 	 F1 Score(Macro): 0.76
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 4 	 learning_rate: 0.01 	 Accuracy: 76.48% 	 F1 Score(Macro): 0.73
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 4 	 learning_rate: 0.05 	 Accuracy: 76.39% 	 F1 Score(Macro): 0.73
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 6 	 learning_rate: 1 	 Accuracy: 82.64% 	 F1 Score(Macro): 0.78
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 6 	 learning_rate: 0.3 	 Accuracy: 81.64% 	 F1 Score(Macro): 0.78
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 6 	 learning_rate: 0.01 	 Accuracy: 77.35% 	 F1 Score(Macro): 0.74
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 6 	 learning_rate: 0.05 	 Accuracy: 78.06% 	 F1 Score(Macro): 0.75
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 8 	 learning_rate: 1 	 Accuracy: 84.72% 	 F1 Score(Macro): 0.79
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 8 	 learning_rate: 0.3 	 Accuracy: 83.72% 	 F1 Score(Macro): 0.8
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 8 	 learning_rate: 0.01 	 Accuracy: 78.88% 	 F1 Score(Macro): 0.76
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 8 	 learning_rate: 0.05 	 Accuracy: 80.34% 	 F1 Score(Macro): 0.77


# ============================================
#  XGBoost Tuning Results on PCA Data
# ============================================
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 3 	 learning_rate: 1 	 Accuracy: 83.12% 	 F1 Score(Macro): 0.74
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 3 	 learning_rate: 0.3 	 Accuracy: 83.79% 	 F1 Score(Macro): 0.73
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 3 	 learning_rate: 0.01 	 Accuracy: 80.25% 	 F1 Score(Macro): 0.59
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 3 	 learning_rate: 0.05 	 Accuracy: 82.3% 	 F1 Score(Macro): 0.67
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 4 	 learning_rate: 1 	 Accuracy: 83.0% 	 F1 Score(Macro): 0.74
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 4 	 learning_rate: 0.3 	 Accuracy: 84.24% 	 F1 Score(Macro): 0.75
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 4 	 learning_rate: 0.01 	 Accuracy: 81.29% 	 F1 Score(Macro): 0.63
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 4 	 learning_rate: 0.05 	 Accuracy: 82.97% 	 F1 Score(Macro): 0.7
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 6 	 learning_rate: 1 	 Accuracy: 83.58% 	 F1 Score(Macro): 0.76
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 6 	 learning_rate: 0.3 	 Accuracy: 85.07% 	 F1 Score(Macro): 0.77
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 6 	 learning_rate: 0.01 	 Accuracy: 82.89% 	 F1 Score(Macro): 0.68
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 6 	 learning_rate: 0.05 	 Accuracy: 83.94% 	 F1 Score(Macro): 0.72
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 8 	 learning_rate: 1 	 Accuracy: 83.62% 	 F1 Score(Macro): 0.76
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 8 	 learning_rate: 0.3 	 Accuracy: 84.98% 	 F1 Score(Macro): 0.77
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 8 	 learning_rate: 0.01 	 Accuracy: 83.53% 	 F1 Score(Macro): 0.7
# n_estimator: 100 	 scale_pos_weight: 1 	 max_depth: 8 	 learning_rate: 0.05 	 Accuracy: 84.57% 	 F1 Score(Macro): 0.74
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 3 	 learning_rate: 1 	 Accuracy: 78.6% 	 F1 Score(Macro): 0.74
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 3 	 learning_rate: 0.3 	 Accuracy: 78.26% 	 F1 Score(Macro): 0.74
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 3 	 learning_rate: 0.01 	 Accuracy: 70.44% 	 F1 Score(Macro): 0.67
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 3 	 learning_rate: 0.05 	 Accuracy: 74.92% 	 F1 Score(Macro): 0.71
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 4 	 learning_rate: 1 	 Accuracy: 78.78% 	 F1 Score(Macro): 0.73
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 4 	 learning_rate: 0.3 	 Accuracy: 79.83% 	 F1 Score(Macro): 0.75
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 4 	 learning_rate: 0.01 	 Accuracy: 72.02% 	 F1 Score(Macro): 0.68
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 4 	 learning_rate: 0.05 	 Accuracy: 75.72% 	 F1 Score(Macro): 0.72
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 6 	 learning_rate: 1 	 Accuracy: 81.42% 	 F1 Score(Macro): 0.75
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 6 	 learning_rate: 0.3 	 Accuracy: 82.19% 	 F1 Score(Macro): 0.77
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 6 	 learning_rate: 0.01 	 Accuracy: 76.02% 	 F1 Score(Macro): 0.72
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 6 	 learning_rate: 0.05 	 Accuracy: 78.62% 	 F1 Score(Macro): 0.74
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 8 	 learning_rate: 1 	 Accuracy: 82.86% 	 F1 Score(Macro): 0.76
# --> n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 8 	 learning_rate: 0.3 	 Accuracy: 84.02% 	 F1 Score(Macro): 0.79 <--
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 8 	 learning_rate: 0.01 	 Accuracy: 78.01% 	 F1 Score(Macro): 0.74
# n_estimator: 100 	 scale_pos_weight: 3.35 	 max_depth: 8 	 learning_rate: 0.05 	 Accuracy: 80.7% 	 F1 Score(Macro): 0.76
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 3 	 learning_rate: 1 	 Accuracy: 75.87% 	 F1 Score(Macro): 0.72
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 3 	 learning_rate: 0.3 	 Accuracy: 73.78% 	 F1 Score(Macro): 0.7
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 3 	 learning_rate: 0.01 	 Accuracy: 63.11% 	 F1 Score(Macro): 0.61
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 3 	 learning_rate: 0.05 	 Accuracy: 68.85% 	 F1 Score(Macro): 0.66
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 4 	 learning_rate: 1 	 Accuracy: 76.65% 	 F1 Score(Macro): 0.72
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 4 	 learning_rate: 0.3 	 Accuracy: 75.86% 	 F1 Score(Macro): 0.72
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 4 	 learning_rate: 0.01 	 Accuracy: 66.41% 	 F1 Score(Macro): 0.64
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 4 	 learning_rate: 0.05 	 Accuracy: 70.5% 	 F1 Score(Macro): 0.68
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 6 	 learning_rate: 1 	 Accuracy: 80.18% 	 F1 Score(Macro): 0.74
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 6 	 learning_rate: 0.3 	 Accuracy: 79.66% 	 F1 Score(Macro): 0.75
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 6 	 learning_rate: 0.01 	 Accuracy: 70.73% 	 F1 Score(Macro): 0.68
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 6 	 learning_rate: 0.05 	 Accuracy: 73.98% 	 F1 Score(Macro): 0.71
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 8 	 learning_rate: 1 	 Accuracy: 82.71% 	 F1 Score(Macro): 0.76
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 8 	 learning_rate: 0.3 	 Accuracy: 82.55% 	 F1 Score(Macro): 0.77
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 8 	 learning_rate: 0.01 	 Accuracy: 73.95% 	 F1 Score(Macro): 0.71
# n_estimator: 100 	 scale_pos_weight: 5 	 max_depth: 8 	 learning_rate: 0.05 	 Accuracy: 77.74% 	 F1 Score(Macro): 0.74
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 3 	 learning_rate: 1 	 Accuracy: 83.12% 	 F1 Score(Macro): 0.74
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 3 	 learning_rate: 0.3 	 Accuracy: 83.79% 	 F1 Score(Macro): 0.73
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 3 	 learning_rate: 0.01 	 Accuracy: 80.25% 	 F1 Score(Macro): 0.59
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 3 	 learning_rate: 0.05 	 Accuracy: 82.3% 	 F1 Score(Macro): 0.67
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 4 	 learning_rate: 1 	 Accuracy: 83.0% 	 F1 Score(Macro): 0.74
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 4 	 learning_rate: 0.3 	 Accuracy: 84.24% 	 F1 Score(Macro): 0.75
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 4 	 learning_rate: 0.01 	 Accuracy: 81.29% 	 F1 Score(Macro): 0.63
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 4 	 learning_rate: 0.05 	 Accuracy: 82.97% 	 F1 Score(Macro): 0.7
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 6 	 learning_rate: 1 	 Accuracy: 83.58% 	 F1 Score(Macro): 0.76
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 6 	 learning_rate: 0.3 	 Accuracy: 85.07% 	 F1 Score(Macro): 0.77
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 6 	 learning_rate: 0.01 	 Accuracy: 82.89% 	 F1 Score(Macro): 0.68
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 6 	 learning_rate: 0.05 	 Accuracy: 83.94% 	 F1 Score(Macro): 0.72
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 8 	 learning_rate: 1 	 Accuracy: 83.62% 	 F1 Score(Macro): 0.76
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 8 	 learning_rate: 0.3 	 Accuracy: 84.98% 	 F1 Score(Macro): 0.77
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 8 	 learning_rate: 0.01 	 Accuracy: 83.53% 	 F1 Score(Macro): 0.7
# n_estimator: 250 	 scale_pos_weight: 1 	 max_depth: 8 	 learning_rate: 0.05 	 Accuracy: 84.57% 	 F1 Score(Macro): 0.74
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 3 	 learning_rate: 1 	 Accuracy: 78.6% 	 F1 Score(Macro): 0.74
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 3 	 learning_rate: 0.3 	 Accuracy: 78.26% 	 F1 Score(Macro): 0.74
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 3 	 learning_rate: 0.01 	 Accuracy: 70.44% 	 F1 Score(Macro): 0.67
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 3 	 learning_rate: 0.05 	 Accuracy: 74.92% 	 F1 Score(Macro): 0.71
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 4 	 learning_rate: 1 	 Accuracy: 78.78% 	 F1 Score(Macro): 0.73
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 4 	 learning_rate: 0.3 	 Accuracy: 79.83% 	 F1 Score(Macro): 0.75
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 4 	 learning_rate: 0.01 	 Accuracy: 72.02% 	 F1 Score(Macro): 0.68
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 4 	 learning_rate: 0.05 	 Accuracy: 75.72% 	 F1 Score(Macro): 0.72
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 6 	 learning_rate: 1 	 Accuracy: 81.42% 	 F1 Score(Macro): 0.75
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 6 	 learning_rate: 0.3 	 Accuracy: 82.19% 	 F1 Score(Macro): 0.77
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 6 	 learning_rate: 0.01 	 Accuracy: 76.02% 	 F1 Score(Macro): 0.72
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 6 	 learning_rate: 0.05 	 Accuracy: 78.62% 	 F1 Score(Macro): 0.74
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 8 	 learning_rate: 1 	 Accuracy: 82.86% 	 F1 Score(Macro): 0.76
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 8 	 learning_rate: 0.3 	 Accuracy: 84.02% 	 F1 Score(Macro): 0.79
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 8 	 learning_rate: 0.01 	 Accuracy: 78.01% 	 F1 Score(Macro): 0.74
# n_estimator: 250 	 scale_pos_weight: 3.35 	 max_depth: 8 	 learning_rate: 0.05 	 Accuracy: 80.7% 	 F1 Score(Macro): 0.76
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 3 	 learning_rate: 1 	 Accuracy: 75.87% 	 F1 Score(Macro): 0.72
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 3 	 learning_rate: 0.3 	 Accuracy: 73.78% 	 F1 Score(Macro): 0.7
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 3 	 learning_rate: 0.01 	 Accuracy: 63.11% 	 F1 Score(Macro): 0.61
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 3 	 learning_rate: 0.05 	 Accuracy: 68.85% 	 F1 Score(Macro): 0.66
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 4 	 learning_rate: 1 	 Accuracy: 76.65% 	 F1 Score(Macro): 0.72
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 4 	 learning_rate: 0.3 	 Accuracy: 75.86% 	 F1 Score(Macro): 0.72
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 4 	 learning_rate: 0.01 	 Accuracy: 66.41% 	 F1 Score(Macro): 0.64
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 4 	 learning_rate: 0.05 	 Accuracy: 70.5% 	 F1 Score(Macro): 0.68
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 6 	 learning_rate: 1 	 Accuracy: 80.18% 	 F1 Score(Macro): 0.74
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 6 	 learning_rate: 0.3 	 Accuracy: 79.66% 	 F1 Score(Macro): 0.75
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 6 	 learning_rate: 0.01 	 Accuracy: 70.73% 	 F1 Score(Macro): 0.68
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 6 	 learning_rate: 0.05 	 Accuracy: 73.98% 	 F1 Score(Macro): 0.71
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 8 	 learning_rate: 1 	 Accuracy: 82.71% 	 F1 Score(Macro): 0.76
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 8 	 learning_rate: 0.3 	 Accuracy: 82.55% 	 F1 Score(Macro): 0.77
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 8 	 learning_rate: 0.01 	 Accuracy: 73.95% 	 F1 Score(Macro): 0.71
# n_estimator: 250 	 scale_pos_weight: 5 	 max_depth: 8 	 learning_rate: 0.05 	 Accuracy: 77.74% 	 F1 Score(Macro): 0.74


def perform_xgboost_tuning(X, param):
    # Load the dataset
    X_train, X_val, _, X_train_pca, X_val_pca, \
                            _, y_train, y_val, _ = X
    
    # Find the optimal ratio for scale_pos_weight
    opt_spw = round(y_train.value_counts()[0] / y_train.value_counts()[1], 2)
    parameters = {
                    "n_estimator": [100, 250],
                    "scale_pos_weight": [1, opt_spw, 5],
                    "max_depth": [3, 4, 6, 8],
                    "learning_rate": [1, 0.3, 0.01, 0.05]
                }

    if param == 'full':
        print("============================================")
        print(f"\033[1m XGBoost Tuning Results on Full Data\033[0m")
        print("============================================")

        for n in parameters['n_estimator']:
            for spw in parameters['scale_pos_weight']:
                for md in parameters['max_depth']:
                    for lr in parameters['learning_rate']:
                        clf_xg = xgb.XGBClassifier(use_label_encoder=False, verbosity = 0, \
                                        random_state=42, n_estimator=n, scale_pos_weight=spw, \
                                        subsample=0.8, colsample_bytree=0.8, max_depth=md, learning_rate=lr)
                        clf_xg.fit(X_train, y_train)
                        y_pred = clf_xg.predict(X_val)
                        score = accuracy_score(y_val, y_pred)
                        f1_s = f1_score(y_val, y_pred, average='macro')
                        print(f"n_estimator: {n} \t scale_pos_weight: {spw} \t max_depth: {md} \t learning_rate: {lr} \t Accuracy: {round(100*score, 2)}% \t F1 Score(Macro): {round(f1_s, 2)}") 
    
    elif param == 'pca':
        print("============================================")
        print(f"\033[1m XGBoost Tuning Results on PCA Data\033[0m")
        print("============================================")

        for n in parameters['n_estimator']:
            for spw in parameters['scale_pos_weight']:
                for md in parameters['max_depth']:
                    for lr in parameters['learning_rate']:
                        clf_xg = xgb.XGBClassifier(use_label_encoder=False, verbosity = 0, \
                                        random_state=42, n_estimator=n, scale_pos_weight=spw, \
                                        subsample=0.8, colsample_bytree=0.8, max_depth=md, learning_rate=lr)
                        clf_xg.fit(X_train_pca, y_train)
                        y_pred = clf_xg.predict(X_val_pca)
                        score = accuracy_score(y_val, y_pred)
                        f1_s = f1_score(y_val, y_pred, average='macro')
                        print(f"n_estimator: {n} \t scale_pos_weight: {spw} \t max_depth: {md} \t learning_rate: {lr} \t Accuracy: {round(100*score, 2)}% \t F1 Score(Macro): {round(f1_s, 2)}") 
    else:
        print("Incorrect argument was passed.")