# Autonomous Benchmark Report

Generated: 2026-09-06T17:11:49

## Experimental Summary Table

| Dataset | Algorithm | Test Accuracy | CV Mean Accuracy | CV Std Dev |
|---|---|---|---|---|
| wine | decision_tree | 0.9444 | 0.8937 | 0.0472 |
| wine | random_forest | 1.0000 | 0.9610 | 0.0221 |
| wine | logistic_regression | 0.9722 | 0.9889 | 0.0136 |
| wine | deep_pytorch_mlp | 0.9722 | N/A | N/A |
| breast_cancer | decision_tree | 0.9386 | 0.9209 | 0.0202 |
| breast_cancer | random_forest | 0.9561 | 0.9543 | 0.0244 |
| breast_cancer | logistic_regression | 0.9825 | 0.9807 | 0.0065 |
| breast_cancer | deep_pytorch_mlp | 0.9649 | N/A | N/A |

## Recommendations

- **wine**: best model is `random_forest` (test accuracy = 1.0000)
- **breast_cancer**: best model is `logistic_regression` (test accuracy = 0.9825)

## Notes

- `decision_tree`, `random_forest`, and `logistic_regression` are evaluated with
  an 80/20 train/test split AND 5-fold cross-validation (see CV columns).
- `deep_pytorch_mlp` (Dropout + BatchNorm + cosine LR scheduling) is evaluated
  with the same 80/20 split; CV columns are N/A because this benchmark uses a
  single held-out split for the neural network for speed. Extend
  `benchmark_runner.py` with a manual K-fold loop around
  `train_deep_pytorch_classifier` if you want CV numbers for it too.
