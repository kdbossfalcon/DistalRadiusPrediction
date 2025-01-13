# Predicting risk of fracture redisplacement using XGBoost model

## Setting up


```python
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
```

### Load and split data, get outcome and predictors


```python
df = pd.read_csv("placeholder.csv")
```


```python
outcome = df.placeholder_column
outcome = abs(1-outcome)
```


```python
predictor = df.iloc[:,[7,8,10,11,13,14,16,17,20,21,32,34,35,36,37,38,39,40,42,43,44,46]]
```


```python
#Split train and test set
X_train, X_test, y_train, y_test = train_test_split(
    predictor, outcome, train_size = 0.75, test_size=0.25, shuffle=False)

print("Split train and test set")
print(f"Length of train set = {len(X_train)}")
print(f"Length of test set = {len(X_test)}")
```

### Function for plotting ROC and Calibration curve


```python
def plot_roc_and_calibration_curve(y_train, y_train_proba, y_test, y_test_proba, n_bins=10):
    """
    Plots the ROC curve and Calibration curve for training and test sets.
    
    Args:
        y_train: Ground truth labels for the training set.
        y_train_proba: Predicted probabilities for the training set.
        y_test: Ground truth labels for the test/validation set.
        y_test_proba: Predicted probabilities for the test/validation set.
        n_bins: Number of bins for the calibration curve.
    """
    # ROC Curve
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)

    auc_train = roc_auc_score(y_train, y_train_proba)
    auc_test = roc_auc_score(y_test, y_test_proba)

    # Calibration Curve
    prob_true_train, prob_pred_train = calibration_curve(y_train, y_train_proba, n_bins=n_bins, strategy='uniform')
    prob_true_test, prob_pred_test = calibration_curve(y_test, y_test_proba, n_bins=n_bins, strategy='uniform')

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot ROC Curve
    axs[0].plot(fpr_train, tpr_train, label=f'Train AUC = {auc_train:.4f}', color='blue')
    axs[0].plot(fpr_test, tpr_test, label=f'Test AUC = {auc_test:.4f}', color='orange')
    axs[0].plot([0, 1], [0, 1], 'k--', label='Random Guess')  # Diagonal line
    axs[0].set_xlabel('False Positive Rate (FPR)')
    axs[0].set_ylabel('True Positive Rate (TPR)')
    axs[0].set_title('ROC Curve')
    axs[0].legend(loc='lower right')
    axs[0].grid()

    # Plot Calibration Curve
    axs[1].plot(prob_pred_train, prob_true_train, marker='o', label='Train', color='blue')
    axs[1].plot(prob_pred_test, prob_true_test, marker='o', label='Test', color='orange')
    axs[1].plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')  # Perfect calibration line
    axs[1].set_xlabel('Mean Predicted Probability')
    axs[1].set_ylabel('Fraction of Positives')
    axs[1].set_title('Calibration Curve')
    axs[1].legend(loc='best')
    axs[1].grid()

    # Show plots
    plt.tight_layout()
    plt.show()

```

## Hyperparameter tuning for base xgboost model

### Create base XGBoost model


```python
xgb_n = xgb.XGBClassifier(
    learning_rate =0.01,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='logloss',
    nthread=10,
    scale_pos_weight=1,
    seed=42
)

print("create xgb model for further hyperparameter tuning")
```

### Tune n_estimators


```python
#Tune n_estimators
tree_param_grid = {
    'n_estimators':range(100,300,1),
}

tree_grid_search = GridSearchCV(
    estimator=xgb_n,
    param_grid=tree_param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1
)

print("performing grid search for n_estimator")
tree_grid_search.fit(X_train, y_train)

best_n_estimators = tree_grid_search.best_params_['n_estimators']
print(f"Best parameters: {tree_grid_search.best_params_}")
```

### Tune max_depth and min_child_weight


```python
xgb1 = xgb.XGBClassifier(
    learning_rate =0.01,
    n_estimators=best_n_estimators,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='logloss',
    nthread=10,
    scale_pos_weight=1,
    seed=42
)
```


```python
#Perform Grid search CV on max_depth and min_child_weight first
first_param_grid = {
    'max_depth':range(3,10,1),
    'min_child_weight':range(1,12,1)
}

first_grid_search = GridSearchCV(
    estimator=xgb1,
    param_grid=first_param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1
)

print("performing grid search for max_dept and min_child_weight")
first_grid_search.fit(X_train, y_train)

best_max_depth = first_grid_search.best_params_['max_depth']
best_min_child_weight = first_grid_search.best_params_['min_child_weight']
print(f"Best parameters: {first_grid_search.best_params_}")
```

### Tune gamma


```python
#Perform Grid search CV on gamma
xgb2 = xgb.XGBClassifier(
    learning_rate =0.01,
    n_estimators=best_n_estimators,
    max_depth=best_max_depth,
    min_child_weight=best_min_child_weight,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='logloss',
    nthread=10,
    scale_pos_weight=1,
    seed=42
)

second_param_grid = {
    'gamma':[i/100.0 for i in range(0,100)]
}

second_grid_search = GridSearchCV(
    estimator=xgb2,
    param_grid=second_param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1
)

print("performing grid search for gamma")
second_grid_search.fit(X_train, y_train)

best_gamma = second_grid_search.best_params_['gamma']
print(f"Best parameters: {second_grid_search.best_params_}")
```

### Tune subsample and colsample_bytree


```python
#Perform Grid search CV on subsample and colsample_bytree
xgb3 = xgb.XGBClassifier(
    learning_rate =0.01,
    n_estimators=best_n_estimators,
    max_depth=best_max_depth,
    min_child_weight=best_min_child_weight,
    gamma=best_gamma,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='logloss',
    nthread=10,
    scale_pos_weight=1,
    seed=42
)

third_param_grid = {
    'subsample':[i/100.0 for i in range(0,100,5)],
    'colsample_bytree':[i/100.0 for i in range(0,100,5)]
}

third_grid_search = GridSearchCV(
    estimator=xgb3,
    param_grid=third_param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1
)

print("performing grid search for subsmple and colsample_bytree")
third_grid_search.fit(X_train, y_train)

best_subsample = third_grid_search.best_params_['subsample']
best_colsample_bytree = third_grid_search.best_params_['colsample_bytree']
print(f"Best parameters: {third_grid_search.best_params_}")
```

### Tune alpha and lambda


```python
#Perform Grid search CV on alpha and lambda
xgb4 = xgb.XGBClassifier(
    learning_rate =0.01,
    n_estimators=best_n_estimators,
    max_depth=best_max_depth,
    min_child_weight=best_min_child_weight,
    gamma=best_gamma,
    subsample=best_subsample,
    colsample_bytree=best_colsample_bytree,
    objective='binary:logistic',
    eval_metric='logloss',
    nthread=10,
    scale_pos_weight=1,
    seed=42
)

forth_param_grid = {
    'alpha':[i*0.005 for i in range(0,100,2)],
    'lambda':[i*0.01 for i in range(50,150,2)]
}

forth_grid_search = GridSearchCV(
    estimator=xgb4,
    param_grid=forth_param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1
)

print("performing grid search for alpha and lambda")
forth_grid_search.fit(X_train, y_train)

best_alpha = forth_grid_search.best_params_['alpha']
best_lambda = forth_grid_search.best_params_['lambda']
print(f"Best parameters: {forth_grid_search.best_params_}")
```

### Final parameters for learning rate = 0.01


```python
final_params = {
    'learning_rate':0.01,
    'n_estimators':best_n_estimators,
    'max_depth':best_max_depth,
    'min_child_weight':best_min_child_weight,
    'gamma':best_gamma,
    'subsample':best_subsample,
    'colsample_bytree':best_colsample_bytree,
    'alpha':best_alpha,
    'lambda':best_lambda
}

final_xgb = xgb.XGBClassifier(
    **final_params,
    objective='binary:logistic',
    eval_metric='logloss',
    nthread=10,
    scale_pos_weight=1,
    seed=42
)
```

## Test full model on test set

Train final model on the whole training dataset, then check ROC curve and Calibration plot for both training and test set


```python
final_xgb.fit(X_train, y_train)
```


```python
y_train_proba = final_xgb.predict_proba(X_train)[:, 1]
y_train_pred = (y_train_proba >= 0.5).astype(int)
y_pred_proba = final_xgb.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)


# Evaluate performance metrics
roc_auc = roc_auc_score(y_train, y_train_proba)
accuracy = accuracy_score(y_train, y_train_pred)
recall = recall_score(y_train, y_train_pred)
precision = precision_score(y_train, y_train_pred)
f1 = f1_score(y_train, y_train_pred)

# Print metrics
print("Train set performance")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")

# Evaluate performance metrics
roc_auc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print metrics
print("\nTest set performance")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nCompare non-calibrated train and test performance")
plot_roc_and_calibration_curve(y_train, y_train_proba, y_test, y_pred_proba, n_bins=10)
```

## Model calibration

Create calibrated probabilities of the training set by performing calibration on each fold of 5-fold cross-validation to get calibrated probabilities for the hold-out set for each folds, use these calibrated probabilities as soft targets for deep learning model training.

Then we plot ROC and calibration curve comparing calibrated probabilities to the non-calibrated model


```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)

calibrated_probs = np.zeros(len(X_train))

for train_idx, cal_idx in kf.split(X_train):
    X_tr, X_cal = X_train.iloc[train_idx], X_train.iloc[cal_idx]
    y_tr, y_cal = y_train.iloc[train_idx], y_train.iloc[cal_idx]
    
    # Train XGBoost on training folds
    model = xgb.XGBClassifier(
        **final_params,
        objective='binary:logistic',
        eval_metric='logloss',
        nthread=10,
        scale_pos_weight=1,
        seed=42
    )
    model.fit(X_tr, y_tr)
    
    # Calibrate on the hold-out fold
    calibrated_model = CalibratedClassifierCV(estimator=model, method='sigmoid', n_jobs = 10)
    calibrated_model.fit(X_cal, y_cal)
    
    # Get calibrated probabilities for the current fold
    calibrated_probs[cal_idx] = calibrated_model.predict_proba(X_cal)[:, 1]
```


```python
y_train_proba = final_xgb.predict_proba(X_train)[:, 1]
y_train_pred = (y_train_proba >= 0.5).astype(int)

calibrated_pred = (calibrated_probs >= 0.5).astype(int)


# Evaluate performance metrics
roc_auc = roc_auc_score(y_train, y_train_proba)
accuracy = accuracy_score(y_train, y_train_pred)
recall = recall_score(y_train, y_train_pred)
precision = precision_score(y_train, y_train_pred)
f1 = f1_score(y_train, y_train_pred)

# Print metrics
print("Train set performance")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")


# Evaluate performance metrics
roc_auc = roc_auc_score(y_train, calibrated_probs)
accuracy = accuracy_score(y_train, calibrated_pred)
recall = recall_score(y_train, calibrated_pred)
precision = precision_score(y_train, calibrated_pred)
f1 = f1_score(y_train, calibrated_pred)

# Print metrics
print("\nCalibrated Train set performance")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")

print("Compare non-calibrated and calibrated train set performance")
plot_roc_and_calibration_curve(y_train, y_train_proba, y_train, calibrated_probs, n_bins=10)
```
