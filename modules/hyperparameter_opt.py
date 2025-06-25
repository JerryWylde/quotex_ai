
 # hyperparameter_opt.py (CORTEX) 

import optuna
import pandas as pd
import numpy as np
import joblib
import json
import os
import argparse
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from datetime import datetime
from modules.feature_engineering import engineer_topnotch_features 

def load_data(data_path, target_col):
    df = pd.read_csv(data_path)
    df.dropna(inplace=True)
    # Explicitly map any -1 label to 0 for binary classification consistency
    df[target_col] = df[target_col].replace(-1, 0).astype(int)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def xgb_objective(trial, X, y):
    y_clean = np.array(y)
    y_clean = np.nan_to_num(y_clean, nan=0)
    y_clean = np.clip(y_clean, 0, 1).astype(int)

    unique_labels = np.unique(y_clean)
    assert set(unique_labels).issubset({0, 1}), f"XGBoost labels must be binary (0 or 1), got: {unique_labels}"

    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'booster': 'gbtree',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-4, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-4, 10.0, log=True),
        'random_state': 42,
        'verbosity': 0,
        'tree_method': 'hist'
    }
    n_estimators = trial.suggest_int('n_estimators', 50, 300)

    tscv = TimeSeriesSplit(n_splits=3)
    scores = []

    for train_idx, valid_idx in tscv.split(X):
        dtrain = xgb.DMatrix(X.iloc[train_idx], label=y_clean[train_idx])
        dvalid = xgb.DMatrix(X.iloc[valid_idx], label=y_clean[valid_idx])

        evals_result = {}
        booster = xgb.train(
            param,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dvalid, "validation")],
            early_stopping_rounds=50,
            evals_result=evals_result,
            verbose_eval=False
        )

        preds = booster.predict(dvalid)
        pred_labels = (preds > 0.5).astype(int)
        acc = accuracy_score(y_clean[valid_idx], pred_labels)
        scores.append(acc)

    return np.mean(scores)


def lgb_objective(trial, X, y):
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-4, 0.1, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-6, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 1500),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'seed': 42
    }

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []

    for train_idx, valid_idx in tscv.split(X):
        lgb_train = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx])
        lgb_valid = lgb.Dataset(X.iloc[valid_idx], label=y.iloc[valid_idx])
        model = lgb.train(param, lgb_train, valid_sets=[lgb_valid], callbacks=[lgb.early_stopping(50),lgb.log_evaluation(0)])
        preds = model.predict(X.iloc[valid_idx])
        pred_labels = (preds > 0.5).astype(int)
        acc = accuracy_score(y.iloc[valid_idx], pred_labels)
        scores.append(acc)

    return np.mean(scores)


def cat_objective(trial, X, y):
    y_clean = np.array(y)
    y_clean = np.nan_to_num(y_clean, nan=0)
    y_clean = np.clip(y_clean, 0, 1).astype(int)

    unique_labels = np.unique(y_clean)
    assert set(unique_labels).issubset({0, 1}), f"CatBoost labels must be binary (0 or 1), got: {unique_labels}"

    param = {
        'loss_function': 'Logloss',
        'eval_metric': 'Accuracy',
        'verbose': 0,
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 128),
        'random_seed': 42
    }

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []

    for train_idx, valid_idx in tscv.split(X):
        y_train = y_clean[train_idx]
        y_valid = y_clean[valid_idx]

        # Skip fold if no class diversity
        if len(np.unique(y_train)) < 2 or len(np.unique(y_valid)) < 2:
            continue

        model = CatBoostClassifier(
            **param,
            iterations=1000,
            early_stopping_rounds=50,
        )

        model.fit(
            X.iloc[train_idx], y_train,
            eval_set=(X.iloc[valid_idx], y_valid),
            use_best_model=True,
            verbose=False
        )

        preds = model.predict(X.iloc[valid_idx])
        pred_labels = preds.astype(int).flatten()
        acc = accuracy_score(y_valid, pred_labels)
        scores.append(acc)

    return np.mean(scores) if scores else 0.0

def run_study(model_name, objective_func, X, y, n_trials, save_dir, n_jobs=1):
    print(f"Running optimization for: {model_name}")

    # Wrap objective function to include X,y fixed args for optuna
    func = lambda trial: objective_func(trial, X, y)

    study = optuna.create_study(direction="maximize")
    study.optimize(func, n_trials=n_trials, n_jobs=n_jobs)

    print(f"Best {model_name} Score: {study.best_value:.5f}")
    print(f"Best {model_name} Params: {study.best_params}")

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"best_{model_name.lower()}_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=4)

    return study.best_params, study.best_value


def run_all_optimizations(X, y, n_trials=50, save_dir="best_params", n_jobs=1):
    print("[DEBUG] Unique Signal labels before Optuna:", y.unique())
    print("[DEBUG] Value counts:\n", y.value_counts(dropna=False))

    # Force repair before it goes to models
    y = y.fillna(0).replace(-1, 0).astype(int).clip(0, 1)

    results = {}

    best_xgb_params, best_xgb_score = run_study(
        "XGBoost", xgb_objective, X, y, n_trials, save_dir, n_jobs)

    best_lgb_params, best_lgb_score = run_study(
        "LightGBM", lgb_objective, X, y, n_trials, save_dir, n_jobs)

    best_cat_params, best_cat_score = run_study(
        "CatBoost", cat_objective, X, y, n_trials, save_dir, n_jobs)

    results["XGBoost"] = {"params": best_xgb_params, "accuracy": best_xgb_score}
    results["LightGBM"] = {"params": best_lgb_params, "accuracy": best_lgb_score}
    results["CatBoost"] = {"params": best_cat_params, "accuracy": best_cat_score}

    print("\n=== Optimization Results ===")
    for model, result in results.items():
        print(f"{model}: Best Accuracy={result['accuracy']:.5f}")
        print(f"Params: {result['params']}")

    return results

  
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to CSV file with features + target")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--save_dir", type=str, default="best_params", help="Directory to save best parameters")
    parser.add_argument("--target_col", type=str, default="Signal", help="Target column name")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs for Optuna")
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)

    if 'Signal' not in df.columns or df['Signal'].isnull().all():
        print("🧠 Detected raw data — auto engineering features + signals...")
        df = engineer_topnotch_features(df, add_signal=True, threshold=0.00015)

    with open("models/feature_cols.json") as f:
        feature_cols = json.load(f)

    df = df[feature_cols + ["Signal"]]


    os.makedirs("data", exist_ok=True)
    df.to_csv("data/eurusd_1min_labelled.csv", index=False)
    print("📁 Saved engineered dataset with signals to 'data/eurusd_1min_ready.csv'")

    print("[DEBUG] After feature engineering:", df.columns)
    print(df[["Signal"]].value_counts(dropna=False))


    if 'Signal' not in df.columns:
        print("❌ 'Signal' column is missing entirely!")
        raise ValueError("🚨 Signal column missing or only one class after generation.")

    print("🧠 Signal column exists. Unique values:")
    print(df['Signal'].value_counts(dropna=False))

    if df['Signal'].nunique() < 2:
        print("❌ Only one class found in 'Signal' column!")
        raise ValueError("🚨 Signal column missing or only one class after generation.")


    df["Signal"] = pd.to_numeric(df["Signal"], errors='coerce')
    df["Signal"] = df["Signal"].fillna(0).replace([np.inf, -np.inf], 0).replace(-1, 0).astype(int)

    features = df.drop(columns=["Signal"])
    features = features.select_dtypes(include=[np.number])
    df = pd.concat([features, df[["Signal"]]], axis=1)

    X = df.drop(columns=["Signal"])
    y = df["Signal"]

    assert "Signal" not in X.columns, "LEAK ALERT: Signal still in X"

    results = run_all_optimizations(X, y, n_trials=args.n_trials, save_dir=args.save_dir, n_jobs=args.n_jobs)

    print("\n=== Optimization Summary ===")
    for model, res in results.items():
        print(f"{model}: Best Accuracy = {res['accuracy']:.5f}")
        print(f"Params: {res['params']}\n")

if __name__ == "__main__":
    main()
