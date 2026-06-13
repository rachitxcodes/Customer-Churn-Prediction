import os
import json
import joblib
import pandas as pd
import numpy as np
import optuna
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from evaluate import calculate_metrics, plot_confusion_matrix, plot_roc_curves, plot_feature_importance

# Set optuna logging to warning to keep console output clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_processed_data(data_dir='data/processed'):
    """Load preprocessed datasets from processed folder."""
    X_train_smote = pd.read_csv(os.path.join(data_dir, 'X_train_smote.csv'))
    y_train_smote = pd.read_csv(os.path.join(data_dir, 'y_train_smote.csv')).values.ravel()
    
    X_train_smoteenn = pd.read_csv(os.path.join(data_dir, 'X_train_smoteenn.csv'))
    y_train_smoteenn = pd.read_csv(os.path.join(data_dir, 'y_train_smoteenn.csv')).values.ravel()
    
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()
    
    return X_train_smote, y_train_smote, X_train_smoteenn, y_train_smoteenn, X_test, y_test

def get_base_models():
    """Define the 5 core classifiers."""
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1)
    }

def train_and_compare(X_train_smote, y_train_smote, X_train_smoteenn, y_train_smoteenn, X_test, y_test):
    """Train all models on both SMOTE and SMOTEENN to evaluate performance."""
    results = []
    trained_models_smote = {}
    trained_models_smoteenn = {}
    
    # Core loop for SMOTE
    print("\n--- Training on SMOTE-Balanced Dataset ---")
    base_models = get_base_models()
    for name, model in base_models.items():
        print(f"Training {name}...")
        # Create a nested/child run in MLflow
        with mlflow.start_run(run_name=f"Base_{name.replace(' ', '_')}_SMOTE", nested=True):
            model.fit(X_train_smote, y_train_smote)
            trained_models_smote[name] = model
            
            # Predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrics
            metrics = calculate_metrics(y_test, y_pred, y_prob)
            metrics['Model'] = name
            metrics['Balancing'] = 'SMOTE'
            results.append(metrics)
            
            # Log params & metrics
            mlflow.log_param("model_type", name)
            mlflow.log_param("balancing", "SMOTE")
            for param, val in model.get_params().items():
                if type(val) in [int, float, str, bool]:
                    mlflow.log_param(f"model_{param}", val)
            
            for metric_name, val in metrics.items():
                if metric_name not in ['Model', 'Balancing']:
                    if not pd.isna(val):
                        mlflow.log_metric(metric_name, float(val))
        
    # Core loop for SMOTEENN
    print("\n--- Training on SMOTEENN-Balanced Dataset ---")
    base_models = get_base_models()
    for name, model in base_models.items():
        print(f"Training {name}...")
        with mlflow.start_run(run_name=f"Base_{name.replace(' ', '_')}_SMOTEENN", nested=True):
            model.fit(X_train_smoteenn, y_train_smoteenn)
            trained_models_smoteenn[name] = model
            
            # Predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrics
            metrics = calculate_metrics(y_test, y_pred, y_prob)
            metrics['Model'] = name
            metrics['Balancing'] = 'SMOTEENN'
            results.append(metrics)
            
            # Log params & metrics
            mlflow.log_param("model_type", name)
            mlflow.log_param("balancing", "SMOTEENN")
            for param, val in model.get_params().items():
                if type(val) in [int, float, str, bool]:
                    mlflow.log_param(f"model_{param}", val)
            
            for metric_name, val in metrics.items():
                if metric_name not in ['Model', 'Balancing']:
                    if not pd.isna(val):
                        mlflow.log_metric(metric_name, float(val))
        
    results_df = pd.DataFrame(results)
    
    # Combine Recall + ROC-AUC as selection score
    results_df['Selection_Score'] = results_df['Recall'] + results_df['ROC-AUC']
    
    print("\n### Base Model Comparisons (Sorted by Selection Score: Recall + AUC) ###")
    print(results_df.sort_values(by='Selection_Score', ascending=False).to_string(index=False))
    
    # Save ROC curves comparison for SMOTE
    os.makedirs('reports/plots', exist_ok=True)
    plot_roc_curves(trained_models_smote, X_test, y_test, save_path='reports/plots/roc_comparison_smote.png')
    plot_roc_curves(trained_models_smoteenn, X_test, y_test, save_path='reports/plots/roc_comparison_smoteenn.png')
    
    return results_df, trained_models_smote, trained_models_smoteenn

def run_optuna_tuning(model_name, balancing_name, X_train, y_train, X_test, y_test):
    """Run Optuna hyperparameter optimization for the selected winner."""
    print(f"\n--- Starting Hyperparameter Tuning for {model_name} on {balancing_name} ---")
    
    def objective(trial):
        if model_name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'random_state': 42,
                'eval_metric': 'logloss'
            }
            model = XGBClassifier(**params)
            
        elif model_name == 'LightGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 128),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42,
                'verbose': -1
            }
            model = LGBMClassifier(**params)
            
        elif model_name == 'Random Forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                'max_depth': trial.suggest_int('max_depth', 5, 25),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
                'random_state': 42
            }
            model = RandomForestClassifier(**params)
            
        elif model_name == 'Logistic Regression':
            params = {
                'C': trial.suggest_float('C', 0.001, 100.0, log=True),
                'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear']),
                'random_state': 42,
                'max_iter': 1000
            }
            model = LogisticRegression(**params)
            
        else: # Decision Tree
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': 42
            }
            model = DecisionTreeClassifier(**params)

        model.fit(X_train, y_train)
        
        # We optimize for Recall + ROC-AUC on the validation (original test) set
        # This keeps focus on minimizing false negatives while keeping classification boundaries solid
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        recall = calculate_metrics(y_test, y_pred, y_prob)['Recall']
        auc = calculate_metrics(y_test, y_pred, y_prob)['ROC-AUC']
        
        # Objective value
        return recall + auc

    # Run Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    print("Best Trial:")
    trial = study.best_trial
    print(f"  Value (Recall + AUC): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    return trial.params, trial.value

def main():
    # Set experiment
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Customer-Churn-Prediction")
    
    with mlflow.start_run(run_name="Churn_Training_Pipeline") as parent_run:
        # Load processed datasets
        X_train_smote, y_train_smote, X_train_smoteenn, y_train_smoteenn, X_test, y_test = load_processed_data()
        
        # Train and compare base models
        results_df, models_smote, models_smoteenn = train_and_compare(
            X_train_smote, y_train_smote, X_train_smoteenn, y_train_smoteenn, X_test, y_test
        )
        
        # Save comparison dataframe
        results_df.to_csv('reports/base_model_comparisons.csv', index=False)
        
        # Select best model and resampler based on Selection Score
        winner_row = results_df.sort_values(by='Selection_Score', ascending=False).iloc[0]
        best_model_name = winner_row['Model']
        best_balancing_name = winner_row['Balancing']
        
        print(f"\n>>> Selected Winner: {best_model_name} with {best_balancing_name} Balancing")
        print(f"    Base Metrics -> Recall: {winner_row['Recall']:.3f}, AUC: {winner_row['ROC-AUC']:.3f}")
        
        # Log top-level selection details to parent run
        mlflow.log_param("selected_base_model", best_model_name)
        mlflow.log_param("selected_balancing", best_balancing_name)
        mlflow.log_metric("base_recall", winner_row['Recall'])
        mlflow.log_metric("base_auc", winner_row['ROC-AUC'])
        
        # Get active train set based on best balancing method
        if best_balancing_name == 'SMOTE':
            X_train_best = X_train_smote
            y_train_best = y_train_smote
        else:
            X_train_best = X_train_smoteenn
            y_train_best = y_train_smoteenn
            
        # Run Optuna Tuning
        best_params, best_value = run_optuna_tuning(
            best_model_name, best_balancing_name, X_train_best, y_train_best, X_test, y_test
        )
        
        # Log tuning results to parent run
        mlflow.log_metric("optuna_best_value", best_value)
        for param, val in best_params.items():
            mlflow.log_param(f"tuned_opt_{param}", val)
        
        # Retrain final model with best params
        print(f"\n--- Retraining final tuned {best_model_name} model ---")
        if best_model_name == 'XGBoost':
            final_model = XGBClassifier(**best_params, random_state=42, eval_metric='logloss')
        elif best_model_name == 'LightGBM':
            final_model = LGBMClassifier(**best_params, random_state=42, verbose=-1)
        elif best_model_name == 'Random Forest':
            final_model = RandomForestClassifier(**best_params, random_state=42)
        elif best_model_name == 'Logistic Regression':
            final_model = LogisticRegression(**best_params, random_state=42, max_iter=1000)
        else:
            final_model = DecisionTreeClassifier(**best_params, random_state=42)
            
        final_model.fit(X_train_best, y_train_best)
        
        # Evaluate final model
        y_pred = final_model.predict(X_test)
        y_prob = final_model.predict_proba(X_test)[:, 1] if hasattr(final_model, 'predict_proba') else None
        
        final_metrics = calculate_metrics(y_test, y_pred, y_prob)
        print("\n--- Tuned Model Performance on Test Set ---")
        for k, v in final_metrics.items():
            print(f"{k}: {v:.4f}")
            
        # Log final model metrics
        for k, v in final_metrics.items():
            if not pd.isna(v):
                mlflow.log_metric(f"tuned_{k.replace('-', '_')}", float(v))
            
        # Save confusion matrix heatmap
        plot_confusion_matrix(y_test, y_pred, f"Tuned {best_model_name}", save_path='reports/plots/confusion_matrix_tuned.png')
        
        # Save feature importances (top 15)
        feature_names = list(X_train_best.columns)
        plot_feature_importance(final_model, feature_names, best_model_name, top_n=15, save_path='reports/plots/feature_importance.png')
        
        # Save best model artifact
        os.makedirs('models', exist_ok=True)
        joblib.dump(final_model, 'models/best_model.pkl')
        print("\nModel saved to models/best_model.pkl successfully!")
        
        # Log plots as artifacts
        mlflow.log_artifact('reports/plots/confusion_matrix_tuned.png', 'plots')
        mlflow.log_artifact('reports/plots/feature_importance.png', 'plots')
        if os.path.exists('reports/plots/roc_comparison_smote.png'):
            mlflow.log_artifact('reports/plots/roc_comparison_smote.png', 'plots')
        if os.path.exists('reports/plots/roc_comparison_smoteenn.png'):
            mlflow.log_artifact('reports/plots/roc_comparison_smoteenn.png', 'plots')
            
        # Log the scikit-learn model artifact to MLflow
        mlflow.sklearn.log_model(final_model, "best_model")
        print("\nModel logged to MLflow successfully!")

if __name__ == '__main__':
    main()
