import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Set style for professional look
sns.set_theme(style="whitegrid")

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate standard classification metrics for the churn class (1)."""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """Plot and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn'])
    plt.title(f'Confusion Matrix - {model_name}', fontweight='bold', pad=10)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_roc_curves(models_dict, X_test, y_test, save_path=None):
    """Plot ROC curves for multiple models on a single chart."""
    plt.figure(figsize=(10, 8))
    
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            continue
            
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curves Comparison', fontweight='bold', pad=15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_feature_importance(model, feature_names, model_name, top_n=15, save_path=None):
    """Extract, plot, and save top N feature importances."""
    plt.figure(figsize=(10, 8))
    
    # Check if model has feature importances or coefficients
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print(f"Model {model_name} does not support feature importances/coefficients directly.")
        return
        
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    sns.barplot(x=top_importances, y=top_features, palette='viridis', hue=top_features, legend=False)
    plt.title(f'Top {top_n} Feature Importances - {model_name}', fontweight='bold', pad=15)
    plt.xlabel('Importance Value')
    plt.ylabel('Features')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
