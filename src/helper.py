import random
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OneHotEncoder, RobustScaler, OrdinalEncoder

from sklearn.model_selection import train_test_split,cross_val_score, StratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, brier_score_loss, log_loss,roc_auc_score,
    f1_score, roc_auc_score, roc_curve, precision_recall_curve, ConfusionMatrixDisplay, classification_report
)



class creditScorePipeline:
    def __init__(self):
        self.metrics = {}
        self.Id_cols = ['ID', 'customer_id', 'country_id', 'tbl_loan_id', 'lender_id']
        self.numerical_cols = ['Total_Amount', 'Total_Amount_to_Repay', 'duration', 
               'Amount_Funded_By_Lender', 'Lender_portion_Funded',
               'Lender_portion_to_be_repaid', 'disbursement_year',
               'disbursement_month', 'disbursement_dayofweek', 'due_year', 'due_month',
               'due_dayofweek','Average precipitation in depth (mm per year)',
               'Deposit interest rate (%)',
               'Fossil fuel energy consumption (% of total)',
               'Inflation, consumer prices (annual %)',
               'Interest rate spread (lending rate minus deposit rate, %)',
               'Lending interest rate (%)',
               'Official exchange rate (LCU per US$, period average)',
               'Real interest rate (%)', 'Unemployment rate']
        self.categorical_cols = ['loan_type','New_versus_Repeat']
 
    # Models evaluation function
    def evaluate_model(self, y_true, y_proba, model_name):
        y_pred = (y_proba >= 0.7).astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)
        cm = confusion_matrix(y_true, y_pred)
        
        self.metrics[model_name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc
        }
    
        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Plot Precision-Recall Curve
        prec_vals, rec_vals, _ = precision_recall_curve(y_true, y_proba)
        plt.figure(figsize=(6, 4))
        plt.plot(rec_vals, prec_vals, label=model_name)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
        # confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'confusion matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
    
    def plotclassification(self, y_true, y_proba, threshold=0.5, bins=50):
        df_eval = pd.DataFrame()
        df_eval["true"] = np.array(y_true)
        df_eval["predicted"] = np.round(y_proba, 4)
    
        # Classification logic
        conditions = [
            (df_eval["predicted"] > threshold) & (df_eval["true"] == 1),
            (df_eval["predicted"] <= threshold) & (df_eval["true"] == 0),
            (df_eval["predicted"] > threshold) & (df_eval["true"] == 0),
            (df_eval["predicted"] <= threshold) & (df_eval["true"] == 1),
        ]
        choices = ["tp", "tn", "fp", "fn"]
    
        df_eval["classification"] = np.select(conditions, choices, default="unknown")
    
        hue_order = ["tp", "tn", "fp", "fn"]
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df_eval, x="predicted", hue="classification", bins=bins, hue_order=hue_order, multiple="stack")
        plt.axvline(threshold, color="black", linestyle="--", label=f"Threshold = {threshold}")
        plt.title("Classification Distribution by Predicted Score")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
        return df_eval.sort_values('predicted', ascending=False, inplace=True)
    
    def predict_and_evaluate(self, model, X_val, y_val, name):
        proba = model.predict_proba(X_val)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, proba)
    
        # Best threshold where recall is high and precision is not too low
        best_threshold = thresholds[np.argmax(recall - (1 - precision))]
        preds = (proba >= best_threshold).astype(int)
    
        print(f"\n=== {name} ===")
        print("Best Threshold:", best_threshold)
        print(classification_report(y_val, preds))
    
        self.evaluate_model(y_val, proba, name)
        self.plotclassification(y_val, proba, threshold=best_threshold)
    
    
    # ECE calc
    def calculate_ece(self, y_true, y_prob, n_bins=10):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
        bin_size = len(y_true) / n_bins
        ece = np.sum(np.abs(prob_true - prob_pred) * bin_size) / len(y_true)
        return ece
    
    
    def feature_engineering_dates(self,df):
        # Disbursement_date
        df['disbursement_date'] = pd.to_datetime(df['disbursement_date'], errors='coerce')
        df['due_date'] = pd.to_datetime(df['due_date'], errors='coerce')
        df['disbursement_year'] = df['disbursement_date'].dt.year
        df['disbursement_month'] = df['disbursement_date'].dt.month
        df['disbursement_dayofweek'] = df['disbursement_date'].dt.dayofweek  # Monday=0, Sunday=6
        # Due_Date
        df['due_year'] = df['due_date'].dt.year
        df['due_month'] = df['due_date'].dt.month
        df['due_dayofweek'] = df['due_date'].dt.dayofweek
        df.drop(columns=['disbursement_date', 'due_date'], inplace=True)
        return df