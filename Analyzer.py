import os
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

#Helper functions
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline

#Preprocessing
from sklearn.preprocessing import StandardScaler

#Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#PDF
from matplotlib.backends.backend_pdf import PdfPages

#Separate dataset into feature matrix and target vector
feature_matrix = pd.read_csv("./FinalDataSet.csv")
target_vector = feature_matrix.pop("HomeWin")

#Split your training and test sets
X_train, X_test, y_train, y_test = train_test_split(feature_matrix, target_vector, test_size=0.1, random_state=0)

#Create models
log_reg = LogisticRegression(max_iter=500, random_state=0)
rand_forest = RandomForestClassifier(max_depth=10, n_estimators=500, random_state=0)
svc = SVC(class_weight='balanced', random_state=0)

#Create pipeline for models
scaled_log = make_pipeline(StandardScaler(), log_reg)
scaled_forest = make_pipeline(StandardScaler(), rand_forest)
scaled_svc = make_pipeline(StandardScaler(), svc)

models = {"Logistic Regression": scaled_log,
          "Random Forest": rand_forest,
          "SVC": scaled_svc}

model_scores = pd.DataFrame({'Model': models.keys(),
                             'Training Accuracy': [None, None, None],
                             'Validation Accuracy': [None, None, None]})

#Cross validate models and record scores
model_scores = model_scores.set_index('Model')
for model in models.keys():
    scores = cross_validate(models[model], X_train, y_train, cv=5, scoring="accuracy", return_train_score=True)
    train_accuracy = sum(scores["train_score"]) / len(scores["train_score"])
    validation_accuracy = sum(scores["test_score"]) / len(scores["test_score"])
    new_row = [train_accuracy, validation_accuracy]
    model_scores.loc[model, :] = new_row

print(model_scores)

param_grid_log = {'logisticregression__C': [0.01, 0.1, 1, 10, 100]}
param_grid_forest = {
    'randomforestclassifier__n_estimators': [5, 10, 100, 1000],
    'randomforestclassifier__max_depth': [5, 10, 20, 50]
}
param_grid_svc = {'svc__C': [0.01, 0.1, 1, 10, 100]}

grid_log = GridSearchCV(scaled_log, param_grid_log, cv=5, return_train_score=True)
grid_forest = GridSearchCV(scaled_forest, param_grid_forest, cv=5, return_train_score=True)
grid_svc = GridSearchCV(scaled_svc, param_grid_svc, cv=5, return_train_score=True)

#Fit each model
grid_log.fit(X_train, y_train)
grid_forest.fit(X_train, y_train)
grid_svc.fit(X_train, y_train)

#Predict values with each model
y_pred_log = grid_log.predict(X_test)
y_pred_forest = grid_forest.predict(X_test)
y_pred_svc = grid_svc.predict(X_test)

#Generate reports for each model
report_log = classification_report(y_test, y_pred_log)
report_forest = classification_report(y_test, y_pred_forest)
report_svc = classification_report(y_test, y_pred_svc)

def save_report(pdf_filename="classification_reports.pdf"):
    with PdfPages(pdf_filename) as pdf:
        #Classification reports
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.text(0.5, 0.95, "Model Performance Report", fontsize=14, ha="center", va="top", weight="bold")
        ax.text(0, 0.85, "Classification Reports:", fontsize=12, va="top", family="monospace")
        ax.text(0, 0.75, f"Logistic Regression:\n{report_log}", fontsize=10, va="top", family="monospace")
        ax.text(0, 0.45, f"Random Forest:\n{report_forest}", fontsize=10, va="top", family="monospace")
        ax.text(0, 0.15, f"SVC:\n{report_svc}", fontsize=10, va="top", family="monospace")
        
        #Model scores to report
        ax.text(0, -0.1, "Model Accuracy Scores:", fontsize=12, va="top", family="monospace")
        ax.text(0, -0.2, f"Logistic Regression - Training Accuracy: {model_scores.loc['Logistic Regression', 'Training Accuracy']:.4f}, Validation Accuracy: {model_scores.loc['Logistic Regression', 'Validation Accuracy']:.4f}", fontsize=10, va="top", family="monospace")
        ax.text(0, -0.3, f"Random Forest - Training Accuracy: {model_scores.loc['Random Forest', 'Training Accuracy']:.4f}, Validation Accuracy: {model_scores.loc['Random Forest', 'Validation Accuracy']:.4f}", fontsize=10, va="top", family="monospace")
        ax.text(0, -0.4, f"SVC - Training Accuracy: {model_scores.loc['SVC', 'Training Accuracy']:.4f}, Validation Accuracy: {model_scores.loc['SVC', 'Validation Accuracy']:.4f}", fontsize=10, va="top", family="monospace")

        ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        #Best parameters for each model
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.text(0.5, 0.95, "Best Parameters for Models", fontsize=14, ha="center", va="top", weight="bold")
        ax.text(0, 0.85, f"Logistic Regression: {grid_log.best_params_}", fontsize=12, va="top", family="monospace")
        ax.text(0, 0.75, f"Random Forest: {grid_forest.best_params_}", fontsize=12, va="top", family="monospace")
        ax.text(0, 0.65, f"SVC: {grid_svc.best_params_}", fontsize=12, va="top", family="monospace")

        ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        #Confusion matrices on separate page
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        confusion_matrices = [confusion_matrix(y_test, y_pred_log),
                              confusion_matrix(y_test, y_pred_forest),
                              confusion_matrix(y_test, y_pred_svc)]
        titles = ["Logistic Regression", "Random Forest", "SVC"]
        colors = ["Blues", "Greens", "Reds"]

        for i, ax in enumerate(axes):
            sn.heatmap(confusion_matrices[i], annot=True, fmt='d', cmap=colors[i], 
                       xticklabels=["Predicted Negative", "Predicted Positive"], 
                       yticklabels=["Actual Negative", "Actual Positive"], ax=ax)
            ax.set_title(f"Confusion Matrix - {titles[i]}")
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"\nPDF report saved as {pdf_filename}")

#Save report as PDF
save_report()