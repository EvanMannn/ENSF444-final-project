import os
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

#Helper functions
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline

#Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#PDF
from matplotlib.backends.backend_pdf import PdfPages

#Pull data from the CSV dataset, and split into train and test datasets
def load_data(filepath):
    feature_matrix = pd.read_csv(filepath)
    target_vector = feature_matrix.pop("HomeWin")

    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, target_vector, test_size=0.1, random_state=0)

    return X_train, X_test, y_train, y_test

#Set up the models, and create scale pipeline for each
def get_models():
    log_reg = LogisticRegression(max_iter=500, random_state=0)
    rand_forest = RandomForestClassifier(max_depth=10, n_estimators=500, random_state=0)
    svc = SVC(class_weight='balanced', random_state=0)

    n_components = 10

    return {
        "Logistic Regression": make_pipeline(StandardScaler(), PCA(n_components=n_components), log_reg),
        "Random Forest": make_pipeline(StandardScaler(), PCA(n_components=n_components), rand_forest),
        "SVC": make_pipeline(StandardScaler(), PCA(n_components=n_components), svc),
    }

#Cross validate models and return results
def evaluate_models(models, X_train, y_train):
    results = []
    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, cv=5, scoring="accuracy", return_train_score=True)
        results.append({
            "Model": name,
            "Training Accuracy": np.mean(scores["train_score"]),
            "Validation Accuracy": np.mean(scores["test_score"])
        })

    return pd.DataFrame(results).set_index("Model")

#Create grid search for each model and tune models for best parameters
def tune_models(X_train, y_train, models):
    param_grids = {
        "Logistic Regression": {'logisticregression__C': [0.01, 0.1, 1, 10, 100]},
        "Random Forest": {
            'randomforestclassifier__n_estimators': [5, 10, 100, 1000],
            'randomforestclassifier__max_depth': [5, 10, 20, 50]
        },
        "SVC": {'svc__C': [0.01, 0.1, 1, 10, 100]}
    }

    tuned_models = {}
    for name in models:
        grid = GridSearchCV(models[name], param_grids[name], cv=5, return_train_score=True)
        grid.fit(X_train, y_train)
        tuned_models[name] = grid

    return tuned_models

#Get scores and predictions from each model
def evaluate_test_performance(tuned_models, X_test, y_test):
    reports = {}
    predictions = {}
    scores = {}

    for name, model in tuned_models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        reports[name] = classification_report(y_test, y_pred)
        scores[name] = accuracy_score(y_test, y_pred)

    return reports, predictions, scores

#Create feature importance plot for each supported model
def plot_feature_importance(model, model_name, feature_names, pdf):
    if "randomforestclassifier" in model.best_estimator_.named_steps:
        importances = model.best_estimator_.named_steps["randomforestclassifier"].feature_importances_
    elif "logisticregression" in model.best_estimator_.named_steps:
        importances = np.abs(model.best_estimator_.named_steps["logisticregression"].coef_[0])
    else:
        print(f"Feature importance not supported for {model_name}")
        return

    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"Feature Importance - {model_name}")
    ax.bar(range(len(importances)), importances[indices])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(np.array(feature_names)[indices], rotation=90)
    pdf.savefig(fig)
    plt.close(fig)


#Add results to PDF report
def create_pdf_report(reports, model_scores, test_scores, tuned_models, predictions, y_test, feature_names, filename="classification_reports.pdf"):
    with PdfPages(filename) as pdf:
        # Report text
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.text(0.5, 1, "Model Performance Report", fontsize=16, ha="center", weight="bold")
        y = 0.95
        for name, report in reports.items():
            ax.text(0, y, f"{name}:\n{report}", fontsize=10, va="top", family="monospace")
            y -= 0.25
        ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

        # Accuracy Table
        model_scores["Test Accuracy"] = model_scores.index.map(lambda x: test_scores.get(x, "N/A"))

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        table = ax.table(cellText=model_scores.values,
                         rowLabels=model_scores.index,
                         colLabels=model_scores.columns,
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.5)
        plt.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.1)
        pdf.savefig(fig)
        plt.close(fig)

        #Confusion Matrices
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        confusion_matrices = [confusion_matrix(y_test, predictions['Logistic Regression'].astype(int)),
                            confusion_matrix(y_test, predictions['Random Forest'].astype(int)),
                            confusion_matrix(y_test, predictions['SVC'].astype(int))]
        titles = ["Logistic Regression", "Random Forest", "SVC"]
        colors = ["Blues", "Greens", "Reds"]

        for i, ax in enumerate(axes):
            sn.heatmap(confusion_matrices[i], annot=True, fmt='d', cmap=colors[i], 
                       xticklabels=["Predicted Negative", "Predicted Positive"], 
                       yticklabels=["Actual Negative", "Actual Positive"], ax=ax)
            ax.set_title(f"Confusion Matrix - {titles[i]}")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        #Feature Importance Graphs
        for name in ["Logistic Regression", "Random Forest"]:
            plot_feature_importance(tuned_models[name], name, feature_names, pdf)

    print(f"\nPDF report saved as {filename}")

#Main function
def run_pipeline(csv_path):
    X_train, X_test, y_train, y_test = load_data(csv_path)
    feature_names = X_train.columns

    models = get_models()
    model_scores = evaluate_models(models, X_train, y_train)
    tuned_models = tune_models(X_train, y_train, models)
    reports, predictions, test_scores = evaluate_test_performance(tuned_models, X_test, y_test)

    print("\n--- Model Scores ---")
    print(model_scores)
    print("\n--- Test Accuracies ---")
    for name, score in test_scores.items():
        print(f"{name}: {score:.4f}")

    create_pdf_report(reports, model_scores, test_scores, tuned_models, predictions, y_test, feature_names)

if __name__ == "__main__":
    run_pipeline("./FinalDataSet.csv")