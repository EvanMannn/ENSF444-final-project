from fpdf import FPDF
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

print("\nBest parameters for Logistic Regression:", grid_log.best_params_)
print("Best parameters for Random Forest:", grid_forest.best_params_)
print("Best parameters for SVC:", grid_svc.best_params_)

#Predict values with each model
y_pred_log = grid_log.predict(X_test)
y_pred_forest = grid_forest.predict(X_test)
y_pred_svc = grid_svc.predict(X_test)

#Generate reports for each model
report_log = classification_report(y_test, y_pred_log)
report_forest = classification_report(y_test, y_pred_forest)
report_svc = classification_report(y_test, y_pred_svc)

print(f"\nLogistic Classification Report:\n {report_log}")
print(f"RandomForest Classification Report:\n {report_forest}")
print(f"SVC Classification Report:\n {report_svc}")

#Generate confusion matrix for each model
confusion_matrix_log = confusion_matrix(y_test, y_pred_log)
confusion_matrix_forest = confusion_matrix(y_test, y_pred_forest)
confusion_matrix_svc = confusion_matrix(y_test, y_pred_svc)

#Display confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

conf_matrices = [confusion_matrix_log, confusion_matrix_forest, confusion_matrix_svc]
titles = ["Logistic Regression", "Random Forest", "SVC"]
colors = ["Blues", "Greens", "Reds"]

for i, ax in enumerate(axes):
    sn.heatmap(conf_matrices[i], annot=True, fmt='d', cmap=colors[i], 
               xticklabels=["Predicted Negative", "Predicted Positive"], 
               yticklabels=["Actual Negative", "Actual Positive"], ax=ax)
    ax.set_title(f"Confusion Matrix - {titles[i]}")

plt.tight_layout()
plt.show()