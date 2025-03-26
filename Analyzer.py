import pandas as pd
import numpy as np
import seaborn as sn

#Helper functions
from sklearn.model_selection import train_test_split
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

featureMatrix = pd.read_csv("./FinalDataSet.csv")
targetVector = featureMatrix.pop("HomeWin")

X_train, X_test, y_train, y_test = train_test_split(featureMatrix, targetVector, test_size=0.1, random_state=0)

logReg = LogisticRegression(max_iter=500, random_state=0)
randForest = RandomForestClassifier(max_depth=10, n_estimators=500, random_state=0)
svc = SVC(random_state=0)

scaledLog = make_pipeline(StandardScaler(), logReg)
scaledForest = make_pipeline(StandardScaler(), randForest)
scaledSVC = make_pipeline(StandardScaler(), svc)

models = {"Logistic Regression": scaledLog,
          "Random Forest": randForest,
          "SVC": scaledSVC}

model_scores = pd.DataFrame({'Model': models.keys(),
                             'Training Accuracy': [None, None, None],
                             'Validation Accuracy': [None, None, None]})

model_scores = model_scores.set_index('Model')
for model in models.keys():
    scores = cross_validate(models[model], X_train, y_train, cv=5, scoring="accuracy", return_train_score=True)
    trainAccuracy = sum(scores["train_score"]) / len(scores["train_score"])
    validationAccuracy = sum(scores["test_score"]) / len(scores["test_score"])
    newRow = [trainAccuracy, validationAccuracy]
    model_scores.loc[model, :] = newRow

print(model_scores)