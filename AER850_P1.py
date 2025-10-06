# all the imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import StratifiedShuffleSplit

#Step 1
data = pd.read_csv("Project 1 Data.csv")

print(data.head(), "\n")
print(data.columns, "\n")

#Step 2

#scatter plots of x, y and z vs. step to understand behaviour withing each class.

#the bar chart shows that the dataset is imbalanced. Some steps have many more 
#samples than others. This means models could become biased towards the majority 
#steps
x = data["X"].to_numpy()
y = data["Y"].to_numpy()
z = data["Z"].to_numpy()
step = data["Step"].to_numpy()

plt.scatter(x, step, alpha=0.5)
plt.title("X vs Step")
plt.xlabel("X")
plt.ylabel("Step")
plt.show()

plt.scatter(y, step, alpha=0.5)
plt.title("Y vs Step")
plt.xlabel("Y")
plt.ylabel("Step")
plt.show()

plt.scatter(z, step, alpha=0.5)
plt.title("Z vs Step")
plt.xlabel("Z")
plt.ylabel("Step")
plt.show()

unique_steps, counts = np.unique(step, return_counts=True)
plt.bar(unique_steps, counts)
plt.title("Number of samples per Step")
plt.xlabel("Step")
plt.ylabel("Count")
plt.show()

#visualizing the count of each step.

#findings show that most of the steps are centered around 7,8,9, meaning that it 
#will lead to bias and requiring addtional optimization.

#Step 3
corr = data[["X", "Y", "Z", "Step"]].corr(method="pearson")
print("Corr Matrix:\n", corr.round(4), "\n")

sb.heatmap(corr, annot=True, cmap="Reds", vmin= -1)
plt.title("correlation matrix")

# X has the strongest negative corrilation with the step, moderately with Y and 
# weakly with Z This means X will have the greatest impact on the model’s 
# predictions, while Y and Z have smaller positive effects. 

#Step 4

#Preparing the data
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=10)

for train_index, test_index in splitter.split(data, data["Step"]):
    strat_data_train = data.loc[train_index]
    strat_data_test  = data.loc[test_index]

y_train = strat_data_train["Step"]
X_train = strat_data_train.drop(columns=["Step"])
y_test  = strat_data_test["Step"]
X_test  = strat_data_test.drop(columns=["Step"])

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

#Logistic Regression

param1 = {
    "C": [0.1, 1, 10],
    "solver": ["lbfgs", "liblinear"],
    "max_iter": [2000]
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
gs1 = GridSearchCV(
    estimator=LogisticRegression(random_state=42),
    param_grid=param1,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    refit=True
)
gs1.fit(X_train, y_train)
mdl1 = gs1.best_estimator_
y_pred_train1 = mdl1.predict(X_test)


#Random Forrest

pipe2 = RandomForestClassifier(random_state=42)
param2 = {
    "n_estimators": [200, 400, 600],
    "max_depth": [None, 6, 10, 16],
    "min_samples_leaf": [1, 2, 4]
}
gs2 = GridSearchCV(pipe2, param_grid=param2, scoring="accuracy", cv=5, n_jobs=-1)
gs2.fit(X_train, y_train)
mdl2 = gs2.best_estimator_
y_pred_train2 = mdl2.predict(X_test)

#SVM

pipe3 = Pipeline([
    ("scaler", StandardScaler()),
    ("mdl3", SVC(probability=True, random_state=42))
])
param3 = {
    "mdl3__kernel": ["linear", "rbf"],
    "mdl3__C": [0.1, 1, 10],
    "mdl3__gamma": ["scale"]
}
gs3 = GridSearchCV(pipe3, param_grid=param3, scoring="accuracy", cv=5, n_jobs=-1)
gs3.fit(X_train, y_train)
mdl3 = gs3.best_estimator_
y_pred_train3 = mdl3.predict(X_test)

#Decision Tree with RandomizedSearchCV

param_dist = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 5, 10, 15, 20, 25],
    "min_samples_split": [2, 3, 5, 7, 10],
    "min_samples_leaf": [1, 2, 3, 4],
    "max_features": [None, "sqrt", "log2"]
}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
rs = RandomizedSearchCV(estimator=DecisionTreeClassifier(random_state=42), 
param_distributions=param_dist, n_iter=25, scoring="accuracy", cv=cv, 
random_state=42, refit=True,
)

# Fit the randomized search
rs.fit(X_train, y_train)
mdl4 = rs.best_estimator_
y_pred_train4 = mdl4.predict(X_test)

from sklearn.metrics import confusion_matrix, precision_score, f1_score

#LR
print("Logistic Regression:")
for i in range(5):
    print("Predicted Step:", y_pred_train1[i], "Real Step:", y_test.iloc[i])

print("Training accuracy:", mdl1.score(X_train, y_train))
print("Test accuracy:", mdl1.score(X_test, y_test))
precision_mdl1 = precision_score(y_test, y_pred_train1, average='macro')
f1_mdl1 = f1_score(y_test, y_pred_train1, average='macro')
print("Precision:", precision_mdl1)
print("F1 Score:", f1_mdl1)
cm_mdl1 = confusion_matrix(y_test, y_pred_train1)
print("Model 1 Confusion Matrix:")
print(cm_mdl1, "\n")

#RF
print("Random Forrest:")
for i in range(5):
    print("Predicted Step:", y_pred_train2[i], "Real Step:", y_test.iloc[i])

print("Training accuracy:", mdl2.score(X_train, y_train))
print("Test accuracy:", mdl2.score(X_test, y_test))
precision_mdl2 = precision_score(y_test, y_pred_train2, average='macro')
f1_mdl2 = f1_score(y_test, y_pred_train2, average='macro')
print("Precision:", precision_mdl2)
print("F1 Score:", f1_mdl2)
cm_mdl2 = confusion_matrix(y_test, y_pred_train2)
print("Model 2 Confusion Matrix:")
print(cm_mdl2, "\n")

#SVM
print("SVM:")
for i in range(5):
    print("Predicted Step:", y_pred_train3[i], "Real Step:", y_test.iloc[i])

print("Training accuracy:", mdl3.score(X_train, y_train))
print("Test accuracy:", mdl3.score(X_test, y_test))
precision_mdl3 = precision_score(y_test, y_pred_train3, average='macro')
f1_mdl3 = f1_score(y_test, y_pred_train3, average='macro')
print("Precision:", precision_mdl3)
print("F1 Score:", f1_mdl3)
cm_mdl3 = confusion_matrix(y_test, y_pred_train3)
print("Model 3 Confusion Matrix:")
print(cm_mdl3, "\n")

#DT
print("Decision Tree with RandomizedSearchCV:")
for i in range(5):
    print("Predicted Step:", y_pred_train4[i], "Real Step:", y_test.iloc[i])

print("Training accuracy:", mdl4.score(X_train, y_train))
print("Test accuracy:", mdl4.score(X_test, y_test))
precision_mdl4 = precision_score(y_test, y_pred_train4, average='macro')
f1_mdl4 = f1_score(y_test, y_pred_train4, average='macro')
print("Precision:", precision_mdl4)
print("F1 Score:", f1_mdl4)
cm_mdl4 = confusion_matrix(y_test, y_pred_train4)
print("Model 4 Confusion Matrix:")
print(cm_mdl4, "\n")

# Model Stacked with LR and RF
from sklearn.ensemble import StackingClassifier

base_models = [('LR', mdl1), ('RF', mdl2)]

mdl5 = LogisticRegression(max_iter=2000, random_state=42)

stack_model = StackingClassifier(estimators=base_models, 
final_estimator=mdl5, cv=5,n_jobs=-1, passthrough=True)

stack_model.fit(X_train, y_train)

y_pred_stack = stack_model.predict(X_test)

print("Model Stacked with LR and RF:")
for i in range(5):
    print("Predicted Step:", y_pred_stack[i], "Real Step:", y_test.iloc[i])

print("Training accuracy:", stack_model.score(X_train, y_train))
print("Test accuracy:", stack_model.score(X_test, y_test))
prec_stack = precision_score(y_test, y_pred_stack, average='macro', zero_division=0)
f1_stack = f1_score(y_test, y_pred_stack, average='macro', zero_division=0)
print("Precision:", prec_stack)
print("F1 Score:", f1_stack)
cm_stack = confusion_matrix(y_test, y_pred_stack)
print("Stack Model Confusion Matrix:")
print(cm_stack, "\n")

# Joblib format

import joblib

joblib.dump(mdl3, "P1_model.joblib")