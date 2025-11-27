import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ===================== LOAD TRAIN AND TEST CSV =====================

train_df = pd.read_csv(r"C:\Users\sheed\OneDrive\Desktop\cvi\ComputerVisionFall2025\Assignment-2\Q2\mnist_train.csv", header=None)
test_df  = pd.read_csv(r"C:\Users\sheed\OneDrive\Desktop\cvi\ComputerVisionFall2025\Assignment-2\Q2\mnist_test.csv", header=None)

# First column = label, others = pixels
y_train = train_df.iloc[:, 0].values        
X_train = train_df.iloc[:, 1:].values       

y_test = test_df.iloc[:, 0].values          
X_test = test_df.iloc[:, 1:].values        

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)

# ===================== SCALING =====================

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

# ===================== MODELS =====================

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "MLP": MLPClassifier(hidden_layer_sizes=(128,), max_iter=100)
}

best_model_name = None
best_model = None
best_acc = 0

for name, model in models.items():
    print(f"\nTraining model: {name}")
    model.fit(X_train_sc, y_train)

    preds = model.predict(X_test_sc)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy for {name}: {acc * 100:.2f}%")

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_model_name = name

# ===================== RESULTS =====================

print("\n==============================")
print(f"Best model: {best_model_name} (Accuracy = {best_acc * 100:.2f}%)")
print("==============================")

preds = best_model.predict(X_test_sc)
print("\nClassification report:")
print(classification_report(y_test, preds))

print("Confusion matrix:")
print(confusion_matrix(y_test, preds))

# ===================== SAVE BEST MODEL =====================

joblib.dump(best_model, "mnist_best_model.z")
joblib.dump(sc, "mnist_scaler.z")

print("\nModel and scaler saved.")
