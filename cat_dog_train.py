import glob
import cv2
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ================== SETTINGS ==================

IMG_SIZE = 64

# پوشه‌ای که همین فایل cat_dog_train.py داخلش است
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(BASE_DIR, "Q1", "train")
TEST_PATH  = os.path.join(BASE_DIR, "Q1", "test")

print("TRAIN_PATH =", TRAIN_PATH)
print("TEST_PATH  =", TEST_PATH)

classes = [("Cat", 0), ("Dog", 1)]  # Cat = 0, Dog = 1

# ================== LOAD TRAIN DATA ==================

data_list = []
label_list = []

for class_name, label in classes:
    pattern = os.path.join(TRAIN_PATH, class_name, "*")
    print("Searching:", pattern)
    files = glob.glob(pattern)
    print(f"  found {len(files)} files")
    for address in files:
        img = cv2.imread(address)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255.0
        img = img.flatten()
        data_list.append(img)
        label_list.append(label)

X = np.array(data_list)
y = np.array(label_list)

print("Train data shape:", X.shape, "Labels shape:", y.shape)

if X.shape[0] == 0:
    raise RuntimeError("هیچ تصویری در TRAIN_PATH پیدا نشد! ساختار پوشه‌ها و نام‌ها (Q1/train/Cat, Q1/train/Dog) را چک کن.")

# ================== TRAIN / VAL SPLIT ==================

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ================== SCALER ==================

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_val_sc = sc.transform(X_val)

# ================== فقط KNN و Logistic Regression ==================

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
}

best_model_name = None
best_model = None
best_acc = 0.0

for name, model in models.items():
    print(f"\nTraining model: {name}")
    model.fit(X_train_sc, y_train)
    preds = model.predict(X_val_sc)
    acc = accuracy_score(y_val, preds)
    print(f"Validation accuracy for {name} = {acc * 100:.2f}%")

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_model_name = name

print("\n==============================")
print(f"Best model on validation set: {best_model_name} with acc = {best_acc * 100:.2f}%")
print("==============================")

val_preds = best_model.predict(X_val_sc)
print("\nClassification report on validation set:")
print(classification_report(y_val, val_preds, target_names=["Cat", "Dog"]))

print("Confusion matrix on validation set:")
print(confusion_matrix(y_val, val_preds))

# ================== TEST SET EVALUATION ==================

test_data = []
test_labels = []

for class_name, label in classes:
    pattern = os.path.join(TEST_PATH, class_name, "*")
    print("Searching test:", pattern)
    files = glob.glob(pattern)
    print(f"  found {len(files)} files in test for {class_name}")
    for address in files:
        img = cv2.imread(address)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255.0
        img = img.flatten()
        test_data.append(img)
        test_labels.append(label)

X_test = np.array(test_data)
y_test = np.array(test_labels)

print("Test data shape:", X_test.shape, "Labels shape:", y_test.shape)

X_test_sc = sc.transform(X_test)
test_preds = best_model.predict(X_test_sc)
test_acc = accuracy_score(y_test, test_preds)
print(f"\nTest accuracy ({best_model_name}) = {test_acc * 100:.2f}%")

print("\nClassification report on test set:")
print(classification_report(y_test, test_preds, target_names=["Cat", "Dog"]))

print("Confusion matrix on test set:")
print(confusion_matrix(y_test, test_preds))

# ================== SAVE BEST MODEL & SCALER ==================

joblib.dump(best_model, "cat_dog_best_model.z")
joblib.dump(sc, "cat_dog_scaler.z")

print("\nBest model and scaler saved as 'cat_dog_best_model.z' and 'cat_dog_scaler.z'")
