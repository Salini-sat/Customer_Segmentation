# Week 7: Support Vector Machine (SVM) - Handwritten Digit Classification

# 1. Import Required Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import SGDClassifier

# 2. Load the MNIST-like Digits Dataset
digits = load_digits()
X = digits.data        # Feature matrix (1797 samples, 64 features)
y = digits.target     # Target labels (0 to 9)

# 3. Split Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create SVM Model (Non-linear Kernel)
svm_model = SVC(kernel='rbf', gamma='scale', C=1.0)

# 5. Train the Model
svm_model.fit(X_train, y_train)

# 6. Predict on Test Data
y_pred = svm_model.predict(X_test)

# 7. Evaluate Model Performance

# A. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of SVM Model:", accuracy)

# B. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# C. Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 8. Display Some Sample Predictions
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f"Predicted: {svm_model.predict([digits.data[i]])[0]}")
    plt.axis('off')
plt.suptitle("SVM Predictions on Handwritten Digits")
plt.show()

# Linear SVM using SGD (epoch-wise training)
linear_svm = SGDClassifier(loss='hinge', random_state=42)

train_acc, val_acc = [], []
train_loss, val_loss = [], []

for epoch in range(10):
    linear_svm.partial_fit(X_train, y_train, classes=np.unique(y))

    # Predictions
    y_train_pred = linear_svm.predict(X_train)
    y_test_pred = linear_svm.predict(X_test)

    # Accuracy
    train_acc.append(accuracy_score(y_train, y_train_pred))
    val_acc.append(accuracy_score(y_test, y_test_pred))

    # Hinge loss (manual)
    train_loss.append(np.mean(np.maximum(0, 1 - y_train * y_train_pred)))
    val_loss.append(np.mean(np.maximum(0, 1 - y_test * y_test_pred)))

# Plot Accuracy
plt.plot(train_acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Linear SVM Accuracy per Epoch")
plt.legend()
plt.show()

# Plot Loss
plt.plot(train_loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Linear SVM Loss per Epoch")
plt.legend()
plt.show()

train_acc_rbf, val_acc_rbf = [], []

for epoch in range(10):
    rbf_svm = SVC(kernel='rbf', gamma='scale')
    rbf_svm.fit(X_train, y_train)

    train_acc_rbf.append(rbf_svm.score(X_train, y_train))
    val_acc_rbf.append(rbf_svm.score(X_test, y_test))

# Plot Accuracy
plt.plot(train_acc_rbf, label="Training Accuracy")
plt.plot(val_acc_rbf, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("RBF SVM Accuracy per Epoch")
plt.legend()
plt.show()