# train.py

from preprocessing import X_train, X_test, Y_train, Y_test
from sklearn.ensemble import RandomForestClassifier
# Add the new metrics here:
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
#  Train the model using the imported data
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=2,criterion="entropy")
model.fit(X_train, Y_train)

predictions = model.predict(X_test)

# 1. Accuracy
acc = accuracy_score(Y_test, predictions)
print(f"Accuracy: {acc * 100:.2f}%\n")

print("--- Confusion Matrix ---")
print(confusion_matrix(Y_test, predictions))
print("\n--- Classification Report ---")
print(classification_report(Y_test, predictions))
#Export the trained model for Streamlit
joblib.dump(model, 'loan_model.pkl')
print("Model saved successfully as loan_model.pkl")