import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib
import numpy as np

# Load the features and labels data
features = pd.read_csv('features.csv')
labels = pd.read_csv('labels.csv')

# Ensure labels are numeric
labels = pd.to_numeric(labels['label'], errors='coerce')

# Check if features and labels match in length
if len(features) != len(labels):
    raise ValueError("The number of samples in features and labels do not match.")

# Ensure that 'filename' column is removed from features (if present)
features = features.drop(columns=['filename'], errors='ignore')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Handle class imbalance by assigning weights to each class
classes = np.unique(y_train)  # Automatically detect all unique classes in y_train
class_weights = compute_class_weight('balanced', classes=classes, y=y_train.values.flatten())
class_weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}

# Initialize the RandomForestClassifier
model = RandomForestClassifier(class_weight=class_weight_dict, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'final_model.pkl')
print("Model training complete. Model saved as final_model.pkl.")
