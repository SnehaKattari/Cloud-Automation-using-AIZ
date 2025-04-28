import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# Load the dataset
df = pd.read_csv(r"C:\Users\HP\Downloads\archive\nsl-kdd\KDDTest-21.txt")  # Update with the correct path

# Handle missing values
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Check after filling missing values
print("Columns after filling missing values:", df.columns)

# Assuming 'guess_passwd' is the correct target column
target_column = 'guess_passwd'  # Update if needed

# Prepare features and labels
X = df.drop(columns=[target_column])  # Drop the target column from the features
y = df[target_column]  # The target column

# Encode categorical features and labels
label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)
X = X.apply(label_encoder.fit_transform)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using multiple metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)  # Handling zero-division
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)  # Handling zero-division
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)  # Handling zero-division
conf_matrix = confusion_matrix(y_test, y_pred)

# Get the predicted probabilities for each class
y_score = model.predict_proba(X_test)

# Ensure that the number of classes in y_test matches the number of columns in y_score
print(f"Unique classes in y_test: {len(set(y_test))}")
print(f"Number of classes in y_score: {y_score.shape[1]}")

# Compute ROC-AUC score if classes match
if y_test.shape[0] == y_score.shape[1]:
    roc_auc = roc_auc_score(y_test, y_score, multi_class='ovr', average='macro')
    print(f"ROC-AUC (macro): {roc_auc:.2f}")
else:
    print("Mismatch between number of classes in y_test and y_score. Please check the model training process.")

# Print other evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision (macro): {precision:.2f}")
print(f"Recall (macro): {recall:.2f}")
print(f"F1-Score (macro): {f1:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
