import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("heart.csv")

print(df)
print(df.head())
print(df.info())
print(df.shape)

print("Column Names")
print(df.columns)

print("Missing Values")
print(df.isnull().sum())

print("Statistical Summary")
print(df.describe())

print("Heart Disease Count")
print(df["target"].value_counts())

# Heart disease count
sns.countplot(x="target", data=df)
plt.title("Heart Disease Count")
plt.xlabel("Target")
plt.ylabel("Count")
plt.show()

# Age distribution
sns.histplot(df["age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

sns.countplot(x="sex", hue="target", data=df)
plt.title("Heart Disease by Gender")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show()

sns.countplot(x="cp", hue="target", data=df)
plt.title("Heart Disease by Chest Pain Type")
plt.xlabel("Chest Pain Type")
plt.ylabel("Count")
plt.show()

sns.scatterplot(x="age", y="chol", hue="target", data=df)
plt.title("Age vs Cholesterol")
plt.xlabel("Age")
plt.ylabel("Cholesterol")
plt.show()

sns.scatterplot(x="age", y="thalach", hue="target", data=df)
plt.title("Age vs Maximum Heart Rate")
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()

# Correlation heatmap
corr = df.corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Used: Logistic Regression")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("Classification Report")
print(classification_report(y_test, y_pred))

sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# New patient prediction
new_patient = pd.DataFrame({
    "age": [55],
    "sex": [1],
    "cp": [0],
    "trestbps": [140],
    "chol": [250],
    "fbs": [0],
    "restecg": [1],
    "thalach": [150],
    "exang": [0],
    "oldpeak": [1.5],
    "slope": [1],
    "ca": [0],
    "thal": [2]
})

new_patient = scaler.transform(new_patient)

prediction = model.predict(new_patient)
probability = model.predict_proba(new_patient)

print("New Patient Prediction")

if prediction[0] == 1:
    print("Patient may have Heart Disease")
else:
    print("Patient may NOT have Heart Disease")

print("No Disease Probability:", round(probability[0][0] * 100, 2), "%")
print("Disease Probability:", round(probability[0][1] * 100, 2), "%")

print("Conclusion")
print("This project analyzes heart disease patient data.")
print("Charts show age, gender, chest pain, cholesterol and heart rate analysis.")
print("Correlation heatmap shows relation between different health features.")
print("Logistic Regression is used for heart disease prediction.")
print("The model predicts whether a patient may have heart disease or not.")