import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = pd.read_csv("dataset/traffic_data.csv")

# Convert time into numeric hour
data["hour"] = data["time"].str.split(":").str[0].astype(int)

# Encode congestion labels
label_encoder = LabelEncoder()
data["congestion_encoded"] = label_encoder.fit_transform(data["congestion_level"])

# Features and target
X = data[["hour", "vehicle_count", "speed"]]
y = data["congestion_encoded"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Test prediction
sample = [[18, 150, 25]]
prediction = model.predict(sample)

print("Predicted Congestion Level:", label_encoder.inverse_transform(prediction)[0])
