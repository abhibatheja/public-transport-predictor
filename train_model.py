import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score

# Load dataset
df = pd.read_csv("transport_data.csv")

# Encoders
le_time = LabelEncoder()
le_day = LabelEncoder()
le_crowd = LabelEncoder()

df["time_of_day"] = le_time.fit_transform(df["time_of_day"])
df["day_type"] = le_day.fit_transform(df["day_type"])
df["crowd_level"] = le_crowd.fit_transform(df["crowd_level"])

# Features and targets
X = df[["distance_km", "time_of_day", "day_type"]]
y_time = df["travel_time_min"]
y_crowd = df["crowd_level"]

# Split
X_train, X_test, y_time_train, y_time_test = train_test_split(X, y_time, test_size=0.2, random_state=42)
_, _, y_crowd_train, y_crowd_test = train_test_split(X, y_crowd, test_size=0.2, random_state=42)

# Models
time_model = LinearRegression()
crowd_model = DecisionTreeClassifier()

# Train
time_model.fit(X_train, y_time_train)
crowd_model.fit(X_train, y_crowd_train)

# Evaluate
time_pred = time_model.predict(X_test)
crowd_pred = crowd_model.predict(X_test)

print("Travel Time MAE:", mean_absolute_error(y_time_test, time_pred))
print("Crowd Accuracy:", accuracy_score(y_crowd_test, crowd_pred))

# Save models
pickle.dump(time_model, open("time_model.pkl", "wb"))
pickle.dump(crowd_model, open("crowd_model.pkl", "wb"))

# Save encoders
pickle.dump(le_time, open("le_time.pkl", "wb"))
pickle.dump(le_day, open("le_day.pkl", "wb"))
pickle.dump(le_crowd, open("le_crowd.pkl", "wb"))

print("Models and encoders saved successfully!")