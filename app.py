import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ----------------------------
# 1️⃣ Load training and test datasets
# ----------------------------
train_df = pd.read_csv("StudentsPerformance_cleaned new123.csv")
test_df = pd.read_csv("test.csv")

st.title("🎓 Student Performance Prediction System")

st.write("✅ Training Data Loaded:", train_df.shape)
st.write("✅ Test Data Loaded:", test_df.shape)

# ----------------------------
# 2️⃣ Encode categorical columns
# ----------------------------
le = LabelEncoder()
for col in ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']:
    if col in train_df.columns:
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))

# ----------------------------
# 3️⃣ Train and Evaluate Model
# ----------------------------
X_train = train_df[['math score', 'reading score', 'writing score']]
y_train = train_df['performance_level'] if 'performance_level' in train_df.columns else train_df.iloc[:, -1]

X_test = test_df[['math score', 'reading score', 'writing score']]
y_test = test_df['performance_level'] if 'performance_level' in test_df.columns else test_df.iloc[:, -1]

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
st.success(f"✅ Model trained successfully! Accuracy: {accuracy}%")

# ----------------------------
# 4️⃣ Streamlit Front-End for Prediction
# ----------------------------
st.write("---")
st.subheader("🔮 Predict a Student's Performance")

math = st.number_input("Enter Math Score", min_value=0, max_value=100, value=70)
reading = st.number_input("Enter Reading Score", min_value=0, max_value=100, value=70)
writing = st.number_input("Enter Writing Score", min_value=0, max_value=100, value=70)

if st.button("Predict Performance Level"):
    new_data = pd.DataFrame([[math, reading, writing]],
                            columns=['math score', 'reading score', 'writing score'])
    prediction = model.predict(new_data)
    st.success(f"🎯 Predicted Performance Level: {prediction[0]}")

st.write("---")
st.caption("Built using Streamlit and RandomForestClassifier")
