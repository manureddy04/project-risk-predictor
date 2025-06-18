import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Project Risk Predictor", layout="centered")

st.title("🧠 Project Risk Prediction Tool")
st.markdown("Enter the project details below to predict the risk level.")

def generate_model():
    np.random.seed(42)
    n = 300
    data = {
        "project_duration": np.random.randint(8, 52, n),
        "team_size": np.random.randint(3, 15, n),
        "avg_sprint_velocity": np.random.randint(10, 60, n),
        "requirement_change_percent": np.random.randint(0, 60, n),
        "delays_percent": np.random.randint(0, 80, n),
        "tech_stack_complexity": np.random.choice(["Low", "Medium", "High"], n),
        "team_experience_level": np.round(np.random.uniform(0.5, 8, n), 1),
        "client_communication_score": np.random.randint(1, 11, n),
    }

    def assign_risk(row):
        if row["delays_percent"] > 50 or row["requirement_change_percent"] > 40:
            return "High"
        elif row["delays_percent"] > 30 or row["requirement_change_percent"] > 20:
            return "Medium"
        else:
            return "Low"

    df = pd.DataFrame(data)
    df["risk_level"] = df.apply(assign_risk, axis=1)

    le = LabelEncoder()
    df["tech_stack_complexity"] = le.fit_transform(df["tech_stack_complexity"])
    X = df.drop("risk_level", axis=1)
    y = LabelEncoder().fit_transform(df["risk_level"])

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model, le

model, tech_encoder = generate_model()

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        project_duration = st.slider("📅 Project Duration (weeks)", 8, 52, 24)
        team_size = st.slider("👥 Team Size", 3, 15, 6)
        avg_sprint_velocity = st.slider("🏃 Sprint Velocity", 10, 60, 30)
        requirement_change_percent = st.slider("🔁 Requirement Change %", 0, 60, 10)
    with col2:
        delays_percent = st.slider("⏱️ Delays %", 0, 80, 20)
        tech_stack_complexity = st.selectbox("💻 Tech Stack Complexity", ["Low", "Medium", "High"])
        team_experience_level = st.slider("👨‍💻 Team Experience (years)", 0.5, 8.0, 3.0, step=0.1)
        client_communication_score = st.slider("📞 Client Communication (1-10)", 1, 10, 7)

    submitted = st.form_submit_button("🔍 Predict Risk")

if submitted:
    input_df = pd.DataFrame([{
        "project_duration": project_duration,
        "team_size": team_size,
        "avg_sprint_velocity": avg_sprint_velocity,
        "requirement_change_percent": requirement_change_percent,
        "delays_percent": delays_percent,
        "tech_stack_complexity": tech_encoder.transform([tech_stack_complexity])[0],
        "team_experience_level": team_experience_level,
        "client_communication_score": client_communication_score,
    }])

    prediction = model.predict(input_df)[0]
    label_map = {0: "High", 1: "Low", 2: "Medium"}
    inv_label_map = {v: k for k, v in label_map.items()}
    final_label = [k for k, v in inv_label_map.items() if v == prediction][0]

    st.success(f"📢 **Predicted Project Risk Level:** {final_label}")
    st.info("This prediction is based on simulated historical project patterns.")

st.markdown("---")
st.caption("Built for final year project — BTech Data Analytics")
