import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load model
model = joblib.load("models/best_diabetes_model.pkl")

# Header with custom styling
st.markdown(
    "<h1 style='text-align: center; color: #FFA500;'>Diabetes Prediction App</h1>",
    unsafe_allow_html=True,
)
st.write("This app predicts the likelihood of diabetes based on medical parameters.")

# Sidebar with app info
st.sidebar.title("How it Works")
st.sidebar.info("""
1. Enter your medical details using the sliders below.
2. The app uses a machine learning model trained on diabetes data.
3. The result is a probability score predicting diabetes risk.
""")

# Input sliders
pregnancies = st.slider("Pregnancies", 0, 20, 2)
glucose = st.slider("Glucose Level", 0, 200, 120)
blood_pressure = st.slider("Blood Pressure", 0, 140, 80)
skin_thickness = st.slider("Skin Thickness", 0, 100, 25)
insulin = st.slider("Insulin Level", 0, 300, 85)
bmi = st.slider("BMI", 0.0, 50.0, 28.5)
diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.slider("Age", 0, 100, 30)

# Prepare input data
input_data = np.array(
    [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]]
)

# Tabs for navigation
tabs = st.tabs(["Input Data", "Results"])

# Input Data Tab
with tabs[0]:
    st.write("### Your Input Data")
    input_df = pd.DataFrame(
        input_data,
        columns=["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", "Insulin", "BMI", "Diabetes Pedigree", "Age"],
    )
    st.write(input_df)

# Results Tab
with tabs[1]:
    st.write("### Prediction Results")
    if st.button("Predict"):
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

        if prediction[0] == 1:
            st.write(f"**Diabetic** with a probability of **{probability:.2f}**")
        else:
            st.write(f"**Non-Diabetic** with a probability of **{1 - probability:.2f}**")

        # Visualization
        fig, ax = plt.subplots()
        ax.pie(
            [probability, 1 - probability],
            labels=["Diabetic", "Non-Diabetic"],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.axis("equal")
        st.pyplot(fig)

        # Model performance metrics
        st.write("### Model Performance")
        st.write("Accuracy: **85%**")
        st.write("ROC-AUC Score: **0.89**")

        # Save results to CSV
        result_df = pd.DataFrame(
            {
                "Pregnancies": [pregnancies],
                "Glucose": [glucose],
                "Blood Pressure": [blood_pressure],
                "Skin Thickness": [skin_thickness],
                "Insulin": [insulin],
                "BMI": [bmi],
                "Diabetes Pedigree": [diabetes_pedigree],
                "Age": [age],
                "Prediction": ["Diabetic" if prediction[0] == 1 else "Non-Diabetic"],
                "Probability": [probability],
            }
        )
        st.download_button(
            label="Download Results as CSV",
            data=result_df.to_csv(index=False),
            file_name="diabetes_prediction.csv",
            mime="text/csv",
        )
