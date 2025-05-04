import streamlit as st
import numpy as np
import cv2
from roboflow import Roboflow
import supervision as sv
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import torch.nn as nn
from check_cycle import predict_value

# Load the pre-trained model
model = joblib.load(r"D:\Aircraft\battery_rul_model.pkl")

# Streamlit UI
st.set_page_config(layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar options
st.sidebar.title('Options')
option = st.sidebar.radio('Select an option:', ('Aircraft Monitor', 'Battery Life Estimation', 'Jet Cycles Prediction'))

# Main content
st.title('Aircraft maintenance Prediction using remaining useful prognostics and aircraft fuselage defect detection')

# Description
st.write("""
This app predicts various parameters related to jets, including aircraft monitoring, battery life estimation, and jet cycles prediction.
""")


st.sidebar.markdown('<div style="margin-top: 15px;">&nbsp;</div>', unsafe_allow_html=True)

if option == 'Aircraft Monitor':
    st.title('Aircraft Monitor')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        
        rf = Roboflow(api_key="Gqf1hrF7jdAh8EsbOoTM")
        project = rf.workspace().project("innovation-hangar-v2")
        model = project.version(1).model

        result = model.predict(image, confidence=20, overlap=30).json()

        labels = [item["class"] for item in result["predictions"]]

        detections = sv.Detections.from_roboflow(result)

        label_annotator = sv.LabelAnnotator()
        bounding_box_annotator = sv.BoxAnnotator()

        annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        st.image(annotated_image, caption='Detected Objects', use_column_width=True)

elif option == 'Battery Life Estimation':
    st.title('Battery Life Estimation')

    # Input form for battery life prediction
    cycle_index = st.number_input('Cycle Index', value=0)
    discharge_time = st.number_input('Discharge Time (s)', value=0.0)
    decrement_time = st.number_input('Decrement 3.6-3.4V Time (s)', value=0.0)
    max_voltage = st.number_input('Max. Voltage Discharged (V)', value=0.0)
    min_voltage = st.number_input('Min. Voltage Charged (V)', value=0.0)
    time_at_415v = st.number_input('Time at 4.15V (s)', value=0.0)
    time_constant_current = st.number_input('Time Constant Current (s)', value=0.0)
    charging_time = st.number_input('Charging Time (s)', value=0.0)

    # Button to predict battery life
    if st.button('Predict Battery Life'):
        predicted_life = model.predict([[cycle_index, discharge_time, decrement_time, max_voltage, min_voltage, time_at_415v, time_constant_current, charging_time]])
        st.write("Predicted Remaining Battery Life:", predicted_life[0], "cycles")

elif option == 'Jet Cycles Prediction':
    st.title("Jet cycles")

    features = ['cycle',
     '(LPC outlet temperature) (◦R)',
     '(LPT outlet temperature) (◦R)',
     '(HPC outlet pressure) (psia)',
     '(HPC outlet Static pressure) (psia)',
     '(Ratio of fuel flow to Ps30) (pps/psia)',
     '(Bypass Ratio) ',
     '(Bleed Enthalpy)',
     '(High-pressure turbines Cool air flow)',
     '(Low-pressure turbines Cool air flow)']

    st.image("Turbofan-operation-lbp.png",caption='Jet Diagram', use_column_width=True)
    st.image('pairplot.png','Pairplot', use_column_width=True)

    in_dict = {}
    for feature in features:
        in_dict[feature] = int(st.number_input(feature))

    a,b= st.columns(2)
    Predict = a.button("Predict")

    if Predict:
        inputs = list(in_dict.values())
        
        value = predict_value(inputs)

        st.write(f"### Number of Cycles left: {value}")
