import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
from datetime import time  # Import the time class

# Load the trained model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('./my_saved_model')
tokenizer = DistilBertTokenizerFast.from_pretrained('./my_saved_model')

# Label encoder to map class IDs back to class names (crime descriptions)
crime_classes = ['BATTERY - SIMPLE ASSAULT', 'BURGLARY', 'BURGLARY FROM VEHICLE', 'THEFT PLAIN - PETTY ($950 & UNDER)', 'VEHICLE - STOLEN']

# Streamlit application layout
st.title("Crime Class Prediction Model")
st.write("Enter the following information to predict the crime class probabilities:")

# Accept user inputs
area = st.number_input("Enter Area Code", min_value=1, max_value=22, step=1)
selected_time = st.time_input("Select the time (HH:MM)", value=time(12, 0))
day_of_week = st.selectbox("Select the Day of the Week", ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])

# Combine input features into a single string as required by the model
combined_features = f"Area: {area} Hour: {selected_time} DayOfWeek: {day_of_week}"

# When the user presses the predict button, process the input
if st.button('Predict Crime Class'):
    # Tokenize the input
    inputs = tokenizer(combined_features, return_tensors="pt", truncation=True, padding=True)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Convert logits to probabilities using softmax
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy().flatten()

    # Display the probabilities for each class
    st.write("Crime Class Probabilities:")
    for crime_class, prob in zip(crime_classes, probabilities):
        st.write(f"{crime_class}: {prob * 100:.2f}%")
