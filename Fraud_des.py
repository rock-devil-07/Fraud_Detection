import streamlit as st
import pickle
import re

# Load the vectorizer and model from pickle files
with open('fraud_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('fraud_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit App
st.title("Fraud Detection App")
st.write("Predict if a transaction is fraudulent based on its description.")

# Input from the user (the text area is more flexible for copy-paste)
description = st.text_area("Enter the transaction description (more than 2 words, no special characters except commas):", "")

# Function to validate input
def validate_input(text):
    # Check for special characters (only allow commas)
    if re.search(r'[^a-zA-Z0-9, ]', text):
        return False
    # Check if there are more than 2 words
    if len(text.split()) <= 2:
        return False
    return True

# Predict button
if st.button("Predict"):
    if description.strip():
        # Validate the description
        if validate_input(description):
            # Transform the input using the vectorizer
            transformed_description = vectorizer.transform([description])
            
            # Make a prediction using the model
            prediction = model.predict(transformed_description)[0]
            
            # Display the result in the format "label: 0" or "label: 1"
            st.write(f"Label: {prediction}")
            
            # Display fraud or normal message
            if prediction == 1:
                st.error("🚨 This transaction is likely FRAUDULENT.")
            else:
                st.success("✅ This transaction is likely NORMAL.")
        else:
            if len(description.split()) <= 2:
                st.warning("Please enter a description with more than 2 words.")
            else:
                st.warning("Please enter a valid description with no special characters (except commas).")
    else:
        st.warning("Please enter a transaction description.")

# Footer
st.write("---")
st.write("Thanks for Using Stremlit")

