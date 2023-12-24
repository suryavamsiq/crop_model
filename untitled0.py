# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 10:56:34 2023
@author: c surya vamsi
"""

import pickle
import numpy as np
import streamlit as st

def load_model():
    try:
        loaded_model = pickle.load(open('1trainedmodel.sav', 'rb'))
        return loaded_model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

def prediction(loaded_model, input_data):
    try:
        # Convert input_data to float
        newValues1 = np.array(input_data, dtype=float)
        newValues1 = newValues1.reshape(1, -1)
        
        prediction_result = loaded_model.predict(newValues1)
        print("Prediction result:", prediction_result)
        return ('The recommended model for your data is', prediction_result)
    except Exception as e:
        return f"Error during prediction: {e}"

def main():
    st.title("CROP RECOMMENDATION SYSTEM WEB APP")
    
    nitrogen = st.text_input('Enter value of N')
    phosphorus = st.text_input('Enter value of P')
    potassium = st.text_input('Enter value of K')
    temp = st.text_input('Enter value of temperature')
    humidity = st.text_input('Enter value of humidity')
    ph = st.text_input('Enter value of pH')
    rainfall = st.text_input('Enter value of rainfall')
    
    loaded_model = load_model()
    
    output = ''
    if loaded_model is not None and st.button('Recommend Crop'):
        output = prediction(loaded_model, [nitrogen, phosphorus, potassium, temp, humidity, ph, rainfall])
    
    st.success(output) 

if __name__ == '__main__':
    main()
