# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 10:56:34 2023

@author: c surya vamsi
"""

import pickle
import numpy as np
try:
    loaded_model = pickle.load(open('1trainedmodel.sav', 'rb'))
except Exception as e:
    print(f"Error loading the model: {e}")
    # Add more detailed information about the exception if needed
    # For example, print(traceback.format_exc()) to print the full traceback

#loaded_model=pickle.load(open('1trainedmodel.sav','rb'))
import streamlit as st

def prediction(input_data):
    newValues1=np.array(input_data)
    newValues1=newValues1.reshape(1,-1)
    try:
        prediction_result = loaded_model.predict(newValues1)
        return ('The recommended model for your data is', prediction_result)
    except Exception as e:
        print(f"Error during prediction: {e}")

    #return ('The recommended model for your data is',loaded_model.predict(newValues1))
    
def main():
    st.title("CROP RECOMMENDATION SYSTEM WEB APP")
    #N	P	K	temperature	humidity	ph	rainfall	label
    Nitrogen=st.text_input('enter value of N')
    Phosporous=st.text_input('enter value of P')
    potassium=st.text_input('enter value of k')
    temp=st.text_input('enter value of temprature')
    humidity=st.text_input('enter value of humidity')
    ph=st.text_input('enter value of ph')
    rainfall=st.text_input('enter value of rainfall')
    
    output=''
    if st.button('recommended crop'):
        output=prediction([Nitrogen,Phosporous,potassium,temp,humidity,ph,rainfall])
    
    st.success(output) 


if __name__=='__main__':
    main()
    
    
    
