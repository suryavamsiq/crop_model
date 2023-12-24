# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 10:56:34 2023

@author: c surya vamsi
"""

import pickle
import numpy as np
loaded_model=pickle.load(open('C:/Users/c surya vamsi/1trainedmodel.sav','rb'))
import streamlit as st

def prediction(input_data):
    newValues1=np.array(input_data)
    newValues1=newValues1.reshape(1,-1)
    return ('The recommended model for your data is',loaded_model.predict(newValues1))
    
def main():
    st.title("CROP RECOMMENDATION SYSTEM WEB APP")
    #N	P	K	temperature	humidity	ph	rainfall	label
    N=st.text_input('enter value of N')
    P=st.text_input('enter value of P')
    K=st.text_input('enter value of k')
    temp=st.text_input('enter value of temprature')
    humidity=st.text_input('enter value of humidity')
    ph=st.text_input('enter value of ph')
    rainfall=st.text_input('enter value of rainfall')
    
    output=''
    if st.button('recommended crop'):
        output=prediction([N,P,K,temp,humidity,ph,rainfall])
    
    st.success(output) 


if __name__=='__main__':
    main()
    
    
    
