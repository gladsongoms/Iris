import streamlit as st 
import joblib 

#load the joblib model 
model_nb = joblib.load('Iris')

#user input 
st.title("Iris Flower Classification ")
ip = st.text_input(" Enter -(sepal_length	sepal_width	petal_length	petal_width	) in this format --eg.[[5.1, 3.5, 1.4, 0.2]] ")

#predict if the entered message is spam or ham 
op = model_nb.predict([ip])
if st.button('PREDICT'):
  st.title(op[0])  #prints the output as spam or ham  

  
                
