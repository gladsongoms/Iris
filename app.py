
import numpy as np

import joblib
import streamlit as st


# loading the saved model
loaded_model = joblib.load('Iris1')


# creating a function for Prediction

def Iris_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] =='Iris-setosa'):
      return 'Iris-setosa flower'
    elif(prediction[0] =='Iris-versicolor'):
      return 'Iris-versicolor flower'
    else:
      return 'Iris-virginica'
  
    
  
def main():
    
    
    # giving a title
    st.title('Iris flower Prediction Web App')
    
    
    # getting the input data from the user
    
    
    sepal_length= st.text_input('Length of Sepal')
    sepal_width= st.text_input(' Enter sepal width')
    petal_length= st.text_input('Enter Length of Petal')
    petal_width= st.text_input('Enter petal width')
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Result'):
        diagnosis = Iris_prediction([[sepal_length,	sepal_width,	petal_length,	petal_width	]])
        
       
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()
    
