import streamlit as st
import pandas as pd
import pickle

# Load the saved model and output label encoder
with open('random_forest_autism3.pickle', 'rb') as f:
    model = pickle.load(f)

with open('output_autism3.pickle', 'rb') as f:
    output_encoder = pickle.load(f)

# Define the function for making predictions
def predict(df):
    # Reindex the input dataframe to make sure it has the same columns as the training data
    df = df.reindex(columns=model.feature_importances_.index, fill_value=0)

    # Make the prediction and map output to labels
    prediction = model.predict(df)
    output = output_encoder.inverse_transform(prediction)

    return output[0]

def calculate_score(input_data):
    score = 0
    score_dict = {
        'Always': 0,
        'Usually': 0,
        'Sometimes': 1,
        'Rarely': 1,
        'Never': 1,
        'Very Easy': 0,
        'Quite Easy': 0,
        'Quite Difficult': 1,
        'Very Difficult': 1,
        'Impossible': 1,
        'Many times a day': 0,
        'A few times a day': 0,
        'A few times a week': 1,
        'Less than once a week': 1,
        'Very typical':0,
        'Quite typical':0,
        'Never': 1,
        'Slightly unusual':1,
        'Very unusual':1,
        'My child does not speak':1,
        
    }

    for key in input_data.keys():
        score += score_dict[input_data[key][0]]

    return score


# Define the Streamlit app
st.title('Simplified Autism Screening Tool')
st.write("This app uses 8 inputs to predict the autism traits using a "
" machine learning model built on the Quantitative Checklist for Autism in Toddlers (Q-CHAT10) dataset."
"The app is using the most important and effective features from the QCHAT10"
"using the Random Forest feature importance with Recursive Feature Elimination method."
"The goal is to provide a more  efficient and simplified alternative to traditional autism  screening and to support early decision making with reliable"
"and accurate results.")

st.write('Enter the following information:')

# Collect input from user
# Get user input
Name = st.text_input("Name :")
Age_Mons = st.text_input("Age : **_format use in month_**.")
Sex = st.selectbox("Gender", ("Female", "Male"))
A1 = st.selectbox("1.Does your child look at you when you call his/her name?", ("Always", "Usually","Sometimes","Rarely","Never"))
A2 = st.selectbox("2.How easy is it for you to get eye contact with your child?", ("Very Easy", "Quite Easy", "Quite Difficult", "Very Difficult","Impossible"))
#A3 = st.selectbox("Does your child point to indicate that he/she wants something?", ("Yes", "No"))
A4 = st.selectbox("3.Does your child point to share interest with you? e.g pointing at an interesting sights", ("Many times a day", "A few times a day", "A few times a week", "Less than once a week", "Never" ))
A5 = st.selectbox("4.Does your child pretend? e.g care for dolls, talk on the toy phone", ("Many times a day", "A few times a day", "A few times a week", "Less than once a week", "Never" ))
A6 = st.selectbox("5.Does your child follow where you're looking?", ("Many times a day", "A few times a day", "A few times a week", "Less than once a week", "Never" ))
A7 = st.selectbox("6.If you or someone else in the family is visibly upset, does your child show signs of wanting to comfort them? eg stroking hair, hugging them",
                   ("Always", "Usually","Sometimes","Rarely","Never"))
A8 = st.selectbox("7.Would you desribe your child's first word as:", ("Very typical", "Quite typical","Slightly unusual","Very unusual","My child does not speak"))
A9 = st.selectbox("8.Does your child use simple gestures? e.g wave goodbye", ("Many times a day", "A few times a day", "A few times a week", "Less than once a week", "Never" ))
#A10 = st.selectbox("Does your child stare at nothing with no apparent purpose?", ("Many times a day", "A few times a day", "A few times a week", "Less than once a week", "Never" ))

# Create a dataframe from the input data
input_data = pd.DataFrame({
    'A1': [A1],
    'A2': [A2],
    'A4': [A4],
    'A5': [A5],
    'A6': [A6],
    'A7': [A7],
    'A8': [A8],
    'A9': [A9]
})

# Calculate the score
score = calculate_score(input_data)

# Make the prediction and display the result
if st.button('Predict'):
    if score >= 3:
        st.write('Prediction: Your child has autism traits.')
    else:
        st.write('Prediction: Your child does not have autism traits.')







