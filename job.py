import streamlit as st
import pandas as pd
import pickle
from PIL import Image
image=Image.open(r"C:\Users\vijit\Downloads\download.png")
st.image(image)

# Load the trained model (Pipeline)
with open(r'C:\Users\vijit\Hr_Analysis.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit app
st.title('Job Change Prediction for Data Scientists')

# User inputs for prediction
st.header('Input the Details')

# Define the categories as they were during model training
gender_categories = ['Male', 'Female']
relevant_experience_categories = ['Has relevent experience', 'No relevent experience']
enrolled_university_categories = ['no_enrollment', 'Full time course', 'Part time course']
education_level_categories = ['Primary School', 'High School', 'Graduate', 'Masters', 'Phd']
major_discipline_categories = ['STEM', 'Humanities', 'Business Degree', 'Arts', 'Other']
company_type_categories = ['Pvt Ltd', 'Funded Startup', 'Public Sector', 'Early Stage Startup', 'NGO']

# User input widgets
gender = st.selectbox('Gender', gender_categories)
relevent_experience = st.selectbox('Relevant Experience', relevant_experience_categories)
enrolled_university = st.selectbox('Enrolled University', enrolled_university_categories)
education_level = st.selectbox('Education Level', education_level_categories)
major_discipline = st.selectbox('Major Discipline', major_discipline_categories)
company_type = st.selectbox('Company Type', company_type_categories)
experience = st.number_input('Years of Experience', min_value=0, max_value=25, value=5)
training_hours = st.number_input('Training Hours', min_value=0, max_value=1000, value=50)

# Create a dataframe from the user inputs
input_data = pd.DataFrame({
    'gender': [gender],
    'relevent_experience': [relevent_experience],
    'enrolled_university': [enrolled_university],
    'education_level': [education_level],
    'major_discipline': [major_discipline],
    'company_type': [company_type],
    'experience': [experience],
    'training_hours': [training_hours]
})

# Ensure that the input data is consistent with the model's expected categories
# Using the .astype('category') ensures that it matches the category dtype if the model expects that
input_data['gender'] = pd.Categorical(input_data['gender'], categories=gender_categories)
input_data['relevent_experience'] = pd.Categorical(input_data['relevent_experience'], categories=relevant_experience_categories)
input_data['enrolled_university'] = pd.Categorical(input_data['enrolled_university'], categories=enrolled_university_categories)
input_data['education_level'] = pd.Categorical(input_data['education_level'], categories=education_level_categories)
input_data['major_discipline'] = pd.Categorical(input_data['major_discipline'], categories=major_discipline_categories)
input_data['company_type'] = pd.Categorical(input_data['company_type'], categories=company_type_categories)

# Predict button
if st.button('Predict'):
    # Predict the outcome using the trained pipeline model
    prediction = model.predict(input_data)

    # Display the result
    if prediction == 1:
        st.success('The candidate is likely to change jobs.')
    else:
        st.error('The candidate is unlikely to change jobs.')
