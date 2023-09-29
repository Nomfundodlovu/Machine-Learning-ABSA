import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler





model_load_path = "mlr_model.pkl"
with open(model_load_path,'rb') as file:
    unpickled_model = pickle.load(file)


# Define the custom CSS style for the frame and border
frame_style = """
<style>
.frame {
    border: 1px solid black;
    padding: 10px;
}
</style>
"""
data = pd.read_csv("absa_df.csv")

# Apply the custom style
st.markdown(frame_style, unsafe_allow_html=True)

# Add the title inside a frame
st.markdown('<div class="frame"><h1>Phoenix Prediction App</h1></div>', unsafe_allow_html=True)

# Add other content inside a frame
st.markdown('<div class="frame"><p> </p></div>', unsafe_allow_html=True)

# Add the form elements inside a frame
st.markdown('<div class="frame">', unsafe_allow_html=True)

income_group = st.selectbox('Select your income group', ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20','21'])
st.write('Your income group is:', income_group)

occupational_status_code = st.selectbox('Select the occupational status', ['1', '2', '3', '4', '5', '6', '7'])
st.write('Your occupational status is:', occupational_status_code)

account_type_code = st.selectbox('Select your account type',[14, 13,  6,  7,  0, 12,  5,  3, 10,  9,  8,  2, 11, 15,  1,  4])
st.write('Your account type code is:', account_type_code)

account_number = st.number_input('Enter your account number', step=1, format='%d')
st.write('Your account number is:', account_number)

record_date_y = st.number_input('Select date')
st.write('Your record date_y is:', record_date_y)

age = st.slider('Select your age', 0, 100, 25)
st.write('Your age is:', age)

customer_identifier = st.number_input('Enter your customer identifier')
st.write('Your customer identifier is:', customer_identifier)

if st.button('Submit'):
    st.write('Button Clicked!')

st.markdown('</div>', unsafe_allow_html=True)  # Close the frame

# encoder = OneHotEncoder(drop="first")
# encoded_data = pd.get_dummies(data, columns=["INCOME_GROUP_CODE", "OCCUPATIONAL_STATUS_CODE", "ACCOUNT_TYPE_CODE", "AGE"])

# X = encoded_data.drop("DECLARED_NET_INCOME", axis=1)
# y = encoded_data["DECLARED_NET_INCOME"]

# #Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if st.button:
    # Create a DataFrame with user input
    user_input = pd.DataFrame({
        "INCOME_GROUP_CODE": [income_group],
        "OCCUPATIONAL_STATUS_CODE": [occupational_status_code],
        "ACCOUNT_TYPE_CODE": [account_type_code],
        "ACCOUNT_NUMBER": [account_number],
        "RECORD_DATE_y": [record_date_y],
        "AGE": [age]
    })
    
    scale = StandardScaler()

    scaledX = scale.fit_transform(user_input)

    # Perform one-hot encoding for user input
    # encoded_user_input = pd.get_dummies(user_input, columns=["INCOME_GROUP_CODE", "OCCUPATIONAL_STATUS_CODE", "ACCOUNT_TYPE_CODE"])
    # encoded_user_input = encoded_user_input.reindex(columns=X.columns, fill_value=0)
    
    # model = LinearRegression()
    # model.fit(X_train, y_train)

    # Make predictions using the trained model
    income_prediction = unpickled_model.predict(scaledX)

    # Display the predicted income
    st.write("Predicted Income:", income_prediction[0])

# Add the image
image_url = "/Users/damac44/Desktop/absa/galleria-105-e1668613757228-1024x751.jpg"  # Replace with your image URL
st.image(image_url, caption="Image Caption")
