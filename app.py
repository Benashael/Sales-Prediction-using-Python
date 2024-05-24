import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Sales Prediction using Python", page_icon="🛒")

st.title("Sales Prediction 🛒")

# Load the dataset
def load_data():
    df = pd.read_csv("advertising.csv")
    return df

df = load_data()

# Display the dataset
st.header("Advertising Dataset")
st.write(df.head())

# Select features and target variable
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# User input features
st.header("User Input Features")

# Options
def user_input_features():
    TV = st.slider('TV (in thousands)', float(df.TV.min()), float(df.TV.max()), float(df.TV.mean()))
    Radio = st.slider('Radio (in thousands)', float(df.Radio.min()), float(df.Radio.max()), float(df.Radio.mean()))
    Newspaper = st.slider('Newspaper (in thousands)', float(df.Newspaper.min()), float(df.Newspaper.max()), float(df.Newspaper.mean()))
    return {'TV': TV, 'Radio': Radio, 'Newspaper': Newspaper}

input_df = pd.DataFrame(user_input_features(), index=[0])

model_type = st.selectbox("Select Model", ["Random Forest", "Linear Regression", "K - Nearest Neighbors"])

# Model
if model_type == "Random Forest":
    model = RandomForestRegressor()
elif model_type == "Linear Regression":
    model = LinearRegression()
elif model_type == "K - Nearest Neighbors":
    model = KNeighborsRegressor()

# Fit the model
model.fit(X_train, y_train)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.subheader('Prediction')
    st.write(f"Predicted Sales (in units): **{prediction[0]:.2f}**")
