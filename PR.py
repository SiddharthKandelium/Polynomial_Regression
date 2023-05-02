#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error

def get_user_input():
    # Get user input for degree of polynomial
    degree = st.sidebar.slider("Select degree of polynomial", min_value=1, max_value=10, step=1)
    
    # Get user input for Excel files to use as input and output data
    file_paths = st.sidebar.file_uploader("Select Excel files for input and output data", type=["xlsx", "xls"], accept_multiple_files=True)
    input_file_path = None
    output_file_path = None
    if file_paths:
        for file_path in file_paths:
            if "input" in file_path.name.lower():
                input_file_path = file_path
            elif "output" in file_path.name.lower():
                output_file_path = file_path
    
    # Get user input for rows and columns to use as input and output data
    if input_file_path is not None:
        input_df = pd.read_excel(input_file_path)
        input_cols = st.sidebar.multiselect("Select input columns for polynomial regression", input_df.columns.tolist())
        input_rows = st.sidebar.multiselect("Select rows for input data", input_df.index.tolist())
        X = input_df.loc[input_rows, input_cols].values
    else:
        X = None
        
    if output_file_path is not None:
        output_df = pd.read_excel(output_file_path)
        output_cols = st.sidebar.multiselect("Select output columns for polynomial regression", output_df.columns.tolist())
        output_rows = st.sidebar.multiselect("Select rows for output data", output_df.index.tolist())
        y = output_df.loc[output_rows, output_cols].values
    else:
        y = None
    
    return degree, X, y

def main():
    st.set_page_config(page_title="Mercury Polynomial Regression", page_icon=":chart_with_upwards_trend:")
    st.title("Polynomial Regression with Mercury")
    
    # Get user input
    degree, X, y = get_user_input()
    
    # Perform polynomial regression and show mean absolute error
    if X is not None and y is not None:
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        linreg = LinearRegression()
        linreg.fit(X_poly, y)
        y_pred = linreg.predict(X_poly)
        mae = mean_absolute_error(y, y_pred)
        st.write("Mean Absolute Error:", mae)
    
    # Get user input for second Excel file to predict on
    predict_file_path = st.file_uploader("Select Excel file to predict on", type=["xlsx", "xls"])
    if predict_file_path is not None:
        predict_df = pd.read_excel(predict_file_path)
        predict_cols = st.multiselect("Select columns to predict on", predict_df.columns.tolist())
        predict_rows = st.multiselect("Select rows for prediction", predict_df.index.tolist())
        X_predict = predict_df.loc[predict_rows, predict_cols].values
        X_predict_poly = poly.transform(X_predict)
        y_predict = linreg.predict(X_predict_poly)
        st.write("Predicted values:", y_predict)

if __name__ == "__main__":
    main()

