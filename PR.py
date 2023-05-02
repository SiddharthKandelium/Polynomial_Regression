import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Polynomial Regression App")

st.title("Polynomial Regression App")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    st.subheader("Input Data")
    st.write(df)

    rows = st.multiselect("Select Rows", df.index.tolist())
    columns = st.multiselect("Select Columns", df.columns.tolist())

    if len(rows) > 0 and len(columns) > 0:
        X = df.loc[rows, columns].values
        y = df.loc[rows, "Output"].values

        degree = st.slider("Polynomial Degree", 1, 10, 2)
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        y_pred = model.predict(X_poly)
        mae = mean_absolute_error(y, y_pred)

        st.subheader("Model Results")
        st.write(f"Polynomial Degree: {degree}")
        st.write(f"Mean Absolute Error: {mae:.2f}")

    else:
        st.warning("Please select at least one row and one column.")

st.markdown("---")

uploaded_file2 = st.file_uploader("Choose another file")

if uploaded_file2 is not None:
    df2 = pd.read_excel(uploaded_file2)

    st.subheader("Input Data")
    st.write(df2)

    rows2 = st.multiselect("Select Rows", df2.index.tolist())
    columns2 = st.multiselect("Select Columns", df2.columns.tolist())

    if len(rows2) > 0 and len(columns2) > 0:
        X2 = df2.loc[rows2, columns2].values

        X2_poly = poly.transform(X2)
        y2_pred = model.predict(X2_poly)

        st.subheader("Model Results")
        st.write(f"Mean Absolute Error: {mean_absolute_error(df2.loc[rows2, 'Output'].values, y2_pred):.2f}")

    else:
        st.warning("Please select at least one row and one column.")
