import streamlit as st
import pandas as pd

# import the excel file
file = st.file_uploader("Upload Excel file", type=["xlsx"])

if file is not None:
    # read the data into a DataFrame
    df = pd.read_excel(file)
    
    # display the DataFrame
    st.dataframe(df)
    
    # select columns and rows
    selected_cols = st.dataframe().multiselect("Select Columns", df.columns.tolist())
    selected_rows = st.dataframe().multiselect("Select Rows", df.index.tolist())
    
    # filter the DataFrame based on selected columns and rows
    filtered_df = df.loc[selected_rows, selected_cols]
    
    # display the filtered DataFrame
    st.dataframe(filtered_df)
