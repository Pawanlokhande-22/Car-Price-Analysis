import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Conclusion",
    layout="wide"
)

df = pd.read_csv("Cars_cleaned.csv")

st.markdown("""
<style>

.title{
    font-size:42px;
    font-weight:bold;
    color:#003566;
    text-align:center;
}

.heading{
    font-size:30px;
    font-weight:bold;
    color:#003566;
    margin-top:25px;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div class='title'>Project Conclusion</div>",
    unsafe_allow_html=True
)

st.write("")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Total Records", df.shape[0])

with c2:
    st.metric("Total Features", df.shape[1])

with c3:
    st.metric("Missing Values", df.isnull().sum().sum())

with c4:
    st.metric("Duplicate Records", df.duplicated().sum())

st.write("")

st.markdown(
    "<div class='heading'>Project Summary</div>",
    unsafe_allow_html=True
)

with st.container(border=True):

    st.markdown("""

- The used cars dataset was successfully cleaned and prepared for analysis.

- Missing values and duplicate records were checked before analysis.

- Numerical and categorical variables were explored using different statistical methods.

- Interactive charts were created to understand the dataset effectively.

- Correlation analysis helped identify relationships between important variables.

- The dataset is suitable for business analysis as well as machine learning applications.

""")

st.write("")

st.markdown(
    "<div class='heading'>Key Findings</div>",
    unsafe_allow_html=True
)

with st.container(border=True):

    st.markdown("""

1. Most vehicles belong to a few well-known automobile brands.

2. Vehicle age and kilometers driven have a strong impact on resale value.

3. Fuel type and transmission influence customer preferences.

4. The cleaned dataset contains no missing values and no duplicate records.

5. Correlation analysis provides useful insights for predictive modeling.

""")

st.write("")

st.markdown(
    "<div class='heading'>Business Recommendations</div>",
    unsafe_allow_html=True
)

with st.container(border=True):

    st.markdown("""

- Compare vehicle prices before making a purchase decision.

- Maintain vehicles regularly to improve resale value.

- Dealers should use historical market data for better pricing strategies.

- Data-driven decision making can improve customer satisfaction and business growth.

""")

st.write("")

st.markdown(
    "<div class='heading'>Future Scope</div>",
    unsafe_allow_html=True
)

with st.container(border=True):

    st.markdown("""

- Develop a Machine Learning model for used car price prediction.

- Deploy the prediction model using Streamlit.

- Integrate real-time used car market data.

- Improve prediction accuracy by adding more vehicle features.

- Build an interactive dashboard with advanced filtering options.

""")

st.write("")

st.markdown(
    "<div class='heading'>Final Conclusion</div>",
    unsafe_allow_html=True
)

with st.container(border=True):

    st.markdown("""

The Used Cars Data Analysis project successfully explored the dataset using Python, Pandas, Plotly and Streamlit. Exploratory Data Analysis helped identify important trends, relationships and business insights. The project demonstrates how data analysis can support better decision-making for buyers, sellers and automobile businesses. Furthermore, the cleaned dataset can be used to build accurate machine learning models for predicting used car prices.

""")

st.write("")

st.success("Project Completed Successfully")