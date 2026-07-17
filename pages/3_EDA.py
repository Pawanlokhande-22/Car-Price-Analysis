import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="EDA",
    layout="wide"
)

df = pd.read_csv("Cars_cleaned.csv")

st.markdown(
    """
    <style>

    .title{
        font-size:42px;
        font-weight:bold;
        color:#003566;
        text-align:center;
    }

    .subtitle{
        font-size:20px;
        color:gray;
        text-align:center;
    }

    .card{
        background:white;
        padding:20px;
        border-radius:12px;
        box-shadow:0px 3px 12px rgba(0,0,0,.15);
        text-align:center;
    }

    .value{
        font-size:32px;
        color:#ff4b4b;
        font-weight:bold;
    }

    .heading{
        font-size:28px;
        font-weight:bold;
        color:#003566;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='title'>Exploratory Data Analysis</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Used Cars Dataset Analysis</div>", unsafe_allow_html=True)

st.write("")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="card">
        <div class="value">{df.shape[0]}</div>
        <div>Total Records</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card">
        <div class="value">{df.shape[1]}</div>
        <div>Total Features</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="card">
        <div class="value">{df.isna().sum().sum()}</div>
        <div>Missing Values</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="card">
        <div class="value">{df.duplicated().sum()}</div>
        <div>Duplicate Records</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

st.markdown("<div class='heading'>Dataset Preview</div>", unsafe_allow_html=True)

rows = st.slider("Select Rows", 5, 50, 10)

st.dataframe(df.head(rows), use_container_width=True)

st.write("")

st.markdown("<div class='heading'>Dataset Shape</div>", unsafe_allow_html=True)

left, right = st.columns(2)

with left:
    st.metric("Rows", df.shape[0])

with right:
    st.metric("Columns", df.shape[1])

st.write("")

st.markdown("<div class='heading'>Data Types</div>", unsafe_allow_html=True)

dtype_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str)
})

st.dataframe(dtype_df, use_container_width=True)

st.write("")

st.markdown("<div class='heading'>Statistical Summary</div>", unsafe_allow_html=True)

st.dataframe(df.describe(), use_container_width=True)

st.write("")

st.markdown("<div class='heading'>Missing Values</div>", unsafe_allow_html=True)

missing = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": df.isnull().sum().values,
    "Percentage": ((df.isnull().sum()/len(df))*100).round(2).values
})

st.dataframe(missing, use_container_width=True)

st.write("")

st.markdown("<div class='heading'>Duplicate Records</div>", unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    st.metric("Duplicate Rows", df.duplicated().sum())

with c2:
    st.metric("Unique Rows", len(df)-df.duplicated().sum())

st.write("")

st.markdown("<div class='heading'>Dataset Information</div>", unsafe_allow_html=True)

info = pd.DataFrame({
    "Property":[
        "Rows",
        "Columns",
        "Numerical Columns",
        "Categorical Columns",
        "Memory Usage (KB)"
    ],
    "Value":[
        df.shape[0],
        df.shape[1],
        len(df.select_dtypes(include="number").columns),
        len(df.select_dtypes(include="object").columns),
        round(df.memory_usage().sum()/1024,2)
    ]
})

st.dataframe(info, use_container_width=True)

st.write("")

st.markdown("<div class='heading'>Feature Information</div>", unsafe_allow_html=True)

feature = pd.DataFrame({
    "Column":df.columns,
    "Datatype":df.dtypes.astype(str),
    "Unique Values":df.nunique().values,
    "Missing":df.isnull().sum().values
})

st.dataframe(feature, use_container_width=True)

numeric_columns = df.select_dtypes(include="number").columns.tolist()

st.write("")

st.markdown("<div class='heading'>Distribution Analysis</div>", unsafe_allow_html=True)

column = st.selectbox(
    "Select Numerical Column",
    numeric_columns
)

fig = px.histogram(
    df,
    x=column,
    nbins=30,
    template="plotly_white"
)

fig.update_layout(height=500)

st.plotly_chart(fig, use_container_width=True)

st.write("")

st.markdown("<div class='heading'>Box Plot</div>", unsafe_allow_html=True)

fig = px.box(
    df,
    y=column,
    template="plotly_white"
)

fig.update_layout(height=500)

st.plotly_chart(fig, use_container_width=True)

st.write("")

st.markdown("<div class='heading'>Correlation Heatmap</div>", unsafe_allow_html=True)

corr = df.select_dtypes(include="number").corr()

fig = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="Blues",
    aspect="auto"
)

fig.update_layout(height=700)

st.plotly_chart(fig, use_container_width=True)

categorical_columns = df.select_dtypes(include="object").columns.tolist()

if len(categorical_columns) > 0:

    st.write("")

    st.markdown("<div class='heading'>Categorical Analysis</div>", unsafe_allow_html=True)

    category = st.selectbox(
        "Select Categorical Column",
        categorical_columns
    )

    value_counts = (
        df[category]
        .value_counts()
        .head(10)
        .reset_index()
    )

    value_counts.columns = [category, "Count"]

    fig = px.bar(
        value_counts,
        x=category,
        y="Count",
        text="Count",
        template="plotly_white"
    )

    fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)

st.write("")

st.markdown("<div class='heading'>Correlation Table</div>", unsafe_allow_html=True)

st.dataframe(
    df.select_dtypes(include="number").corr().round(2),
    use_container_width=True
)

st.write("")

st.markdown("<div class='heading'>Download Dataset</div>", unsafe_allow_html=True)

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download CSV",
    csv,
    file_name="Cars_cleaned.csv",
    mime="text/csv"
)

st.write("")

st.markdown("<div class='heading'>Key Insights</div>", unsafe_allow_html=True)

st.success(f"Total Records : {df.shape[0]}")
st.success(f"Total Features : {df.shape[1]}")
st.success(f"Numerical Features : {len(df.select_dtypes(include='number').columns)}")
st.success(f"Categorical Features : {len(df.select_dtypes(include='object').columns)}")
st.success(f"Missing Values : {df.isnull().sum().sum()}")
st.success(f"Duplicate Records : {df.duplicated().sum()}")

st.write("")

st.markdown("<div class='heading'>Top 10 Car Brands</div>", unsafe_allow_html=True)

if "Name" in df.columns:

    brand = (
        df["Name"]
        .astype(str)
        .str.split()
        .str[0]
        .value_counts()
        .head(10)
        .reset_index()
    )

    brand.columns = ["Brand", "Count"]

    fig = px.bar(
        brand,
        x="Brand",
        y="Count",
        text="Count",
        color="Count",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

st.write("")

if "Fuel_Type" in df.columns:

    st.markdown("<div class='heading'>Fuel Type Distribution</div>", unsafe_allow_html=True)

    fuel = df["Fuel_Type"].value_counts().reset_index()
    fuel.columns = ["Fuel Type", "Count"]

    fig = px.pie(
        fuel,
        names="Fuel Type",
        values="Count",
        hole=0.45
    )

    st.plotly_chart(fig, use_container_width=True)

st.write("")

if "Transmission" in df.columns:

    st.markdown("<div class='heading'>Transmission Analysis</div>", unsafe_allow_html=True)

    transmission = df["Transmission"].value_counts().reset_index()
    transmission.columns = ["Transmission", "Count"]

    fig = px.bar(
        transmission,
        x="Transmission",
        y="Count",
        text="Count",
        color="Transmission",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

st.write("")

if "Location" in df.columns:

    st.markdown("<div class='heading'>Location Wise Cars</div>", unsafe_allow_html=True)

    location = (
        df["Location"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    location.columns = ["Location", "Count"]

    fig = px.bar(
        location,
        x="Location",
        y="Count",
        text="Count",
        color="Count",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

st.write("")

if "Year" in df.columns:

    st.markdown("<div class='heading'>Year Wise Cars</div>", unsafe_allow_html=True)

    year = (
        df["Year"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    year.columns = ["Year", "Count"]

    fig = px.line(
        year,
        x="Year",
        y="Count",
        markers=True,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

st.write("")

st.markdown("<div class='heading'>Dataset Explorer</div>", unsafe_allow_html=True)

search = st.text_input("Search")

if search:
    result = df.astype(str).apply(
        lambda x: x.str.contains(search, case=False)
    ).any(axis=1)

    st.dataframe(df[result], use_container_width=True)
else:
    st.dataframe(df, use_container_width=True)

st.write("")

st.markdown("<div class='heading'>Final Summary</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.info(f"Dataset contains {df.shape[0]} records.")

with c2:
    st.info(f"Dataset contains {df.shape[1]} features.")

with c3:
    st.info(f"{len(df.select_dtypes(include='number').columns)} numerical features available.")