import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the dataset
voice_data = pd.read_excel("V_15MIN_VOICE.xlsx", skiprows=1)

# Drop unnecessary columns
voice_data.drop(columns=["load_datetime"], inplace=True)

# Convert date and datetime columns to appropriate data types
date_columns = ["event_date", "event_datetime"]
voice_data[date_columns] = voice_data[date_columns].apply(pd.to_datetime)

# Set Streamlit theme
st.set_page_config(
    page_title="AI-ML Integration with Streamlit",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")


def display_dataset():
    # Remove 'nan' values from the 'bssname_region' column
    voice_data_cleaned = voice_data.dropna(subset=['bssname_region'])

    # Display the voice dataset
    st.subheader("Voice Data")
    st.write(voice_data_cleaned)

def display_overall():
    # Remove 'nan' values from the 'bssname_region' column
    voice_data_cleaned = voice_data.dropna(subset=['bssname_region'])

    # Data analysis and visualization
    st.subheader("Call Duration Distribution")
    plt.hist(voice_data_cleaned["call_duration"], bins=20)
    plt.xlabel("Call Duration")
    plt.ylabel("Frequency")
    plt.title("Distribution of Call Duration")
    st.pyplot()

    st.subheader("Call Success Rate")
    success_rate = voice_data_cleaned["successfully_setup_call"].mean() * 100
    st.write(f"Call Success Rate: {success_rate:.2f}%")

def display_filter():
    # Remove 'nan' values from the 'bssname_region' column
    voice_data_cleaned = voice_data.dropna(subset=['bssname_region'])

    # Filtering options
    bssname_regions = voice_data_cleaned["bssname_region"].unique()
    bssname_region = st.sidebar.selectbox("Select BSS Name Region", bssname_regions)

    # Filter the data based on the selected BSS
    filtered_by_bss = voice_data_cleaned[voice_data_cleaned["bssname_region"] == bssname_region]

    # Get the available RAT types for the selected BSS
    rat_types = filtered_by_bss["rattype"].unique()
    rattype = st.sidebar.selectbox("Select RAT Type", rat_types)

    # Filter the data further based on the selected RAT
    filtered_data = filtered_by_bss[filtered_by_bss["rattype"] == rattype]

    # Insights based on filtering
    st.subheader("Filtered Data Insights")

    st.subheader("Call Duration Distribution (Filtered)")
    plt.hist(filtered_data["call_duration"], bins=20)
    plt.xlabel("Call Duration")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Call Duration ({bssname_region} - {rattype})")
    st.pyplot()

    st.subheader("Call Success Rate (Filtered)")
    filtered_success_rate = filtered_data["successfully_setup_call"].mean() * 100
    st.write(f"Call Success Rate ({bssname_region} - {rattype}): {filtered_success_rate:.2f}%")

def display_prediction():
    # Remove 'nan' values from the 'bssname_region' column
    voice_data_cleaned = voice_data.dropna(subset=['bssname_region'])

    # Forecasting
    st.subheader("Call Duration Forecasting")

    # Prepare the data for forecasting
    siteid_data = voice_data_cleaned.groupby("siteid")["call_duration"].mean().reset_index()
    siteid_data = siteid_data.nlargest(15, "call_duration")  # Select top 15 sites

    # Plot the forecasted values
    plt.figure(figsize=(10, 6))
    plt.bar(siteid_data["siteid"], siteid_data["call_duration"])
    plt.xlabel("Site ID")
    plt.ylabel("Call Duration")
    plt.title("Forecasted Call Durations (Top 15 sites)")
    plt.xticks(rotation=90, fontsize=8)  # Adjust x-axis font size and rotation
    st.pyplot()

    # Show forecasted values in table for all sites
    st.subheader("Forecasted Call Durations (All Sites)")
    st.write(voice_data_cleaned.groupby("siteid")["call_duration"].mean().reset_index())

def contact_form():
    st.write("---")
    st.header("Get In Touch With Us!")
    st.write("##")

    # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
    form = """
    <form action="https://formsubmit.co/kjyuaan@maxis.com.my" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here" required></textarea>
        <button type="submit">Send</button>
    </form>
    """
    st.markdown(form, unsafe_allow_html=True)

def main():
    st.title("Data Visualization and Forecasting Using AI-ML")

    # Create a sidebar
    st.sidebar.image("logo.png", use_column_width=True)

    # Create a container for the main content
    main_container = st.container()

    # Create tabs
    tabs = ["📦 Dataset", "📊 Overall Call Distribution", "🛒 Filtered Data Insights", "📈 Sites' Call Prediction"]
    selected_tab = st.sidebar.selectbox("Select Tab", tabs)

    if selected_tab == "📦 Dataset":
        with main_container:
            display_dataset()
            contact_form()
    elif selected_tab == "📊 Overall Call Distribution":
        with main_container:
            display_overall()
            contact_form()
    elif selected_tab == "🛒 Filtered Data Insights":
        with main_container:
            display_filter()
            contact_form()
    elif selected_tab == "📈 Sites' Call Prediction":
        with main_container:
            display_prediction()
            contact_form()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Powered by Streamlit")

if __name__ == "__main__":
    main()