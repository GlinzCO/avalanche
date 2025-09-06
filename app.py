# import packages
from dotenv import load_dotenv
import openai
import streamlit as st
import time
import os
import pandas as pd
from pathlib import Path
import string
from collections import deque
import altair as alt


# ------------------- RATE LIMITING SETUP (move to top!) -------------------
max_requests = 5
time_window = 60  # seconds
request_times = deque()

def rate_limited():
    now = time.time()
    # Remove timestamps older than time_window
    while request_times and now - request_times[0] > time_window:
        request_times.popleft()
    if len(request_times) < max_requests:
        request_times.append(now)
        return False  # Not rate limited
    else:
        return True   # Rate limited
# --------------------------------------------------------------------------

# load environment variables from .env file
load_dotenv()


st.set_page_config(page_title="GenAI App", page_icon=":robot_face:", layout="wide") 


# CSS get Roboto
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css?family=Roboto:400,700&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"]  {
            font-family: 'Roboto', sans-serif;
        }
        textarea, .stTextArea textarea {
            background-color: #FFF !important;
            color: #000 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Function to clean text
def clean_text(text):
    """
    Cleans input text by removing punctuation, converting to lowercase, and stripping whitespace.
    """
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Strip leading/trailing whitespace
    text = text.strip()
    return text

# Initialize OpenAI client (move before get_response)
client = openai.OpenAI()

def get_response(user_promt, temperature, max_tokens):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": user_promt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response

def get_datset_path():
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct path to the CSV file
    csv_path = os.path.join(current_dir, 'data', 'customer_reviews.csv')
    return csv_path 

def typewriter(text: str, speed: int):
    tokens = text.split()
    container = st.empty()
    for index in range(len(tokens) + 1):
        curr_full_text = " ".join(tokens[:index])
        container.markdown(curr_full_text)
        time.sleep(1 / speed)

# Main page 
st.title("Daniel's GenAI App")
st.write("This is a simple app that uses OpenAI's GPT-4o model.")

# Layout: two columns
col1, col2 = st.columns(2)

with col1:
    if st.button("Ingest Dataset"):
        try:
            csv_path = get_datset_path()
            st.session_state["df"] = pd.read_csv(csv_path)
            st.success("Dataset ingested successfully!")
        except FileNotFoundError:
            st.error("Dataset not found. Please check the file path ")
        except Exception as e:
            st.error(f"An error occurred: {e}")

with col2:
    if st.button("Parse Reviews"):
        if "df" in st.session_state:
            st.session_state["df"]['CLEANED_SUMMARY'] = st.session_state["df"]['SUMMARY'].apply(clean_text)
            st.success("Reviews parsed and cleaned successfully!")
    else:
        st.warning("Please ingest the dataset first.")

# Sidebar: show logo and dataset filter (filter only visible when dataset is loaded)
with st.sidebar:
    # moved logo up by using inline HTML with a negative top margin
    st.markdown(
        """
        <div style="margin-top:-50px; padding-top:0;">
            <img src="https://www.glinz.co/wp-content/uploads/2017/11/GCO_Logo_v1_WebHeaderWhite_PDF7.png"
                 style="max-width:240px; width:100%; display:block;" alt="logo"/>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Define product here so it's available when we render the main area below
    if "df" in st.session_state:
        st.subheader("Filter by Feature")
        product = st.selectbox(
            "Choose a feature",
            ["All Features"] + list(st.session_state["df"]["PRODUCT"].unique()),
            index=0
        )
    else:
        product = "All Features"

st.divider()  # 👈 Draws a horizontal rule

# --- Display dataset preview and visualizations (visible only when dataset is loaded) ---
if "df" in st.session_state:
    df = st.session_state["df"]

    # Apply product filter from sidebar (product is defined in the sidebar block)
    if product != "All Features":
        filtered_df = df[df["PRODUCT"] == product]
    else:
        filtered_df = df

    st.subheader("Dataset Preview")
    st.dataframe(filtered_df.head(50))

    # Visualization(s)
    # 1) Sentiment score distribution (if column exists)
    if "SENTIMENT_SCORE" in filtered_df.columns:
        st.subheader("Sentiment Score Distribution")
        hist = (
            alt.Chart(filtered_df)
            .mark_bar(color="#57c8e3")
            .encode(
                alt.X("SENTIMENT_SCORE:Q", bin=alt.Bin(maxbins=20), title="Sentiment Score"),
                y="count():Q",
                tooltip=["SENTIMENT_SCORE"]
            )
            .properties(width=700, height=300)
        )
        st.altair_chart(hist, use_container_width=True)

        # 2) Average sentiment by product (use full df grouping so axis includes all products)
        st.subheader("Average Sentiment by Product")
        mean_df = df.groupby("PRODUCT", as_index=False)["SENTIMENT_SCORE"].mean()
        bar = (
            alt.Chart(mean_df)
            .mark_bar(color="#57c8e3")
            .encode(
                x=alt.X("PRODUCT:N", sort="-y", title="Product"),
                y=alt.Y("SENTIMENT_SCORE:Q", title="Average Sentiment"),
                tooltip=["PRODUCT", "SENTIMENT_SCORE"]
            )
            .properties(width=700, height=300)
        )
        st.altair_chart(bar, use_container_width=True)
    else:
        st.info("Column 'SENTIMENT_SCORE' not found — showing raw table only.")
# Layout: two columns for user input and model settings
col1, col2 = st.columns(2, gap="large")

with col1:
    user_promt = st.text_area(
        "Enter your prompt:",
        height=150
    )
    submit = st.button("Submit Prompt")

with col2:
    temperature = st.slider(
        "Model temperature (creativity):",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Controls randomness of the output. Lower values make output more focused and deterministic."
    )
    max_tokens = st.slider("Select max output tokens:", 50, 500, 100, 10)

if submit:
    if rate_limited():
        st.warning(f"Rate limit reached: Only {max_requests} requests allowed every {time_window} seconds.")
    else:
        timer_placeholder = st.empty()
        start_time = time.time()
        with st.spinner("Generating response..."):
            response = get_response(user_promt, temperature, max_tokens)
            output_text = response.choices[0].message.content  # Correct for OpenAI chat API
        elapsed = time.time() - start_time
        timer_placeholder.write(f"Model responded in {elapsed:.2f} seconds.")

        #Sample Example
        text =  output_text
        speed = 10


        #with st.chat_message("AI", avatar="assets/VALIDANT_AI_logo_v0.3_Icon.png"): 
        typewriter(text=text, speed=speed)






