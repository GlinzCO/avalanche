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
        except FileExistsError:
            st.error("Dataset not found. Pleae check the file path ") 

with col2:
    if st.button("Parse Reviews"):
        if "df" in st.session_state:
            st.session_state["df"]['CLEANED_SUMMARY'] = st.session_state["df"]['SUMMARY'].apply(clean_text)
            st.success("Reviews parsed and cleaned successfully!")
    else:
        st.warning("Please ingest the dataset first.")

# Display the dataset if ingested
if "df" in st.session_state:
    st.subheader("Filter by Feature:")
    col_a, col_b = st.columns([1,1])
    with col_a:
        product = st.selectbox(
            "Choose a feature",
            ["All Features"] + list(st.session_state["df"]["PRODUCT"].unique())
        )
    st.subheader(f"Dataset Preview:")

    if product != "All Features":
        filtered_df = st.session_state["df"][st.session_state["df"]["PRODUCT"] == product]
    else:
        filtered_df = st.session_state["df"]

    st.dataframe(filtered_df.head(10))  # Display first 10 rows of the filtered dataset

    st.subheader("Sentiment Score by Feature:")
    grouped = st.session_state["df"].groupby("PRODUCT")["SENTIMENT_SCORE"].mean() 
    st.bar_chart(grouped)


# Plotting with Altair
    chart = alt.Chart(filtered_df).mark_bar().add_selection(
            alt.selection_interval()
        ).encode(
            alt.X("SENTIMENT_SCORE:Q", bin=alt.Bin(maxbins=10), title="Sentiment Score"),
            alt.Y("count():Q", title="Frequency"), tooltip=["count():Q"]
        ).properties(
            width=600,
            height=400,
            title="Sentiment Score Distribution"
        )
    st.altair_chart(chart, use_container_width=True)











# Sidebar: just with a logo 
with st.sidebar:
    st.image("https://www.glinz.co/wp-content/uploads/2017/11/GCO_Logo_v1_WebHeaderWhite_PDF7.png", use_container_width=False)

st.divider()  # 👈 Draws a horizontal rule

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

        # Name using AI key from environment variable
        #with st.chat_message("AI", avatar="assets/VALIDANT_AI_logo_v0.3_Icon.png"): 
        typewriter(text=text, speed=speed)






