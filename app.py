# import packages
import streamlit as st
from dotenv import load_dotenv
import openai
import time
import os
import pandas as pd
from pathlib import Path
import string
from collections import deque
import altair as alt
import plotly.express as px

# load environment variables from .env file
load_dotenv()

# Initialize OpenAI client (move before get_response)
client = openai.OpenAI()


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




st.set_page_config(page_title="GenAI App", page_icon=":robot_face:", layout="wide") 

# Reduce top padding so the page title appears higher
st.markdown(
    """
    <style>
        /* Move main content (including the title) higher */
        .block-container {
            padding-top: 0.25rem;
        }
        /* optional: tighten spacing for header elements */
        .stHeader, header, .css-1v3fvcr {
            margin-top: 0rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

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

# ------------------- Helper function to get dataset path  -----------------
def get_datset_path():
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct path to the CSV file
    csv_path = os.path.join(current_dir, 'data', 'customer_reviews.csv')
    return csv_path 
# --------------------------------------------------------------------------


# ------------------- Function to get sentiment using GenAI  ----------------
@st.cache_data
def get_sentiment(text):
    if not text or pd.isna(text):
        return "Neutral"
    try:
        response = client.responses.create(
            model="gpt-4o",  # Use the latest chat model
            input=[
                {"role": "system", "content": "Classify the sentiment of the following review as exactly one word: Positive, Negative, or Neutral."},
                {"role": "user", "content": f"What's the sentiment of this review? {text}"}
            ],
            temperature=0,  # Deterministic output
            max_output_tokens=100  # Limit response length
        )
        return response.output[0].content[0].text.strip()
    except Exception as e:
        st.error(f"API error: {e}")
        return "Neutral"
# --------------------------------------------------------------------------



# Typewriter effect function
def typewriter(text: str, speed: int):
    tokens = text.split()
    container = st.empty()
    for index in range(len(tokens) + 1):
        curr_full_text = " ".join(tokens[:index])
        container.markdown(curr_full_text)
        time.sleep(1 / speed)

# Main page 
st.title("Features Sentiment Analysis Dashboard")
st.write("A GenAI-powered data processing app for customer reviews analysis.")

# ------------------- Layout: two columns for dataset ingestion and sentiment analysis -------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("Ingest Dataset"):
        try:
            csv_path = get_datset_path()
            df = pd.read_csv(csv_path)
            st.session_state["df"] = df.head(10)  # Store only first 10 rows for performance
            st.success("Dataset ingested successfully!")
        except FileNotFoundError:
            st.error("Dataset not found. Please check the file path ")
        except Exception as e:
            st.error(f"An error occurred: {e}")

with col2:
    if st.button("Analyze Sentiment"):
        if "df" in st.session_state:
            try:
                with st.spinner("Analyzing sentiment..."):
                    st.session_state["df"].loc[:, "Sentiment"] = st.session_state["df"]["SUMMARY"].apply(get_sentiment)
                    st.success("Sentiment analysis completed!")
            except Exception as e:
                st.error(f"Something went wrong: {e}")
        else:
            st.warning("Please ingest the dataset first.")
# --------------------------------------------------------------------------

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
        product = st.selectbox("Choose a product", ["All Products"] + list(st.session_state["df"]["PRODUCT"].unique()))

    else:
        # standardized default to match later checks
        product = "All Products"

st.divider()  # 👈 Draws a horizontal rule

# --- Display dataset preview and visualizations (visible only when dataset is loaded) ---
if "df" in st.session_state:
    # Ensure product variable exists (set by sidebar), otherwise default
    try:
        product  # noqa: F821
    except NameError:
        product = "All Products"

    # Apply product filter
    if product != "All Products":
        filtered_df = st.session_state["df"][st.session_state["df"]["PRODUCT"] == product]
    else:
        filtered_df = st.session_state["df"]

    # show table
    st.dataframe(filtered_df)

    # Visualization using Plotly if sentiment analysis has been performed
    if "Sentiment" in filtered_df.columns:
        st.subheader(f"📊 Sentiment Breakdown for {product}")
        
        # Create Plotly bar chart for sentiment distribution using filtered data
        sentiment_counts = filtered_df["Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']

        # Define custom order and colors
        sentiment_order = ['Negative', 'Neutral', 'Positive']
        sentiment_colors = {'Negative': '#57c8e3', 'Neutral': 'lightgray', 'Positive': 'green'}
        
        # Only include sentiment categories that actually exist in the data
        existing_sentiments = sentiment_counts['Sentiment'].unique()
        filtered_order = [s for s in sentiment_order if s in existing_sentiments]
        filtered_colors = {s: sentiment_colors[s] for s in existing_sentiments if s in sentiment_colors}
        
        # Reorder the data according to our custom order (only for existing sentiments)
        sentiment_counts['Sentiment'] = pd.Categorical(sentiment_counts['Sentiment'], categories=filtered_order, ordered=True)
        sentiment_counts = sentiment_counts.sort_values('Sentiment')
        
        fig = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Count",
            title=f"Distribution of Sentiment Classifications - {product}",
            labels={"Sentiment": "Sentiment Category", "Count": "Number of Reviews"},
            color="Sentiment",
            color_discrete_map=filtered_colors
        )
        fig.update_layout(
            xaxis_title="Sentiment Category",
            yaxis_title="Number of Reviews",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
# --------------------------------------------------------------------------


# --- Display dataset preview and visualizations (visible only when dataset is loaded) ---
if "df" in st.session_state:
    df = st.session_state["df"]

    # Apply product filter from sidebar (product is defined in the sidebar block)
    if product != "All Features":
        filtered_df = df[df["PRODUCT"] == product]
    else:
        filtered_df = df

#   st.subheader("Dataset Preview")
#   st.dataframe(filtered_df.head(50))

    # Visualization(s)
    # 1) Sentiment score distribution (if column exists)
    if "SENTIMENT_SCORE" in filtered_df.columns:
        st.subheader("Sentiment Score Distribution Altair")
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



# --- GenAI Prompting Section ---
        
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


        #with st.chat_message("AI", avatar="assets/VALIDANT_AI_logo_v0.3_Icon.png): 
        typewriter(text=text, speed=speed)






