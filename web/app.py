import streamlit as st
from transformers import pipeline
import torch
import time

# Set page config
st.set_page_config(
    page_title="SmartNote HF - Online Text Summarizer",
    page_icon="📝",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        max-width: 1000px;
        margin: 0 auto;
        padding: 1rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .summary-box {
        background-color: #f8f9fa;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("📝 SmartNote HF")
st.markdown("### AI-Powered Text Summarization Tool")
st.markdown("Quickly summarize long articles, research papers, or any text with our AI-powered summarizer.")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox(
        "Select Model",
        ["facebook/bart-large-cnn", "google/pegasus-xsum"],
        help="Choose the model for summarization. BART is good for general purpose, while Pegasus is better for extreme summarization."
    )
    max_length = st.slider("Summary Length", 50, 500, 150, step=10, 
                         help="Maximum number of words in the summary")
    min_length = st.slider("Minimum Length", 10, 200, 30, step=5,
                         help="Minimum number of words in the summary")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("SmartNote HF uses state-of-the-art NLP models to generate concise summaries of your text.")
    st.markdown("Built with ❤️ using [Hugging Face Transformers](https://huggingface.co/transformers/)")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Text")
    input_text = st.text_area("Paste your text here", height=300, 
                            placeholder="Enter the text you want to summarize...")
    
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        submit_btn = st.button("Generate Summary")
    with col1_2:
        clear_btn = st.button("Clear")
        if clear_btn:
            input_text = ""

with col2:
    st.subheader("Summary")
    summary_placeholder = st.empty()
    
    if 'summary' in st.session_state and st.session_state.summary:
        with st.expander("View Summary", expanded=True):
            st.markdown(f'<div class="summary-box">{st.session_state.summary}</div>', 
                      unsafe_allow_html=True)
            
            # Add copy to clipboard button
            st.download_button(
                label="Download Summary",
                data=st.session_state.summary,
                file_name="summary.txt",
                mime="text/plain"
            )

# Handle form submission
if submit_btn and input_text.strip():
    with st.spinner("Generating summary..."):
        try:
            # Initialize the model
            summarizer = pipeline("summarization", model=model_name)
            
            # Generate summary
            summary = summarizer(
                input_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )[0]['summary_text']
            
            # Store the summary in session state
            st.session_state.summary = summary
            
            # Rerun to update the UI
            st.rerun()
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Add some space at the bottom
st.markdown("\n\n---")
st.markdown("### How to Use")
st.markdown("""
1. Paste your text in the input box
2. Adjust the summary length using the sliders
3. Click 'Generate Summary'
4. View, copy, or download your summary
""")
