import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import time
from typing import Tuple, Optional

# Set page config first
st.set_page_config(
    page_title="SmartNote HF",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to fix text visibility and improve UI
st.markdown("""
    <style>
        .stApp {
            background-color: #f8f9fa;
        }
        .stTextArea textarea {
            color: #333333 !important;
        }
        .stTextInput input {
            color: #333333 !important;
        }
        .stButton>button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px;
            border-radius: 5px;
            cursor: pointer;
        }
        .stButton>button:disabled {
            background-color: #cccccc;
        }
    </style>
""", unsafe_allow_html=True)

def load_model(model_name: str) -> Tuple[Optional[AutoModelForSeq2SeqLM], Optional[AutoTokenizer], str]:
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with appropriate settings
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        if not torch.cuda.is_available():
            model = model.to(device)
            
        return model, tokenizer, device
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, ""

def generate_summary(model, tokenizer, device: str, text: str, max_length: int, min_length: int) -> str:
    try:
        inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        raise Exception(f"Error generating summary: {str(e)}")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.tokenizer = None
    st.session_state.device = ""
    st.session_state.summary = ""

# UI
st.title("📝 SmartNote HF")
st.markdown("### AI-Powered Text Summarization Tool")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox(
        "Select Model",
        ["facebook/bart-large-cnn", "google/pegasus-cnn_dailymail"],
        help="BART for general purpose, Pegasus for extreme summarization"
    )
    
    if st.button("Load Model"):
        with st.spinner(f"Loading {model_name} (this may take a few minutes for first load)..."):
            st.session_state.model, st.session_state.tokenizer, st.session_state.device = load_model(model_name)
            if st.session_state.model:
                st.success(f"✅ {model_name} loaded successfully!")
            else:
                st.error("Failed to load model")

    st.markdown("---")
    st.markdown("### Summary Settings")
    max_length = st.slider("Max Length", 50, 500, 150)
    min_length = st.slider("Min Length", 10, 200, 30)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Text")
    input_text = st.text_area("Paste your text here", height=300, 
                            placeholder="Enter the text you want to summarize...")
    
    if st.button("Generate Summary", disabled=st.session_state.model is None):
        if input_text.strip():
            with st.spinner("Generating summary..."):
                try:
                    start_time = time.time()
                    summary = generate_summary(
                        st.session_state.model,
                        st.session_state.tokenizer,
                        st.session_state.device,
                        input_text,
                        max_length,
                        min_length
                    )
                    st.session_state.summary = summary
                    st.success(f"Summary generated in {time.time() - start_time:.2f} seconds")
                except Exception as e:
                    st.error(str(e))
        else:
            st.warning("Please enter some text to summarize")

with col2:
    st.subheader("Summary")
    if st.session_state.summary:
        st.text_area("Generated Summary", st.session_state.summary, height=300, key="summary_output")
        if st.download_button(
            label="Download Summary",
            data=st.session_state.summary,
            file_name="summary.txt",
            mime="text/plain"
        ):
            st.success("Summary downloaded!")
    else:
        st.info("Your summary will appear here")

st.markdown("---")
st.markdown("### How to Use")
st.markdown("""
1. Select a model and click "Load Model"
2. Wait for the model to load (you'll see a success message)
3. Paste your text in the input box
4. Adjust the summary length using the sliders
5. Click 'Generate Summary'
6. View, copy, or download your summary
""")
