import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import time
from typing import Tuple, Optional

# CSS to make input and summary text white
st.markdown("""
    <style>
        .stTextArea textarea {
            color: white !important;
        }
        .stTextInput input {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

def load_model() -> Tuple[Optional[AutoModelForSeq2SeqLM], Optional[AutoTokenizer], str]:
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}")
        
        # Using BART model which is more stable
        model_name = "facebook/bart-large-cnn"
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None
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
st.set_page_config(page_title="SmartNote HF", page_icon="📝", layout="wide")

st.title("📝 SmartNote HF")
st.markdown("### AI-Powered Text Summarization Tool (BART Model)")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Load model button
    if st.button("Load Model"):
        with st.spinner("Loading BART model (this may take a minute for first load)..."):
            st.session_state.model, st.session_state.tokenizer, st.session_state.device = load_model()
            if st.session_state.model:
                st.success("✅ BART model loaded successfully!")
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
        st.text_area("Generated Summary", st.session_state.summary, height=300)
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
1. Click "Load Model" (only needed once per session)
2. Paste your text in the input box
3. Adjust the summary length using the sliders
4. Click 'Generate Summary'
5. View, copy, or download your summary
""")
