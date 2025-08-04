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

@st.cache_resource(show_spinner="Loading BART model (this may take a minute for first load)...")
def load_model():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Using a smaller BART model variant
        model_name = "facebook/bart-large-cnn"
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
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
if 'summary' not in st.session_state:
    st.session_state.summary = ""

# Load model automatically
model, tokenizer, device = load_model()

# UI
st.set_page_config(page_title="SmartNote HF", page_icon="📝", layout="wide")

st.title("📝 SmartNote HF")
st.markdown("### AI-Powered Text Summarization Tool")

# Sidebar
with st.sidebar:
    st.header("⚙️ Summary Settings")
    max_length = st.slider("Max Length", 50, 500, 150)
    min_length = st.slider("Min Length", 10, 200, 30)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Text")
    input_text = st.text_area("Paste your text here", height=300, 
                            placeholder="Enter the text you want to summarize...")
    
    if st.button("Generate Summary", disabled=model is None):
        if input_text.strip():
            with st.spinner("Generating summary..."):
                try:
                    start_time = time.time()
                    st.session_state.summary = generate_summary(
                        model,
                        tokenizer,
                        device,
                        input_text,
                        max_length,
                        min_length
                    )
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
1. Paste your text in the input box
2. Adjust the summary length using the sliders
3. Click 'Generate Summary'
4. View, copy, or download your summary
""")
