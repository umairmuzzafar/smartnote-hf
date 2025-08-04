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

def load_model(model_name: str) -> Tuple[Optional[AutoModelForSeq2SeqLM], Optional[AutoTokenizer], str]:
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with appropriate settings
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

# Rest of your existing code remains the same...
[Previous content continues exactly as before...]
