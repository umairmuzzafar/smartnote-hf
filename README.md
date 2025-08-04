# SmartNote HF - Text Summarization Tool

A simple yet powerful text summarization tool powered by Hugging Face's Transformers library. Perfect for students, researchers, and anyone who needs to quickly understand long articles or documents.

## ✨ Features
- **Easy-to-use** command line interface
- **Customizable** summary length
- **Fast** processing with optimized models
- **No API keys** required - runs completely offline

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/smartnote-hf.git
   cd smartnote-hf
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage
```bash
python src/summarize.py --text "Your long text here..." --max_length 150
```

### Example
Check out the example in the `examples/` directory to see it in action!

## 🌐 Web Interface

SmartNote HF includes a user-friendly web interface built with Streamlit. To use it:

1. Navigate to the web directory:
   ```bash
   cd web
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Open your browser and go to `http://localhost:8501`

## 📚 Documentation

### Command Line Arguments
- `--text`: The text you want to summarize (or use `--input_file` for large texts)
- `--max_length`: Maximum length of the summary (default: 150)
- `--min_length`: Minimum length of the summary (default: 30)
- `--model`: Model to use for summarization (default: "facebook/bart-large-cnn")

## 🤝 Contributing
I'm just starting out with open source, and I'd love your help! Here's how you can contribute:
- Report bugs
- Suggest new features
- Improve documentation
- Submit pull requests

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Made with ❤️ by [Your Name] - Feel free to connect with me on [LinkedIn/Twitter]!
