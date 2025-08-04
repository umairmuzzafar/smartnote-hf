#!/usr/bin/env python3
"""
SmartNote HF - Text Summarization Tool

A simple command-line tool for summarizing text using Hugging Face's Transformers.
"""

import click
from transformers import pipeline
import torch
import sys
from pathlib import Path

def load_model(model_name="facebook/bart-large-cnn"):
    """Load the summarization model."""
    print(f"Loading {model_name}... (This may take a moment)")
    return pipeline("summarization", model=model_name, device=0 if torch.cuda.is_available() else -1)

def summarize_text(model, text, max_length=150, min_length=30):
    """Generate a summary of the input text."""
    try:
        # Split text into chunks if it's too long
        if len(text) > 10000:  # Arbitrary chunk size
            chunks = [text[i:i+10000] for i in range(0, len(text), 10000)]
            summaries = []
            for chunk in chunks:
                summary = model(chunk, max_length=max_length, min_length=min_length, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            return " ".join(summaries)
        else:
            summary = model(text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']
    except Exception as e:
        print(f"Error during summarization: {str(e)}", file=sys.stderr)
        return None

@click.command()
@click.option('--text', help='Text to summarize')
@click.option('--input_file', type=click.Path(exists=True), help='Input file containing text to summarize')
@click.option('--output_file', type=click.Path(), help='Output file for the summary')
@click.option('--max_length', default=150, help='Maximum length of the summary')
@click.option('--min_length', default=30, help='Minimum length of the summary')
@click.option('--model', default="facebook/bart-large-cnn", help='Model to use for summarization')
def main(text, input_file, output_file, max_length, min_length, model):
    """SmartNote HF - Generate summaries from text or files."""
    # Validate input
    if not text and not input_file:
        print("Error: Either --text or --input_file must be provided", file=sys.stderr)
        sys.exit(1)
    
    # Read input
    if input_file:
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading input file: {str(e)}", file=sys.stderr)
            sys.exit(1)
    
    # Load model and generate summary
    model = load_model(model)
    print("Generating summary...")
    summary = summarize_text(model, text, max_length, min_length)
    
    if summary:
        # Output the result
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(summary)
                print(f"Summary saved to {output_file}")
            except Exception as e:
                print(f"Error writing to output file: {str(e)}", file=sys.stderr)
                sys.exit(1)
        else:
            print("\n=== SUMMARY ===")
            print(summary)
            print("===============")
    else:
        print("Failed to generate summary.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
