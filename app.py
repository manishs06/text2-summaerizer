from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import os
from pathlib import Path

# Add src directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.extractive.extractive_summarizer import ExtractiveSummarizer
from src.abstractive.abstractive_summarizer import AbstractiveSummarizer

app = Flask(__name__)
CORS(app)

# Initialize summarizers lazily to avoid delay at startup if models are large
extractive_summarizer = None
abstractive_summarizer = None

def get_extractive():
    global extractive_summarizer
    if extractive_summarizer is None:
        extractive_summarizer = ExtractiveSummarizer()
    return extractive_summarizer

def get_abstractive():
    global abstractive_summarizer
    if abstractive_summarizer is None:
        abstractive_summarizer = AbstractiveSummarizer()
    return abstractive_summarizer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    text = data.get('text', '')
    method = data.get('method', 'extractive')
    
    if not text or len(text.strip()) < 10:
        return jsonify({'error': 'Text is too short to summarize'}), 400
    
    try:
        if method == 'extractive':
            algorithm = data.get('algorithm', 'textrank')
            num_sentences = int(data.get('num_sentences', 3))
            
            summarizer = get_extractive()
            summary = summarizer.summarize(
                text, 
                method=algorithm, 
                num_sentences=num_sentences
            )
        elif method == 'abstractive':
            max_length = int(data.get('max_length', 130))
            min_length = int(data.get('min_length', 30))
            
            summarizer = get_abstractive()
            summary = summarizer.summarize(
                text,
                max_length=max_length,
                min_length=min_length
            )
        else:
            return jsonify({'error': 'Invalid summarization method'}), 400
            
        return jsonify({
            'summary': summary,
            'original_length': len(text),
            'summary_length': len(summary),
            'reduction': round(((len(text) - len(summary)) / len(text)) * 100, 2)
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(error_details)  # This will show up in Vercel logs
        return jsonify({
            'error': str(e),
            'details': error_details
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
