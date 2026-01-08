# SummAIze: Premium Text Summarization

SummAIze is a powerful, web-based application designed to transform long content into clear, concise summaries instantly. It offers a premium, glassmorphic user interface and a lag-free experience.

## Key Features

- **Extractive Summarization**: Selects the most important sentences from the original text using algorithms like TextRank, TF-IDF, and Word Frequency.
- **Abstractive Summarization**: Generates new, human-like summaries using advanced Transformer-based machine learning models.
- **Premium UI/UX**: A modern, responsive design with smooth animations, dark mode, and glassmorphism.
- **Real-time Statistics**: View reduction percentages and character count comparisons instantly.
- **Zero Lag**: Highly optimized backend using Flask for fast, asynchronous processing.
- **One-Click Copy**: Easily copy your generated summary to the clipboard.

## Local Setup

To run this project on your local machine, follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/manishs06/Text-summaerizer.git
cd Text-summaerizer
```

### 2. Install dependencies
It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/scripts/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the application
```bash
python app.py
```
The app will be available at `http://127.0.0.1:5000`.

## How it Works

Simply paste your text, choose your preferred summarization method and parameters, and let SummAIze handle the rest. Whether it's a long article or a complex document, get the insights you need in seconds.