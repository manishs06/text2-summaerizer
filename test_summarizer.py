import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.extractive.extractive_summarizer import ExtractiveSummarizer

text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. 
Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. 
As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect. 
For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology. 
Modern machine learning techniques involve deep learning and transformer models.
"""

try:
    summarizer = ExtractiveSummarizer()
    print("Testing TF-IDF...")
    summary = summarizer.tfidf_summarize(text, num_sentences=2)
    print("Summary:", summary)
    
    print("\nTesting TextRank...")
    summary = summarizer.textrank_summarize(text, num_sentences=2)
    print("Summary:", summary)
    
    print("\nTesting Frequency...")
    summary = summarizer.frequency_based_summarize(text, num_sentences=2)
    print("Summary:", summary)
    
    print("\nAll tests passed!")
except Exception as e:
    print(f"Test failed: {e}")
    import traceback
    traceback.print_exc()
