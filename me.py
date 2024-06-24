import os
import re
import random
from datetime import datetime
from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize Flask app
app = Flask(__name__)

# Initialize GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

# List of domains to prioritize for content retrieval
priority_domains = [
    "https://simple.wikipedia.org/wiki/",
    "https://en.wikipedia.org/wiki/",
    "https://scholar.google.com/scholar?q=",
    "https://sw.wikipedia.org/wiki/",
    "https://www.byjus.com/",
    "https://www.wikihow.com/",
    "https://www.britannica.com/"
]

# Route for home page
@app.route('/')
def home():
    return render_template('scrpe.html')

# Route for handling search request
@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query')
    query_type = data.get('query_type')
    
    if not query or not query_type:
        return jsonify({"error": "Please provide both query and query type."}), 400
    
    # Attempt to fetch content from prioritized sources
    content, images, related_keywords = fetch_content_from_prioritized_sources(query)
    
    if not content:
        # If content not found, generate using GPT-2
        content = generate_text_with_gpt2(query, query_type)
        images = []
        related_keywords = []
    
    return jsonify({"content": content, "images": images, "related_keywords": related_keywords})

# Function to fetch content from prioritized sources
def fetch_content_from_prioritized_sources(query):
    content = None
    images = []
    related_keywords = []
    
    for domain in priority_domains:
        url = domain + query
        response = requests.get(url)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            content_div = soup.find('div', class_='content') or soup.find('div', {'id': 'mw-content-text'})
            
            if content_div:
                paragraphs = content_div.find_all('p', limit=25)
                content = ' '.join([clean_text(p.text.strip()) for p in paragraphs if p.text.strip()])
                
                for img in content_div.find_all('img', limit=5):
                    img_url = img.get('src')
                    if img_url:
                        if img_url.startswith('//'):
                            img_url = 'https:' + img_url
                        images.append(img_url)
                
                if content:
                    related_keywords = extract_keywords(soup.title.string)
                    break
    
    return content, images, related_keywords

# Function to clean text from unnecessary symbols and content
def clean_text(text):
    cleaned_text = re.sub(r'\[.*?\]', '', text)  # Remove text within square brackets
    cleaned_text = re.sub(r'\(.*?\)', '', cleaned_text)  # Remove text within parentheses
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Remove extra whitespaces
    return cleaned_text.strip()

# Function to extract keywords from a query
def extract_keywords(query):
    return re.findall(r'\b\w+\b', query)

# Function to generate text using GPT-2
def generate_text_with_gpt2(query, query_type):
    input_text = f"Generate {query_type} about {query}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate text based on input
    output = model.generate(input_ids, max_length=500, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_text

# Run Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4000))
    app.run(host="0.0.0.0", port=port)
