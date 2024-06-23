import os
from flask import Flask, render_template, request, send_file
from bs4 import BeautifulSoup
import requests
from fpdf import FPDF
from pptx import Presentation
from docx import Document
from pptx.util import Inches
from transformers import pipeline, BertTokenizer, BertForMaskedLM

app = Flask(__name__)

# Initialize BERT model for masked language modeling
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

# Function to scrape information from different sources
def scrape_information(query, query_type):
    sources = [
        f"https://www.byjus.com/{query}",
        f"https://www.wikihow.com/{query}",
        f"https://simple.wikipedia.org/wiki/{query}",
        f"https://scholar.google.com/scholar?q={query}",
        f"https://en.wikipedia.org/wiki/{query}",
        f"https://sw.wikipedia.org/wiki/{query}",
        f"https://www.britannica.com/{query}"
    ]
    content_list = []
    images = []
    related_keywords = []

    for url in sources:
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Scraping entire text content from the page
            paragraphs = soup.find_all('p')
            content = ' '.join([p.text.strip() for p in paragraphs])
            content_list.append(content)
            
            # Scraping images
            for img in soup.find_all('img', limit=5):
                img_url = img.get('src')
                if img_url:
                    images.append(f"https:{img_url}")
                    
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")

    # Join all scraped content
    text = ' '.join(content_list)
    
    # Use BERT for masked language modeling to summarize content
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    masked_index = inputs[0].tolist().index(tokenizer.mask_token_id)
    outputs = model(inputs)
    predicted_tokens = tokenizer.convert_ids_to_tokens(outputs.logits.argmax(dim=-1))
    summary = tokenizer.decode(predicted_tokens[0][masked_index:], skip_special_tokens=True)

    # Extract keywords using BERT
    keywords = []
    entities = tokenizer.tokenize(summary)
    for entity in entities:
        if entity.startswith('##'):
            keywords.append(entity[2:])
    
    return {
        "content": summary,
        "images": images,
        "related_keywords": keywords[:3]
    }

# Function to create PDF
def create_pdf(query, query_type, result):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    title_text = f"{query_type.capitalize()} of {query}"
    content_text = result['content']
    
    try:
        pdf.cell(200, 10, txt=title_text, ln=True, align='C')
        pdf.multi_cell(0, 10, txt=content_text)
        
        if result['images']:
            for img in result['images']:
                pdf.image(img, w=100)
        
        pdf.output("output.pdf")
        print("PDF created successfully.")
    except Exception as e:
        print(f"An error occurred while creating the PDF: {e}")

# Function to create PowerPoint presentation
def create_ppt(query, query_type, result):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    title = slide.shapes.title
    title.text = f"{query_type.capitalize()} of {query}"
    
    left_inch = Inches(1)
    top_inch = Inches(1.2)
    width_inch = Inches(8)
    height_inch = Inches(1.5)
    
    textbox = slide.shapes.add_textbox(left_inch, top_inch, width_inch, height_inch)
    text_frame = textbox.text_frame
    text_frame.word_wrap = True
    p = text_frame.add_paragraph()
    p.text = result['content']
    
    if result['images']:
        for img in result['images']:
            slide.shapes.add_picture(img, Inches(1), Inches(2), height=Inches(1.5))
    
    prs.save('Nobestudy.pptx')

# Function to create DOCX document
def create_docx(query, query_type, result):
    doc = Document()
    doc.add_heading(f"{query_type.capitalize()} of {query}", level=1)
    doc.add_paragraph(result['content'])
    
    if result['images']:
        for img in result['images']:
            doc.add_picture(img, width=Inches(2))
    
    doc.save('Nobestudy.docx')

# Route for home page
@app.route('/')
def home():
    return render_template('scrpe.html')

# Route for handling search request
@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    query_type = request.form.get('query_type')
    
    if not query or not query_type:
        return render_template('scrpe.html', error="Please provide both query and query type.")
    
    result = scrape_information(query, query_type)
    
    return render_template('scrpe.html', query=query, query_type=query_type, result=result)

# Route for downloading files
@app.route('/download/<file_type>', methods=['POST'])
def download(file_type):
    query = request.form.get('query')
    query_type = request.form.get('query_type')
    content = request.form.get('content')
    
    if not query or not query_type or not content:
        return render_template('scrpe.html', error="Missing data to generate file.")
    
    result = {"content": content, "images": []}
    
    if file_type == 'pdf':
        create_pdf(query, query_type, result)
        return send_file("output.pdf", as_attachment=True)
    elif file_type == 'ppt':
        create_ppt(query, query_type, result)
        return send_file("Nobestudy.pptx", as_attachment=True)
    elif file_type == 'docx':
        create_docx(query, query_type, result)
        return send_file("Nobestudy.docx", as_attachment=True)
    else:
        return render_template('scrpe.html', error="Invalid file type.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4000))
    app.run(host="0.0.0.0", port=port)
