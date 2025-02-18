# Text Summarizer

![Hugging Face](https://img.shields.io/badge/HuggingFace-🤗-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-red?logo=pytorch)

![GitHub last commit](https://img.shields.io/github/last-commit/malorieiovino/text_summarizer)
![GitHub repo size](https://img.shields.io/github/repo-size/malorieiovino/text_summarizer)
![GitHub issues](https://img.shields.io/github/issues/malorieiovino/text_summarizer)


## 📌 Project Overview
This project is a **text summarization tool** that uses **Natural Language Processing (NLP)** to generate concise summaries of articles and long-form text. It leverages **Hugging Face Transformers** and the **BART Large CNN** model for abstractive summarization.

## 🚀 Features
- Extracts and summarizes key information from long text.
- Uses **BART Large CNN** for high-quality abstractive summarization.
- Supports customization of summary length.
- Implemented in **Python** using `transformers` and `torch`.
- Can fetch live news articles using an API (optional feature).

## 🛠 Installation & Setup
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/malorieiovino/text_summarizer.git
cd text_summarizer
```

### **2️⃣ Install Dependencies**
Create a virtual environment (optional but recommended):
```sh
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
Then install required packages:
```sh
pip install -r requirements.txt
```

## 📜 Usage
### **Running the Jupyter Notebook**
Launch Jupyter Notebook and open the summarizer script:
```sh
jupyter notebook
```

### **Example Code for Summarization**
```python
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

def summarize_text(text, max_length=60, min_length=20):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, num_beams=5, length_penalty=2.0, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

article = "Your long text here..."
print(summarize_text(article))
```

## 📌 Example Output
**Original Text:**
> A lobe of the polar vortex, a ring of freezing air typically found near the North Pole, will once again move into the central and eastern states this week, bringing some of the coldest air of the winter season.

**Generated Summary:**
> A lobe of the polar vortex will bring coldest air of the winter season to central and eastern states.

## 🤝 Contributing
Feel free to submit **issues** or **pull requests** to improve this project!

## 📄 License
This project is open-source and available under the **MIT License**.

---
🔗 **GitHub Repository:** [Text Summarizer](https://github.com/malorieiovino/text_summarizer)

