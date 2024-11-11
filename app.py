import streamlit as st
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import numpy as np

# Define your categories
categories = ['Finance', 'Geopolitical', 'Natural Disasters', 'Regulatory']

# Define the model class (you can also import this if it's in another file)
class BertMultiLabelClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BertMultiLabelClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.out(output)

# Initialize model and load trained weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertMultiLabelClassifier(num_labels=len(categories))
model.load_state_dict(torch.load('bert_model_10epochs.pth', map_location=device))
model.eval().to(device)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prediction function
def predict(text):
    encoding = tokenizer.encode_plus(
        text,
        max_length=512,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
    
    results = {category: prob for category, prob in zip(categories, probs)}
    return results

# Streamlit UI
st.title("Supplier News Categorization")
st.write("Enter news content, and this tool will classify it into multiple categories: "
         "Finance, Geopolitical, Natural Disasters, and Regulatory. Each category is given a probability score.")

# Use Streamlit session state to store and update the news content
if "news_content" not in st.session_state:
    st.session_state.news_content = ""

# Text input for news content
news_content = st.text_area("News Content", st.session_state.news_content, height=200)

# Predict and display results when the "Classify" button is pressed
if st.button("Classify"):
    if news_content:
        results = predict(news_content)
        st.subheader("Category Probabilities")
        for category, prob in results.items():
            st.write(f"{category}: {prob:.2f}")
    else:
        st.warning("Please enter some news content.")

# Example texts
st.sidebar.header("Example News Contents")
example_texts = [
    "New regulations on cryptocurrency trading are being discussed by the financial authorities.",
    "A powerful earthquake has struck the coastal region, causing widespread damage.",
    "The ongoing trade negotiations between major economies have raised geopolitical concerns.",
    "The central bank announced changes in the interest rate policies affecting the financial sector."
]

for i, example in enumerate(example_texts, 1):
    if st.sidebar.button(f"Example {i}"):
        st.session_state.news_content = example  # Update session state with selected example text
        st.experimental_rerun()  # Rerun the app to update the text area with the example text

# Run the Streamlit app
# To run this app, use the command in your terminal:
# streamlit run app.py
