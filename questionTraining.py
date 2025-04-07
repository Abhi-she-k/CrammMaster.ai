import openai
from transformers import BertTokenizer, BertModel
import torch
import pdfplumber
import spacy
import torch.nn.functional as F
# from sklearn.metrics.pairwise import cosine_similarity


# Set OpenAI API key
# openai.api_key = 'YOUR_OPENAI_API_KEY'

nlp = spacy.load("en_core_web_sm")

with pdfplumber.open("PDFs\A Brief Overview of the History of Computers.pdf") as pdf:
    text = ""
    for page in pdf.pages:
        text += page.extract_text()

text = text.lower()

doc = nlp(text)
sentences = [sent.text.strip() for sent in doc.sents]



tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

bert_embeddings = []


def getEmbeddings(sentence):
    input = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**input)
    return outputs.last_hidden_state.mean(dim=1)
    

for sentence in sentences:
    sentence_embed = getEmbeddings(sentence)
    bert_embeddings.append(sentence_embed)

reference = ["what are computers"]

reference_embed = getEmbeddings(reference[0])

similarities = [F.cosine_similarity(reference_embed, sentence_embedding, dim=1) for sentence_embedding in bert_embeddings]

similarities_tensor = torch.tensor(similarities)

top_k = 5
top_k_values, top_k_indices = torch.topk(similarities_tensor, k=top_k)

# Print the top 5 most relevant sentences
for i in range(top_k):
    print(f"Rank {i+1}: {sentences[top_k_indices[i].item()]} (Similarity Score: {top_k_values[i].item():.4f})")