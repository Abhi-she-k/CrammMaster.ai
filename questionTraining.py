import openai
from transformers import BertTokenizer, BertModel
import torch
import pdfplumber
import spacy
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
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


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


bert_embeddings = []

for sentence in sentences:
    sentence_embed = model.encode(sentence)
    bert_embeddings.append(sentence_embed)

reference = ["what made the first computers so special?"]

reference_embed = model.encode(reference[0])

# Calculate similarities
similarities = [F.cosine_similarity(torch.tensor(reference_embed), torch.tensor(embed), dim=0) for embed in bert_embeddings]

# Get top-k results
top_k = 10
top_k_indices = torch.topk(torch.tensor(similarities), k=top_k).indices

# Extract relevant sentences
relative_sentences = [sentences[idx] for idx in top_k_indices]

print("Top-k relevant sentences:" )
for i, sentence in enumerate(relative_sentences):
    print(f"Sentence {i+1}: {sentence}")