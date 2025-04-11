import openai
from sentence_transformers import SentenceTransformer

# Set OpenAI API key
# openai.api_key = 'YOUR_OPENAI_API_KEY'

def getSentenceVector():
    text = text.lower()
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    embeddings = []

    for sentence in sentences:
        sentence_embed = model.encode(sentence)
        embeddings.append(sentence_embed)

    reference_embed = model.encode(reference[0])

