import torch.nn.functional as F

def calculateSimilarity(embeddings):
    # Calculate similarities
    similarities = [F.cosine_similarity(torch.tensor(reference_embed), torch.tensor(embed), dim=0) for embed in embeddings]

    # Get top-k results
    top_k = 10
    top_k_indices = torch.topk(torch.tensor(similarities), k=top_k).indices

    # Extract relevant sentences
    relative_sentences = [sentences[idx] for idx in top_k_indices]

    print("Top-k relevant sentences:" )
    for i, sentence in enumerate(relative_sentences):
        print(f"Sentence {i+1}: {sentence}")