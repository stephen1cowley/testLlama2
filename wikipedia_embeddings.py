from datasets import load_dataset
import torch
import cohere

# Add your cohere API key from www.cohere.com
co = cohere.Client("")

#Load at max 1000 documents + embeddings
max_docs = 1000
docs_stream = load_dataset(f"Cohere/wikipedia-22-12-simple-embeddings", split="train", streaming=True)

docs = []
doc_embeddings = []

for doc in docs_stream:
    docs.append(doc)
    doc_embeddings.append(doc['emb'])
    if len(docs) >= max_docs:
        break

doc_embeddings = torch.tensor(doc_embeddings)