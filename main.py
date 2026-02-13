import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import matplotlib.pyplot as plt
import json

def main():
    # 1. Load Data
    # The CSV appears to be tab-separated based on the raw file formatting
    df = pd.read_csv("classmates.csv", sep="\t", header=None, names=["Name", "Description"])
    
    # 2. Generate Embeddings
    # Using the exact model specified in the repository and assignment
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['Description'].tolist())
    
    # Save the person_embeddings dictionary to a local file
    person_embeddings = {name: emb.tolist() for name, emb in zip(df['Name'], embeddings)}
    with open("embeddings.json", "w") as f:
        json.dump(person_embeddings, f)
        
    # 3. Dimension Reduction (UMAP)
    # The assignment strictly requires setting the seed to 42 for reproducibility
    reducer = umap.UMAP(random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    # 4. Visualization
    plt.figure(figsize=(12, 10))
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1])
    
    # Label each point with the classmate's name
    for i, name in enumerate(df['Name']):
        plt.annotate(name, (embedding_2d[i, 0], embedding_2d[i, 1]), fontsize=9)
        
    plt.title("Classmate Interests Embedding Space")
    plt.savefig("visualization.png")
    plt.close()

if __name__ == "__main__":
    main()