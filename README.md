# pdf-qna-chatbot
!pip install transformers sentence-transformers annoy PyMuPDF
!pip install PyMuPDF
!pip install annoy
!pip install sentence-transformers
!pip install transformers
!pip search fitzz

import os
import fitz  # PyMuPDF for PDF processing
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
import numpy as np

# Step 0: Install requirements and set Hugging Face token
HUGGINGFACE_TOKEN = "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"
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
TITLE = "THE BOT"

# Step 1: Load PDF document
def load_pdf(file_path):
    """Extract text from a PDF file."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Step 2: Text Splitter
def split_text(text, chunk_size=500, overlap=50):
    """Split text into chunks of a specific size with overlap."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

import numpy as np
import torch

def embed_texts(model, tokenizer, texts):
    """Generate embeddings for a list of texts using a causal language model."""
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)  # CausalLMOutputWithPast
            # Extract logits or hidden states as embeddings
            logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)
            embedding = logits.mean(dim=1).squeeze().numpy()  # Mean over sequence length
        embeddings.append(embedding)
    return np.array(embeddings)





# Step 4: Store Embeddings in ANNOY Vector DB
def build_annoy_index(embeddings, dimension, index_path):
    """Build and save an Annoy index."""
    index = AnnoyIndex(dimension, 'angular')
    for i, embed in enumerate(embeddings):
        index.add_item(i, embed)
    index.build(10)  # 10 trees
    index.save(index_path)
    return index

# Step 5: Load Query, Retrieve Relevant Chunks, and Generate Response
def retrieve_and_respond(query, model, tokenizer, annoy_index, text_chunks, embeddings):
    """Retrieve the nearest chunk and respond using the model."""
    # Generate the embedding for the query
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        # For causal models, use the input embeddings or logits
        outputs = model(**inputs)
        query_embedding = outputs.logits.mean(dim=1).squeeze().numpy()

    # Search for nearest neighbors in the ANNOY index
    nearest_indices = annoy_index.get_nns_by_vector(query_embedding, 1)
    nearest_chunk = text_chunks[nearest_indices[0]]

    # Use the model to generate a response based on the retrieved chunk
    prompt = f"Context: {nearest_chunk}\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=200, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
    
#Step 6: Main Workflow
def main():
    # Load Qwen2-1.5B-Instruct model and tokenizer
    print("Loading Qwen model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HUGGINGFACE_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_auth_token=HUGGINGFACE_TOKEN)
    print("Model loaded successfully!")

    # User uploads the PDF
    from google.colab import files
    print("Please upload a PDF document:")
    uploaded = files.upload()
    file_path = list(uploaded.keys())[0]

    # Step 1: Process the PDF
    print("Processing the document...")
    document_text = load_pdf(file_path)
    text_chunks = split_text(document_text)
    print(f"Document split into {len(text_chunks)} chunks.")

    # Step 2: Embed the chunks
    print("Generating embeddings...")
    embeddings = embed_texts(model, tokenizer, text_chunks)
    print("Embeddings generated successfully!")

    # Step 3: Store embeddings in ANNOY
    print("Building ANNOY index...")
    index_path = "document_index.ann"
    annoy_index = build_annoy_index(embeddings, embeddings.shape[1], index_path)
    print("ANNOY index built and saved!")

    # Query phase
    while True:
        query = input("\nAsk a question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        response = retrieve_and_respond(query, model, tokenizer, annoy_index, text_chunks, embeddings)
        print(f"\nResponse: {response}")

if __name__ == "__main__":
    main()

