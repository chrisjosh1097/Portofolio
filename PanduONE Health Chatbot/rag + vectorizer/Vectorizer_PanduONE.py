import os
import glob
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

VECTOR_INDEX_PATH = "C:\\Users\\chris\\OneDrive\\Documents\\GitHub\\pandu.one-sehat\\PanduOne_Resources\\Chunks\\pdf_index.faiss"
CHUNKS_PATH = "C:\\Users\\chris\\OneDrive\\Documents\\GitHub\\pandu.one-sehat\\PanduOne_Resources\\FAISS\\doc_chunks.txt"
PDF_FOLDER = "C:\\Users\\chris\\OneDrive\\Documents\\GitHub\\pandu.one-sehat\\PanduOne_Resources\\Pdfs"  # Folder containing your PDFs

CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50

def read_pdfs_from_folder(folder_path):
    all_text = []
    pdf_files = glob.glob(os.path.join(folder_path, "**", "*.pdf"), recursive=True)
    for pdf_file in pdf_files:
        try:
            reader = PdfReader(pdf_file)
            full_text = " ".join(page.extract_text() or "" for page in reader.pages)
            filename = os.path.basename(pdf_file)  # get filename only, not full path
            all_text.append((filename, full_text))
        except Exception as e:
            print(f"Error reading {pdf_file}: {e}")
    return all_text


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def embed_chunks(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    
    # Extract just the chunk text (not filename)
    if isinstance(chunks[0], tuple):  # (filename, text)
        texts = [chunk[1] for chunk in chunks]
    else:
        texts = chunks

    if not texts:
        raise ValueError("No text chunks provided for embedding.")

    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings)


def save_index(embeddings, chunks):
    if os.path.exists(VECTOR_INDEX_PATH):
        os.remove(VECTOR_INDEX_PATH)
    if os.path.exists(CHUNKS_PATH):
        os.remove(CHUNKS_PATH)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, VECTOR_INDEX_PATH)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for filename, chunk in chunks:
            clean_chunk = chunk.strip().replace("\n", " ")
            f.write(f"{filename}\t{clean_chunk}\n")  # Tab-separated: filename + chunk

if __name__ == "__main__":
    print("📥 Reading PDFs...")
    texts = read_pdfs_from_folder(PDF_FOLDER)

    print("🧩 Chunking text...")
    all_chunks = []
    for filename, text in texts:
        chunks = chunk_text(text)
        for chunk in chunks:
            all_chunks.append((filename, chunk))

    print("🧠 Embedding chunks...")
    embeddings = embed_chunks(all_chunks)

    print("💾 Saving FAISS index and chunks...")
    save_index(embeddings, all_chunks)

    print("✅ Done! Vector index and text chunks saved.")
