import re
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from modules.utils import get_document_display_name

# =============================================================================
# TEXT & VECTOR PROCESSING
# =============================================================================

def clean_text(text):
    text = text.encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text, num_chunks=5):
    sentences = sent_tokenize(text)
    if len(sentences) < num_chunks:
        return sentences if sentences else [text]
    per_chunk = len(sentences) // num_chunks
    remainder = len(sentences) % num_chunks
    chunks, start = [], 0
    for i in range(num_chunks):
        size = per_chunk + (1 if i < remainder else 0)
        chunks.append(' '.join(sentences[start:start+size]))
        start += size
    return chunks

def build_vector_db(documents):
    all_chunks, chunks_meta = [], []
    for doc in documents:
        owner = get_document_display_name(doc)
        for idx, chunk in enumerate(doc['chunks']):
            all_chunks.append(chunk)
            chunks_meta.append({
                'text': chunk, 
                'owner': owner, 
                'doc_id': doc.get('doc_id'),
                'file_type': doc.get('file_type', 'unknown')
            })
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(all_chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks_meta, model
