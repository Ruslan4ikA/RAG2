from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm
from config import EMBEDDING_MODEL, DB_DIR
from docx_processor import process_docx_files


def create_vector_db():
    """Создает векторную БД из чанков"""
    chunks = process_docx_files()
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection("knowledge_base")

    # Добавляем документы батчами для экономии памяти
    batch_size = 32
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i + batch_size]
        embeddings = embedding_model.encode([chunk['text'] for chunk in batch])
        collection.add(
            embeddings=[embed.tolist() for embed in embeddings],
            documents=[chunk['text'] for chunk in batch],
            metadatas=[{'source': chunk['source']} for chunk in batch],
            ids=[f"id_{i + j}" for j in range(len(batch))]
        )

    print(f"Векторная БД создана, добавлено {len(chunks)} чанков")


if __name__ == "__main__":
    create_vector_db()