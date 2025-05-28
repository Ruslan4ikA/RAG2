from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm
from config import EMBEDDING_MODEL, DB_DIR
from docx_processor import process_docx_files
import torch


def create_vector_db():
    print("[INFO] Извлечение и подготовка данных...")
    chunks = process_docx_files()

    if not chunks:
        raise ValueError("Не найдено ни одного чанка. Проверьте входные документы.")

    # Проверка структуры чанков
    required_keys = {'text', 'source', 'section'}
    for chunk in chunks[:10]:  # Проверяем первые 10
        if not all(key in chunk for key in required_keys):
            raise ValueError(f"Чанк {chunk} не содержит всех обязательных полей")

    print(f"[INFO] Пример чанка: {chunks[0]['text'][:100]}...")

    # Загрузка модели с проверкой
    try:
        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        test_embedding = embedding_model.encode("test")
        assert len(test_embedding) == 384  # Для all-MiniLM-L6-v2
    except Exception as e:
        raise RuntimeError(f"Ошибка загрузки модели: {e}")

    # Создание коллекции с оптимизированными параметрами
    client = chromadb.PersistentClient(path=DB_DIR)

    if "knowledge_base" in [col.name for col in client.list_collections()]:
        client.delete_collection("knowledge_base")

    collection = client.create_collection(
        name="knowledge_base",
        metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 400,
            "hnsw:M": 32
        }
    )

    # Векторизация с прогресс-баром и валидацией
    batch_size = 64
    for i in tqdm(range(0, len(chunks), batch_size), desc="Векторизация"):
        batch = chunks[i:i + batch_size]
        texts = [chunk['text'] for chunk in batch]

        try:
            embeddings = embedding_model.encode(
                texts,
                convert_to_tensor=False,
                normalize_embeddings=True,  # Критически важно!
                show_progress_bar=False
            )

            # Проверка качества эмбеддингов
            if any(embed.sum() == 0 for embed in embeddings):
                raise ValueError("Обнаружены нулевые эмбеддинги")

            collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=[
                    {
                        'source': chunk['source'],
                        'section': chunk.get('section', 'unknown'),
                        'chunk_id': f"{chunk['source']}_{i + j}"
                    } for j, chunk in enumerate(batch)
                ],
                ids=[f"id_{i + j}" for j in range(len(batch))]
            )
        except Exception as e:
            print(f"Ошибка в батче {i // batch_size}: {e}")
            continue

    # Валидация
    test_query = "Политика безопасности"
    test_embedding = embedding_model.encode(test_query, normalize_embeddings=True)
    results = collection.query(
        query_embeddings=[test_embedding.tolist()],
        n_results=3,
        include=["distances", "documents"]
    )
    print("\n[TEST] Результаты тестового запроса:")
    for doc, dist in zip(results['documents'][0], results['distances'][0]):
        print(f"Документ ({1 - dist:.2f}): {doc[:100]}...")

    print(f"\n[SUCCESS] База создана. Чанков: {collection.count()}")

if __name__ == "__main__":
    create_vector_db()
