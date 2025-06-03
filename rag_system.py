from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from config import MODEL_PATH, MODEL_KWARGS, DB_DIR, EMBEDDING_MODEL
from typing import List, Dict, Optional
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAGSystem")

class RAGSystem:
    def __init__(self):
        """
        Инициализация компонентов системы:
        - модель эмбеддингов (для поиска);
        - reranking модель (для уточнения релевантности);
        - языковая модель (для генерации ответа);
        - векторная база знаний.
        """

        logger.info("Загрузка модели эмбеддингов...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        logger.info("Инициализация reranking модели...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        logger.info(f"Подключение к векторной БД: {DB_DIR}")
        self.db_client = chromadb.PersistentClient(path=DB_DIR)

        try:
            self.collection = self.db_client.get_collection(name="knowledge_base")
            if not self.collection.count():
                raise ValueError("База знаний пуста. Проверьте create_vector_db.py")
        except Exception as e:
            logger.error(f"Ошибка при подключении к коллекции: {e}")
            raise

        logger.info("Загрузка языковой модели...")
        try:
            self.llm = Llama(model_path=MODEL_PATH, **MODEL_KWARGS)
        except Exception as e:
            logger.error(f"Не удалось загрузить языковую модель: {e}")
            raise

    def retrieve(self, query: str, top_k: int = 5, min_similarity: float = 0.65) -> List[Dict]:
        try:
            # Нормализованные эмбеддинги
            query_embedding = self.embedding_model.encode(
                query,
                normalize_embeddings=True,
                show_progress_bar=False
            ).tolist()

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 2,
                include=["documents", "metadatas", "distances"]
            )

            # Фильтрация по порогу схожести
            filtered = []
            for doc, meta, dist in zip(results['documents'][0],
                                       results['metadatas'][0],
                                       results['distances'][0]):
                similarity = 1 - dist
                if similarity >= min_similarity:
                    filtered.append({
                        'text': doc,
                        'source': meta['source'],
                        'section': meta.get('section', 'unknown'),
                        'similarity': similarity,
                        'chunk_id': meta.get('chunk_id', '')
                    })
            # logger.info(f'retrieve: {filtered}')
            # Сортировка по убыванию релевантности
            return sorted(filtered, key=lambda x: x['similarity'], reverse=True)[:top_k]

        except Exception as e:
            logger.error(f"Retrieve error: {str(e)}", exc_info=True)
            return []

    def rerank(self, query: str, documents: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        Уточнение релевантности найденных документов на основе reranking модели.
        Возвращает топ-K самых релевантных документов.
        """
        logger.info("Повторное ранжирование найденных документов...")

        texts = [doc['text'] for doc in documents]
        pairs = [[query, text] for text in texts]

        scores = self.reranker.predict(pairs)
        scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)

        reranked_docs = [doc for _, doc in scored_docs[:top_k]]
        logger.info(f"Выбраны топ-{top_k} самых релевантных чанков {reranked_docs[0:top_k+1]}")

        return reranked_docs

    def generate_response(self, user_query: str) -> str:
        """
        Генерация ответа на основе найденного контекста.
        Если подходящего ответа нет — возвращается стандартное сообщение.
        """
        try:
            # Поиск релевантных чанков
            raw_results = self.retrieve(user_query, top_k=5)

            if not raw_results:
                logger.warning("Не найдено релевантных документов")
                return "Не могу найти информацию в базе знаний"

            # Переранжирование и выбор топ-3
            reranked_results = self.rerank(user_query, raw_results, top_k=3)

            # Формирование контекста с учётом разделов
            context = "\n\n".join([
                f"Раздел: {doc.get('section', 'без раздела')}\n{doc['text']}"
                for doc in reranked_results
            ])

            # Строгий промпт для модели
            prompt = f"""Ты — внутренний помощник компании «Факт». Отвечай только по предоставленному контексту.

ИНСТРУКЦИЯ:
- Опираемся только на КОНТЕКСТ.
- Не добавляем ничего от себя.
- Если информации нет — пишем: "Не могу найти информацию в базе знаний".
- Ответ должен быть понятным и точным.

КОНТЕКСТ:
{context}

ВОПРОС:
{user_query}

ОТВЕТ:"""

            logger.info("Генерация ответа...")
            response = self.llm(prompt, max_tokens=512, temperature=0.1, stop=["\n", "Примечание.", "Примечание", "Примечания:", "Примечания."])
            answer = response['choices'][0]['text'].strip()

            # Проверяем, не возвращена ли заглушка
            if not answer:
                logger.warning("Модель не нашла информации в контекстееееее")
                return "Не могу найти информацию в базе знаний"

            return answer

        except Exception as e:
            logger.error(f"Ошибка при генерации: {e}", exc_info=True)
            return "Произошла ошибка при обработке запроса."

    def get_sources(self, user_query: str, top_k: int = 3) -> List[Dict]:
        """
        Возвращает источники, использованные для ответа на запрос.
        Полезно для предоставления пользователю дополнительной информации.
        """
        logger.info("Получение источников для текущего запроса...")

        raw_results = self.retrieve(user_query, top_k=top_k * 2)
        if not raw_results:
            logger.warning("Не найдено релевантных документов для получения источников")
            return []

        # Создаем пары запрос-документ для reranker
        pairs = [[user_query, doc['text']] for doc in raw_results]
        scores = self.reranker.predict(pairs)

        # Сортируем по scores
        scored_docs = sorted(zip(scores, raw_results), key=lambda x: x[0], reverse=True)

        sources = [{
            'section': doc.get('section', 'раздел не указан'),
            'source': doc.get('source', 'неизвестный источник'),
            'relevance_score': float(score),
            'document_text': doc.get('text', '')[:200] + "..." if len(doc.get('text', '')) > 200 else ""
        } for score, doc in scored_docs[:top_k]]

        return sources


if __name__ == "__main__":
    rag = RAGSystem()
    test_query = "Кому подчиняется руководитель отдела?"
    print(f"\nЗапрос: {test_query}")
    print(f"Ответ: {rag.generate_response(test_query)}")
    print("\nИспользованные источники:")
    for source in rag.get_sources(test_query):
        print(f"- {source['section']} ({source['source']}) — релевантность: {source['relevance_score']:.4f}")