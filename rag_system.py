from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import chromadb
from config import *


class RAGSystem:
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.llm = Llama(
            model_path=MODEL_PATH,
            **MODEL_KWARGS
        )
        self.db_client = chromadb.PersistentClient(path=DB_DIR)
        self.collection = self.db_client.get_collection("knowledge_base")

    def retrieve(self, query: str, top_k: int = 3):
        """Поиск релевантных чанков"""
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results['documents'][0]

    def generate_response(self, query: str):
        """Генерация ответа с использованием контекста"""
        context = self.retrieve(query)
        prompt = f"""Ты - ассистент компании. Ответь на вопрос сотрудника, используя только предоставленный контекст. Если ответа нет в контексте, скажи "Не могу найти информацию в базе знаний".

        Контекст: {' '.join(context)}
        Вопрос: {query}
        Ответ:"""

        response = self.llm(
            prompt,
            max_tokens=256,
            temperature=0.1,  # Для более детерминированных ответов
            stop=["\n"]
        )
        return response['choices'][0]['text'].strip()


if __name__ == "__main__":
    rag = RAGSystem()
    test_query = "Как оформить отпуск?"
    print(rag.generate_response(test_query))