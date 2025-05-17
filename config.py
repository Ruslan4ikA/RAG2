# Пути
DOCUMENTS_DIR = r"C:\Users\Mi\OneDrive\Рабочий стол\PARIS\факт\БЗ\Инструкции"
DB_DIR = "db/"
MODEL_PATH = "models/Qwen3-1.7B-BF16.gguf"

# Настройки обработки текста
CHUNK_SIZE = 500  # Примерный размер чанка в символах
OVERLAP = 50      # Перекрытие между чанками

# Настройки модели
MODEL_KWARGS = {
    "n_ctx": 2048,
    "n_threads": 4,  # Количество потоков CPU
    "n_gpu_layers": 0  # Для CPU-режима
}

# Настройки эмбеддингов
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"