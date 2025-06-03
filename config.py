import os
import torch
# Пути
DOCUMENTS_DIR = r"C:\Users\Mi\OneDrive\Рабочий стол\PARIS\факт\БЗ\Инструкции"
DB_DIR = "db/"
MODEL_PATH = "models/Q4_K_M.gguf"

# Настройки обработки текста
CHUNK_SIZE = 500  # Примерный размер чанка в символах
OVERLAP = 50  # Перекрытие между чанками

# Настройки модели
MODEL_KWARGS = {
    "n_ctx": 4096,  # Увеличить контекст
    "n_threads": 8 if os.cpu_count() >= 8 else 4,  # Автовыбор потоков
    "n_gpu_layers": 20 if torch.cuda.is_available() else 0  # Автодетект GPU
}

# Настройки эмбеддингов
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
