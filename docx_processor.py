from docx import Document
import os
from config import DOCUMENTS_DIR, CHUNK_SIZE, OVERLAP

def process_docx_files():
    """Извлекает текст из docx и разбивает по абзацам"""
    chunks = []
    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.endswith('.docx'):
            doc = Document(os.path.join(DOCUMENTS_DIR, filename))
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:  # Игнорируем пустые абзацы
                    chunks.append({
                        'text': text,
                        'source': filename
                    })
    return chunks

if __name__ == "__main__":
    chunks = process_docx_files()
    print(f"Обработано {len(chunks)} абзацев")