from docx import Document
import os
import re
from config import DOCUMENTS_DIR

def is_section_header(text):
    """
    Проверяет, является ли строка заголовком/разделом.
    Поддерживает форматы:
        - '1.5. Обязанности'
        - 'Обязанности менеджера проекта:'
        - 'Требования к знаниям'
    """
    if not text:
        return False

    # Форматы: "1.5. Требования", "2. Рекомендации"
    if re.match(r'^\d+[\.]?\d*[\.:]?\s', text):
        return True

    # Заглавная буква + точка или двоеточие — возможный заголовок
    if re.match(r'^[А-ЯA-Z][^\.]*[\.:\–]', text):
        return True

    # Частые случаи: "ОБЯЗАННОСТИ", "ЧТО ДОЛЖЕН ЗНАТЬ СОТРУДНИК"
    if text.upper() == text and len(text.split()) < 10:
        return True

    return False


def process_docx_files():
    """
    Извлекает текст из .docx и разбивает на смысловые чанки по разделам
    """
    chunks = []

    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.endswith('.docx'):
            print(f"\n[INFO] Обработка файла: {filename}")
            doc = Document(os.path.join(DOCUMENTS_DIR, filename))
            current_section = None
            current_text = ""
            paragraph_counter = 0

            for para in doc.paragraphs:
                paragraph_counter += 1
                text = para.text.strip()
                if not text:
                    continue  # Пропускаем пустые строки

                # Для отладки: вывод первых 5 параграфов
                if paragraph_counter <= 5:
                    print(f"  Параграф {paragraph_counter}: \"{text}\"")

                # Если найден новый заголовок/секция
                if is_section_header(text):
                    if current_section:
                        chunks.append({
                            'text': current_text.strip(),
                            'source': filename,
                            'section': current_section
                        })
                    current_section = text
                    current_text = text + "\n\n"  # Включаем заголовок в содержимое
                else:
                    current_text += text + "\n"

            # Добавляем последний чанк
            if current_section and current_text.strip():
                chunks.append({
                    'text': current_text.strip(),
                    'source': filename,
                    'section': current_section
                })

    print(f"\n[INFO] Обработано {len(chunks)} чанков")
    return chunks


if __name__ == "__main__":
    chunks = process_docx_files()

    # Для тестирования: вывод первых 5 чанков
    print("\n[DEBUG] Первые 5 чанков:")
    for i, chunk in enumerate(chunks[:5], 1):
        print(f"\n--- Чанк {i} ---")
        print(f"Файл: {chunk['source']}")
        print(f"Заголовок: {chunk['section']}")
        print(f"Текст (обрезанный): {chunk['text'][:200]}...")