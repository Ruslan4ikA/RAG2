# 🤖 RAG Knowledge Base Telegram Bot

**Telegram-бот для ответов на вопросы сотрудников на основе базы знаний с использованием RAG (Retrieval-Augmented Generation)**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 🔍 Описание

Система позволяет сотрудникам получать точные ответы на вопросы, основанные на корпоративных документах (DOCX). Использует:
- Локальную языковую модель Qwen-1.7B
- Векторную базу данных ChromaDB
- Технологию RAG для поиска релевантных данных

## ✨ Возможности

- 📄 Загрузка и обработка документов DOCX
- 🔍 Семантический поиск по базе знаний
- 💬 Интеграция с Telegram
- 🧠 Локальное выполнение на CPU
- 🛡️ Работа с конфиденциальными данными без облачных сервисов

## 🛠 Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/Ruslan4ikA/RAG2.git
cd RAG2
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. В файле `config.py` укажите папку для документов в переменной `DOCUMENTS_DIR`

4. Запустите обработку:
```bash
python docx_processor.py
python embedding_generator.py
```

## 🚀 Запуск бота

1. Создайте бота через [@BotFather](https://t.me/BotFather)
2. Запустите бота:
```bash
python telegram_bot.py YOUR_TELEGRAM_BOT_TOKEN
```

## 🗂 Структура проекта

```
RAG2/
├── documents/          # Исходные документы DOCX
├── db/                # Векторная база данных
├── models/            # Локальные модели AI
├── config.py          # Настройки
├── docx_processor.py  # Обработка документов
├── embedding_generator.py # Генерация эмбеддингов
├── rag_system.py      # Ядро RAG-системы
└── telegram_bot.py    # Telegram-интерфейс
```

## 📝 Пример использования

```
Пользователь: Как оформить отпуск?
Бот: Согласно документу "Правила внутреннего распорядка":
1. Подайте заявление за 2 недели
2. Согласуйте с руководителем
3. Оформите приказ в HR-системе
```

## 🤝 Как внести вклад

1. Форкните репозиторий
2. Создайте ветку (`git checkout -b feature/AmazingFeature`)
3. Сделайте коммит (`git commit -m 'Add some AmazingFeature'`)
4. Запушьте (`git push origin feature/AmazingFeature`)
5. Откройте Pull Request

---

> 📧 **Контакты**: Абзелилов Руслан · [@unbefangenheit](https://t.me/unbefangenheit) · ruslanr26@mail.ru
