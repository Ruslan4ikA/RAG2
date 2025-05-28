from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from rag_system import RAGSystem
import asyncio
import logging

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("TelegramBot")

class KnowledgeBot:
    def __init__(self, token):
        """
        Инициализация Telegram-бота и подключение к RAG-системе.
        """
        logger.info("Инициализация RAG-системы...")
        self.rag = RAGSystem()
        logger.info("RAG-система успешно загружена")

        logger.info("Создание Telegram-приложения...")
        self.app = Application.builder().token(token).build()

        # Регистрация команд
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

    async def start(self, update: Update, _):
        """
        Приветственное сообщение при запуске бота.
        """
        logger.info("/start получен от пользователя %s", update.message.from_user.id)
        await update.message.reply_text(
            "Привет! Я бот для внутренней поддержки сотрудников компании «Факт».\n"
            "Задайте ваш вопрос, и я найду информацию в базе знаний."
        )

    async def handle_message(self, update: Update, _):
        """
        Обработка текстовых сообщений от пользователей.
        """
        user_id = update.message.from_user.id
        user_query = update.message.text.strip()

        logger.info(f"Получен запрос от пользователя {user_id}: '{user_query}'")

        if not user_query:
            await update.message.reply_text("Пожалуйста, введите текстовый запрос.")
            return

        try:
            # Асинхронная генерация ответа
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.rag.generate_response, user_query)

            logger.info(f"Ответ сгенерирован для пользователя {user_id}: '{response}'")

            if "Не могу найти информацию" in response:
                await update.message.reply_text(
                    "К сожалению, я не нашёл информации по вашему запросу. "
                    "Попробуйте уточнить вопрос или проверьте актуальность данных."
                )
            else:
                await update.message.reply_text(response)

        except Exception as e:
            logger.error(f"Ошибка при обработке запроса от пользователя {user_id}: {e}", exc_info=True)
            await update.message.reply_text(
                "Произошла ошибка при обработке вашего запроса. Повторите попытку позже."
            )

    def run(self):
        """
        Запуск бота в режиме polling
        """
        logger.info("Бот запущен и ожидает сообщения...")
        self.app.run_polling()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python telegram_bot.py <TELEGRAM_BOT_TOKEN>")
        sys.exit(1)

    bot_token = sys.argv[1]
    bot = KnowledgeBot(bot_token)
    bot.run()