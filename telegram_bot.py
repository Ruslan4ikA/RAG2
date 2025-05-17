from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from rag_system import RAGSystem
import asyncio


class KnowledgeBot:
    def __init__(self, token):
        self.rag = RAGSystem()
        self.app = Application.builder().token(token).build()

        # Обработчики команд
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

    async def start(self, update: Update, _):
        await update.message.reply_text("Привет! Я бот для ответов на вопросы по базе знаний. Задайте ваш вопрос.")

    async def handle_message(self, update: Update, _):
        user_query = update.message.text
        try:
            # Асинхронно запускаем генерацию ответа
            response = await asyncio.to_thread(
                self.rag.generate_response,
                user_query
            )
            await update.message.reply_text(response)
        except Exception as e:
            await update.message.reply_text(f"Произошла ошибка: {str(e)}")

    def run(self):
        self.app.run_polling()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python telegram_bot.py <TELEGRAM_BOT_TOKEN>")
        sys.exit(1)

    bot = KnowledgeBot(sys.argv[1])
    bot.run()