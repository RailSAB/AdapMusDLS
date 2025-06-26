from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
from dotenv import load_dotenv
import os
def dummy_search(prompt, author, similar_to, not_similar_to, limit):
    songs = [
        {"title": f"Song {i}", "author": f"Author {i}", "url": f"http://example.com/song{i}"}
        for i in range(1, 21)
    ]
    return songs[:limit]

class MusicBot:
    def __init__(self):
        self.user_data = {}

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Welcome to the Music Bot! Use /search to find music and /liked to view your liked songs.")

    async def search(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.message.chat_id
        self.user_data[chat_id] = {
            "search_results": [],
            "current_index": 0,
            "liked_songs": self.user_data.get(chat_id, {}).get("liked_songs", []),
            "search_state": {
                "params": set(),
                "step": "choose_params",
                "prompt": None,
                "author": None,
                "limit": None,
            },
        }
        await self.send_search_param_selection(update, context)

    async def send_search_param_selection(self, update, context):
        chat_id = update.message.chat_id if update.message else update.callback_query.message.chat_id
        state = self.user_data[chat_id]["search_state"]
        params = state["params"]
        def btn(label, key):
            if key in params:
                return InlineKeyboardButton(f"🟢 {label}", callback_data=f"searchparam_{key}")
            else:
                return InlineKeyboardButton(label, callback_data=f"searchparam_{key}")
        def btn_similar(label, key):
            if key in params:
                return InlineKeyboardButton(f"🟢 {label}", callback_data=f"searchparam_{key}")
            elif ("similar" in params or "not_similar" in params) and key != list(params & {"similar", "not_similar"})[0]:
                return InlineKeyboardButton(f"🔴 {label}", callback_data=f"searchparam_{key}")
            else:
                return InlineKeyboardButton(label, callback_data=f"searchparam_{key}")
        keyboard = [
            [btn("By prompt", "prompt")],
            [btn("By author", "author")],
            [btn_similar("Similar by liked songs", "similar")],
            [btn_similar("Not similar by liked songs", "not_similar")],
            [InlineKeyboardButton("Continue", callback_data="searchparam_continue")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        text = "Which parameters to use for search?\n(You can select several)"
        if update.message:
            await update.message.reply_text(text, reply_markup=reply_markup)
        else:
            await update.callback_query.edit_message_text(text, reply_markup=reply_markup)

    async def search_param_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        chat_id = query.message.chat_id
        state = self.user_data[chat_id]["search_state"]
        params = state["params"]
        data = query.data
        if data == "searchparam_continue":
            if not params:
                await query.answer("Please select at least one parameter!", show_alert=True)
                return
            if "similar" in params and "not_similar" in params:
                await query.answer("You can't select both: Similar and Not similar!", show_alert=True)
                return
            state["step"] = "ask_params"
            state["ask_queue"] = []
            if "prompt" in params:
                state["ask_queue"].append("prompt")
            if "author" in params:
                state["ask_queue"].append("author")
            await self.ask_next_param(update, context)
            return
        key = data.replace("searchparam_", "")
        if key == "similar":
            params.discard("not_similar")
            if "similar" in params:
                params.remove("similar")
            else:
                params.add("similar")
        elif key == "not_similar":
            params.discard("similar")
            if "not_similar" in params:
                params.remove("not_similar")
            else:
                params.add("not_similar")
        else:
            if key in params:
                params.remove(key)
            else:
                params.add(key)
        await self.send_search_param_selection(update, context)

    async def ask_next_param(self, update, context):
        chat_id = update.message.chat_id if update.message else update.callback_query.message.chat_id
        state = self.user_data[chat_id]["search_state"]
        if not state["ask_queue"]:
            state["step"] = "ask_limit"
            await context.bot.send_message(chat_id, "How many songs to show? (1-15)")
            return
        next_param = state["ask_queue"].pop(0)
        state["current_ask"] = next_param
        if next_param == "prompt":
            await context.bot.send_message(chat_id, "Enter the prompt for search:")
        elif next_param == "author":
            await context.bot.send_message(chat_id, "Enter the author for search:")

    async def handle_search_param_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.message.chat_id
        state = self.user_data[chat_id]["search_state"]
        if state.get("step") == "ask_params":
            current = state.get("current_ask")
            if current == "prompt":
                state["prompt"] = update.message.text
            elif current == "author":
                state["author"] = update.message.text
            await self.ask_next_param(update, context)
        elif state.get("step") == "ask_limit":
            try:
                limit = int(update.message.text)
                if not (1 <= limit <= 15):
                    raise ValueError
            except Exception:
                await update.message.reply_text("Please enter a number from 1 to 15.")
                return
            state["limit"] = limit
            await self.run_search(update, context)

    async def run_search(self, update, context):
        chat_id = update.message.chat_id if update.message else update.callback_query.message.chat_id
        state = self.user_data[chat_id]["search_state"]
        params = state["params"]
        prompt = state.get("prompt") if "prompt" in params else None
        author = state.get("author") if "author" in params else None
        similar = "similar" if "similar" in params else None
        not_similar = "not_similar" if "not_similar" in params else None
        limit = state.get("limit", 10)
        results = dummy_search(prompt, author, similar, not_similar, limit)
        self.user_data[chat_id]["search_results"] = results
        self.user_data[chat_id]["current_index"] = 0
        await self.send_song(update, context)

    async def send_song(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.message.chat_id if update.message else update.callback_query.message.chat_id
        data = self.user_data[chat_id]
        if not data["search_results"]:
            await context.bot.send_message(chat_id, "No search results found.")
            return
        index = data["current_index"]
        song = data["search_results"][index]
        keyboard = [
            [
                InlineKeyboardButton("Like", callback_data="like"),
                InlineKeyboardButton("Dislike", callback_data="dislike"),
            ],
            [
                InlineKeyboardButton("Previous", callback_data="previous"),
                InlineKeyboardButton("Next", callback_data="next"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await context.bot.send_message(
            chat_id,
            f"Title: {song['title']}\nAuthor: {song['author']}\nLink: {song['url']}",
            reply_markup=reply_markup,
        )

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        chat_id = query.message.chat_id
        data = self.user_data[chat_id]
        if query.data == "like":
            song = data["search_results"][data["current_index"]]
            if song not in data["liked_songs"]:
                data["liked_songs"].append(song)
            await query.answer("Liked!")
        elif query.data == "dislike":
            await query.answer("Disliked!")
        elif query.data == "next":
            if data["current_index"] < len(data["search_results"]) - 1:
                data["current_index"] += 1
                await query.message.delete()
                await self.send_song(update, context)
            else:
                await query.answer("No more songs.")
        elif query.data == "previous":
            if data["current_index"] > 0:
                data["current_index"] -= 1
                await query.message.delete()
                await self.send_song(update, context)
            else:
                await query.answer("This is the first song.")

    async def liked(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.message.chat_id
        liked_songs = self.user_data.get(chat_id, {}).get("liked_songs", [])
        if not liked_songs:
            await update.message.reply_text("You haven't liked any songs yet.")
            return
        await self.send_liked_list(update, context, page=0)

    async def send_liked_list(self, update, context, page=0):
        chat_id = update.message.chat_id if update.message else update.callback_query.message.chat_id
        liked_songs = self.user_data.get(chat_id, {}).get("liked_songs", [])
        page_size = 3
        start = page * page_size
        end = start + page_size
        page_songs = liked_songs[start:end]
        keyboard = []
        for idx, song in enumerate(page_songs):
            btn_idx = start + idx
            keyboard.append([
                InlineKeyboardButton(f"{song['title']} - {song['author']}", callback_data=f"liked_song_{btn_idx}")
            ])
        nav_buttons = []
        if start > 0:
            nav_buttons.append(InlineKeyboardButton("Previous", callback_data=f"liked_prev_{page-1}"))
        if end < len(liked_songs):
            nav_buttons.append(InlineKeyboardButton("Next", callback_data=f"liked_next_{page+1}"))
        if nav_buttons:
            keyboard.append(nav_buttons)
        reply_markup = InlineKeyboardMarkup(keyboard)
        text = f"Your liked songs (page {page+1}/{(len(liked_songs)-1)//page_size+1}):"
        if update.message:
            sent = await update.message.reply_text(text, reply_markup=reply_markup)
            return
        else:
            try:
                await update.callback_query.edit_message_text(text, reply_markup=reply_markup)
            except Exception as e:
                await context.bot.send_message(chat_id, text, reply_markup=reply_markup)

    async def liked_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        chat_id = query.message.chat_id
        liked_songs = self.user_data.get(chat_id, {}).get("liked_songs", [])
        data = query.data
        if data.startswith("liked_song_"):
            idx = int(data.split("_")[-1])
            if 0 <= idx < len(liked_songs):
                song = liked_songs[idx]
                prompt = "Your hardcoded prompt here"
                await query.answer()
                keyboard = [[InlineKeyboardButton("Search similar", callback_data=f"search_similar_{idx}")]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await context.bot.send_message(
                    chat_id,
                    f"Title: {song['title']}\nAuthor: {song['author']}\nPrompt: {prompt}\nLink: {song['url']}",
                    reply_markup=reply_markup
                )
        elif data.startswith("liked_prev_") or data.startswith("liked_next_"):
            page = int(data.split("_")[-1])
            await self.send_liked_list(update, context, page=page)

    async def search_similar_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        chat_id = query.message.chat_id
        liked_songs = self.user_data.get(chat_id, {}).get("liked_songs", [])
        data = query.data
        if data.startswith("search_similar_"):
            idx = int(data.split("_")[-1])
            if 0 <= idx < len(liked_songs):
                song = liked_songs[idx]
                results = dummy_search(None, None, song, None, 10)
                self.user_data[chat_id]["search_results"] = results
                self.user_data[chat_id]["current_index"] = 0
                await query.answer()
                await self.send_song(update, context)

load_dotenv()
token = os.getenv("TG_BOT_TOKEN")
music_bot = MusicBot()

app = ApplicationBuilder().token(token).build()
app.add_handler(CommandHandler("start", music_bot.start))
app.add_handler(CommandHandler("search", music_bot.search))
app.add_handler(CommandHandler("liked", music_bot.liked))
app.add_handler(CallbackQueryHandler(music_bot.button_callback, pattern=r"^(like|dislike|next|previous)$"))
app.add_handler(CallbackQueryHandler(music_bot.liked_callback, pattern=r"^liked_(song|prev|next)_"))
app.add_handler(CallbackQueryHandler(music_bot.search_param_callback, pattern=r"^searchparam_"))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, music_bot.handle_search_param_input))
app.add_handler(CallbackQueryHandler(music_bot.search_similar_callback, pattern=r"^search_similar_"))

app.run_polling()
