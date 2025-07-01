from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
from dotenv import load_dotenv
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'similarity_search'))

from search_logger import SearchLogger

try:
    import weaviate
    from similarity import Similarity
    SIMILARITY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import similarity modules: {e}")
    SIMILARITY_AVAILABLE = False
    weaviate = None
    Similarity = None

class MusicBot:
    def __init__(self):
        self.user_data = {}
        self.similarity = None
        self.weaviate_client = None
        self.logger = SearchLogger("search_logs.jsonl")
        
        if SIMILARITY_AVAILABLE:
            try:
                self.weaviate_client = weaviate.Client("http://localhost:8080")
                self.similarity = Similarity(self.weaviate_client)
                
                base_path = os.path.join(os.path.dirname(__file__), '..', 'similarity_search')
                faiss_path = os.path.join(base_path, 'faiss_index_csv.bin')
                mapping_path = os.path.join(base_path, 'song_id_mapping_csv.pkl')
                metadata_path = os.path.join(base_path, 'metadata_csv.pkl')
                
                if not self.similarity.load_faiss_index(faiss_path, mapping_path, metadata_path):
                    print("Warning: Failed to load FAISS index")
                
                reducer_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'embedding_reducer.pkl')
                if not self.similarity.load_embedding_tools(reducer_path):
                    print("Warning: Failed to load embedding tools")
                    
                print("Similarity search initialized successfully")
            except Exception as e:
                print(f"Failed to initialize similarity search: {e}")
                self.similarity = None
        else:
            print("Similarity search not available - using fallback mode")
    
    def dummy_search(self, prompt, tags, similar_to, not_similar_to, limit):
        songs = []
        for i in range(1, min(limit + 1, 21)):
            song = {
                "song_id": str(i),
                "title": f"Song {i}",
                "author": f"Author {i}", 
                "url": f"http://example.com/song{i}",
                "tags": ["pop", "rock"] if i % 2 == 0 else ["jazz", "blues"],
                "prompt": f"A sample song {i} for testing"
            }
            songs.append(song)
        return songs
    
    def extract_song_ids_from_liked_songs(self, liked_songs):
        song_ids = []
        for song in liked_songs:
            if 'song_id' in song:
                song_ids.append(song['song_id'])
        return song_ids
    
    def format_search_results(self, results):
        formatted = []
        for result in results:
            formatted.append({
                'song_id': result.get('song_id', ''),
                'title': result.get('title', ''),
                'author': result.get('author', ''),
                'url': result.get('url', ''),
                'tags': result.get('tags', []),
                'prompt': result.get('prompt', ''),
                'similarity_score': result.get('similarity_score', 0),
                'match_score': result.get('match_score', 0)
            })
        return formatted
    
    def get_song_status(self, song, liked_songs, disliked_songs):
        song_id = song.get('song_id')
        if song_id:
            for liked_song in liked_songs:
                if liked_song.get('song_id') == song_id:
                    return "liked", "🟢"
        else:
            for liked_song in liked_songs:
                if (liked_song.get('title') == song.get('title') and 
                    liked_song.get('author') == song.get('author')):
                    return "liked", "🟢"
        if song_id:
            for disliked_song in disliked_songs:
                if disliked_song.get('song_id') == song_id:
                    return "disliked", "🔴"
        else:
            for disliked_song in disliked_songs:
                if (disliked_song.get('title') == song.get('title') and 
                    disliked_song.get('author') == song.get('author')):
                    return "disliked", "🔴"
        return "none", ""
    
    def remove_from_list(self, song, song_list):
        song_id = song.get('song_id')
        if song_id:
            for i, list_song in enumerate(song_list):
                if list_song.get('song_id') == song_id:
                    song_list.pop(i)
                    return True
        else:
            for i, list_song in enumerate(song_list):
                if (list_song.get('title') == song.get('title') and 
                    list_song.get('author') == song.get('author')):
                    song_list.pop(i)
                    return True
        return False

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Welcome to the Music Bot! Use /search to find music and /liked to view your liked songs.")

    async def search(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.message.chat_id
        if chat_id in self.user_data and "current_session_id" in self.user_data[chat_id]:
            self.logger.end_search_session()
        self.user_data[chat_id] = {
            "search_results": [],
            "current_index": 0,
            "liked_songs": self.user_data.get(chat_id, {}).get("liked_songs", []),
            "disliked_songs": self.user_data.get(chat_id, {}).get("disliked_songs", []),
            "search_state": {
                "params": set(),
                "step": "choose_params",
                "prompt": None,
                "tags": None,
                "limit": None,
            },
            "similar_search_state": {
                "step": None,
                "song_idx": None,
                "limit": None,
            },
        }
        await self.send_search_param_selection(update, context)

    async def send_search_param_selection(self, update, context):
        chat_id = update.message.chat_id if update.message else update.callback_query.message.chat_id
        state = self.user_data[chat_id]["search_state"]
        params = state["params"]
        def btn(label, key):
            if "tags" in params and key != "tags":
                return InlineKeyboardButton(f"🔴 {label}", callback_data=f"disabled_{key}")
            elif key in params:
                return InlineKeyboardButton(f"🟢 {label}", callback_data=f"searchparam_{key}")
            else:
                return InlineKeyboardButton(label, callback_data=f"searchparam_{key}")
        def btn_similar(label, key):
            if "tags" in params:
                return InlineKeyboardButton(f"🔴 {label}", callback_data=f"disabled_{key}")
            elif key in params:
                return InlineKeyboardButton(f"🟢 {label}", callback_data=f"searchparam_{key}")
            else:
                return InlineKeyboardButton(label, callback_data=f"searchparam_{key}")
        keyboard = [
            [btn("By prompt", "prompt")],
            [btn("By tags", "tags")],
            [btn_similar("Similar by liked songs", "similar")],
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
        if data.startswith("disabled_"):
            await query.answer("This option is disabled when 'By tags' is selected!", show_alert=True)
            return
        if data == "searchparam_continue":
            if not params:
                await query.answer("Please select at least one parameter!", show_alert=True)
                return
            state["step"] = "ask_params"
            state["ask_queue"] = []
            if "prompt" in params:
                state["ask_queue"].append("prompt")
            if "tags" in params:
                state["ask_queue"].append("tags")
            await self.ask_next_param(update, context)
            return
        key = data.replace("searchparam_", "")
        if key == "tags":
            if "tags" in params:
                params.remove("tags")
            else:
                params.clear()
                params.add("tags")
        elif key == "similar":
            if "tags" in params:
                await query.answer("Cannot select this when 'By tags' is active!", show_alert=True)
                return
            if "similar" in params:
                params.remove("similar")
            else:
                params.add("similar")
        else:
            if "tags" in params and key != "tags":
                await query.answer("Cannot select this when 'By tags' is active!", show_alert=True)
                return
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
        elif next_param == "tags":
            await context.bot.send_message(chat_id, "Enter the tags, splitted by comma for search:")

    async def handle_search_param_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.message.chat_id
        if chat_id in self.user_data and self.user_data[chat_id].get("similar_search_state", {}).get("step") == "ask_limit":
            try:
                limit = int(update.message.text)
                if not (1 <= limit <= 15):
                    raise ValueError
                similar_state = self.user_data[chat_id]["similar_search_state"]
                similar_state["limit"] = limit
                similar_state["step"] = "execute"
                await self.execute_similar_search(update, context)
                return
            except ValueError:
                await update.message.reply_text("Please enter a number from 1 to 15.")
                return
        state = self.user_data.get(chat_id, {}).get("search_state", {})
        if state.get("step") == "ask_params":
            current = state.get("current_ask")
            if current == "prompt":
                state["prompt"] = update.message.text
            elif current == "tags":
                state["tags"] = update.message.text
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

    async def execute_similar_search(self, update, context):
        chat_id = update.message.chat_id
        user_id = str(update.effective_user.id) if update.effective_user else str(chat_id)
        similar_state = self.user_data[chat_id]["similar_search_state"]
        liked_songs = self.user_data.get(chat_id, {}).get("liked_songs", [])
        if "disliked_songs" not in self.user_data[chat_id]:
            self.user_data[chat_id]["disliked_songs"] = []
        song_idx = similar_state["song_idx"]
        limit = similar_state["limit"]
        if 0 <= song_idx < len(liked_songs):
            song = liked_songs[song_idx]
            song_id = song.get('song_id')
            search_params = {
                "search_type": "similar_song",
                "base_song_id": song_id,
                "base_song_title": song.get('title', ''),
                "base_song_author": song.get('author', ''),
                "limit": limit,
                "search_method": None
            }
            session_id = self.logger.start_search_session(user_id, search_params)
            search_start_time = time.time()
            try:
                if self.similarity and song_id:
                    search_params["search_method"] = "search_by_song_id"
                    results = self.similarity.search_by_song_id(song_id, limit=limit)
                    formatted_results = self.format_search_results(results)
                else:
                    search_params["search_method"] = "dummy_search"
                    results = self.dummy_search(None, None, song, None, limit)
                    formatted_results = self.format_search_results(results)
                search_time = time.time() - search_start_time
                self.logger.log_search_results(formatted_results, search_time)
                self.user_data[chat_id]["current_session_id"] = session_id
                self.user_data[chat_id]["search_results"] = formatted_results
                self.user_data[chat_id]["current_index"] = 0
                similar_state["step"] = None
                similar_state["song_idx"] = None
                similar_state["limit"] = None
                if formatted_results:
                    await self.send_song(update, context)
                else:
                    await update.message.reply_text("No similar songs found.")
                    self.logger.end_search_session()
            except Exception as e:
                print(f"Similar search error: {e}")
                await update.message.reply_text(f"Search failed: {str(e)}")
                self.logger.end_search_session()

    async def run_search(self, update, context):
        chat_id = update.message.chat_id if update.message else update.callback_query.message.chat_id
        user_id = str(update.effective_user.id) if update.effective_user else str(chat_id)
        state = self.user_data[chat_id]["search_state"]
        params = state["params"]
        prompt = state.get("prompt") if "prompt" in params else None
        tags = state.get("tags") if "tags" in params else None
        similar = "similar" if "similar" in params else None
        limit = state.get("limit", 10)
        liked_songs = self.user_data.get(chat_id, {}).get("liked_songs", [])
        liked_song_ids = self.extract_song_ids_from_liked_songs(liked_songs)
        search_params = {
            "search_type": "regular_search",
            "prompt": prompt,
            "tags": tags,
            "similar": similar,
            "limit": limit,
            "liked_songs_count": len(liked_song_ids),
            "search_method": None
        }
        session_id = self.logger.start_search_session(user_id, search_params)
        search_start_time = time.time()
        try:
            results = []
            if self.similarity:
                if "tags" in params and tags:
                    search_params["search_method"] = "search_by_tags_simple"
                    tag_list = [tag.strip() for tag in tags.split(',')]
                    results = self.similarity.search_by_tags_simple(tag_list, limit=limit, expand_synonyms=True)
                elif prompt and similar and liked_song_ids:
                    search_params["search_method"] = "search_by_multiple_song_ids_and_query"
                    results = self.similarity.search_by_multiple_song_ids_and_query(
                        liked_song_ids, prompt, limit=limit, weight_songs=0.4, weight_query=0.6
                    )
                elif prompt:
                    search_params["search_method"] = "search_by_query"
                    results = self.similarity.search_by_query(prompt, limit=limit)
                elif similar and liked_song_ids:
                    search_params["search_method"] = "search_by_multiple_song_ids"
                    results = self.similarity.search_by_multiple_song_ids(liked_song_ids, limit=limit)
                else:
                    await context.bot.send_message(chat_id, "No valid search parameters provided.")
                    return
                formatted_results = self.format_search_results(results)
            else:
                search_params["search_method"] = "dummy_search"
                await context.bot.send_message(chat_id, "Using fallback search (similarity not available)...")
                results = self.dummy_search(prompt, tags, similar, None, limit)
                formatted_results = self.format_search_results(results)
            search_time = time.time() - search_start_time
            self.logger.log_search_results(formatted_results, search_time)
            self.user_data[chat_id]["current_session_id"] = session_id
            self.user_data[chat_id]["search_results"] = formatted_results
            self.user_data[chat_id]["current_index"] = 0
            if formatted_results:
                await self.send_song(update, context)
            else:
                await context.bot.send_message(chat_id, "No songs found matching your criteria.")
                self.logger.end_search_session()
        except Exception as e:
            print(f"Search error: {e}")
            await context.bot.send_message(chat_id, f"Search failed: {str(e)}")
            self.logger.end_search_session()
            return

    async def send_song(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.message.chat_id if update.message else update.callback_query.message.chat_id
        data = self.user_data[chat_id]
        if not data["search_results"]:
            await context.bot.send_message(chat_id, "No search results found.")
            return
        index = data["current_index"]
        song = data["search_results"][index]
        if "current_session_id" in data:
            self.logger.log_user_interaction("view", index, song)
        liked_songs = data.get("liked_songs", [])
        disliked_songs = data.get("disliked_songs", [])
        status, marker = self.get_song_status(song, liked_songs, disliked_songs)
        song_info = f"{marker} Title: {song['title']}\nAuthor: {song['author']}"
        if song.get('prompt'):
            prompt = song['prompt']
            if len(prompt) > 150:
                prompt = prompt[:150] + "..."
            song_info += f"\nPrompt: {prompt}"
        if song.get('tags'):
            tags_str = ', '.join(song['tags'][:3])
            if len(song['tags']) > 3:
                tags_str += f" (+{len(song['tags'])-3} more)"
            song_info += f"\nTags: {tags_str}"
        song_info += f"\nLink: {song['url']}"
        song_info += f"\n\nSong {index + 1} of {len(data['search_results'])}"
        like_text = "Like"
        dislike_text = "Dislike"
        if status == "liked":
            like_text = "🟢 Liked"
        elif status == "disliked":
            dislike_text = "🔴 Disliked"
        keyboard = [
            [
                InlineKeyboardButton(like_text, callback_data="like"),
                InlineKeyboardButton(dislike_text, callback_data="dislike"),
            ],
            [
                InlineKeyboardButton("Previous", callback_data="previous"),
                InlineKeyboardButton("Next", callback_data="next"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await context.bot.send_message(
            chat_id,
            song_info,
            reply_markup=reply_markup,
        )

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        chat_id = query.message.chat_id
        data = self.user_data[chat_id]
        current_song = data["search_results"][data["current_index"]]
        current_position = data["current_index"]
        if query.data == "like":
            song = data["search_results"][data["current_index"]]
            liked_songs = data.get("liked_songs", [])
            disliked_songs = data.get("disliked_songs", [])
            self.remove_from_list(song, disliked_songs)
            status, _ = self.get_song_status(song, liked_songs, disliked_songs)
            if status != "liked":
                liked_songs.append(song)
                await query.answer("Liked! 🟢")
                if "current_session_id" in data:
                    self.logger.log_user_interaction("like", current_position, current_song)
            else:
                await query.answer("Already liked!")
        elif query.data == "dislike":
            song = data["search_results"][data["current_index"]]
            liked_songs = data.get("liked_songs", [])
            disliked_songs = data.get("disliked_songs", [])
            self.remove_from_list(song, liked_songs)
            status, _ = self.get_song_status(song, liked_songs, disliked_songs)
            if status != "disliked":
                disliked_songs.append(song)
                await query.answer("Disliked! 🔴")
                if "current_session_id" in data:
                    self.logger.log_user_interaction("dislike", current_position, current_song)
            else:
                await query.answer("Already disliked!")
        elif query.data == "next":
            if data["current_index"] < len(data["search_results"]) - 1:
                data["current_index"] += 1
                await query.message.delete()
                await self.send_song(update, context)
                new_song = data["search_results"][data["current_index"]]
                new_position = data["current_index"]
                if "current_session_id" in data:
                    self.logger.log_user_interaction("view", new_position, new_song)
            else:
                await query.answer("No more songs.")
        elif query.data == "previous":
            if data["current_index"] > 0:
                data["current_index"] -= 1
                await query.message.delete()
                await self.send_song(update, context)
                new_song = data["search_results"][data["current_index"]]
                new_position = data["current_index"]
                if "current_session_id" in data:
                    self.logger.log_user_interaction("view", new_position, new_song)
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
                await query.answer()
                song_info = f"Title: {song['title']}\nAuthor: {song['author']}"
                if song.get('prompt'):
                    prompt = song['prompt']
                    if len(prompt) > 100:
                        prompt = prompt[:100] + "..."
                    song_info += f"\nPrompt: {prompt}"
                if song.get('tags'):
                    tags_str = ', '.join(song['tags'][:5])
                    if len(song['tags']) > 5:
                        tags_str += f" (+{len(song['tags'])-5} more)"
                    song_info += f"\nTags: {tags_str}"
                song_info += f"\nLink: {song['url']}"
                keyboard = [[InlineKeyboardButton("Search similar", callback_data=f"search_similar_{idx}")]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await context.bot.send_message(
                    chat_id,
                    song_info,
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
                if chat_id not in self.user_data:
                    self.user_data[chat_id] = {
                        "similar_search_state": {},
                        "disliked_songs": []
                    }
                elif "similar_search_state" not in self.user_data[chat_id]:
                    self.user_data[chat_id]["similar_search_state"] = {}
                if "disliked_songs" not in self.user_data[chat_id]:
                    self.user_data[chat_id]["disliked_songs"] = []
                self.user_data[chat_id]["similar_search_state"]["step"] = "ask_limit"
                self.user_data[chat_id]["similar_search_state"]["song_idx"] = idx
                await query.answer()
                await context.bot.send_message(
                    chat_id, 
                    "How many similar songs to find? (1-15)"
                )

    async def metrics(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.message.chat_id
        if chat_id in self.user_data and "current_session_id" in self.user_data[chat_id]:
            self.logger.end_search_session()
        try:
            from search_logger import calculate_metrics_from_logs
            metrics = calculate_metrics_from_logs("search_logs.jsonl")
            if "error" in metrics:
                await update.message.reply_text(f"Error getting metrics: {metrics['error']}")
                return
            report = f"""**Search Metrics Report**

**Search Statistics:**
• Total sessions: {metrics.get('total_sessions', 0)}
• Regular searches: {metrics.get('total_searches', 0)}
• Similar searches: {metrics.get('total_similar_searches', 0)}

**Performance:**
• Avg search time: {metrics.get('avg_search_time_seconds', 0):.3f}s
• Avg memory usage: {metrics.get('avg_memory_usage_mb', 0):.2f} MB
• Avg CPU usage: {metrics.get('avg_cpu_usage_percent', 0):.2f}%

**User Engagement:**
• Total likes: {metrics.get('total_likes', 0)}
• Total dislikes: {metrics.get('total_dislikes', 0)}
• Like rate: {(metrics.get('overall_like_rate', 0) * 100):.2f}%
• Dislike rate: {(metrics.get('overall_dislike_rate', 0) * 100):.2f}%
• Engagement rate: {(metrics.get('avg_engagement_rate', 0) * 100):.2f}%

**Relevance Metrics:**
• MAP@1: {metrics.get('MAP@1', 0):.3f}
• MAP@3: {metrics.get('MAP@3', 0):.3f}
• MAP@5: {metrics.get('MAP@5', 0):.3f}
• MAP@10: {metrics.get('MAP@10', 0):.3f}

Analysis time: {metrics.get('analysis_timestamp', 'Unknown')}
"""
            await update.message.reply_text(report, parse_mode='Markdown')
        except Exception as e:
            await update.message.reply_text(f"Error calculating metrics: {str(e)}")

def cleanup_sessions():
    if music_bot.logger.current_search_session:
        music_bot.logger.end_search_session()

import signal
import atexit

atexit.register(cleanup_sessions)
signal.signal(signal.SIGINT, lambda sig, frame: cleanup_sessions())
signal.signal(signal.SIGTERM, lambda sig, frame: cleanup_sessions())

load_dotenv()
token = os.getenv("TG_BOT_TOKEN")
music_bot = MusicBot()

app = ApplicationBuilder().token(token).build()
app.add_handler(CommandHandler("start", music_bot.start))
app.add_handler(CommandHandler("search", music_bot.search))
app.add_handler(CommandHandler("liked", music_bot.liked))
app.add_handler(CommandHandler("metrics", music_bot.metrics))
app.add_handler(CallbackQueryHandler(music_bot.button_callback, pattern=r"^(like|dislike|next|previous)$"))
app.add_handler(CallbackQueryHandler(music_bot.liked_callback, pattern=r"^liked_(song|prev|next)_"))
app.add_handler(CallbackQueryHandler(music_bot.search_param_callback, pattern=r"^searchparam_"))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, music_bot.handle_search_param_input))
app.add_handler(CallbackQueryHandler(music_bot.search_similar_callback, pattern=r"^search_similar_"))

app.run_polling()
