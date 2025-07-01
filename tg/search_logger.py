import json
import time
import psutil
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

class SearchLogger:
    def __init__(self, log_file_path: str = "search_logs.jsonl"):
        self.log_file_path = log_file_path
        self.current_search_session = None
        
    def start_search_session(self, user_id: str, search_params: Dict[str, Any]) -> str:
        session_id = f"{user_id}_{int(time.time() * 1000)}"
        
        self.current_search_session = {
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "search_params": search_params,
            "start_time": time.time(),
            "start_memory": psutil.virtual_memory().used,
            "start_cpu": psutil.cpu_percent(),
            "results": [],
            "user_interactions": [],
            "performance_metrics": {}
        }
        
        return session_id
    
    def log_search_results(self, results: List[Dict[str, Any]], search_time: float):
        if not self.current_search_session:
            return
            
        end_memory = psutil.virtual_memory().used
        end_cpu = psutil.cpu_percent()
        memory_usage = end_memory - self.current_search_session["start_memory"]
        cpu_usage = end_cpu - self.current_search_session["start_cpu"]
        
        self.current_search_session["results"] = [
            {
                "position": i,
                "song_id": result.get("song_id", ""),
                "title": result.get("title", ""),
                "author": result.get("author", ""),
                "similarity_score": result.get("similarity_score", 0),
                "match_score": result.get("match_score", 0),
                "tags": result.get("tags", [])
            }
            for i, result in enumerate(results)
        ]
        
        self.current_search_session["performance_metrics"] = {
            "search_time_seconds": search_time,
            "memory_usage_bytes": memory_usage,
            "cpu_usage_percent": cpu_usage,
            "total_results": len(results),
            "timestamp": datetime.now().isoformat()
        }
    
    def log_user_interaction(self, action: str, song_position: int, song_data: Dict[str, Any]):
        if not self.current_search_session:
            return
            
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "action": action,  # "like", "dislike", "view"
            "song_position": song_position,
            "song_id": song_data.get("song_id", ""),
            "song_title": song_data.get("title", ""),
            "song_author": song_data.get("author", ""),
            "relevance_label": 1 if action == "like" else 0 if action == "dislike" else None
        }
        
        self.current_search_session["user_interactions"].append(interaction)
    
    def end_search_session(self):
        if not self.current_search_session:
            return
            
        self.current_search_session["end_time"] = time.time()
        self.current_search_session["total_session_duration"] = (
            self.current_search_session["end_time"] - self.current_search_session["start_time"]
        )
        
        # calculate relevance metrics
        interactions = self.current_search_session["user_interactions"]
        likes = [i for i in interactions if i["action"] == "like"]
        dislikes = [i for i in interactions if i["action"] == "dislike"]
        views = [i for i in interactions if i["action"] == "view"]
        
        unique_viewed_positions = set(i["song_position"] for i in views)
        rating_interactions = len(likes) + len(dislikes)
        
        self.current_search_session["session_summary"] = {
            "total_interactions": len(interactions),
            "likes_count": len(likes),
            "dislikes_count": len(dislikes),
            "liked_positions": [i["song_position"] for i in likes],
            "disliked_positions": [i["song_position"] for i in dislikes],
            "engagement_rate": rating_interactions / max(len(unique_viewed_positions), 1)
        }
        
        # write to log file
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(self.current_search_session, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Error writing to log file: {e}")
        
        self.current_search_session = None
    
    def log_similar_search(self, user_id: str, base_song_id: str, results: List[Dict[str, Any]], search_time: float):
        session_id = self.start_search_session(user_id, {
            "search_type": "similar_song",
            "base_song_id": base_song_id
        })
        
        self.log_search_results(results, search_time)
        return session_id


def calculate_metrics_from_logs(log_file_path: str = "search_logs.jsonl") -> Dict[str, Any]:
    def calculate_ap_at_k(relevant_positions: List[int], k: int, total_results: int) -> float:
        if not relevant_positions or k <= 0:
            return 0.0
        
        relevant_at_k = [pos for pos in relevant_positions if pos < k]
        if not relevant_at_k:
            return 0.0
        
        ap = 0.0
        num_relevant = 0
        
        for i in range(k):
            if i in relevant_at_k:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                ap += precision_at_i
        
        return ap / len(relevant_at_k)
    
    def calculate_map_at_k(sessions: List[Dict], k: int) -> float:
        ap_scores = []
        
        for session in sessions:
            liked_positions = session.get("session_summary", {}).get("liked_positions", [])
            total_results = session.get("performance_metrics", {}).get("total_results", 0)
            
            if total_results > 0:
                ap = calculate_ap_at_k(liked_positions, k, total_results)
                ap_scores.append(ap)
        
        return sum(ap_scores) / len(ap_scores) if ap_scores else 0.0
    
    try:
        sessions = []
        with open(log_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    sessions.append(json.loads(line))
        
        if not sessions:
            return {"error": "No sessions found in log file"}
        
        k_values = [1, 3, 5, 10]
        metrics = {}
        
        for k in k_values:
            metrics[f"MAP@{k}"] = calculate_map_at_k(sessions, k)
        
        total_sessions = len(sessions)
        total_searches = sum(1 for s in sessions if s.get("search_params", {}).get("search_type") != "similar_song")
        total_similar_searches = total_sessions - total_searches
        
        avg_search_time = sum(s.get("performance_metrics", {}).get("search_time_seconds", 0) for s in sessions) / total_sessions
        avg_memory_usage = sum(s.get("performance_metrics", {}).get("memory_usage_bytes", 0) for s in sessions) / total_sessions
        avg_cpu_usage = sum(s.get("performance_metrics", {}).get("cpu_usage_percent", 0) for s in sessions) / total_sessions
        
        total_likes = sum(s.get("session_summary", {}).get("likes_count", 0) for s in sessions)
        total_dislikes = sum(s.get("session_summary", {}).get("dislikes_count", 0) for s in sessions)
        total_results = sum(s.get("performance_metrics", {}).get("total_results", 0) for s in sessions)
        
        total_engagement_interactions = 0
        total_viewed_songs = 0
        
        for session in sessions:
            session_summary = session.get("session_summary", {})
            total_engagement_interactions += session_summary.get("likes_count", 0) + session_summary.get("dislikes_count", 0)
            
            interactions = session.get("user_interactions", [])
            viewed_positions = set(i["song_position"] for i in interactions if i["action"] == "view")
            total_viewed_songs += len(viewed_positions)
        
        avg_engagement = total_engagement_interactions / max(total_viewed_songs, 1)
        
        metrics.update({
            "total_sessions": total_sessions,
            "total_searches": total_searches,
            "total_similar_searches": total_similar_searches,
            "avg_search_time_seconds": avg_search_time,
            "avg_memory_usage_mb": avg_memory_usage / (1024 * 1024),
            "avg_cpu_usage_percent": avg_cpu_usage,
            "total_likes": total_likes,
            "total_dislikes": total_dislikes,
            "total_results_shown": total_results,
            "overall_like_rate": total_likes / max(total_results, 1),
            "overall_dislike_rate": total_dislikes / max(total_results, 1),
            "avg_engagement_rate": avg_engagement,
            "analysis_timestamp": datetime.now().isoformat()
        })
        
        return metrics
        
    except FileNotFoundError:
        return {"error": "Log file not found"}
    except Exception as e:
        return {"error": f"Error calculating metrics: {str(e)}"}


if __name__ == "__main__":
    # example 
    logger = SearchLogger("test_logs.jsonl")
    
    session_id = logger.start_search_session("user123", {
        "search_type": "query",
        "prompt": "relaxing music",
        "limit": 10
    })
    
    results = [
        {"song_id": "1", "title": "Calm Song", "author": "Artist A", "similarity_score": 0.9},
        {"song_id": "2", "title": "Peaceful Melody", "author": "Artist B", "similarity_score": 0.8},
        {"song_id": "3", "title": "Relaxing Tune", "author": "Artist C", "similarity_score": 0.7}
    ]
    
    logger.log_search_results(results, 0.5)
    
    logger.log_user_interaction("like", 0, results[0])
    logger.log_user_interaction("dislike", 1, results[1])
    logger.log_user_interaction("like", 2, results[2])
    
    logger.end_search_session()
    
    metrics = calculate_metrics_from_logs("test_logs.jsonl")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
