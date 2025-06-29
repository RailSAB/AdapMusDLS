from nltk.corpus import wordnet as wn
import weaviate
import nltk
import numpy as np
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

nltk.download('wordnet')

class Similarity:
    def __init__(self, client):
        self.client = client
        self.faiss_index = None
        self.song_id_mapping = None
        self.metadata = None
        self.embedding_reducer = None
        self.sentence_transformer = None
        
    def get_total_songs(self):
        try:
            response = (
                self.client.query
                .aggregate("Song")
                .with_meta_count()
                .do()
            )
            return response['data']['Aggregate']['Song'][0]['meta']['count']
        except Exception as e:
            print(f"Error getting total songs: {e}")
            return 0
        
    @staticmethod
    def expand_tags_with_synonyms(tags):
        expanded_tags = set(tags)
        for tag in tags:
            clean_tag = tag.lower().replace(" ", "_")
            for synset in wn.synsets(clean_tag):
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace("_", " ").lower()
                    expanded_tags.add(synonym)
        return list(expanded_tags)
    
    def search_by_tags_simple(self, tags, limit=10, expand_synonyms=True):
        try:
            if expand_synonyms:
                search_tags = self.expand_tags_with_synonyms(tags)
                print(f"Original tags: {tags}")
                print(f"Expanded tags: {search_tags}")
            else:
                search_tags = tags
            
            response = (
                self.client.query
                .get("Song", ["song_id", "title", "author", "tags", "url"])
                .with_limit(limit)
                .do()
            )
            
            if 'data' not in response or 'Get' not in response['data'] or 'Song' not in response['data']['Get']:
                return []
                
            songs = response['data']['Get']['Song']
            
            results = []
            for song in songs:
                song_tags = song.get('tags', [])
                
                if any(tag in song_tags for tag in search_tags):
                    matched_tags = set(song_tags) & set(search_tags)
                    match_score = len(matched_tags) / len(set(song_tags) | set(search_tags)) if song_tags else 0
                    
                    results.append({
                        'song_id': song.get('song_id', ''),
                        'title': song.get('title', ''),
                        'author': song.get('author', ''),
                        'tags': song_tags,
                        'url': song.get('url', ''),
                        'matched_tags': list(matched_tags),
                        'match_score': match_score
                    })
            
            results.sort(key=lambda x: x['match_score'], reverse=True)

            if len(results) < limit:
                remaining_limit = limit - len(results)
                query_results = self.search_by_query(" ".join(search_tags), limit=remaining_limit)

                for query_result in query_results:
                    if 'match_score' not in query_result:
                        query_result['match_score'] = query_result.get('similarity_score', 0)
                results.extend(query_results)
            return results[:limit]
        
        except Exception as e:
            print(f"Error in simple tag search: {e}")
            return []
    
    def load_faiss_index(self, index_path="faiss_index_csv.bin", mapping_path="song_id_mapping_csv.pkl", metadata_path="metadata_csv.pkl"):
        try:
            self.faiss_index = faiss.read_index(index_path)
            print(f"FAISS index loaded with {self.faiss_index.ntotal} songs")
          
            with open(mapping_path, 'rb') as f:
                self.song_id_mapping = pickle.load(f)
            print(f"Mapping loaded for {len(self.song_id_mapping)} songs")
            
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"Metadata loaded for {len(self.metadata)} songs")
            
            return True
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            return False
    
    def load_embedding_tools(self, reducer_path="../dataset/embedding_reducer.pkl", model_name="all-MiniLM-L6-v2"):
        try:
            with open(reducer_path, 'rb') as f:
                self.embedding_reducer = pickle.load(f)
            print(f"Embedding reducer loaded")
            
            self.sentence_transformer = SentenceTransformer(model_name)
            print(f"SentenceTransformer loaded: {model_name}")
            
            return True
        except Exception as e:
            print(f"Error loading embedding tools: {e}")
            return False
    
    def create_query_embedding(self, query):
        if not self.sentence_transformer or not self.embedding_reducer:
            raise ValueError("Embedding tools not loaded. Call load_embedding_tools()")
        
        embedding = self.sentence_transformer.encode(query)
        
        if embedding is None:
            raise ValueError("Failed to create embedding for query")
        scaled = self.embedding_reducer['scaler'].transform([embedding])
        
        reduced = self.embedding_reducer['transformer'].transform(scaled)
        
        reduced_normalized = reduced.astype(np.float32)
        faiss.normalize_L2(reduced_normalized)
        
        return reduced_normalized[0]
    
    def get_song_embedding_by_id(self, song_id):
        try:
            song_id_int = song_id
            
            response = (
                self.client.query
                .get("Song", ["song_id"])
                .with_additional(["vector"])
                .with_where({
                    "path": ["song_id"],
                    "operator": "Equal",
                    "valueText": song_id_int
                })
                .with_limit(1)
                .do()
            )
            
            if ('data' in response and 
                'Get' in response['data'] and 
                'Song' in response['data']['Get'] and 
                len(response['data']['Get']['Song']) > 0):
                
                song = response['data']['Get']['Song'][0]
                vector = song.get('_additional', {}).get('vector')
                if vector is not None and len(vector) > 0:
                    return np.array(vector, dtype=np.float32)
                else:
                    print(f"Empty vector for song_id {song_id}")
                    return None
            else:
                print(f"Song not found for song_id {song_id}")
                return None
        except Exception as e:
            print(f"Error getting embedding for song_id {song_id}: {e}")
            return None
    
    def get_songs_metadata_by_ids(self, song_ids):
        results = []
        for song_id in song_ids:
            try:
                song_id_int = song_id               
                response = (
                    self.client.query
                    .get("Song", ["song_id", "title", "author", "prompt", "url", "tags"])
                    .with_where({
                        "path": ["song_id"],
                        "operator": "Equal",
                        "valueText": song_id_int
                    })
                    .with_limit(1)
                    .do()
                )
                
                if ('data' in response and 
                    'Get' in response['data'] and 
                    'Song' in response['data']['Get'] and 
                    len(response['data']['Get']['Song']) > 0):
                    
                    song = response['data']['Get']['Song'][0]
                    results.append({
                        'song_id': song.get('song_id'),
                        'title': song.get('title', ''),
                        'author': song.get('author', ''),
                        'tags': song.get('tags', []),
                        'prompt': song.get('prompt', ''),
                        'url': song.get('url', '')
                    })
            except Exception as e:
                print(f"Error getting metadata for song_id {song_id}: {e}")
        return results
    
    def search_by_query(self, query, limit=10):
        if not self.faiss_index:
            raise ValueError("FAISS index not loaded. Call load_faiss_index()")
        
        query_vector = self.create_query_embedding(query)
        
        distances, indices = self.faiss_index.search(np.array([query_vector]), limit)
        
        found_song_ids = []
        for idx in indices[0]:
            if idx < len(self.song_id_mapping):
                found_song_ids.append(self.song_id_mapping[idx])
        
        metadata_results = self.get_songs_metadata_by_ids(found_song_ids)
        
        for i, result in enumerate(metadata_results):
            if i < len(distances[0]):
                result['similarity_score'] = float(distances[0][i])
        
        return metadata_results
    
    
    def search_by_song_id(self, song_id, limit=10):
        if not self.faiss_index:
            raise ValueError("FAISS index not loaded. Call load_faiss_index()")
        
        song_vector = self.get_song_embedding_by_id(song_id)
        if song_vector is None:
            return []
        
        faiss.normalize_L2(np.array([song_vector]))

        distances, indices = self.faiss_index.search(np.array([song_vector]), limit + 1)

        found_song_ids = []
        for idx in indices[0]:
            if idx < len(self.song_id_mapping):
                found_song_id = self.song_id_mapping[idx]
                if found_song_id != song_id:
                    found_song_ids.append(found_song_id)
        
        found_song_ids = found_song_ids[:limit]
        
        metadata_results = self.get_songs_metadata_by_ids(found_song_ids)
        
        score_idx = 0
        for i, result in enumerate(metadata_results):
            while score_idx < len(distances[0]) and score_idx < len(indices[0]):
                if self.song_id_mapping[indices[0][score_idx]] == song_id:
                    score_idx += 1
                    continue
                result['similarity_score'] = float(distances[0][score_idx])
                score_idx += 1
                break
        
        return metadata_results
    
    def search_by_multiple_song_ids(self, song_ids, limit=10):
        if not self.faiss_index:
            raise ValueError("FAISS index not loaded. Call load_faiss_index()")
        
        song_vectors = []
        for song_id in song_ids:
            vector = self.get_song_embedding_by_id(song_id)
            if vector is not None:
                song_vectors.append(vector)
        
        if not song_vectors:
            return []
        
        avg_vector = np.mean(song_vectors, axis=0)
        
        faiss.normalize_L2(np.array([avg_vector]))

        distances, indices = self.faiss_index.search(np.array([avg_vector]), limit + len(song_ids))

        found_song_ids = []
        for idx in indices[0]:
            if idx < len(self.song_id_mapping):
                found_song_id = self.song_id_mapping[idx]
                if found_song_id not in song_ids:
                    found_song_ids.append(found_song_id)
        
        found_song_ids = found_song_ids[:limit]
        
        metadata_results = self.get_songs_metadata_by_ids(found_song_ids)
        
        score_idx = 0
        for i, result in enumerate(metadata_results):
            while score_idx < len(distances[0]) and score_idx < len(indices[0]):
                if self.song_id_mapping[indices[0][score_idx]] in song_ids:
                    score_idx += 1
                    continue
                result['similarity_score'] = float(distances[0][score_idx])
                score_idx += 1
                break
        
        return metadata_results
    
    def search_by_multiple_song_ids_and_query(self, song_ids, query, limit=10, weight_songs=0.4, weight_query=0.6):
        if not self.faiss_index:
            raise ValueError("FAISS index not loaded. Call load_faiss_index()")
        
        song_vectors = []
        for song_id in song_ids:
            vector = self.get_song_embedding_by_id(song_id)
            if vector is not None:
                song_vectors.append(vector)
        
        query_vector = self.create_query_embedding(query)
        
        combined_vector = None
        if song_vectors and query_vector is not None:
            avg_songs_vector = np.mean(song_vectors, axis=0)
            combined_vector = weight_songs * avg_songs_vector + weight_query * query_vector
        elif song_vectors:
            combined_vector = np.mean(song_vectors, axis=0)
        elif query_vector is not None:
            combined_vector = query_vector
        else:
            return []
        
        faiss.normalize_L2(np.array([combined_vector]))

        distances, indices = self.faiss_index.search(np.array([combined_vector]), limit + len(song_ids))

        found_song_ids = []
        for idx in indices[0]:
            if idx < len(self.song_id_mapping):
                found_song_id = self.song_id_mapping[idx]
                if found_song_id not in song_ids:
                    found_song_ids.append(found_song_id)
        
        found_song_ids = found_song_ids[:limit]
        
        metadata_results = self.get_songs_metadata_by_ids(found_song_ids)
        
        score_idx = 0
        for i, result in enumerate(metadata_results):
            while score_idx < len(distances[0]) and score_idx < len(indices[0]):
                if self.song_id_mapping[indices[0][score_idx]] in song_ids:
                    score_idx += 1
                    continue
                result['similarity_score'] = float(distances[0][score_idx])
                score_idx += 1
                break
        
        return metadata_results

def main():
    
    client = weaviate.Client("http://localhost:8080")
    
    similarity = Similarity(client)
    
    print("\n1. Loading FAISS index...")
    if not similarity.load_faiss_index():
        print("Failed to load FAISS index. Make sure the files exist.")
        return
    
    print("\n2. Loading embedding tools...")
    if not similarity.load_embedding_tools():
        print("Failed to load embedding tools. Make sure the files exist.")
        return

    # Uncomment and use the following for testing as needed

    # print("\n3. Test search by query...")
    # try:
    #     query = "Music for studying and concentration"
    #     print(f"Query: '{query}'")
    #     print(similarity.search_by_query(query, limit=5))
    # except Exception as e:
    #     print(f"Error searching by query: {e}")
    
    # print("\n4. Test search by song_id ...")
    # try:
    #    print(similarity.search_by_song_id("234"))
    # except Exception as e:
    #     print(f"Error searching: {e}")
    
    # print("\n5. Test search by multiple song_ids...")
    # try:
    #     test_song_ids = similarity.song_id_mapping[:3] if len(similarity.song_id_mapping) >= 3 else similarity.song_id_mapping
    #     if len(test_song_ids) >= 2:
    #         results = similarity.search_by_multiple_song_ids(test_song_ids, limit=3)
    #         print(f"Search for similar songs for song_ids: {test_song_ids}")
    #         print(f"Found {len(results)} similar songs:")
    #         for i, result in enumerate(results, 1):
    #             print(f"  {i}. {result['title']} - {result['author']}")
    #             print(f"     Similarity: {result.get('similarity_score', 'N/A'):.3f}")
    #             print(f"     Song ID: {result['song_id']}")
    #     else:
    #         print("Not enough song_id for testing multiple search")
    # except Exception as e:
    #     print(f"Error searching by multiple song_ids: {e}")
    
    # print("\n6. Test search by song_id + query...")
    # try:
    #     test_song_ids = similarity.song_id_mapping[:2] if len(similarity.song_id_mapping) >= 2 else similarity.song_id_mapping[:1]
    #     query = "upbeat electronic dance"
    #     if test_song_ids:
    #         results = similarity.search_by_multiple_song_ids_and_query(
    #             test_song_ids, query, limit=6, weight_songs=0.6, weight_query=0.4
    #         )
    #         print(f"Combined search:")
    #         print(f"  Song IDs: {test_song_ids}")
    #         print(f"  Query: '{query}'")
    #         print(f"  Weights: songs=0.6, query=0.4")
    #         print(f"Found {len(results)} songs:")
    #         for i, result in enumerate(results, 1):
    #             print(f"  {i}. {result['title']} - {result['author']}")
    #             print(f"     Similarity: {result.get('similarity_score', 'N/A'):.3f}")
    #             print(f"     Song ID: {result['song_id']}")
    #     else:
    #         print("No available song_id for combined search")
    # except Exception as e:
    #     print(f"Error in combined search: {e}")
    
    # print("\n7. Test search by tags...")
    # try:
    #     tags = ["rock", "pop"]
    #     results = similarity.search_by_tags_simple(tags, limit=3, expand_synonyms=True)
    #     print(f"Search by tags: {tags}")
    #     print(f"Found {len(results)} songs:")
    #     for i, result in enumerate(results, 1):
    #         print(f"  {i}. {result['title']} - {result['author']}")
    #         print(f"     Match score: {result.get('match_score', 'N/A'):.3f}")
    #         print(f"     Matched tags: {result.get('matched_tags', [])}")
    # except Exception as e:
    #     print(f"Error searching by tags: {e}")

if __name__ == "__main__":
    main()
