import weaviate
import numpy as np
import faiss
import pickle
import os
import pandas as pd
from tqdm import tqdm


def get_all_songs_from_csv_and_weaviate(csv_path="youtube-cc-by-music_cleaned.csv"):
    df = pd.read_csv(csv_path)
    client = weaviate.Client("http://localhost:8080")
    
    all_songs = []
    failed_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Songs Processing"):
        url = row['url']
        
        try:
            response = (
                client.query
                .get("Song", ["song_id", "url"])
                .with_additional(["vector"])
                .with_where({
                    "path": ["url"],
                    "operator": "Equal",
                    "valueText": url
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
                song_id = song.get('song_id')
                
                if vector is not None and song_id is not None:
                    song_data = {
                        'song_id': song_id,
                        'url': url,
                        'title': row.get('title', ''),
                        'author': row.get('author', ''),
                        'tags': row.get('tags', []),
                        'prompt': row.get('prompt', ''),
                        'vector': vector
                    }
                    all_songs.append(song_data)
                else:
                    failed_count += 1
            else:
                failed_count += 1
                
        except Exception as e:
            failed_count += 1
            if idx < 10: 
                print(f"Error processing song with URL {url}: {e}")
    
    return all_songs


def build_faiss_index(embeddings_array, nlist=100):
    dimension = embeddings_array.shape[1]
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    index.train(embeddings_array)
    index.add(embeddings_array)
    
    return index


def save_index_and_mapping(index, song_ids, index_path="faiss_index.bin", mapping_path="song_id_mapping.pkl"):
    faiss.write_index(index, index_path)
    
    with open(mapping_path, 'wb') as f:
        pickle.dump(song_ids, f)


def main():
    
    all_songs = get_all_songs_from_csv_and_weaviate()

    if not all_songs:
        print("Unlucky")
        return

    embeddings = []
    song_ids = []
    metadata = []

    for song in all_songs:
        vector = song.get('vector')
        song_id = song.get('song_id')
        
        if vector is not None and song_id is not None:
            embeddings.append(vector)
            song_ids.append(song_id)
            metadata.append({
                'url': song.get('url', ''),
                'title': song.get('title', ''),
                'author': song.get('author', ''),
                'genre': song.get('genre', ''),
                'album': song.get('album', '')
            })

    if not embeddings:
        print("Unlucky")
        return

    embeddings_array = np.array(embeddings, dtype=np.float32)

    faiss.normalize_L2(embeddings_array)
    
    nlist = min(int(np.sqrt(len(embeddings))), len(embeddings) // 39)
    nlist = max(nlist, 10)
    
    index = build_faiss_index(embeddings_array, nlist=nlist)
    
    save_index_and_mapping(index, song_ids, "faiss_index_csv.bin", "song_id_mapping_csv.pkl")

    with open("metadata_csv.pkl", 'wb') as f:
        pickle.dump(metadata, f)


if __name__ == "__main__":
    main()
