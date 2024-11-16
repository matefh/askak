import os
import shutil
import concurrent.futures
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pytube import Playlist, YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound
from langchain.schema import Document
from xml.etree.ElementTree import ParseError
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders.youtube import YoutubeLoader, TranscriptFormat
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv
from pytube.exceptions import PytubeError
import time
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import isodate  # for parsing duration
from datetime import datetime
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from googleapiclient.http import HttpRequest
import httplib2
import ssl
import certifi
import socket
from googleapiclient.discovery import build
import httplib2
from http.client import IncompleteRead
load_dotenv()  # Add this at the beginning of your script
# Set up environment variables
openai_api_key = os.environ.get("OPENAI_API_KEY")
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")

ssl_context = ssl.create_default_context(cafile=certifi.where())
print("Certificates installed successfully!")
print(f"Certificate path: {certifi.where()}")

# Recreate the YouTube API client with retry settings
def create_youtube_client():
    # Set a global timeout
    socket.setdefaulttimeout(30)
    
    # Create custom HTTP object with extended timeout
    http = httplib2.Http(
        timeout=30,
        ca_certs=certifi.where()
    )
    
    return build(
        'youtube', 
        'v3', 
        developerKey=YOUTUBE_API_KEY,
        http=http,
        cache_discovery=False
    )

# Initialize the client with better error handling
try:
    youtube = create_youtube_client()
    print("YouTube client initialized successfully")
except Exception as e:
    print(f"Failed to initialize YouTube client: {e}")
    raise

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Reset the chroma_db directory
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

def get_video_metadata(video_id, max_retries=3):
    for attempt in range(max_retries):
        try:
            # Add delay between attempts
            if attempt > 0:
                time.sleep(2 ** attempt)
            
            # Create a new client for each request to avoid memory issues
            youtube = create_youtube_client()
            
            video_response = youtube.videos().list(
                part='snippet,contentDetails,statistics',
                id=video_id
            ).execute()

            if not video_response.get('items'):
                print(f"No video data found for ID: {video_id}")
                return None

            video_data = video_response['items'][0]
            snippet = video_data['snippet']
            content_details = video_data['contentDetails']
            statistics = video_data['statistics']

            # Clean up the client
            del youtube

            return {
                'title': snippet['title'],
                'description': snippet['description'],
                'publish_date': datetime.strptime(
                    snippet['publishedAt'], 
                    '%Y-%m-%dT%H:%M:%SZ'
                ).strftime('%Y-%m-%d'),
                'thumbnail_url': snippet['thumbnails']['maxres']['url'] 
                    if 'maxres' in snippet['thumbnails'] 
                    else snippet['thumbnails']['high']['url'],
                'length': int(isodate.parse_duration(content_details['duration']).total_seconds()),
                'view_count': int(statistics.get('viewCount', 0)),
                'author': snippet['channelTitle']
            }
            
        except IncompleteRead:
            print(f"IncompleteRead error on attempt {attempt + 1} for video {video_id}")
            if attempt == max_retries - 1:
                return None
            continue
        except socket.timeout:
            print(f"Timeout on attempt {attempt + 1} for video {video_id}")
            continue
        except HttpError as e:
            print(f"HTTP error on attempt {attempt + 1} for video {video_id}: {str(e)}")
            if e.resp.status in [429, 500, 502, 503, 504]:
                continue
            return None
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1} for video {video_id}: {str(e)}")
            if attempt == max_retries - 1:
                return None
            continue
        finally:
            # Ensure cleanup
            try:
                del video_response
                del video_data
            except:
                pass
    
    return None

# Function to process YouTube video
def process_video(video_url, episode_number):
    try:
        video_id = video_url.split("v=")[1]
        
        # Get metadata with retries
        video_info = get_video_metadata(video_id)
        
        # If metadata fetch fails, use basic info
        if video_info is None:
            print(f"Using fallback metadata for video {video_id}")
            video_info = {
                'title': "Unknown",
                'description': "No description",
                'publish_date': "Unknown",
                'thumbnail_url': f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
                'length': 0,
                'view_count': 0,
                'author': "Unknown"
            }

        # Add additional info
        video_info.update({
            'source': video_url,
            'video_id': video_id,
            'episode_number': episode_number
        })

        # Add delay between requests to avoid rate limiting
        time.sleep(0.5)

        # Use YoutubeLoader for transcript
        loader = YoutubeLoader.from_youtube_url(
            video_url,
            add_video_info=False,
            transcript_format=TranscriptFormat.CHUNKS,
            chunk_size_seconds=30,
            language='ar'
        )
        
        documents_langchain = loader.load()

        # Update metadata for all documents
        prev_doc_id = None
        for i, doc in enumerate(documents_langchain):
            doc_id = f"{video_id}_{i}"
            metadata = {
                **video_info,
                'id': doc_id,
                'prev_id': prev_doc_id
            }
            doc.metadata.update(metadata)
            filtered_metadata = {}
            for key, value in doc.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    filtered_metadata[key] = value
                elif isinstance(value, (list, dict)):
                    filtered_metadata[key] = str(value)
            doc.metadata = filtered_metadata
            prev_doc_id = doc_id
        return documents_langchain

    except Exception as e:
        print(f"Error processing video {video_url}: {str(e)}")
        return []

# Function to load and process YouTube playlist
def load_playlist(playlist_url):
    playlist = Playlist(playlist_url)
    all_documents = []

    # Process videos sequentially instead of in parallel
    for i, url in enumerate(playlist.video_urls):
        try:
            documents = process_video(url, i)
            all_documents.extend(documents)
            print(f"Processed video: {url}")
            # Add delay between videos
            time.sleep(1)
        except Exception as exc:
            print(f"Error processing {url}: {exc}")
            continue

    if all_documents:
        text_splitter = SemanticChunker(
            OpenAIEmbeddings(), breakpoint_threshold_type="gradient"
        )
        chunks = text_splitter.split_documents(all_documents)
        
        vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
        vectorstore.persist()
        return vectorstore
    else:
        print("No valid documents were created from the videos.")
        return None

if __name__ == "__main__":
    playlist_urls = [
        "https://www.youtube.com/playlist?list=PLhbs8A5De9zSB471YWmrzKMyU1zMM4TH4",
        "https://www.youtube.com/playlist?list=PLhbs8A5De9zQQ2RjNZvIMEQ_JWYaBGP8e",
        "https://www.youtube.com/playlist?list=PLhbs8A5De9zSVObgh99rhe7FDHdw69Ua2",
        "https://www.youtube.com/playlist?list=PLhbs8A5De9zSvoxMrljw59xlkZXPWTB46",
        "https://www.youtube.com/playlist?list=PLhbs8A5De9zTnrwu4_lJvwrhS07oWudQl"
    ]
    for playlist_url in playlist_urls:
        vectorstore = load_playlist(playlist_url)
        if vectorstore:
            print("Vector store created and persisted successfully.")
        else:
            print("Failed to create vector store.")
