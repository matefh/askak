import os, json, pickle
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
from openai import OpenAI
import tempfile
import random
import yt_dlp
import subprocess
load_dotenv()  # Add this at the beginning of your script
# Set up environment variables

db_directory = "chroma_db"
openai_api_key = os.environ.get("OPENAI_API_KEY")
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")

ssl_context = ssl.create_default_context(cafile=certifi.where())
print("Certificates installed successfully!")
print(f"Certificate path: {certifi.where()}")

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
                # 'description': snippet['description'],
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
                # 'description': "No description",
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

def create_documents_from_transcript(transcript, video_info):
    documents = []
    prev_doc_id = None
    
    for i, segment in enumerate(transcript.segments):
        doc_id = f"{video_info['video_id']}_{i}"
        
        # Create metadata similar to YoutubeLoader format
        metadata = {
            **video_info,
            'id': doc_id,
            'prev_id': prev_doc_id,
            'start_timestamp': segment.start,
            'end_timestamp': segment.end
        }
        
        # Filter metadata to remove complex types
        filtered_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                filtered_metadata[key] = value
            elif isinstance(value, (list, dict)):
                filtered_metadata[key] = str(value)
        
        # Create document
        doc = Document(
            page_content=segment.text,
            metadata=filtered_metadata
        )
        documents.append(doc)
        prev_doc_id = doc_id
    
    return documents

def get_transcript(output_filename):
    # Initialize OpenAI client and transcribe
    client = OpenAI()
    
    with open(output_filename, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            language="ar"
        )
    return transcript

# Function to process YouTube video without captions
def process_video_without_captions(video_url, episode_number):
    try:
        video_id = video_url.split("v=")[1]
        
        # Get metadata with retries (reusing existing function)
        video_info = get_video_metadata(video_id)
        
        if video_info is None:
            print(f"Using fallback metadata for video {video_id}")
            video_info = {
                'title': "Unknown",
                # 'description': "No description",
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

        output_filename = os.path.join('audios', f"audio_{video_id}.mp3")
        
        # Configure yt-dlp options with lower quality audio
        ydl_opts = {
            'format': 'worstaudio/worst',  # Get smallest audio stream
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '32',  # Lower bitrate (32kbps)
            }],
            'outtmpl': output_filename[:-4],
            'postprocessor_args': [
                '-ac', '1',  # Mono channel
                '-ar', '16000',  # 16kHz sample rate
            ],
            'verbose': True,
        }
        
        try:
            if not os.path.exists(output_filename):
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
                    
            transcript_filename = os.path.join('transcripts', f'transcript_{video_id}.pkl')
            if os.path.exists(transcript_filename):
                print(f"Loading existing transcript for video {video_id}")
                with open(transcript_filename, 'rb') as f:
                    transcript = pickle.load(f)
            else:
                print(f"Creating new transcript for video {video_id}")
                transcript = get_transcript(output_filename)
                with open(transcript_filename, 'wb') as f:
                    pickle.dump(transcript, f)
            return create_documents_from_transcript(transcript, video_info)

        except Exception as e:
            print(f"Error during download/transcription: {str(e)}")
            raise

    except Exception as e:
        print(f"Error processing video without captions {video_url}: {str(e)}")
        return []

# Function to load and process YouTube playlist
def load_playlist(playlist_url, with_captions=True):
    playlist = Playlist(playlist_url)
    all_documents = []

    for i, url in enumerate(playlist.video_urls):
        try:
            # documents = process_video_without_captions("https://www.youtube.com/watch?v=7-Kt5wptOog", i)
            documents = process_video_without_captions(url, i) if not with_captions else process_video(url, i)
            all_documents.extend(documents)
            print(f"Processed video: {url}")
            # Add delay between videos
            time.sleep(1)
        except Exception as exc:
            print(f"Error processing {url}: {exc}")
            continue
    print(len(all_documents), all_documents[:5])
    # Merge documents that are too short (less than 100 characters)
    merged_documents = []
    current_doc = None
    
    for doc in all_documents:
        if not current_doc:
            current_doc = doc
            continue
            
        # If current document is too short, merge with next
        if len(current_doc.page_content) < 500:
            # Combine metadata
            merged_metadata = current_doc.metadata.copy()
            if 'end_timestamp' in doc.metadata:
                merged_metadata['end_timestamp'] = doc.metadata['end_timestamp']
            
            # Combine content
            merged_content = current_doc.page_content + " " + doc.page_content
            
            # Create new merged document
            current_doc = Document(
                page_content=merged_content,
                metadata=merged_metadata
            )
        else:
            merged_documents.append(current_doc)
            current_doc = doc
    
    # Add final document
    if current_doc:
        merged_documents.append(current_doc)
        
    all_documents = merged_documents
    if all_documents:
        text_splitter = SemanticChunker(
            OpenAIEmbeddings(), breakpoint_threshold_type="gradient"
        )
        chunks = text_splitter.split_documents(all_documents)

        vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=db_directory)
        vectorstore.persist()
        return vectorstore
    else:
        print("No valid documents were created from the videos.")
        return None

if __name__ == "__main__":
    # Initialize the client with better error handling
    try:
        youtube = create_youtube_client()
        print("YouTube client initialized successfully")
    except Exception as e:
        print(f"Failed to initialize YouTube client: {e}")
        raise

    # Initialize embeddings
    embeddings = OpenAIEmbeddings()

    # if os.path.exists(db_directory):
    #     shutil.rmtree(db_directory)

    # transcript = get_transcript("audio_20O8gzvRsmE.mp3")
    # print(transcript)

    playlist_urls_with_captions = [
        "https://www.youtube.com/playlist?list=PLhbs8A5De9zSB471YWmrzKMyU1zMM4TH4",
        "https://www.youtube.com/playlist?list=PLhbs8A5De9zQQ2RjNZvIMEQ_JWYaBGP8e",
    ]
    playlist_urls_without_captions = [
        # "https://www.youtube.com/playlist?list=PLhbs8A5De9zSVObgh99rhe7FDHdw69Ua2",
        # "https://www.youtube.com/playlist?list=PLhbs8A5De9zSvoxMrljw59xlkZXPWTB46",
        # "https://www.youtube.com/playlist?list=PLhbs8A5De9zTnrwu4_lJvwrhS07oWudQl"
    ]
    for playlist_url in playlist_urls_with_captions:
        vectorstore = load_playlist(playlist_url, with_captions=True)
        if vectorstore:
            print("Vector store created and persisted successfully.")
        else:
            print("Failed to create vector store.")
    for playlist_url in playlist_urls_without_captions:
        vectorstore = load_playlist(playlist_url, with_captions=False)
        if vectorstore:
            print("Vector store created and persisted successfully.")
        else:
            print("Failed to create vector store.")
