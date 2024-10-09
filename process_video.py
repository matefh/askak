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
load_dotenv()  # Add this at the beginning of your script
# Set up environment variables
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Reset the chroma_db directory
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

# Function to process YouTube video
def process_video(video_url, episode_number):
    video_id = video_url.split("v=")[1]
    # try:
    #     # Get Arabic transcript
    #     transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ar'])
    # except (NoTranscriptFound, ParseError) as e:
    #     print(f"Error processing video {video_url}: {str(e)}")
    #     return []

    # # Create documents with individual transcript entries
    # documents = []
    # prev_doc_id = None
    # for i, entry in enumerate(transcript):
    #     doc_id = f"{video_id}_{i}"
    #     metadata = {
    #         "id": doc_id,
    #         "prev_id": prev_doc_id,
    #         "source": video_url,
    #         "start": entry['start'],
    #         "duration": entry['duration']
    #     }
    #     # Filter out None values
    #     metadata = {k: v for k, v in metadata.items() if v is not None}
        
    #     doc = Document(
    #         page_content=entry['text'],
    #         metadata=metadata
    #     )
    #     documents.append(doc)
    #     prev_doc_id = doc_id

    loader = YoutubeLoader.from_youtube_url(
        video_url,
        add_video_info=True,
        transcript_format=TranscriptFormat.CHUNKS,
        chunk_size_seconds=30,
        language='ar'
    )
    documents_langchain = loader.load()


    try:
        # Get video info
        video = YouTube(video_url)
        video_info = {
            'source': video_url,
            'title': video.title or "Untitled",
            'description': video.description or "No description",
            'view_count': video.views or 0,
            'thumbnail_url': video.thumbnail_url or "",
            'publish_date': str(video.publish_date) if video.publish_date else "Unknown",
            'length': video.length or 0,
            'author': video.author or "Unknown",
            'episode_number': episode_number
        }

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
            # Manually filter complex metadata
            filtered_metadata = {}
            for key, value in doc.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    filtered_metadata[key] = value
                elif isinstance(value, (list, dict)):
                    # Convert complex types to string representation
                    filtered_metadata[key] = str(value)
            doc.metadata = filtered_metadata
            prev_doc_id = doc_id
        return documents_langchain
        
    except Exception as e:
        print(f"Error processing video info for {video_url}: {str(e)}")
        return []

# Function to load and process YouTube playlist
def load_playlist(playlist_url):
    playlist = Playlist(playlist_url)
    all_documents = []

    # Process videos in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(process_video, url, i): url for i, url in enumerate(playlist.video_urls)}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                documents = future.result()
                all_documents.extend(documents)
                print(f"Processed video: {url}")
            except Exception as exc:
                print(f"Error processing {url}: {exc}")

    if all_documents:
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
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
    playlist_url = "https://www.youtube.com/playlist?list=PLhbs8A5De9zSB471YWmrzKMyU1zMM4TH4"
    vectorstore = load_playlist(playlist_url)
    if vectorstore:
        print("Vector store created and persisted successfully.")
    else:
        print("Failed to create vector store.")
