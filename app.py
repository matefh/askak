import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from langchain.schema import Document
import re, os, json
from arabic_support import support_arabic_text
from dotenv import load_dotenv
load_dotenv()  # Add this at the beginning of your script
# Function to get answer from Claude
def get_claude_response(question, context):
    system_prompt = """
# Instructions
- You are a helpful assistant that can answer questions about a video series by Dr. Amr Khaled.
- The videos are in Arabic.
- The series is called "الفهم عن الله".
- You will be provided the transcripts of the most relevant clips to the user's question.
- You should answer the user's question ONLY IN ARABIC.
- You should reference the clips by the episode number and minute number and timestamp in this format `(2 - 00:41:00)`.
- You should answer the question directly and concisely.
- You should structure your answer into numbered bullet points.
- You should ensure that the referenced transcripts are diverse and from different parts of the series to maximize coverage and diversity.
- Do not mention the same part of the same episode more than once.
"""

    prompt = f"""{HUMAN_PROMPT}
Relevant transcripts:
{context}

Now answer this question ONLY IN ARABIC by referencing quotes from the context along with the episode number and minute number and timestamp in this format `(2 - 00:41:00)`:
{question}

{AI_PROMPT}"""

    print(system_prompt)
    print(prompt)
    response = anthropic.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1000,
        system=system_prompt,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.content[0].text

# Function to get unique videos from the vector store
def get_unique_videos():
    all_docs = vectorstore.get()
    unique_videos = {}
    for metadata in all_docs['metadatas']:
        video_id = metadata['source'].split('v=')[1]
        if video_id not in unique_videos:
            unique_videos[video_id] = metadata.copy()
    return unique_videos

def replace_timestamps(answer, videos_dict):
    def timestamp_to_link(match):
        title, episode_number, start_timestamp = match.groups()
        start_seconds = sum(int(x) * 60 ** i for i, x in enumerate(reversed(start_timestamp.split(':'))))
        video_url = f"{videos_dict[int(episode_number) - 1]['source']}&t={start_seconds - 5}s"
        return f"[{title} {episode_number} - {start_timestamp}]({video_url})"
    print(answer)
    pattern = r"(\*\*.*?\*\*)\s*\((\d+)\s*-\s*(\d{2}:\d{2}:\d{2})\)"
    replaced_answer = re.sub(pattern, timestamp_to_link, answer)
    return replaced_answer

# Function to get surrounding context
def get_surrounding_context(doc, all_docs):
    current_id = doc.metadata['id']
    prev_id = doc.metadata.get('prev_id')
    
    # Get previous document
    prev_doc = None
    if prev_id:
        prev_index = next((i for i, m in enumerate(all_docs['metadatas']) if m.get('id') == prev_id), None)
        if prev_index is not None:
            prev_doc = Document(
                page_content=all_docs['documents'][prev_index],
                metadata=all_docs['metadatas'][prev_index]
            )
    
    # Get following document
    following_doc = None
    following_index = next((i for i, m in enumerate(all_docs['metadatas']) if m.get('prev_id') == current_id), None)
    if following_index is not None:
        following_doc = Document(
            page_content=all_docs['documents'][following_index],
            metadata=all_docs['metadatas'][following_index]
        )
    
    # Compile surrounding documents
    surrounding_docs = []
    if prev_doc:
        surrounding_docs.append(prev_doc)
    surrounding_docs.append(doc)
    if following_doc:
        surrounding_docs.append(following_doc)
    
    return surrounding_docs

def format_context(context_chunks):
    formatted_chunks = []
    for chunk in context_chunks:
        data = json.loads(chunk)
        timestamp = data['timestamp']
        content = data['content']
        title = re.sub(r'#\w+', '', data['episode_title']).strip()
        # Remove everything before the first '|' in the title
        # title_parts = title.split('|')
        # if len(title_parts) > 1:
        #     title = '|'.join(title_parts[1:]).strip()
        formatted_chunks.append(f"- **{title}** ({timestamp}) - \"{content}\"")
    return "\n".join(formatted_chunks)

st.set_page_config(page_title="اسأل عمرو خالد | الفهم عن الله ٢")

# Support Arabic text alignment in all components
support_arabic_text(all=True)

# Set up environment variables
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Initialize Anthropic client
anthropic = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Streamlit UI
st.markdown("# اسأل عمرو خالد | الفهم عن الله ٢")

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
# Get all documents for the video
if 'all_docs' not in st.session_state:
    st.session_state.all_docs = vectorstore.get()
all_docs = st.session_state.all_docs

# Display list of videos in the vector store in a collapsible container
with st.expander("الفيديوهات", expanded=False):
    unique_videos = get_unique_videos()
    sorted_videos = sorted(unique_videos.values(), key=lambda x: x['publish_date'])
    for video_info in sorted_videos:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(video_info['thumbnail_url'], width=120)
        with col2:
            st.write(f"**[{video_info['title']}]({video_info['source']})**")
            st.write(f"{video_info['publish_date']}")
        st.write("---")

# User question input
question = st.text_input("سؤالك:")

if question:
    # Get relevant documents
# Load the persisted vector store
    relevant_docs = vectorstore.similarity_search(question, k=5)
    
    if relevant_docs:
        # Combine the content of relevant documents and their surrounding context
        context_chunks = []
        for doc in relevant_docs:
            surrounding_docs = [doc]#get_surrounding_context(doc, all_docs)
            for d in surrounding_docs:
                episode_number = int(d.metadata.get('episode_number', -1) + 1)
                start_timestamp = d.metadata.get('start_timestamp', 'Unknown')
                episode_title = d.metadata.get('title', 'Unknown')
                content = d.page_content
                context_chunks.append(json.dumps({
                    "timestamp": f"{episode_number}-{start_timestamp}",
                    "content": content,
                    "episode_title": episode_title
                }))
        # Remove duplicates while preserving order
        context_chunks = list(dict.fromkeys(context_chunks))
        context = format_context(context_chunks)  # Use the new function here

        # answer = get_claude_response(question, context)
        # st.write(answer)
        videos_dict = {int(video['episode_number']): video for video in sorted_videos}
        answer = replace_timestamps(context, videos_dict)

        # Display answer
        st.markdown(answer)

        # Display relevant video info
        # st.subheader("Relevant Video Segments:")
        # print(len(surrounding_docs))
        # for i, relevant_doc in enumerate(relevant_docs, 1):
        #     st.write(f"**Relevant Document {i} and Surrounding Context:**")
        #     surrounding_docs = get_surrounding_context(relevant_doc, all_docs)
        #     for j, doc in enumerate(surrounding_docs):
        #         video_info = doc.metadata
        #         start_seconds = int(video_info['start_seconds']) - 10
        #         start_timestamp = video_info['start_timestamp']
        #         base_url = video_info['source']
        #         timestamp_param = f"t={start_seconds}"
                
        #         # Check if the URL already contains a query parameter
        #         if '?' in base_url:
        #             video_url = f"{base_url}&{timestamp_param}"
        #         else:
        #             video_url = f"{base_url}?{timestamp_param}"
                
        #         context_type = "Previous" if j == 0 else "Relevant" if j == 1 else "Following"
        #         st.write(f"- {context_type}: [{video_info['title']} (at {start_timestamp})]({video_url})")
        #         st.write(f"  Content: {doc.page_content[:100]}...")  # Display first 100 characters of content
        #     st.write("---")
        
        # # Display the most relevant video info
        # st.subheader("Most Relevant Video:")
        # video_info = relevant_docs[0].metadata
        # st.write(f"**{video_info['title']}**")
        # st.image(video_info['thumbnail_url'], width=200)
        # st.write(f"Author: {video_info['author']}")
        # st.write(f"View Count: {video_info['view_count']}")
        # st.write(f"Published: {video_info['publish_date']}")

    else:
        st.error("No relevant information found for this question.")

# Add a footer with information about the app
st.markdown("---")

def format_context(context_chunks):
    formatted_chunks = []
    for chunk in context_chunks:
        data = json.loads(chunk)
        timestamp = data['timestamp']
        content = data['content']
        formatted_chunks.append(f"- {content} ({timestamp})")
    return "\n".join(formatted_chunks)
