import streamlit as st
import openai
import json
import faiss
import numpy as np

# Sample tool dataset (can be replaced with an actual database or API)
tools = [
    {"name": "ChatGPT", "description": "An AI chatbot by OpenAI that generates human-like text responses."},
    {"name": "DALL·E", "description": "An AI model by OpenAI that generates images from text descriptions."},
    {"name": "Codex", "description": "An AI model by OpenAI that translates natural language into code."},
    {"name": "Stable Diffusion", "description": "An AI-powered image generation model by Stability AI."},
    {"name": "Claude", "description": "An AI assistant developed by Anthropic for conversational AI."},
    {"name": "Bard", "description": "An AI chatbot developed by Google for generating text-based responses."},
    {"name": "Whisper", "description": "An AI-powered automatic speech recognition (ASR) system by OpenAI."},
    {"name": "GPT-4", "description": "A large language model by OpenAI for generating and understanding text."},
    {"name": "Midjourney", "description": "An AI-based image generation tool for creative artwork."},
    {"name": "DeepL", "description": "An AI-powered language translation tool with high accuracy."},
    {"name": "Hugging Face Transformers", "description": "A library of pre-trained AI models for NLP and ML applications."},
    {"name": "Runway ML", "description": "A creative AI tool for video editing and special effects generation."},
    {"name": "AutoGPT", "description": "An AI agent that autonomously performs complex tasks using GPT models."},
    {"name": "Synthesia", "description": "An AI-based tool for creating synthetic videos with virtual avatars."},
    {"name": "Lumen5", "description": "An AI-powered video creation platform for marketing and content generation."},
    {"name": "Pika Labs", "description": "An AI-powered video generation tool for animation and creative content."},
    {"name": "DeepBrain AI", "description": "An AI tool for video synthesis and realistic AI avatars."},
    {"name": "Descript", "description": "An AI-powered tool for audio and video editing with transcription features."},
    {"name": "Notion AI", "description": "An AI-powered assistant integrated into Notion for writing and organization."},
    {"name": "Quillbot", "description": "An AI-based paraphrasing and writing enhancement tool."},
    {"name": "Grammarly", "description": "An AI-powered writing assistant for grammar and style improvement."},
    {"name": "Retell", "description": "An AI-powered tool for real-time voice and speech synthesis."},
    {"name": "Vapi", "description": "An AI-driven API for generating and managing voice interactions."},
    {"name": "OpenAI", "description": "A leading AI research and deployment company providing advanced AI models."}
]


openai.api_key = st.secrets["OPENAI_API_KEY"]

# Function to get OpenAI embeddings (Updated API v1)
def get_openai_embedding(text):
    client = openai.OpenAI()  # Ensure you're using the latest OpenAI library
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding)


def build_index(tools):
    """Builds a FAISS index for tool descriptions using OpenAI embeddings."""
    descriptions = [tool["description"] for tool in tools]
    embeddings = np.array([get_openai_embedding(desc) for desc in descriptions])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

# Build search index
index, embeddings = build_index(tools)

def search_tools(query, top_k=3):
    """Searches for relevant tools based on user query using OpenAI embeddings."""
    query_embedding = get_openai_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    results = [tools[i] for i in indices[0] if i < len(tools)]
    return results

# Streamlit UI
st.title("AI-Powered Tool Search")
query = st.text_input("Enter your search query:")

if query:
    results = search_tools(query)
    if results:
        st.subheader("Relevant Tools:")
        for tool in results:
            st.write(f"**{tool['name']}**: {tool['description']}")
    else:
        st.write("No relevant tools found.")
