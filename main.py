import streamlit as st
import openai
import json
import faiss
import numpy as np

# Sample tool dataset (can be replaced with an actual database or API)
tools = [
    {"name": "Python", "description": "A popular, interpreted, high-level, general-purpose dynamic programming language."},
    {"name": "Java", "description": "An object-oriented, class-based, general-purpose programming language."},
    {"name": "C++", "description": "A high-performance, compiled, general-purpose programming language."},
    {"name": "GPT-1", "description": "A small, simple, neural network-based language model."},
    {"name": "GPT-2", "description": "A medium-sized, neural network-based language model."},
    {"name": "GPT-3", "description": "A large, neural network-based language model."},
    {"name": "JavaScript", "description": "A high-level, interpreted, dynamic, and flexible scripting language."},
    {"name": "C#", "description": "An object-oriented, statically typed, general-purpose programming language."},
    {"name": "GPT-Neo", "description": "A series of transformer-based language models."},
    {"name": "GPT-J", "description": "A large, transformer-based language model."},
    {"name": "Rust", "description": "A systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety."},
    {"name": "Go", "description": "A statically typed, compiled, concurrent, and garbage-collected programming language."},
    {"name": "TypeScript", "description": "A superset of JavaScript that adds optional static typing and other features."},
    {"name": "Kotlin", "description": "A modern, statically typed, general-purpose programming language."},
    {"name": "Swift", "description": "A general-purpose, multi-paradigm, compiled programming language."},
    {"name": "Julia", "description": "A high-level, high-performance, just-in-time compiled, garbage-collected language."},
    {"name": "Haskell", "description": "A statically typed, purely functional programming language."},
    {"name": "R", "description": "A programming language and environment for statistical computing and graphics."},
    {"name": "MATLAB", "description": "A high-level programming language and environment for numerical computation and data analysis."},
    {"name": "PHP", "description": "A mature, open source, loosely typed, server-side scripting language."},
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
