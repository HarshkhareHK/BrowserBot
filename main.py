import streamlit as st
import openai

# Load API key securely
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Dummy AI tools data
dummy_tools = {
    "java": ["Jasper AI", "Tabnine", "Codium", "Kite", "DeepCode"],
    "python": ["ChatGPT", "Hugging Face Transformers", "PyCaret", "Auto-sklearn", "DVC"],
    "image processing": ["OpenCV", "DeepDream", "Runway ML", "Artbreeder", "DeepAI"],
    "text analysis": ["GPT-4", "BERT", "NLTK", "SpaCy", "TextBlob"]
}

def get_dummy_ai_tools(query):
    """Return dummy AI tools based on query."""
    query_lower = query.lower()
    for key, tools in dummy_tools.items():
        if key in query_lower:
            return "\n".join([f"- {tool}" for tool in tools])
    return None  # Return None so we can fall back to OpenAI

def get_ai_tools(query):
    """Fetch relevant AI tools using OpenAI API."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant that suggests relevant AI tools based on user queries."},
                {"role": "user", "content": f"List the top AI tools for {query}."}
            ]
        )
        tools = response['choices'][0]['message']['content'].strip()
        return tools
    except openai.error.OpenAIError as e:
        return f"Error fetching AI tools: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

# Streamlit UI
st.title("AI Tools Finder")
st.write("Enter your query to find relevant AI tools!")

query = st.text_input("Search for AI tools:")

if st.button("Search"):
    if query.strip():
        with st.spinner("Fetching AI tools..."):
            tools_list = get_dummy_ai_tools(query)
            if not tools_list:  # If no dummy data, fallback to OpenAI
                tools_list = get_ai_tools(query)
        st.markdown(tools_list)
    else:
        st.error("Please enter a valid search query.")
