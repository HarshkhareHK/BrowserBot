import streamlit as st
import openai

# Load API key securely from Streamlit Secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

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
        tools = response['choices'][0]['message']['content'].strip().split("\n")
        return [tool.strip("- ") for tool in tools if tool.strip()]
    except openai.error.OpenAIError as e:
        return [f"Error: {str(e)}"]
    except Exception as e:
        return [f"Unexpected error: {str(e)}"]

# Streamlit UI
st.title("🔍 AI Tools Finder")
st.write("Enter your query to find relevant AI tools!")

# Search input
search_query = st.text_input("Search for AI tools:", key="search")

# Get suggestions from GPT-4
suggestions = get_ai_tools(search_query) if search_query else []

# Show suggestions dynamically
if suggestions:
    selected_tool = st.selectbox("Suggestions:", suggestions)
else:
    selected_tool = search_query  # Use input if no suggestions yet

if st.button("Search"):
    if selected_tool.strip():
        with st.spinner("Fetching AI tools..."):
            tools_list = get_ai_tools(selected_tool)
        st.markdown("\n".join([f"- {tool}" for tool in tools_list]))
    else:
        st.error("Please enter a valid search query.")
