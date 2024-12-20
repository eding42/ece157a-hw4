import streamlit as st
import numpy as np
import json

from database import SimpleDatabase
from encoders import SillyEncoder, HuggingFaceEncoder
from generative import SillyLLM, LocalLLM
from parsers import parse_pdf

from io import BytesIO



#%% Page config has to be first

st.set_page_config(
    page_title="My LLM PDF App",  # Title of the web page
    page_icon="🧊",               # Icon displayed in the browser tab
    layout="wide",                # Use the "wide" layout instead of the "centered" one
)




#%% Initialization

def init():
    st.session_state['database'] = SimpleDatabase()
    st.session_state['encoder'] = HuggingFaceEncoder(model_name="sentence-transformers/all-mpnet-base-v2")
    st.session_state['LLM'] = LocalLLM(model_name="mistral")
    st.session_state['files'] = {}
    reset()

def reset():
    st.session_state['chat_history'] = []

if 'database' not in st.session_state:
    init()


#%% Application layout - always happens first so prevent render problems.
# User dependent rendering can happen later
 

bar = st.sidebar
with bar:
    st.title( "Edward's LLM PDF App")
    clear_chat = st.button("Clear chat")
    st.markdown('---')
    files = st.file_uploader(label='## Upload a PDF', type='pdf', accept_multiple_files=True)
    st.markdown('---\n\n## Debug Info')
    debug_container = st.container()

chat_input = st.chat_input()
chat_container = st.container()






#%%  Function definitions
    
def user_message(text):
    chat_container.chat_message('user').write(text)

    st.session_state['chat_history'].append(
        {'role': 'user', 'content': text}
    )

def assistant_message(generator):
    with chat_container.chat_message('assistant'):
        text = st.write_stream(generator)
    st.session_state['chat_history'].append(
        {'role': 'assistant', 'content': text}
    )

def debug_message(*args):
    with debug_container.chat_message('system'):
        st.write(*args)


def run_parser(file):
    """Parser a pdf and put the result in the database"""
    # The parser needs file in IO form since its an in-memory file
    pages = parse_pdf(BytesIO(file.getvalue()))
    # add to the database
    st.session_state['database'].add_documents(
        pages,
        st.session_state['encoder'].encode(pages)
    )
    # record that we have parsed
    st.session_state['files'][file.name] = True
    debug_message(f'File `{file.name}` loaded to database')


def get_top_documents_for_RAG(query, n=3):
    """Get the best matching documents and put them into one big string"""
    query_vector = st.session_state['encoder'].encode(query)
    database_response = st.session_state['database'].retrieve_documents(query_vector, n=n)

    if database_response:
        # Documents, each marked by """ quotation
        context = '\n\n'.join(f'''"""{d}"""''' for d in database_response)
        context_for_visual_debug = f"PDF snippets\n```json\n{json.dumps(database_response, indent=2)}\n```"
    else:
        # No Documents
        context, context_for_visual_debug = '', 'No Documents Available'

    # Can be commented later but for debug allow the user to see database documents
    debug_message(context_for_visual_debug)
    return context


def add_context_to_query(context: str, user_input: str) -> list[dict[str, str]]:
    """
    Returns extra messages to add to a chat history given a user input and document context.

    This could be done a number of ways, with the context being provided as in independent messages,
    or being concatenated with the user input to product one single message.

    Remember, every message is {'role': ..., 'content': ...}
    """

    user_with_context = (
        "-----------------------PDF snippets------------------------\n"
        + context
        + '\n\n-----------------------User Question-----------------------\n'
        + user_input
    )
    
    extra_messages = [
        {'role': 'user', 'content': user_with_context}
    ]
    return extra_messages

    

#%% Running the application"""


# RESET messages
if clear_chat:
    reset()

# Show History
for message in st.session_state['chat_history']:
    chat_container.chat_message(message['role']).write(message['content'])

# UPLOADING
for file in files:
    if not st.session_state['files'].get(file.name):
        run_parser(file)
# can be commented later
debug_message(f"There are {len(st.session_state['database'].documents)} document snippets in the database")


# Responding to user query
if chat_input:
    # get a string that combines the closest matching PDF snippets
    context = get_top_documents_for_RAG(chat_input)
    
    # setup the messages to send to the chat. We add a extra prompt - the system prompt
    if context:
        # With PDFS - focus on PDFs
        SYSTEM_PROMPT = {
            'role': 'system', 
            'content': (
                "Answer the user's questions based on retrieved snippets from the PDF. "
            )  
        }
        new_messages = add_context_to_query(context, chat_input)
        
    else:
        # No PDFs - act like normal ChatGPT
        SYSTEM_PROMPT = {
            'role': 'system', 
            'content': "You are a helpful AI assistant. Please answer the user's questions concisely and precisely."
        }
        new_messages = [{'role': 'user', 'content': chat_input}]
    
    messages = [SYSTEM_PROMPT] + st.session_state['chat_history'] + new_messages

    # Visually, we only need to record the user message
    user_message(chat_input)
    # generate a new message, inside which the we added the context to the message history
    LLM_response = st.session_state['LLM'].generate(messages)
    assistant_message(LLM_response)



# Absolutely nothing has happened so we show a greeting
if not st.session_state['chat_history']:
    with chat_container.chat_message('assistant'):
        st.write('''
            **Welcome to My LLM PDF App!** 🎉
            
            - Upload PDF files using the sidebar.
            - Ask questions based on the uploaded content.
            - Clear the chat history at any time with the "Clear chat" button.
            
            **Start by uploading a PDF or asking a question.**
            '''
        )