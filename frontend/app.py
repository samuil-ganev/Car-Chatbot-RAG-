import json
import streamlit as st
import re
from datetime import datetime
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag.ask_llm import CarAssistant 
from rag.search import *

CREDENTIALS_PATH = project_root / 'assets' / 'secrets' / 'credentials.json'
LOGS_PATH = project_root / 'rag' / 'interaction_logs.jsonl'

def log_interaction(query, answer, chunks, prompt):
    '''
    Logs the interaction with query, answer, context, chunks and the prompt used
    '''
    
    log_entry = {
        'question': query,
        'answer': answer,
        'contexts': chunks if chunks else [],
        'prompt': prompt,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(LOGS_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

@st.cache_resource
def initialize_resources():
    model = initialize_embed_model(
        model_name='BAAI/bge-base-en-v1.5',
        device=None,
        model_kwargs={}
    )
    vector_index = load_vector_index(model)
    return model, vector_index


class RAGCarChatbotApp:
    def __init__(self):
        self._setup_page()
        self.embed_model, self.index = initialize_resources()
        self.assistant = CarAssistant(query='', nodes=None, credentials_path=str(CREDENTIALS_PATH))
        self._init_session_state()

    def _setup_page(self):
        st.set_page_config(page_title='RAG Car Chatbot', page_icon='🚗', layout='wide')
        with st.sidebar:
            st.title('🚙 RAG Car Chatbot')
            st.markdown('### Ask about car models and features!')
            st.markdown('Ask questions and get answers based on your car data.')
        st.title('Welcome to our RAG Car Chatbot 🚗')

    def _init_session_state(self):
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

    def _get_chat_history_for_prompt(self, last_n=5):
        history_str = ''
        recent_history = st.session_state.chat_history[-(last_n*2):]

        for msg in recent_history:
            role = 'User' if msg['role'] == 'user' else 'Assistant'
            history_str += f'{role}: {msg['content']}\n'
        return history_str.strip()
    
    def _get_standalone_query(self, current_query: str):
        if not st.session_state.chat_history:
            return current_query
        
        chat_history_str = self._get_chat_history_for_prompt(last_n=3)

        rephrase_prompt = f'''You are a query rewriting expert. Your task is to rephrase the "Follow-up Question" to be a standalone question that incorporates necessary context from the "Chat History".
                            If the "Follow-up Question" is already standalone or the history does not seem relevant to it, return the original "Follow-up Question" unchanged.
                            Only output the rephrased standalone question, without any preamble or explanation.

                            Chat History:
                            {chat_history_str}

                            Follow-up Question: {current_query}

                            Standalone Question:'''
        
        standalone_query = self.assistant.llm.generate_answer(rephrase_prompt).strip()
        print(f'Original query: {current_query}\nStandalone query: {standalone_query}')
        return standalone_query

    def run(self):
        query = st.chat_input('Ask something...')
        standalone_query = self._get_standalone_query(query)

        prompt = f'''
            Your task is to convert user questions into concise statements or descriptive phrases suitable for semantic search in a vector database. Focus on the core topic, removing conversational filler and question structure.
            ---
            User Question: "Hey, can you tell me what the main benefits of using Retrieval-Augmented Generation are?"
            Optimized Search Statement: "Benefits of Retrieval-Augmented Generation (RAG)"
            ---
            User Question: "I'm trying to understand how photosynthesis works in plants."
            Optimized Search Statement: "Process of photosynthesis in plants"
            ---
            User Question: "What's the difference between the iPhone 15 Pro and the Samsung S24 Ultra cameras?"
            Optimized Search Statement: "Comparison of iPhone 15 Pro and Samsung S24 Ultra cameras"
            ---
            User Question: "How do I reset my forgotten password for my online banking account?"
            Optimized Search Statement: "Resetting forgotten online banking password"
            ---
            User Question: "Could you explain the historical context surrounding the fall of the Berlin Wall?"
            Optimized Search Statement: "Historical context of the fall of the Berlin Wall"
            ---
            User Question: "{standalone_query}"
            Optimized Search Statement:
        '''
        
        query_to_statement = self.assistant.llm.generate_answer(prompt)
        
        if query:
            st.session_state.chat_history.append({'role': 'user', 'content': query})

            results = search_in_index(query_str=query_to_statement, index=self.index)

            if results:
                self.assistant.set_query(standalone_query)
                self.assistant.set_nodes(new_nodes=results)
                answer, retrieved_chunks = self.assistant.get_answer()

                fallback_responses = [
                    'Based on the provided context, I am unable to provide an answer.',
                    "I'm sorry, but I couldn't find enough context to answer your question."
                ]

                if any(fallback.lower() in answer.lower() for fallback in fallback_responses):
                    retrieved_chunks = None

                log_interaction(standalone_query, answer, retrieved_chunks, prompt)
                
            else:
                answer = "Sorry, I couldn't find relevant information."
                retrieved_chunks = None

            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': answer,
                'chunks': retrieved_chunks if retrieved_chunks else None
            })

        self._display_chat_history()

    def _display_chat_history(self):
        for i, msg in enumerate(st.session_state.chat_history):
            if msg['role'] == 'user':
                with st.chat_message('user', avatar='👤'):
                    st.markdown(msg['content'])
            elif msg['role'] == 'assistant':
                with st.chat_message('assistant', avatar='🤖'):
                    st.markdown(msg['content'])
                    if msg.get('chunks'):
                        clean_query = re.sub(r'\W+', '_', msg['content'].strip().lower())[:50]
                        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                        chunk_file_name = f'chunks_{clean_query}_{timestamp}.txt'
                        st.download_button(
                            label='📄 Download Retrieved Chunks',
                            data=msg['chunks'],
                            file_name=chunk_file_name,
                            mime='text/plain',
                            key=f'download_button_{i}'
                        )

if __name__ == '__main__':
    app = RAGCarChatbotApp()
    app.run()
