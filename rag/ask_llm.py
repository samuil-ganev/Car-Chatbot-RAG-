'''Module for prompting the llm.'''
import logging
from typing import List, Tuple
from llama_index.core.schema import NodeWithScore
from rag.llm_connector import LLMConnector

class CarAssistant:
    '''Class for defining the prompt to the llm.'''
    def __init__(self, query: str, nodes: List[NodeWithScore],
                 credentials_path: str = 'credentials.json') -> None:
        self.query = query
        self.nodes = nodes
        self.credentials_path = credentials_path
        self.llm = LLMConnector(credentials_path)

    def set_nodes(self, new_nodes: List[NodeWithScore]):
        '''Updates the nodes.'''
        logging.info('Updating nodes')
        self.nodes = new_nodes

    def set_query(self, new_query: str):
        '''Updates the query.'''
        logging.info('Query updated')
        self.query = new_query

    def _build_prompt(self) -> str:
        if not self.nodes:
            logging.warning('No context provided to build prompt.')
            return ''
        context = '\n\n'.join([
            node.node.get_content()
            for node in self.nodes
            if node.node.get_content()
        ])

        prompt = f'''You are a helpful AI assistant specializing in answering questions about cars.
                    Your primary goal is to answer the user's question using ONLY the information provided in the "Context" section.

                    IMPORTANT: The "Context" may contain textual descriptions of one or more images. You should treat these descriptions as factual information about the visual aspects of the car(s) or relevant scenes. For example, if an image description states "a red sports car with a black interior," you can use this to answer questions about the car's color or interior.

                    Carefully and thoroughly review the ENTIRE provided context before answering.

                    If the information required to answer the question is explicitly present or can be directly inferred from the provided context (including any image descriptions), please provide a concise answer.

                    Do not use any external knowledge or make assumptions beyond what is explicitly stated in the context.

                    Context:
                    {context}

                    Question:
                    {self.query}

                    Answer:
                    '''
        return prompt

    def get_answer(self) -> Tuple[str, str]:
        '''Returns the answer. If no prompt is given it returns a defined answer.'''
        prompt = self._build_prompt()

        if not prompt:
            return 'Could not generate a prompt from the provided context.', ''

        try:
            answer = self.llm.generate_answer(prompt)
            retrieved_chunks = '\n\n---\n\n'.join([
                node.node.get_content()
                for node in self.nodes
                if node.node.get_content()
            ])
            return answer, retrieved_chunks
        except Exception as e:
            logging.error('Failed to get LLM response: %s', e, exc_info=True)
            return 'An error occurred while contacting the LLM.', ''
        