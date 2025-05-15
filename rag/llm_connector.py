import json
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client

class LLMConnector:
    def __init__(self, credentials_path: str, model_name: str = 'gpt-4o', temperature: float = 0.0):
        with open(credentials_path, 'r') as f:
            creds = json.load(f)

        self.proxy_client = get_proxy_client(
            proxy_version='gen-ai-hub',
            base_url=creds['serviceurls']['AI_API_URL'],
            auth_url=creds['url'],
            client_id=creds['clientid'],
            client_secret=creds['clientsecret']
        )

        self.llm = ChatOpenAI(
            proxy_client=self.proxy_client,
            proxy_model_name=model_name,
            temperature=temperature
        )

    def generate_answer(self, prompt: str) -> str:
        return self.llm.invoke(prompt).content

