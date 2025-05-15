import re
import logging
import base64
import os
from pathlib import Path

from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from langchain_core.messages import HumanMessage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MarkdownImageProcessor:
    '''
    Processes markdown files in a directory to replace 
    image tags with AI-generated descriptions based on 
    the image and surrounding text
    '''

    def __init__(self, hub_base_url: str, hub_auth_url: str, hub_client_id: str, hub_client_secret: str, context_words: int = 150):
        
        proxy_client_instance = get_proxy_client(
            proxy_version='gen-ai-hub',
            base_url=hub_base_url,
            auth_url=hub_auth_url,
            client_id=hub_client_id,
            client_secret=hub_client_secret
        )

        self.llm = ChatOpenAI(
            proxy_client=proxy_client_instance,
            proxy_model_name='gpt-4o',
            temperature=0.1,
            max_tokens=200
        )
        
        self.context_words = context_words
        self.image_tag_pattern = re.compile(r'!\[(.*?)\]\((.*?)\)')
        logging.info(f'MarkdownImageProcessor initialized. Context window: {self.context_words} words')

    def _describe_image(self, image_path: Path, context_text: str) -> str:
        '''
        Describes an image using the configured multimodal model
        '''
            
        if not image_path.exists():
            logging.error(f'    - Image file not found: {image_path}')
            raise FileNotFoundError('Can\'t find the image file')
            
        try:

            with open(image_path, 'rb') as image_file:
                image_bytes = image_file.read()

                mime_type = f'image/{image_path.suffix.lower().strip('.')}'
                if mime_type == 'image/jpg': mime_type = 'image/jpeg'
                if mime_type not in ['image/png', 'image/jpeg', 'image/gif', 'image/webp']:
                    logging.warning(f'    - Potentially unsupported image type for API: {mime_type}. Using image/png')
                    mime_type = 'image/png'

                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                image_url = f'data:{mime_type};base64,{base64_image}'

                prompt = f'''
                    Describe the following image concisely. Use the accompanying text context from the PDF page
                    where the image appeared to inform the description. Focus on what the image visually shows
                    and how it relates to the text, if possible. Avoid stating "The image shows...".

                    Text Context:
                    ---
                    {context_text[:2000]}
                    ---

                    Image:
                    '''
                
                messages = [
                    HumanMessage(
                        content=[
                            {
                                'type': 'text', 
                                'text': prompt
                            },
                            {
                            'type': 'image_url', 
                            'image_url': 
                                {
                                    'url': image_url, 
                                    'detail': 'low'
                                }
                            },
                        ]
                    )
                ]

                response = self.llm.invoke(messages) 
                description = response.content.strip()
                
                clean_description = description.replace('\n', ' ').strip()
                return clean_description
            
        except Exception as e:

            logging.error(f'    - Error describing image {image_path.name}: {e}', exc_info=True)
            return f'Error describing image {image_path.name}'
        
    def _extract_and_clean_context(self, full_text: str, match_start: int, match_end: int) -> str:
        '''
        Extracts text around the match and removes other image tags.
        Uses word count for context window
        '''

        text_before_match = full_text[:match_start]
        text_after_match = full_text[match_end:]

        words_before = text_before_match.split()
        words_after = text_after_match.split()

        context_before = ' '.join(words_before[-self.context_words:])
        context_after = ' '.join(words_after[:self.context_words])
        context = self.image_tag_pattern.sub('', context_before + ' ' + context_after)
        context = re.sub(r'\s+', ' ', context).strip()

        return context
    
    def process_markdown_file(self, md_file_path: Path):
        '''
        Reads a markdown file, finds image tags, gets description and overwrites the file
        '''

        logging.info(f'Processing markdown file: {md_file_path}')
        try:

            content = md_file_path.read_text(encoding='utf-8')
            original_content = content
            matches = list(self.image_tag_pattern.finditer(content))

            logging.info(f'  - Found {len(matches)} image tags in {md_file_path.name}')
            num_processed = 0

            for match in reversed(matches):
                num_processed += 1
                alt_text = match.group(1)
                relative_image_path_str = match.group(2)
                tag_start, tag_end = match.span()
                original_tag_text = match.group(0)

                image_path = (md_file_path.parent / Path(relative_image_path_str)).resolve()
                cleaned_context = self._extract_and_clean_context(
                    original_content,
                    match.start(),
                    match.end()
                )
                if not cleaned_context:
                    cleaned_context = alt_text if alt_text else 'No text context available'
                description = self._describe_image(image_path, cleaned_context)
                content = content[:tag_start] + description + content[tag_end:]

                md_file_path.write_text(content, encoding='utf-8')
                logging.info(f'  - Finished processing. Updated file saved: {md_file_path.name}')
        
        except Exception as e:
            logging.error(f'Failed to process file {md_file_path}: {e}', exc_info=True)
    
    def process_dir(self, input_dir: Path):
        '''
        Finds all .md files in the input directory and processes them
        '''

        if not input_dir.is_dir():
            logging.error(f'Input path is not a valid directory: {input_dir}')
            return
        
        md_files_processed = 0
        for md_file in input_dir.glob('*.md'):
            self.process_markdown_file(md_file)
            md_files_processed += 1
        
        logging.info(f'Finished processing directory. {md_files_processed} markdown files processed')
    