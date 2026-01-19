import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ContentProcessor:
    """
    Processes the parsed content - summarizes images, creates llm based semantic chunks
    """
    def __init__(self, config):
        """
        Initialize the response generator.
        
        Args:
            llm: Large language model for image summarization
        """
        self.logger = logging.getLogger(__name__)
        self.summarizer_model = config.rag.summarizer_model     # temperature 0.5
        self.chunker_model = config.rag.chunker_model     # temperature 0.0
    
    def summarize_images(self, images: List[str]) -> List[str]:
        """
        NOTE: The current LLM (TinyLlama) is text-only and cannot summarize images.
            This function will return a placeholder and skip any LLM calls.
        """
        self.logger.warning("Image summarization is being SKIPPED because the configured LLM is text-only.")
        results = [f"Placeholder for image {i+1}" for i, _ in enumerate(images)]
        return results
            
    def format_document_with_images(self, parsed_document: Any, image_summaries: List[str]) -> str:
        """
        Format the parsed document by replacing image placeholders with image summaries.
        
        Args:
            parsed_document: Parsed document from doc_parser
            image_summaries: List of image summaries
            
        Returns:
            Formatted document text with image summaries
        """
        IMAGE_PLACEHOLDER = "<!-- image_placeholder -->"
        PAGE_BREAK_PLACEHOLDER = "<!-- page_break -->"
        
        formatted_parsed_document = parsed_document.export_to_markdown(
            page_break_placeholder=PAGE_BREAK_PLACEHOLDER, 
            image_placeholder=IMAGE_PLACEHOLDER
        )
        
        formatted_document = self._replace_occurrences(
            formatted_parsed_document, 
            IMAGE_PLACEHOLDER, 
            image_summaries
        )
        
        return formatted_document
    
    def _replace_occurrences(self, text: str, target: str, replacements: List[str]) -> str:
        """
        Replace occurrences of a target placeholder with corresponding replacements.
        
        Args:
            text: Text containing placeholders
            target: Placeholder to replace
            replacements: List of replacements for each occurrence
            
        Returns:
            Text with replacements
        """
        result = text
        for counter, replacement in enumerate(replacements):
            if target in result:
                if replacement.lower() != 'non-informative':
                    result = result.replace(
                        target, 
                        f'picture_counter_{counter}' + ' ' + replacement, 
                        1
                    )
                else:
                    result = result.replace(target, '', 1)
            else:
                # Instead of raising an error, just break the loop when no more occurrences are found
                break
        
        return result

    def chunk_document(self, formatted_document: str) -> List[str]:
        """
        Split the document into chunks using a standard text splitter.

        NOTE: The LLM-based semantic chunking has been replaced to prevent
                Out-of-Memory errors with large documents and small models.
        """
        self.logger.info("Chunking document using RecursiveCharacterTextSplitter.")

        # Initialize a reliable text splitter from LangChain
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,      # The size of each chunk in characters
            chunk_overlap=50,    # How many characters to overlap between chunks
            length_function=len,
        )

        # Split the document and return the list of text chunks
        chunks = text_splitter.split_text(formatted_document)

        if not chunks:
            self.logger.warning("Text splitter returned no chunks. The document might be empty.")
            return []
            
        return chunks
    
    def _split_text_by_llm_suggestions(self, chunked_text: str, llm_response: str) -> List[str]:
        """
        Split text according to LLM suggested split points.
        
        Args:
            chunked_text: Text with chunk markers
            llm_response: LLM response with split suggestions
            
        Returns:
            List of document chunks
        """
        # Extract split points from LLM response
        split_after = [] 
        if "split_after:" in llm_response:
            split_points = llm_response.split("split_after:")[1].strip()
            split_after = [int(x.strip()) for x in split_points.replace(',', ' ').split()] 

        # If no splits were suggested, return the whole text as one section
        if not split_after:
            return [chunked_text]

        # Find all chunk markers in the text
        chunk_pattern = r"<\|start_chunk_(\d+)\|>(.*?)<\|end_chunk_\1\|>"
        chunks = re.findall(chunk_pattern, chunked_text, re.DOTALL)

        # Group chunks according to split points
        sections = []
        current_section = [] 

        for chunk_id, chunk_text in chunks:
            current_section.append(chunk_text)
            if int(chunk_id) in split_after:
                sections.append("".join(current_section).strip())
                current_section = [] 
        
        # Add the last section if it's not empty
        if current_section:
            sections.append("".join(current_section).strip())

        return sections