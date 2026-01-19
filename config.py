"""
Configuration file for the Multi-Agent Medical Chatbot.
This file has been modified to use ChromaDB and a local Hugging Face model (TinyLlama).
"""

import os
import torch
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

_SHARED_LLM = None

def get_shared_llm(temperature):
    global _SHARED_LLM
    if _SHARED_LLM is None:
        _SHARED_LLM = get_huggingface_llm(temperature=temperature)
    return _SHARED_LLM


# Load environment variables from .env file
load_dotenv()

def get_huggingface_llm(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", temperature: float = 0.5, max_new_tokens: int = 512):
    """
    Loads the TinyLlama model with 4-bit quantization for efficiency
    and creates a LangChain-compatible pipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Configure 4-bit quantization using BitsAndBytesConfig
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     llm_int8_enable_fp32_cpu_offload=True
    # )
    
    # Load model with quantization config
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float32
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95,
        repetition_penalty=1.15
    )
    return HuggingFacePipeline(pipeline=pipe)


class AgentDecisoinConfig:
    def __init__(self):
        self.llm = get_shared_llm(temperature=0.3)

class ConversationConfig:
    def __init__(self):
        self.llm = get_shared_llm(temperature=0.3)

class WebSearchConfig:
    def __init__(self):
        self.llm = get_shared_llm(temperature=0.3)
        self.context_limit = 20

class RAGConfig:
    def __init__(self):
        self.vector_db_type = "chromadb"
        self.embedding_dim = 384  
        self.distance_metric = "Cosine"
        self.use_local = True
        self.vector_local_path = "./data/chroma_db"  # New path for ChromaDB
        self.doc_local_path = "./data/docs_db"
        self.parsed_content_dir = "./data/parsed_docs"
        # Qdrant URL and API Key are no longer needed
        self.collection_name = "medical_assistance_rag"
        self.chunk_size = 512
        self.chunk_overlap = 50

        # --- MODIFIED: Switched to a local Hugging Face embedding model ---
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # Use 'cuda' if GPU is available and desired for embeddings
        )

        # --- MODIFIED: Switched all LLMs to the local TinyLlama model ---
        self.llm = get_huggingface_llm(temperature=0.3)
        self.summarizer_model = get_huggingface_llm(temperature=0.5)
        self.chunker_model = get_huggingface_llm(temperature=0.0)  # Factual
        self.response_generator_model = get_huggingface_llm(temperature=0.3)
        
        self.top_k = 5
        self.vector_search_type = 'similarity'

        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        self.reranker_model = "cross-encoder/ms-marco-TinyBERT-L-6"
        self.reranker_top_k = 3

        self.max_context_length = 2048  # Adjusted for smaller local models
        self.include_sources = True

        self.min_retrieval_confidence = 0.40
        self.context_limit = 20

class MedicalCVConfig:
    def __init__(self):
        self.brain_tumor_model_path = "./agents/image_analysis_agent/brain_tumor_agent/models/brain_tumor_segmentation.pth"
        self.chest_xray_model_path = "./agents/image_analysis_agent/chest_xray_agent/models/covid_chest_xray_model.pth"
        self.skin_lesion_model_path = "./agents/image_analysis_agent/skin_lesion_agent/models/checkpointN25_.pth.tar"
        self.skin_lesion_segmentation_output_path = "./uploads/skin_lesion_output/segmentation_plot.png"
        
        # --- MODIFIED: Switched to the local TinyLlama model ---
        self.llm = get_huggingface_llm(temperature=0.1)

class SpeechConfig:
    def __init__(self):
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.eleven_labs_voice_id = "21m00Tcm4TlvDq8ikWAM"

class ValidationConfig:
    def __init__(self):
        self.require_validation = {
            "CONVERSATION_AGENT": False,
            "RAG_AGENT": False,
            "WEB_SEARCH_AGENT": False,
            "BRAIN_TUMOR_AGENT": True,
            "CHEST_XRAY_AGENT": True,
            "SKIN_LESION_AGENT": True
        }
        self.validation_timeout = 300
        self.default_action = "reject"

class APIConfig:
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 8000
        self.debug = True
        self.rate_limit = 10
        self.max_image_upload_size = 5

class UIConfig:
    def __init__(self):
        self.theme = "light"
        self.enable_speech = True
        self.enable_image_upload = True

class Config:
    def __init__(self):
        self.agent_decision = AgentDecisoinConfig()
        self.conversation = ConversationConfig()
        self.rag = RAGConfig()
        self.medical_cv = MedicalCVConfig()
        self.web_search = WebSearchConfig()
        self.api = APIConfig()
        self.speech = SpeechConfig()
        self.validation = ValidationConfig()
        self.ui = UIConfig()
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.max_conversation_history = 20