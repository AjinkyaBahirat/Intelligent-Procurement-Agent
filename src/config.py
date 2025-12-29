import os
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

class Config:
    """
    Centralized configuration management for the application.
    Loads values from environment variables.
    """

    # --- LLM Configuration ---
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "vertex_ai")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-1.5-pro")
    
    # Construct the model string for LiteLLM (e.g., "vertex_ai/gemini-1.5-pro" or "groq/llama3-70b-8192")
    # We ensure the string follows 'provider/model_name' format.
    if LLM_PROVIDER and not LLM_MODEL_NAME.startswith(f"{LLM_PROVIDER}/"):
         LLM_MODEL_STRING = f"{LLM_PROVIDER}/{LLM_MODEL_NAME}"
    else:
         LLM_MODEL_STRING = LLM_MODEL_NAME

    # --- Embedding Configuration ---
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "vertex_ai")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-004")
    
    if EMBEDDING_PROVIDER and not EMBEDDING_MODEL_NAME.startswith(f"{EMBEDDING_PROVIDER}/"):
        EMBEDDING_MODEL_STRING = f"{EMBEDDING_PROVIDER}/{EMBEDDING_MODEL_NAME}"
    else:
        EMBEDDING_MODEL_STRING = EMBEDDING_MODEL_NAME

    # --- Vector Database ---
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./chroma_db")

    # --- API Keys (Optional accessors, though libraries usually read env vars directly) ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # --- Ollama Specific ---
    OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    
    @staticmethod
    def validate():
        """Optional: Check for required keys based on provider."""
        if Config.LLM_PROVIDER == "groq" and not Config.GROQ_API_KEY:
            print("Warning: LLM_PROVIDER is 'groq' but GROQ_API_KEY is missing.")
