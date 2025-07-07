from mem0 import Memory
import os
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global client instance to prevent multiple collection creation
_mem0_client = None

# Custom instructions for memory processing
# These aren't being used right now but Mem0 does support adding custom prompting
# for handling memory retrieval and processing.
CUSTOM_INSTRUCTIONS = """
Extract the Following Information:

- Key Information: Identify and save the most important details.
- Context: Capture the surrounding context to understand the memory's relevance.
- Connections: Note any relationships to other topics or memories.
- Importance: Highlight why this information might be valuable in the future.
- Source: Record where this information came from when applicable.
"""

def get_mem0_client():
    global _mem0_client

    # Return existing client if already created
    if _mem0_client is not None:
        return _mem0_client

    # Get LLM provider and configuration
    llm_provider = os.getenv('LLM_PROVIDER')
    llm_api_key = os.getenv('LLM_API_KEY')
    llm_model = os.getenv('LLM_CHOICE')
    embedding_model = os.getenv('EMBEDDING_MODEL_CHOICE')

    # Initialize config dictionary
    config = {}

    # Configure LLM based on provider
    if llm_provider == 'openai' or llm_provider == 'openrouter':
        config["llm"] = {
            "provider": "openai",
            "config": {
                "model": llm_model,
                "temperature": 0.2,
                "max_tokens": 2000,
            }
        }

        # Set API key in environment if not already set
        if llm_api_key and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = llm_api_key

        # For OpenRouter, set the specific API key
        if llm_provider == 'openrouter' and llm_api_key:
            os.environ["OPENROUTER_API_KEY"] = llm_api_key

    elif llm_provider == 'ollama':
        config["llm"] = {
            "provider": "ollama",
            "config": {
                "model": llm_model,
                "temperature": 0.2,
                "max_tokens": 2000,
            }
        }

        # Set base URL for Ollama if provided
        llm_base_url = os.getenv('LLM_BASE_URL')
        if llm_base_url:
            config["llm"]["config"]["ollama_base_url"] = llm_base_url

    # Configure embedder based on provider
    if llm_provider == 'openai':
        config["embedder"] = {
            "provider": "openai",
            "config": {
                "model": embedding_model or "text-embedding-3-small",
                "embedding_dims": 1536  # Default for text-embedding-3-small
            }
        }

        # Set API key in environment if not already set
        if llm_api_key and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = llm_api_key

    elif llm_provider == 'ollama':
        config["embedder"] = {
            "provider": "ollama",
            "config": {
                "model": embedding_model or "nomic-embed-text",
                "embedding_dims": 768  # Default for nomic-embed-text
            }
        }

        # Set base URL for Ollama if provided
        embedding_base_url = os.getenv('LLM_BASE_URL')
        if embedding_base_url:
            config["embedder"]["config"]["ollama_base_url"] = embedding_base_url

    # Configure Supabase vector store
    config["vector_store"] = {
        "provider": "supabase",
        "config": {
            "connection_string": os.environ.get('DATABASE_URL', ''),
            "collection_name": "mem0_memories",
            "embedding_model_dims": 1536 if llm_provider == "openai" else 768,
            # Add connection pooling configuration to prevent connection exhaustion
            "pool_size": int(os.environ.get('DB_POOL_SIZE', '5')),  # Number of connections to maintain in the pool
            "max_overflow": int(os.environ.get('DB_MAX_OVERFLOW', '10')),  # Maximum number of connections that can be created beyond pool_size
            "pool_timeout": int(os.environ.get('DB_POOL_TIMEOUT', '30')),  # Timeout for getting a connection from the pool
            "pool_recycle": int(os.environ.get('DB_POOL_RECYCLE', '3600')),  # Recycle connections after 1 hour
            "pool_pre_ping": os.environ.get('DB_POOL_PRE_PING', 'true').lower() == 'true',  # Verify connections before use
        }
    }

    # config["custom_fact_extraction_prompt"] = CUSTOM_INSTRUCTIONS

    # Create and cache the Memory client with retry logic
    max_retries = int(os.environ.get('DB_CONNECTION_RETRIES', '3'))
    retry_delay = int(os.environ.get('DB_RETRY_DELAY', '5'))

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to create Mem0 client (attempt {attempt + 1}/{max_retries})")
            _mem0_client = Memory.from_config(config)
            logger.info("Successfully created Mem0 client")
            return _mem0_client
        except Exception as e:
            logger.warning(f"Failed to create Mem0 client (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("Failed to create Mem0 client after all retries")
                raise

def reset_mem0_client():
    """Reset the global Mem0 client instance. Useful for testing or container restarts."""
    global _mem0_client
    _mem0_client = None
