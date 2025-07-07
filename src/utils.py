"""
Utility functions for the MCP-Mem0 server.

This module provides functions for creating and managing the Mem0 client instance
with proper singleton pattern and retry logic for database connections.
"""

from mem0 import Memory
import os
import time
import logging

# Set up logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mcp-mem0.log')
    ]
)
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


def get_mem0_client() -> Memory:
    """
    Get or create the Mem0 client instance with singleton pattern and retry logic.

    This function implements a singleton pattern to ensure only one Mem0 client
    is created per application instance, preventing connection pool exhaustion.
    It also includes retry logic to handle temporary database connection failures.

    Returns:
        Memory: The Mem0 client instance

    Raises:
        Exception: If client creation fails after all retry attempts
    """
    global _mem0_client

    # Return existing client if already created
    if _mem0_client is not None:
        return _mem0_client

    # Get LLM provider and configuration
    llm_provider = os.getenv('LLM_PROVIDER')
    llm_api_key = os.getenv('LLM_API_KEY')
    llm_model = os.getenv('LLM_CHOICE')
    embedding_model = os.getenv('EMBEDDING_MODEL_CHOICE')

    logger.info(f"Configuring Mem0 client with LLM provider: {llm_provider}")
    logger.debug(f"LLM model: {llm_model}, Embedding model: {embedding_model}")

    # Initialize config dictionary
    config = {}

    # Configure LLM based on provider
    if llm_provider == 'openai' or llm_provider == 'openrouter':
        logger.info(f"Configuring OpenAI/OpenRouter LLM with model: {llm_model}")
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
            logger.debug("Set OPENAI_API_KEY from environment variable")

        # For OpenRouter, set the specific API key
        if llm_provider == 'openrouter' and llm_api_key:
            os.environ["OPENROUTER_API_KEY"] = llm_api_key
            logger.debug("Set OPENROUTER_API_KEY from environment variable")

    elif llm_provider == 'ollama':
        logger.info(f"Configuring Ollama LLM with model: {llm_model}")
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
            logger.debug(f"Set Ollama base URL: {llm_base_url}")

    # Configure embedder based on provider
    if llm_provider == 'openai':
        embed_model = embedding_model or "text-embedding-3-small"
        logger.info(f"Configuring OpenAI embedder with model: {embed_model}")
        config["embedder"] = {
            "provider": "openai",
            "config": {
                "model": embed_model,
                "embedding_dims": 1536  # Default for text-embedding-3-small
            }
        }

        # Set API key in environment if not already set
        if llm_api_key and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = llm_api_key
            logger.debug("Set OPENAI_API_KEY for embedder")

    elif llm_provider == 'ollama':
        embed_model = embedding_model or "nomic-embed-text"
        logger.info(f"Configuring Ollama embedder with model: {embed_model}")
        config["embedder"] = {
            "provider": "ollama",
            "config": {
                "model": embed_model,
                "embedding_dims": 768  # Default for nomic-embed-text
            }
        }

        # Set base URL for Ollama if provided
        embedding_base_url = os.getenv('LLM_BASE_URL')
        if embedding_base_url:
            config["embedder"]["config"]["ollama_base_url"] = embedding_base_url
            logger.debug(f"Set Ollama base URL for embedder: {embedding_base_url}")

    # Configure Supabase vector store
    connection_string = os.environ.get('DATABASE_URL', '')

    if not connection_string:
        logger.warning("No DATABASE_URL provided - this will cause memory operations to fail")
    else:
        logger.info("Using connection string for vector store")
        # Log connection string without sensitive info
        safe_connection = connection_string.split('@')[1] if '@' in connection_string else "unknown"
        logger.debug(f"Database connection: ...@{safe_connection}")

    embedding_dims = 1536 if llm_provider == "openai" else 768
    logger.info(f"Configuring Supabase vector store with embedding dimensions: {embedding_dims}")

    config["vector_store"] = {
        "provider": "supabase",
        "config": {
            "connection_string": connection_string,
            "collection_name": "mem0_memories",
            "embedding_model_dims": embedding_dims,
        }
    }

    # config["custom_fact_extraction_prompt"] = CUSTOM_INSTRUCTIONS

    # Create and cache the Memory client with retry logic
    max_retries = int(os.environ.get('DB_CONNECTION_RETRIES', '3'))
    retry_delay = int(os.environ.get('DB_RETRY_DELAY', '5'))

    logger.info(f"Creating Mem0 client with {max_retries} retry attempts and {retry_delay}s delay")

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to create Mem0 client (attempt {attempt + 1}/{max_retries})")
            _mem0_client = Memory.from_config(config)
            logger.info("Successfully created Mem0 client")
            return _mem0_client
        except Exception as e:
            logger.warning(f"Failed to create Mem0 client (attempt {attempt + 1}/{max_retries}): {e}")
            logger.debug(f"Error type: {type(e).__name__}")

            # Provide more specific error information
            if "connection" in str(e).lower() or "database" in str(e).lower():
                logger.error("Database connection issue detected")
            elif "api" in str(e).lower() or "key" in str(e).lower():
                logger.error("API/authentication issue detected")
            elif "embedding" in str(e).lower():
                logger.error("Embedding model issue detected")

            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("Failed to create Mem0 client after all retries")
                logger.error(f"Final error: {e}")
                raise


def reset_mem0_client() -> None:
    """
    Reset the global Mem0 client instance.

    This function is useful for testing or when the application needs to be restarted.
    It clears the singleton instance, allowing a new client to be created on the next
    call to get_mem0_client().
    """
    global _mem0_client
    _mem0_client = None
