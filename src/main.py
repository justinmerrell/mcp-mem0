"""
MCP-Mem0 Server

This module implements a Model Context Protocol (MCP) server that provides
long-term memory storage and retrieval capabilities using the Mem0 library.

The server supports both SSE and stdio transport protocols and provides tools
for saving, searching, and retrieving memories with optional user and agent isolation.
"""

from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from dotenv import load_dotenv
from mem0 import Memory
import asyncio
import json
import os
import signal
import sys
import logging
import traceback
from typing import Optional

from utils import get_mem0_client, reset_mem0_client

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
logger.info(f"Logging level set to: {log_level}")

# Load environment variables
load_dotenv()

# Global shutdown event
shutdown_event = asyncio.Event()

@dataclass
class Mem0Context:
    """Context for the Mem0 MCP server containing the memory client."""
    mem0_client: Memory

@asynccontextmanager
async def mem0_lifespan(server: FastMCP) -> AsyncIterator[Mem0Context]:
    """
    Manage the Mem0 client lifecycle with proper cleanup.

    This function creates a singleton Mem0 client instance and ensures proper
    cleanup of database connections when the server shuts down.

    Args:
        server: The FastMCP server instance

    Yields:
        Mem0Context: The context containing the Mem0 client

    Note:
        The get_mem0_client function handles the singleton pattern and retry logic.
        This lifespan function only manages the lifecycle and cleanup.
    """
    logger.info("Initializing Mem0 client for MCP server")

    # Create and return the Memory client with the helper function in utils.py
    # The get_mem0_client function already handles singleton pattern and retry logic
    try:
        mem0_client = get_mem0_client()
        logger.info("Successfully initialized Mem0 client")
    except Exception as e:
        logger.error(f"Failed to initialize Mem0 client: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

    try:
        yield Mem0Context(mem0_client=mem0_client)
    finally:
        # Cleanup: Close any open connections
        logger.info("Cleaning up Mem0 client connections")
        try:
            # Try to close the Mem0 client directly
            if hasattr(mem0_client, 'close'):
                await mem0_client.close()
                logger.debug("Closed Mem0 client directly")
            elif hasattr(mem0_client, '_client') and hasattr(mem0_client._client, 'close'):
                await mem0_client._client.close()
                logger.debug("Closed Mem0 client._client")

            # Try to close underlying database connections
            if hasattr(mem0_client, '_vector_store') and hasattr(mem0_client._vector_store, 'close'):
                await mem0_client._vector_store.close()
                logger.debug("Closed vector store")

            # Try to close vecs client database connections
            if hasattr(mem0_client, '_vector_store') and hasattr(mem0_client._vector_store, 'db'):
                vecs_client = mem0_client._vector_store.db
                if hasattr(vecs_client, 'engine') and hasattr(vecs_client.engine, 'dispose'):
                    vecs_client.engine.dispose()
                    logger.debug("Disposed vecs client engine")
                elif hasattr(vecs_client, 'close'):
                    await vecs_client.close()
                    logger.debug("Closed vecs client")

            # Try to close any HTTP clients
            if hasattr(mem0_client, '_llm_client') and hasattr(mem0_client._llm_client, 'close'):
                await mem0_client._llm_client.close()
                logger.debug("Closed LLM client")
            if hasattr(mem0_client, '_embedder_client') and hasattr(mem0_client._embedder_client, 'close'):
                await mem0_client._embedder_client.close()
                logger.debug("Closed embedder client")

            logger.info("Successfully cleaned up all Mem0 client connections")

        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
            logger.debug(f"Cleanup traceback: {traceback.format_exc()}")
        finally:
            # Only reset the global client instance if we're shutting down
            # Don't reset it on every request to maintain the singleton pattern
            pass

# Initialize FastMCP server with the Mem0 client as context
mcp = FastMCP(
    "mcp-mem0",
    description="MCP server for long term memory storage and retrieval with Mem0",
    lifespan=mem0_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=os.getenv("PORT", "8050")
)

@mcp.tool()
async def save_memory(ctx: Context, text: str, user_id: Optional[str] = None, agent_id: Optional[str] = None, memory_type: Optional[str] = "") -> str:
    """Store information in long-term memory for future retrieval using semantic search.

    This tool saves any type of information that might be useful in the future. The content
    is processed, embedded, and indexed for later retrieval through semantic search. Use this
    when you want to remember conversations, facts, preferences, or any contextual information.

    **When to use:**
    - Storing conversation history or important details
    - Saving user preferences or settings
    - Recording factual information for future reference
    - Creating a knowledge base of interactions
    - Storing procedural memories (use memory_type="procedural_memory" with agent_id)

    **Examples:**
    - "User prefers dark mode and uses Python for backend development"
    - "Discussed project requirements: need authentication, user management, and API endpoints"
    - "User's name is John, works at TechCorp, interested in AI and machine learning"
    - For procedural memories: "Agent analyzed user requirements, created database schema with 3 tables,
      implemented REST API endpoints, and deployed to staging environment. Process took 2 hours."

    **Limitations:**
    - Content is limited to text format
    - Maximum text length depends on the underlying Mem0 implementation
    - Memories are isolated by user_id for privacy

    Args:
        ctx: The MCP server provided context which includes the Mem0 client
        text: The content to store in memory. Include relevant details, context, and any
              information that would be useful for future reference. Be descriptive and
              comprehensive to improve search relevance.
        user_id: Optional user identifier to isolate memories for different users.
                 Use unique IDs for different users to maintain privacy. If None or not provided,
                 memories will be stored as anonymous/shared memories.
        agent_id: Optional agent identifier for procedural memory. Required when
                  memory_type is "procedural_memory" to associate memories with specific agents.
        memory_type: Optional memory type classification. Use "procedural_memory" for
                     storing agent actions and workflows.

    Returns:
        str: Success message with details about the saved memory
    """
    try:
        logger.info(f"Attempting to save memory - Text length: {len(text)}, User ID: {user_id}, Agent ID: {agent_id}, Memory Type: {memory_type}")

        mem0_client = ctx.request_context.lifespan_context.mem0_client

        # Prepare add parameters
        add_params = {"user_id": user_id, "agent_id": agent_id, "memory_type": memory_type}
        logger.debug(f"Memory add parameters: {add_params}")

        # Attempt to add memory
        result = mem0_client.add(text, **add_params)
        logger.info(f"Successfully added memory to Mem0 client")

        # Build response message
        parts = ["Successfully saved memory"]
        if memory_type:
            parts.append(f"({memory_type})")
        if user_id:
            parts.append(f"for user '{user_id}'")
        else:
            parts.append("(anonymous/shared)")
        if agent_id:
            parts.append(f"for agent '{agent_id}'")

        message = " ".join(parts) + ": "
        message += text[:100] + "..." if len(text) > 100 else text

        logger.info(f"Memory saved successfully: {message}")
        return message

    except Exception as e:
        error_msg = f"Error saving memory: {str(e)}"
        logger.error(f"Failed to save memory: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Full traceback: {traceback.format_exc()}")

        # Provide more specific error messages based on exception type
        if "connection" in str(e).lower() or "database" in str(e).lower():
            error_msg += "\n\nPossible causes:\n- Database connection issue\n- Invalid DATABASE_URL\n- Database server is down\n- Network connectivity problem"
        elif "api" in str(e).lower() or "key" in str(e).lower():
            error_msg += "\n\nPossible causes:\n- Invalid LLM API key\n- API quota exceeded\n- LLM service is down\n- Incorrect LLM_PROVIDER configuration"
        elif "embedding" in str(e).lower():
            error_msg += "\n\nPossible causes:\n- Embedding model configuration issue\n- Invalid EMBEDDING_MODEL_CHOICE\n- Embedding service is down"
        elif "memory" in str(e).lower() or "text" in str(e).lower():
            error_msg += "\n\nPossible causes:\n- Text too long for the model\n- Invalid text format\n- Memory processing failed"

        return error_msg

@mcp.tool()
async def get_all_memories(ctx: Context, user_id: Optional[str] = None, agent_id: Optional[str] = None) -> str:
    """Retrieve all stored memories for a user, optionally filtered by agent.

    This tool returns a complete list of all memories stored for a specific user. Use this
    when you need comprehensive context of all previous interactions, facts, and procedural
    memories. Results are returned in JSON format with creation timestamps and content.

    **When to use:**
    - Getting complete context for a new conversation
    - Reviewing all stored information about a user
    - Analyzing patterns in stored memories
    - Debugging or auditing memory contents
    - Creating backups or exports of memory data

    **Examples:**
    - Get all memories for user "john_doe"
    - Get all procedural memories for agent "code_assistant" belonging to user "alice"
    - Retrieve complete memory history for analysis

    **Limitations:**
    - Results are paginated (default 50 items per page)
    - Large memory sets may impact performance
    - Returns all memory types mixed together unless filtered by agent_id
    - JSON output format for structured data access

    Args:
        ctx: The MCP server provided context which includes the Mem0 client
        user_id: Optional user identifier to retrieve memories for a specific user.
                 Use the same user_id that was used when saving memories. If None or not provided,
                 retrieves anonymous/shared memories.
        agent_id: Optional agent identifier to filter memories for a specific agent.
                  When provided, only returns memories associated with this agent.
                  Useful for isolating procedural memories or agent-specific data.

    Returns:
        str: JSON string containing all memories for the specified user/agent
    """
    try:
        logger.info(f"Attempting to retrieve all memories - User ID: {user_id}, Agent ID: {agent_id}")

        mem0_client = ctx.request_context.lifespan_context.mem0_client

        # Prepare search parameters
        search_params = {"user_id": user_id, "agent_id": agent_id}
        logger.debug(f"Memory retrieval parameters: {search_params}")

        memories = mem0_client.get_all(**search_params)
        logger.info(f"Successfully retrieved memories from Mem0 client")

        # Handle response format
        if isinstance(memories, dict) and "results" in memories:
            flattened_memories = [memory["memory"] for memory in memories["results"]]
            logger.debug(f"Retrieved {len(flattened_memories)} memories from results dict")
        else:
            flattened_memories = memories
            logger.debug(f"Retrieved {len(flattened_memories) if hasattr(flattened_memories, '__len__') else 'unknown'} memories")

        result = json.dumps(flattened_memories, indent=2)
        logger.info(f"Successfully formatted {len(flattened_memories) if hasattr(flattened_memories, '__len__') else 'unknown'} memories as JSON")
        return result

    except Exception as e:
        error_msg = f"Error retrieving memories: {str(e)}"
        logger.error(f"Failed to retrieve memories: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Full traceback: {traceback.format_exc()}")

        # Provide more specific error messages based on exception type
        if "connection" in str(e).lower() or "database" in str(e).lower():
            error_msg += "\n\nPossible causes:\n- Database connection issue\n- Invalid DATABASE_URL\n- Database server is down\n- Network connectivity problem"
        elif "json" in str(e).lower():
            error_msg += "\n\nPossible causes:\n- Memory data format issue\n- Invalid memory structure\n- JSON serialization problem"
        elif "permission" in str(e).lower() or "access" in str(e).lower():
            error_msg += "\n\nPossible causes:\n- Database permission issue\n- Invalid user_id/agent_id\n- Access denied to memory collection"

        return error_msg

@mcp.tool()
async def search_memories(ctx: Context, query: str, limit: int = 3, user_id: Optional[str] = None, agent_id: Optional[str] = None) -> str:
    """Search memories using semantic search to find relevant information from stored knowledge.

    This tool performs semantic search across all stored memories to find the most relevant
    information based on your query. Results are ranked by relevance and can include both
    general memories and procedural memories. Always search your memories before making
    decisions to leverage existing knowledge effectively.

    **When to use:**
    - Looking for specific information or facts
    - Finding relevant past conversations or interactions
    - Retrieving user preferences or settings
    - Searching for procedural knowledge or workflows (use agent_id to filter)
    - Getting context for current tasks or decisions

    **Examples:**
    - Search for "authentication setup" to find related procedures
    - Query "user preferences" to find stored settings
    - Search "project requirements" to recall discussed features
    - Look for "debugging steps" to find troubleshooting procedures
    - Search for "deployment process" with agent_id to find specific agent workflows

    **Best practices:**
    - Use natural language queries for better semantic matching
    - Be specific but not overly restrictive in your search terms
    - Start with broader queries and refine if needed
    - Use the same user_id that was used when saving memories
    - Use agent_id to filter for procedural memories from specific agents

    **Limitations:**
    - Search quality depends on how well memories were originally described
    - Results limited to 3 items by default (adjustable via limit parameter)
    - Semantic search may not find exact text matches
    - Performance may vary with large memory sets

    Args:
        ctx: The MCP server provided context which includes the Mem0 client
        query: Search query in natural language describing what you're looking for.
               Be descriptive and use relevant keywords. Examples: "user authentication setup",
               "project requirements discussion", "debugging steps for API errors".
        limit: Maximum number of results to return (default: 3, max recommended: 10).
               Higher limits may impact performance and relevance.
        user_id: Optional user identifier to search memories for a specific user.
                 Use the same user_id that was used when saving memories. If None or not provided,
                 searches anonymous/shared memories.
        agent_id: Optional agent identifier to filter memories for a specific agent.
                  When provided, only searches memories associated with this agent.
                  Useful for finding agent-specific procedures or workflows.

    Returns:
        str: JSON string containing search results ranked by relevance
    """
    try:
        logger.info(f"Attempting to search memories - Query: '{query[:50]}{'...' if len(query) > 50 else ''}', Limit: {limit}, User ID: {user_id}, Agent ID: {agent_id}")

        mem0_client = ctx.request_context.lifespan_context.mem0_client

        # Prepare search parameters
        search_params = {"query": query, "limit": limit, "user_id": user_id, "agent_id": agent_id}
        logger.debug(f"Memory search parameters: {search_params}")

        memories = mem0_client.search(**search_params)
        logger.info(f"Successfully searched memories from Mem0 client")

        # Handle response format
        if isinstance(memories, dict) and "results" in memories:
            flattened_memories = [memory["memory"] for memory in memories["results"]]
            logger.debug(f"Found {len(flattened_memories)} memories from results dict")
        else:
            flattened_memories = memories
            logger.debug(f"Found {len(flattened_memories) if hasattr(flattened_memories, '__len__') else 'unknown'} memories")

        result = json.dumps(flattened_memories, indent=2)
        logger.info(f"Successfully formatted {len(flattened_memories) if hasattr(flattened_memories, '__len__') else 'unknown'} search results as JSON")
        return result

    except Exception as e:
        error_msg = f"Error searching memories: {str(e)}"
        logger.error(f"Failed to search memories: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Full traceback: {traceback.format_exc()}")

        # Provide more specific error messages based on exception type
        if "connection" in str(e).lower() or "database" in str(e).lower():
            error_msg += "\n\nPossible causes:\n- Database connection issue\n- Invalid DATABASE_URL\n- Database server is down\n- Network connectivity problem"
        elif "embedding" in str(e).lower():
            error_msg += "\n\nPossible causes:\n- Embedding model configuration issue\n- Invalid EMBEDDING_MODEL_CHOICE\n- Embedding service is down\n- Query too long for embedding model"
        elif "api" in str(e).lower() or "key" in str(e).lower():
            error_msg += "\n\nPossible causes:\n- Invalid LLM API key\n- API quota exceeded\n- LLM service is down\n- Incorrect LLM_PROVIDER configuration"
        elif "query" in str(e).lower() or "search" in str(e).lower():
            error_msg += "\n\nPossible causes:\n- Invalid search query format\n- Query too long or empty\n- Search index not available"
        elif "json" in str(e).lower():
            error_msg += "\n\nPossible causes:\n- Memory data format issue\n- Invalid memory structure\n- JSON serialization problem"

        return error_msg

@mcp.tool()
async def debug_memory_system(ctx: Context) -> str:
    """Debug the memory system to check configuration and connectivity.

    This tool provides detailed information about the current memory system
    configuration, including database connectivity, LLM provider status, and
    any potential issues. Use this when troubleshooting memory operations.

    **When to use:**
    - Diagnosing memory save/retrieve failures
    - Checking system configuration
    - Verifying database connectivity
    - Troubleshooting LLM provider issues
    - Validating environment variables

    Returns:
        str: Detailed diagnostic information about the memory system
    """
    try:
        logger.info("Running memory system diagnostics")

        mem0_client = ctx.request_context.lifespan_context.mem0_client

        # Collect diagnostic information
        diagnostics = {
            "timestamp": str(asyncio.get_event_loop().time()),
            "environment": {},
            "client_status": {},
            "recommendations": []
        }

        # Check environment variables
        env_vars = [
            'LLM_PROVIDER', 'LLM_CHOICE', 'EMBEDDING_MODEL_CHOICE',
            'DATABASE_URL', 'DB_CONNECTION_RETRIES', 'DB_RETRY_DELAY',
            'LOG_LEVEL', 'TRANSPORT'
        ]

        for var in env_vars:
            value = os.getenv(var)
            if value:
                # Mask sensitive values
                if 'KEY' in var or 'URL' in var:
                    if '@' in value:
                        # For database URLs, show only the host part
                        safe_value = f"...@{value.split('@')[1]}"
                    else:
                        safe_value = f"{value[:8]}..." if len(value) > 8 else "***"
                else:
                    safe_value = value
                diagnostics["environment"][var] = safe_value
            else:
                diagnostics["environment"][var] = "NOT_SET"
                if var in ['LLM_PROVIDER', 'DATABASE_URL']:
                    diagnostics["recommendations"].append(f"Set {var} environment variable")

        # Check client attributes
        try:
            diagnostics["client_status"]["has_vector_store"] = hasattr(mem0_client, '_vector_store')
            diagnostics["client_status"]["has_llm_client"] = hasattr(mem0_client, '_llm_client')
            diagnostics["client_status"]["has_embedder_client"] = hasattr(mem0_client, '_embedder_client')

            # Test basic operations
            if hasattr(mem0_client, 'add'):
                diagnostics["client_status"]["add_method_available"] = True
            else:
                diagnostics["client_status"]["add_method_available"] = False
                diagnostics["recommendations"].append("Mem0 client missing 'add' method")

            if hasattr(mem0_client, 'search'):
                diagnostics["client_status"]["search_method_available"] = True
            else:
                diagnostics["client_status"]["search_method_available"] = False
                diagnostics["recommendations"].append("Mem0 client missing 'search' method")

        except Exception as e:
            diagnostics["client_status"]["error"] = str(e)
            diagnostics["recommendations"].append(f"Client status check failed: {e}")

        # Add general recommendations
        if not diagnostics["environment"].get('DATABASE_URL') or diagnostics["environment"].get('DATABASE_URL') == "NOT_SET":
            diagnostics["recommendations"].append("DATABASE_URL is required for memory operations")

        if not diagnostics["environment"].get('LLM_PROVIDER') or diagnostics["environment"].get('LLM_PROVIDER') == "NOT_SET":
            diagnostics["recommendations"].append("LLM_PROVIDER is required for memory processing")

        logger.info("Memory system diagnostics completed")
        return json.dumps(diagnostics, indent=2)

    except Exception as e:
        error_msg = f"Error running diagnostics: {str(e)}"
        logger.error(f"Failed to run diagnostics: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return error_msg

async def main() -> None:
    """
    Main entry point for the MCP server.

    Sets up signal handlers for graceful shutdown and runs the server
    with the appropriate transport protocol (SSE or stdio).
    """
    transport = os.getenv("TRANSPORT", "sse")
    logger.info(f"Starting MCP-Mem0 server with transport: {transport}")

    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}. Shutting down gracefully...")
        shutdown_event.set()
        # Reset the global client instance on shutdown
        reset_mem0_client()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        if transport == 'sse':
            logger.info("Starting MCP server with SSE transport")
            # Run the MCP server with sse transport
            await mcp.run_sse_async()
        else:
            logger.info("Starting MCP server with stdio transport")
            # Run the MCP server with stdio transport
            await mcp.run_stdio_async()
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully due to keyboard interrupt...")
        reset_mem0_client()
    except Exception as e:
        logger.error(f"Error running server: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        reset_mem0_client()
        sys.exit(1)

if __name__ == "__main__":
    try:
        logger.info("MCP-Mem0 server starting up...")
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)
