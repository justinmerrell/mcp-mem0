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
from typing import Optional

from utils import get_mem0_client, reset_mem0_client

load_dotenv()

# Global shutdown event
shutdown_event = asyncio.Event()

# Create a dataclass for our application context
@dataclass
class Mem0Context:
    """Context for the Mem0 MCP server."""
    mem0_client: Memory

@asynccontextmanager
async def mem0_lifespan(server: FastMCP) -> AsyncIterator[Mem0Context]:
    """
    Manages the Mem0 client lifecycle.

    Args:
        server: The FastMCP server instance

    Yields:
        Mem0Context: The context containing the Mem0 client
    """
    # Create and return the Memory client with the helper function in utils.py
    mem0_client = get_mem0_client()

    try:
        yield Mem0Context(mem0_client=mem0_client)
    finally:
        # Cleanup: Close any open connections
        try:
            # Try to close the Mem0 client directly
            if hasattr(mem0_client, 'close'):
                await mem0_client.close()
            elif hasattr(mem0_client, '_client') and hasattr(mem0_client._client, 'close'):
                await mem0_client._client.close()

            # Try to close underlying database connections
            if hasattr(mem0_client, '_vector_store') and hasattr(mem0_client._vector_store, 'close'):
                await mem0_client._vector_store.close()

            # Try to close any HTTP clients
            if hasattr(mem0_client, '_llm_client') and hasattr(mem0_client._llm_client, 'close'):
                await mem0_client._llm_client.close()
            if hasattr(mem0_client, '_embedder_client') and hasattr(mem0_client._embedder_client, 'close'):
                await mem0_client._embedder_client.close()

        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")
        finally:
            # Reset the global client instance
            reset_mem0_client()

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
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client

        # Prepare add parameters
        add_params = {"user_id": user_id, "agent_id": agent_id, "memory_type": memory_type}

        mem0_client.add(text, **add_params)

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
        return message
    except Exception as e:
        return f"Error saving memory: {str(e)}"

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
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client

        # Prepare search parameters
        search_params = {"user_id": user_id, "agent_id": agent_id}

        memories = mem0_client.get_all(**search_params)

        # Handle response format
        if isinstance(memories, dict) and "results" in memories:
            flattened_memories = [memory["memory"] for memory in memories["results"]]
        else:
            flattened_memories = memories

        return json.dumps(flattened_memories, indent=2)
    except Exception as e:
        return f"Error retrieving memories: {str(e)}"

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
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client

        # Prepare search parameters
        search_params = {"query": query, "limit": limit, "user_id": user_id, "agent_id": agent_id}

        memories = mem0_client.search(**search_params)

        # Handle response format
        if isinstance(memories, dict) and "results" in memories:
            flattened_memories = [memory["memory"] for memory in memories["results"]]
        else:
            flattened_memories = memories

        return json.dumps(flattened_memories, indent=2)
    except Exception as e:
        return f"Error searching memories: {str(e)}"

async def main():
    transport = os.getenv("TRANSPORT", "sse")

    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}. Shutting down gracefully...")
        shutdown_event.set()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        if transport == 'sse':
            # Run the MCP server with sse transport
            await mcp.run_sse_async()
        else:
            # Run the MCP server with stdio transport
            await mcp.run_stdio_async()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error running server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
