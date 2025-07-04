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

from utils import get_mem0_client, reset_mem0_client

load_dotenv()

# Default user ID for memory operations
DEFAULT_USER_ID = "user"

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
async def save_memory(ctx: Context, text: str, user_id: str = DEFAULT_USER_ID, agent_id: str = None, memory_type: str = None) -> str:
    """Save information to your long-term memory.

    This tool is designed to store any type of information that might be useful in the future.
    The content will be processed and indexed for later retrieval through semantic search.

    Args:
        ctx: The MCP server provided context which includes the Mem0 client
        text: The content to store in memory, including any relevant details and context
        user_id: Optional user ID to isolate memories for different users (default: "user")
        agent_id: Optional agent ID for procedural memory (required when memory_type is "procedural_memory")
        memory_type: Optional memory type - use "procedural_memory" for procedural memory creation
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client
        messages = [{"role": "user", "content": text}]

        # Prepare add parameters
        add_params = {"user_id": user_id}
        if agent_id:
            add_params["agent_id"] = agent_id
        if memory_type:
            add_params["memory_type"] = memory_type

        mem0_client.add(messages, **add_params)

        memory_type_str = f" ({memory_type})" if memory_type else ""
        agent_str = f" for agent '{agent_id}'" if agent_id else ""
        return f"Successfully saved memory{memory_type_str} for user '{user_id}'{agent_str}: {text[:100]}..." if len(text) > 100 else f"Successfully saved memory{memory_type_str} for user '{user_id}'{agent_str}: {text}"
    except Exception as e:
        return f"Error saving memory: {str(e)}"

@mcp.tool()
async def save_procedural_memory(ctx: Context, text: str, agent_id: str, user_id: str = DEFAULT_USER_ID) -> str:
    """Save procedural memory for an AI agent's actions and interactions.

    This tool is specifically designed for storing structured summaries of AI agent actions,
    interactions, and their outcomes during specific tasks or processes. It creates a
    detailed log or "how-to" guide that can be referenced later.

    Args:
        ctx: The MCP server provided context which includes the Mem0 client
        text: The content describing the agent's actions, interactions, and outcomes
        agent_id: Required agent ID to associate the procedural memory with a specific agent
        user_id: Optional user ID to isolate memories for different users (default: "user")
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client
        messages = [{"role": "user", "content": text}]

        mem0_client.add(
            messages,
            user_id=user_id,
            agent_id=agent_id,
            memory_type="procedural_memory"
        )

        return f"Successfully saved procedural memory for agent '{agent_id}' (user '{user_id}'): {text[:100]}..." if len(text) > 100 else f"Successfully saved procedural memory for agent '{agent_id}' (user '{user_id}'): {text}"
    except Exception as e:
        return f"Error saving procedural memory: {str(e)}"

@mcp.tool()
async def get_all_memories(ctx: Context, user_id: str = DEFAULT_USER_ID, agent_id: str = None) -> str:
    """Get all stored memories for the user.

    Call this tool when you need complete context of all previously memories.

    Args:
        ctx: The MCP server provided context which includes the Mem0 client
        user_id: Optional user ID to retrieve memories for a specific user (default: "user")
        agent_id: Optional agent ID to filter memories for a specific agent

    Returns a JSON formatted list of all stored memories, including when they were created
    and their content. Results are paginated with a default of 50 items per page.
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client
        memories = mem0_client.get_all(user_id=user_id)

        if isinstance(memories, dict) and "results" in memories:
            # Filter by agent_id if provided
            if agent_id:
                filtered_memories = [
                    memory for memory in memories["results"]
                    if memory.get("metadata", {}).get("agent_id") == agent_id
                ]
                flattened_memories = [memory["memory"] for memory in filtered_memories]
            else:
                flattened_memories = [memory["memory"] for memory in memories["results"]]
        else:
            flattened_memories = memories

        return json.dumps(flattened_memories, indent=2)
    except Exception as e:
        return f"Error retrieving memories: {str(e)}"

@mcp.tool()
async def search_memories(ctx: Context, query: str, limit: int = 3, user_id: str = DEFAULT_USER_ID, agent_id: str = None) -> str:
    """Search memories using semantic search.

    This tool should be called to find relevant information from your memory. Results are ranked by relevance.
    Always search your memories before making decisions to ensure you leverage your existing knowledge.

    Args:
        ctx: The MCP server provided context which includes the Mem0 client
        query: Search query string describing what you're looking for. Can be natural language.
        limit: Maximum number of results to return (default: 3)
        user_id: Optional user ID to search memories for a specific user (default: "user")
        agent_id: Optional agent ID to filter memories for a specific agent
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client
        memories = mem0_client.search(query, user_id=user_id, limit=limit)

        if isinstance(memories, dict) and "results" in memories:
            # Filter by agent_id if provided
            if agent_id:
                filtered_memories = [
                    memory for memory in memories["results"]
                    if memory.get("metadata", {}).get("agent_id") == agent_id
                ]
                flattened_memories = [memory["memory"] for memory in filtered_memories]
            else:
                flattened_memories = [memory["memory"] for memory in memories["results"]]
        else:
            flattened_memories = memories

        return json.dumps(flattened_memories, indent=2)
    except Exception as e:
        return f"Error searching memories: {str(e)}"

@mcp.tool()
async def search_procedural_memories(ctx: Context, query: str, agent_id: str = None, limit: int = 3, user_id: str = DEFAULT_USER_ID) -> str:
    """Search procedural memories using semantic search.

    This tool searches specifically for procedural memories - structured summaries of AI agent
    actions, interactions, and outcomes. Useful for finding relevant procedures or workflows.

    Args:
        ctx: The MCP server provided context which includes the Mem0 client
        query: Search query string describing what procedural memory you're looking for
        agent_id: Optional agent ID to search memories for a specific agent
        limit: Maximum number of results to return (default: 3)
        user_id: Optional user ID to search memories for a specific user (default: "user")
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client

        # Search with memory type filter if supported by the underlying implementation
        # For now, we'll use the standard search and filter results
        memories = mem0_client.search(query, user_id=user_id, limit=limit)

        if isinstance(memories, dict) and "results" in memories:
            # Filter for procedural memories if agent_id is provided
            if agent_id:
                filtered_memories = [
                    memory for memory in memories["results"]
                    if memory.get("metadata", {}).get("agent_id") == agent_id
                ]
                flattened_memories = [memory["memory"] for memory in filtered_memories]
            else:
                flattened_memories = [memory["memory"] for memory in memories["results"]]
        else:
            flattened_memories = memories

        return json.dumps(flattened_memories, indent=2)
    except Exception as e:
        return f"Error searching procedural memories: {str(e)}"

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
