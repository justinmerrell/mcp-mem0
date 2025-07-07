<h1 align="center">MCP-Mem0: Long-Term Memory for AI Agents</h1>

<p align="center">
  <img src="public/Mem0AndMCP.png" alt="Mem0 and MCP Integration" width="600">
</p>

A template implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server integrated with [Mem0](https://mem0.ai) for providing AI agents with persistent memory capabilities.

Use this as a reference point to build your MCP servers yourself, or give this as an example to an AI coding assistant and tell it to follow this example for structure and code correctness!

## Overview

This project demonstrates how to build an MCP server that enables AI agents to store, retrieve, and search memories using semantic search. It serves as a practical template for creating your own MCP servers, simply using Mem0 and a practical example.

The implementation follows the best practices laid out by Anthropic for building MCP servers, allowing seamless integration with any MCP-compatible client.

## Features

The server provides five essential memory management tools:

1. **`save_memory`**: Store any information in long-term memory with semantic indexing
2. **`save_procedural_memory`**: Store structured summaries of AI agent actions and interactions
3. **`get_all_memories`**: Retrieve all stored memories for comprehensive context
4. **`search_memories`**: Find relevant memories using semantic search
5. **`search_procedural_memories`**: Search specifically for procedural memories

All tools support optional user isolation through a `user_id` parameter, allowing you to maintain separate memory spaces for different users or sessions. Procedural memory tools also support `agent_id` for tracking specific AI agents.

## Prerequisites

- Python 3.12+
- Supabase or any PostgreSQL database (for vector storage of memories)
- API keys for your chosen LLM provider (OpenAI, OpenRouter, or Ollama)
- Docker if running the MCP server as a container (recommended)

## Installation

### Using uv

1. Install uv if you don't have it:

   ```bash
   pip install uv
   ```

2. Clone this repository:

   ```bash
   git clone https://github.com/coleam00/mcp-mem0.git
   cd mcp-mem0
   ```

3. Install dependencies:

   ```bash
   uv pip install -e .
   ```

4. Create a `.env` file based on `.env.example`:

   ```bash
   cp .env.example .env
   ```

5. Configure your environment variables in the `.env` file (see Configuration section)

### Using Docker (Recommended)

1. Build the Docker image:

   ```bash
   docker build -t mcp/mem0 --build-arg PORT=8050 .
   ```

2. Create a `.env` file based on `.env.example` and configure your environment variables

## Configuration

The following environment variables can be configured in your `.env` file:

| Variable                 | Description                                  | Example                               |
| ------------------------ | -------------------------------------------- | ------------------------------------- |
| `TRANSPORT`              | Transport protocol (sse or stdio)            | `sse`                                 |
| `HOST`                   | Host to bind to when using SSE transport     | `0.0.0.0`                             |
| `PORT`                   | Port to listen on when using SSE transport   | `8050`                                |
| `LLM_PROVIDER`           | LLM provider (openai, openrouter, or ollama) | `openai`                              |
| `LLM_BASE_URL`           | Base URL for the LLM API                     | `https://api.openai.com/v1`           |
| `LLM_API_KEY`            | API key for the LLM provider                 | `sk-...`                              |
| `LLM_CHOICE`             | LLM model to use                             | `gpt-4o-mini`                         |
| `EMBEDDING_MODEL_CHOICE` | Embedding model to use                       | `text-embedding-3-small`              |
| `DATABASE_URL`           | PostgreSQL connection string                 | `postgresql://user:pass@host:port/db` |
| `DB_CONNECTION_RETRIES`  | Number of connection retry attempts          | `3`                                   |
| `DB_RETRY_DELAY`         | Delay between retry attempts (seconds)       | `5`                                   |

## Running the Server

### Using uv

#### SSE Transport

```bash
# Set TRANSPORT=sse in .env then:
uv run src/main.py
```

The MCP server will essentially be run as an API endpoint that you can then connect to with config shown below.

#### Stdio Transport

With stdio, the MCP client iself can spin up the MCP server, so nothing to run at this point.

### Using Docker

#### SSE Transport

```bash
docker run --env-file .env -p:8050:8050 mcp/mem0
```

The MCP server will essentially be run as an API endpoint within the container that you can then connect to with config shown below.

#### Stdio Transport

With stdio, the MCP client iself can spin up the MCP server container, so nothing to run at this point.

## Integration with MCP Clients

### SSE Configuration

Once you have the server running with SSE transport, you can connect to it using this configuration:

```json
{
  "mcpServers": {
    "mem0": {
      "transport": "sse",
      "url": "http://localhost:8050/sse"
    }
  }
}
```

#### User ID and Agent ID Parameters (Optional)

You can optionally specify a user ID and/or agent ID as parameters to each tool to isolate and filter memories:

- `save_memory(text, user_id="alice")` - Save memory for user "alice"
- `get_all_memories(user_id="alice", agent_id="coding_agent_001")` - Get all memories for user "alice" from agent "coding_agent_001"
- `search_memories(query, user_id="alice", agent_id="coding_agent_001")` - Search memories for user "alice" from agent "coding_agent_001"

If no `user_id` parameter is provided, the server will use a default user ID of "user". If no `agent_id` is provided, all memories (regardless of agent) will be included. This allows you to:

- Separate memories between different users or sessions
- Filter memories by specific AI agents
- Maintain privacy and isolation of memory data
- Support multi-tenant scenarios

#### Procedural Memory Support

The server supports procedural memory for AI coding systems and agents. Procedural memory stores structured summaries of AI agent actions, interactions, and outcomes:

- `save_procedural_memory(text, agent_id="coding_agent_001", user_id="alice")` - Save procedural memory for a specific agent
- `search_procedural_memories(query, agent_id="coding_agent_001", user_id="alice")` - Search procedural memories for a specific agent
- `get_all_memories(user_id="alice", agent_id="coding_agent_001")` - Get all memories (including procedural) for a specific agent

**Procedural Memory Use Cases:**

- Track coding workflows and debugging processes
- Store successful problem-solving approaches
- Maintain context across development sessions
- Create reusable procedure libraries for AI agents

> **Note for Windsurf users**: Use `serverUrl` instead of `url` in your configuration:
>
> ```json
> {
>   "mcpServers": {
>     "mem0": {
>       "transport": "sse",
>       "serverUrl": "http://localhost:8050/sse"
>     }
>   }
> }
> ```

> **Note for n8n users**: Use host.docker.internal instead of localhost since n8n has to reach outside of it's own container to the host machine:
>
> So the full URL in the MCP node would be: http://host.docker.internal:8050/sse

Make sure to update the port if you are using a value other than the default 8050.

### Python with Stdio Configuration

Add this server to your MCP configuration for Claude Desktop, Windsurf, or any other MCP client:

```json
{
  "mcpServers": {
    "mem0": {
      "command": "your/path/to/mcp-mem0/.venv/Scripts/python.exe",
      "args": ["your/path/to/mcp-mem0/src/main.py"],
      "env": {
        "TRANSPORT": "stdio",
        "LLM_PROVIDER": "openai",
        "LLM_BASE_URL": "https://api.openai.com/v1",
        "LLM_API_KEY": "YOUR-API-KEY",
        "LLM_CHOICE": "gpt-4o-mini",
        "EMBEDDING_MODEL_CHOICE": "text-embedding-3-small",
        "DATABASE_URL": "YOUR-DATABASE-URL"
      }
    }
  }
}
```

### Docker with Stdio Configuration

```json
{
  "mcpServers": {
    "mem0": {
      "command": "docker",
      "args": [
        "run",
        "--rm",
        "-i",
        "-e",
        "TRANSPORT",
        "-e",
        "LLM_PROVIDER",
        "-e",
        "LLM_BASE_URL",
        "-e",
        "LLM_API_KEY",
        "-e",
        "LLM_CHOICE",
        "-e",
        "EMBEDDING_MODEL_CHOICE",
        "-e",
        "DATABASE_URL",
        "mcp/mem0"
      ],
      "env": {
        "TRANSPORT": "stdio",
        "LLM_PROVIDER": "openai",
        "LLM_BASE_URL": "https://api.openai.com/v1",
        "LLM_API_KEY": "YOUR-API-KEY",
        "LLM_CHOICE": "gpt-4o-mini",
        "EMBEDDING_MODEL_CHOICE": "text-embedding-3-small",
        "DATABASE_URL": "YOUR-DATABASE-URL"
      }
    }
  }
}
```

## Building Your Own Server

This template provides a foundation for building more complex MCP servers. To build your own:

1. Add your own tools by creating methods with the `@mcp.tool()` decorator
2. Create your own lifespan function to add your own dependencies (clients, database connections, etc.)
3. Modify the `utils.py` file for any helper functions you need for your MCP server
4. Feel free to add prompts and resources as well with `@mcp.resource()` and `@mcp.prompt()`

## Troubleshooting

### Database Connection Issues

If you encounter database connection errors like "remaining connection slots are reserved for non-replication superuser connections", this indicates that your database has reached its connection limit. Here are some solutions:

1. **Reduce Concurrent Connections**: The server now uses a singleton pattern to prevent multiple client instances. If you're still having issues, consider:

   - Running fewer instances of the application
   - Upgrading your database plan to support more connections

2. **Upgrade Your Database Plan**: If using DigitalOcean, consider upgrading to a plan with more connections.

3. **Check for Connection Leaks**: The server includes automatic cleanup and retry logic. Monitor your database connections to ensure they're being properly managed.

4. **Use Connection Recycling**: The server automatically handles connection cleanup and retries failed connections.

5. **Enable Connection Pre-ping**: The server includes retry logic that will attempt to reconnect if connections fail.

**Note**: The server now uses a singleton pattern to ensure only one Mem0 client instance is created, which helps prevent connection pool exhaustion. The retry logic will automatically attempt to reconnect if the database is temporarily unavailable.

### Common Environment Variables for Connection Issues

```bash
# Retry settings for connection issues
DB_CONNECTION_RETRIES=5
DB_RETRY_DELAY=10
```
