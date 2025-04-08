# client_app_separate.py

import asyncio
import os
import json
from dotenv import load_dotenv

# MCP and Adapter Imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

# LangChain/LangGraph Imports
from langchain_anthropic import ChatAnthropic # Using Anthropic instead of OpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage # Or use tuples ("user", content)
from langchain_core.tools import BaseTool
from typing import List

# --- Configuration ---
load_dotenv() # Load API keys (ANTHROPIC_API_KEY, etc.) from .env

# >>> IMPORTANT: Set the correct paths to your server scripts <<<
# Can be relative (if client is run from the same dir) or absolute
INVEST_SERVER_SCRIPT_PATH = "invest_tool_server.py" # <-- ADJUST THIS PATH if needed
BPG_SERVER_SCRIPT_PATH = "bpg_tool_server.py"       # <-- ADJUST THIS PATH if needed

# Consider using absolute paths for robustness:
# INVEST_SERVER_SCRIPT_PATH = os.path.abspath("invest_tool_server.py")
# BPG_SERVER_SCRIPT_PATH = os.path.abspath("bpg_tool_server.py")

# --- Helper: Check if scripts exist ---
for script_path in [INVEST_SERVER_SCRIPT_PATH, BPG_SERVER_SCRIPT_PATH]:
    if not os.path.exists(script_path):
        print(f"❌ Error: Server script not found at '{script_path}'")
        print("Please update the script path variables in this client script.")
        exit(1)
    elif not os.path.isfile(script_path):
        print(f"❌ Error: Path '{script_path}' is not a file.")
        exit(1)


# --- Main Async Function ---
async def run_agent_with_separate_mcp_tools():
    print("🐍 Client App: Preparing to connect to two separate MCP servers...")

    # --- Configure the LLM for the agent (Anthropic) ---
    try:
        model = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
        # Anthropic SDK checks for ANTHROPIC_API_KEY (or CLAUDE_API_KEY)
        if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("CLAUDE_API_KEY"):
             print("⚠️ Warning: ANTHROPIC_API_KEY/CLAUDE_API_KEY not found. Agent LLM may fail.")
             print("   Ensure it's set in your environment or .env file.")
             # Consider exiting if the key is absolutely required
             # return

    except Exception as e:
        print(f"❌ Error initializing Anthropic chat model: {e}")
        print("   Ensure 'langchain-anthropic' is installed and the required API key is set.")
        return

    # --- Define Server Parameters ---
    # Use "python3" as the command based on previous error
    server_params_invest = StdioServerParameters(
        command="python3", # <-- CORRECTED
        args=[INVEST_SERVER_SCRIPT_PATH],
    )
    server_params_bpg = StdioServerParameters(
        command="python3", # <-- CORRECTED
        args=[BPG_SERVER_SCRIPT_PATH],
    )

    all_tools: List[BaseTool] = []
    invest_tools: List[BaseTool] = []
    bpg_tools: List[BaseTool] = []

    print(f"🚀 Starting MCP client connections...")
    try:
        # Establish connections to BOTH servers using nested async with
        # This ensures both connections are active when the agent might need them
        # Update print statements to reflect using python3
        print(f"   Connecting to INVEST server: python3 {INVEST_SERVER_SCRIPT_PATH}")
        async with stdio_client(server_params_invest) as (read_invest, write_invest):
            print(f"   Connecting to BPG server: python3 {BPG_SERVER_SCRIPT_PATH}")
            async with stdio_client(server_params_bpg) as (read_bpg, write_bpg):
                print("✅ Connections established. Initializing MCP sessions...")

                # Create MCP client sessions for each connection
                async with ClientSession(read_invest, write_invest) as session_invest, \
                           ClientSession(read_bpg, write_bpg) as session_bpg:

                    # Initialize both sessions (can be done concurrently)
                    await asyncio.gather(
                        session_invest.initialize(),
                        session_bpg.initialize()
                    )
                    print("✅ MCP Sessions initialized.")

                    # --- Discover Tools from Both Servers ---
                    print("🔎 Discovering tools from both servers via MCP...")

                    # Load tools from INVEST server
                    invest_tools = await load_mcp_tools(session_invest)
                    if not invest_tools:
                         print(f"⚠️ Warning: No tools found on INVEST server ({INVEST_SERVER_SCRIPT_PATH}). Check server logs.")
                    else:
                        print(f"   ✅ Found INVEST tools: {[tool.name for tool in invest_tools]}")

                    # Load tools from BPG server
                    bpg_tools = await load_mcp_tools(session_bpg)
                    if not bpg_tools:
                         print(f"⚠️ Warning: No tools found on BPG server ({BPG_SERVER_SCRIPT_PATH}). Check server logs.")
                    else:
                        print(f"   ✅ Found BPG tools: {[tool.name for tool in bpg_tools]}")

                    # Combine tools from both servers
                    all_tools = invest_tools + bpg_tools

                    if not all_tools:
                        print("❌ Error: No tools found on *either* server.")
                        print("   Cannot proceed without tools. Check server logs for errors.")
                        return # Exit if no tools are available at all

                    print(f"🛠️ Combined tools available for agent: {[tool.name for tool in all_tools]}")

                    # --- Create Agent ---
                    print("🤖 Creating ReAct agent with combined tools (using Anthropic)...")
                    agent_executor = create_react_agent(model, all_tools)
                    print("✅ Agent created.")

                    # --- Define the User Task ---
                    user_story = "As an online shopper, I want to add items to a shopping cart so that I can purchase multiple items at once."
                    prompt = f"""
                    Here is a user story:
                    "{user_story}"

                    Your task is to:
                    1. First, analyze this user story using the INVEST criteria. Show the full analysis.
                    2. Second, generate a Business Process Guide (BPG) document based *only* on the original user story provided above.
                    Provide the paths to the generated BPG files.
                    """
                    print(f"\n💬 Sending task to agent:\n--- START PROMPT ---\n{prompt}\n--- END PROMPT ---")

                    # --- Invoke Agent and Stream Output ---
                    print("\n⏳ Agent starting work (streaming output follows)...")
                    inputs = {"messages": [("user", prompt)]}

                    async for event in agent_executor.astream_events(inputs, version="v1"):
                        kind = event["event"]
                        if kind == "on_chat_model_stream":
                            content = event["data"]["chunk"].content
                            if content:
                                print(content, end="", flush=True)
                        elif kind == "on_tool_start":
                            print("\n" + "="*15 + f"🛠️ Calling Tool: {event['name']} " + "="*15)
                            tool_input = event['data'].get('input')
                            print("Tool Input:")
                            try:
                                # Attempt to format if it's valid JSON string
                                print(json.dumps(json.loads(tool_input), indent=2))
                            except:
                                print(tool_input) # Otherwise print raw
                            print("-"*(32 + len(event['name'])))
                        elif kind == "on_tool_end":
                            tool_output = event["data"].get("output")
                            print("\n" + "="*15 + f"✅ Tool Result ({event['name']}): " + "="*15)
                            try:
                                # Attempt to format if it's valid JSON string
                               print(json.dumps(json.loads(tool_output), indent=2))
                            except:
                               print(tool_output) # Otherwise print raw
                            print("="*(33 + len(event['name'])) + "\n")

                    print("\n\n🏁 Agent execution finished.")
                    print("   Review the streamed output above for the full process and final results.")

    except ConnectionRefusedError as e:
        # This might be harder to pinpoint which server failed with nested contexts
        print(f"\n❌ Connection Refused: Could not connect to one of the MCP servers.")
        print(f"   Ensure both '{INVEST_SERVER_SCRIPT_PATH}' and '{BPG_SERVER_SCRIPT_PATH}' scripts are running correctly in separate terminals.")
        print(f"   Check the console output of *both* servers for any startup errors.")
        print(f"   Specific error detail (may not be precise): {e}")
    except asyncio.TimeoutError:
         print("\n❌ Timeout Error: Connection to one of the servers timed out.")
         print(f"   Are both server scripts running and responsive?")
    except FileNotFoundError as e:
        # Specifically catch FileNotFoundError for the command
        if e.filename in ["python", "python3"]:
             print(f"\n❌ FileNotFoundError: Could not find the Python executable '{e.filename}'.")
             print(f"   Make sure '{e.filename}' is installed and in your system's PATH.")
             print(f"   Tried to run: {e.filename} {e.args}") # Show what it tried to run
        else:
             # Handle other potential FileNotFoundError during subprocess creation
             print(f"\n❌ FileNotFoundError: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during client execution: {e}")
        import traceback
        traceback.print_exc()

# --- Run the main async function ---
if __name__ == "__main__":
    print("Starting Asynchronous Client Application (Separate Servers)...")
    try:
        asyncio.run(run_agent_with_separate_mcp_tools())
    except KeyboardInterrupt:
        print("\n🛑 Client execution cancelled by user.")
    print("Client Application Finished.")