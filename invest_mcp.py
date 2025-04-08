import os
import json
import asyncio
import threading
import re
import traceback
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langchain_together import ChatTogether
from langchain_core.messages import SystemMessage, HumanMessage
from mcp.server.fastmcp import FastMCP
from fastapi import FastAPI, Request
from fastapi import Body
import uvicorn
import logging
import httpx
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("InvestAnalyzer")

# Load environment variables
load_dotenv()

# Initialize MCP server (not run directly, used for tool registration)
mcp = FastMCP("userstory-invest-analyzer")

# Validate environment variables
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY environment variable not set")

# Initialize FastAPI app
app = FastAPI(title="User Story INVEST Analyzer")

# Pydantic model for user story input
class UserStoryInput(BaseModel):
    Title: str
    Description: str
    AcceptanceCriteria: list
    AdditionalInformation: Optional[str] = None

class UserStoryInvestAnalyzer:
    def __init__(self):
        self.chat = ChatTogether(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            together_api_key=TOGETHER_API_KEY,
            temperature=0.2,
        )

    def analyze_user_story(self, user_story: Dict[str, Any]) -> Dict[str, Any]:
        try:
            user_story_str = json.dumps(user_story)
            prompt = self.create_prompt(user_story_str)
            response = self.chat.invoke([SystemMessage(content="You are an expert agile coach."), HumanMessage(content=prompt)])
            content = response.content.strip()
            clean_json = self.sanitize_json(content)
            result = json.loads(clean_json)
            self.validate(result)
            logger.debug(f"Analysis result: {result}")
            return result
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            traceback.print_exc()
            return self.fallback_result()

    def create_prompt(self, user_story: str) -> str:
        return f"""
You are an expert agile coach specializing in analyzing user stories using the INVEST criteria.

# Task
1. Analyze the original user story and calculate its INVEST score (Independent, Negotiable, Valuable, Estimable, Small, Testable — 1 to 5).
2. Suggest an improved version.
3. Explain what changed and why.

# Input User Story
{user_story}

# Output JSON format (strict):
{{
  "OriginalUserStory": {{
    "Title": "...",
    "Description": "...",
    "AcceptanceCriteria": ["...", "..."],
    "AdditionalInformation": "..."
  }},
  "ImprovedUserStory": {{
    "Title": "...",
    "Description": "...",
    "AcceptanceCriteria": ["...", "..."],
    "AdditionalInformation": "..."
  }},
  "Independent": {{ "score": 1, "explanation": "...", "recommendation": "..." }},
  "Negotiable": {{ "score": 1, "explanation": "...", "recommendation": "..." }},
  "Valuable": {{ "score": 1, "explanation": "...", "recommendation": "..." }},
  "Estimable": {{ "score": 1, "explanation": "...", "recommendation": "..." }},
  "Small":      {{ "score": 1, "explanation": "...", "recommendation": "..." }},
  "Testable":   {{ "score": 1, "explanation": "...", "recommendation": "..." }},
  "overall": {{
    "score": 12,
    "improved_score": 26,
    "summary": "...",
    "refinement_summary": "* item1\\n* item2\\n* item3\\nINVEST Score improved from X/30 to Y/30"
  }}
}}

Only output the JSON, no extra text.
"""

    def sanitize_json(self, text: str) -> str:
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group()
        raise ValueError("No valid JSON block found in response")

    def validate(self, result: Dict[str, Any]):
        required = ["OriginalUserStory", "ImprovedUserStory", "Independent", "Negotiable",
                    "Valuable", "Estimable", "Small", "Testable", "overall"]
        for field in required:
            if field not in result:
                raise ValueError(f"Missing field: {field}")

    def fallback_result(self) -> Dict[str, Any]:
        return {
            "OriginalUserStory": {"Title": "", "Description": "", "AcceptanceCriteria": [], "AdditionalInformation": ""},
            "ImprovedUserStory": {"Title": "", "Description": "", "AcceptanceCriteria": [], "AdditionalInformation": ""},
            "Independent": {"score": 0, "explanation": "", "recommendation": ""},
            "Negotiable": {"score": 0, "explanation": "", "recommendation": ""},
            "Valuable": {"score": 0, "explanation": "", "recommendation": ""},
            "Estimable": {"score": 0, "explanation": "", "recommendation": ""},
            "Small": {"score": 0, "explanation": "", "recommendation": ""},
            "Testable": {"score": 0, "explanation": "", "recommendation": ""},
            "overall": {"score": 0, "improved_score": 0, "summary": "", "refinement_summary": ""}
        }

# MCP Server Tool
@mcp.tool(description="Analyzes user stories against INVEST criteria and suggests improvements.")
async def analyze_user_story_mcp(user_story: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug(f"Processing user story via MCP: {user_story}")
    analyzer = UserStoryInvestAnalyzer()
    result = analyzer.analyze_user_story(user_story)
    return {"input": user_story, "evaluation": result}

# FastAPI Endpoint with Pydantic model
@app.post("/analyze_story")
async def analyze_story_endpoint(user_story: UserStoryInput = Body(...)):
    logger.debug(f"Received request at endpoint: {user_story.dict()}")
    
    # Convert Pydantic model to dict for MCP processing
    user_story_dict = user_story.dict()
    
    # Simulate MCP request
    mcp_request = {
        "tool": "analyze_user_story_mcp",
        "params": {"user_story": user_story_dict}
    }
    
    # Process using MCP tool
    response = await analyze_user_story_mcp(**mcp_request["params"])
    logger.debug(f"Endpoint response: {response}")
    return response

# Function to run the FastAPI server in a separate thread
def run_server():
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="info")
    server = uvicorn.Server(config)
    server.run()

# Client runner
async def run_client():
    logger.info("Running client...")
    print("📥 Paste your User Story JSON (e.g., {\"Title\": \"...\", \"Description\": \"...\", ...}):")
    user_input = input()
    
    try:
        user_story = json.loads(user_input)
        async with httpx.AsyncClient() as client:
            # Wait for server to start
            await asyncio.sleep(2)
            response = await client.post("http://127.0.0.1:8000/analyze_story", json=user_story)
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Client received response: {result}")
            print("✅ INVEST Analysis Result:")
            print(json.dumps(result, indent=2))

            # Optional BPG generation (uncomment if you have the nofast module)
            # print("\n⚙️ Generating Business Process Guide from ImprovedUserStory...")
            # improved_json = json.dumps({"UserStory": result["evaluation"]["ImprovedUserStory"]})
            # pdf_path, word_path = generate_and_save_bpg(improved_json)
            # print(f"📄 PDF saved: {pdf_path}")
            # print(f"📝 Word saved: {word_path}")

    except json.JSONDecodeError:
        print("❌ Error: Invalid JSON input")
    except httpx.RequestError as e:
        print(f"❌ Error connecting to server: {str(e)}. Is the server running?")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()

# Main function to run server and client
async def main():
    # Start FastAPI server in a separate thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    logger.info("Starting FastAPI server on http://127.0.0.1:8000 in a separate thread")
    server_thread.start()
    
    # Run client
    await run_client()

if __name__ == "__main__":
    # Ensure Windows compatibility
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())