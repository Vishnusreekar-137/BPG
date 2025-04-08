import os
import json
import asyncio
import threading
from typing import Dict, Any, Optional, List
from langchain_groq import ChatGroq
from dotenv import load_dotenv
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
logger = logging.getLogger("MCP-Script")

# Load environment variables
load_dotenv()

# Initialize MCP server (not run directly, used for tool registration)
mcp = FastMCP("userstory-invest-mcp")

# Validate environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")

if not GROQ_API_KEY or not GROQ_MODEL:
    raise ValueError("GROQ_API_KEY or GROQ_MODEL environment variables not set")

# Initialize FastAPI app
app = FastAPI(title="User Story Evaluator")

# Log available routes for debugging
@app.on_event("startup")
async def startup_event():
    routes = [route.path for route in app.routes]
    logger.info(f"Registered routes: {routes}")

# Pydantic model for user story input
class UserStoryInput(BaseModel):
    title: str
    description: str
    acceptance_criteria: List[str]
    additional_information: Optional[str] = None

# Core evaluation function
def evaluate_story(user_story_dict: Dict) -> Dict:
    llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.2)
    
    prompt = f"""
    Evaluate the following user story based on the INVEST principles using the detailed criteria provided:
    
    USER STORY:
    Title: {user_story_dict.get('title', 'No title provided')}
    Description: {user_story_dict.get('description', 'No description provided')}
    Acceptance Criteria: {user_story_dict.get('acceptance_criteria', 'No criteria provided')}
    Additional Information: {user_story_dict.get('additional_information', 'No additional info')}
    
    INVEST PRINCIPLES CRITERIA:
    Independent:
    - The story should be self-contained
    - It should not have inherent dependencies on other stories
    - It can be developed and delivered separately from other stories
    - It avoids "this and that" formulations that combine multiple features

    Negotiable:
    - The story is not an explicit contract for features
    - Details should be co-created by the customer and development team
    - It leaves room for conversation and refinement
    - It avoids overly prescriptive implementation details

    Valuable:
    - The story delivers value to stakeholders
    - It clearly describes a benefit, not just a feature
    - It explains WHY the feature is needed
    - A real user or customer would care about this story

    Estimable:
    - The team can estimate how much effort the story will take
    - It has enough detail for estimation without being over-specified
    - It doesn't contain unknowns that prevent reasonable estimation
    - It's clear enough that developers understand the scope

    Small:
    - The story is small enough to be completed in a single sprint
    - It represents a vertical slice of functionality
    - It's focused on a single capability or feature
    - It can be completed by a single developer in a few days

    Testable:
    - The story includes clear acceptance criteria
    - Success can be verified objectively
    - It’s possible to write automated tests for the story
    - The team understands what "done" looks like
    
    OUTPUT GUIDELINES:
    - For each principle, provide a meaningful analysis based on the criteria
    - Your explanations should be concise but substantive, addressing key points from the criteria
    - Recommendations should be specific and actionable for the team
    - The summary should provide a clear assessment of the overall quality with appropriate rationale
    
    For each principle, provide:
    1. A score out of 5 (where 1 is poor and 5 is excellent)
    2. A concise but substantive explanation for the score, referencing the specific criteria
    3. A specific, actionable recommendation for improvement
    
    Then provide:
    - The overall score out of 30 (sum of all individual scores)
    - A summary statement in this format: "Current INVEST Score: [X]/30 ([Poor/Fair/Good/Excellent]). [Assessment with rationale]"
    - 2-3 key prioritized recommendations focused on the weakest areas
    
    Response format:
    {{
      "Independent": {{
        "score": X,
        "explanation": "Your concise explanation referencing the criteria...",
        "recommendation": "Your specific, actionable recommendation..."
      }},
      "Negotiable": {{ ... }},
      "Valuable": {{ ... }},
      "Estimable": {{ ... }},
      "Small": {{ ... }},
      "Testable": {{ ... }},
      "overall": {{
        "score": X,
        "summary": "Current INVEST Score: X/30 (Rating). Assessment with rationale.",
        "key_recommendations": ["Prioritized rec 1", "Prioritized rec 2", "Prioritized rec 3"]
      }}
    }}
    
    Provide ONLY the JSON response, no additional text.
    """
    
    messages = [
        SystemMessage(content="""You are an experienced agile coach who evaluates user stories against INVEST principles.
You provide balanced, fair evaluations based on specific criteria, not personal opinions.
Your explanations reference the specific criteria and your recommendations are actionable.
You ensure the overall score accurately reflects the sum of individual principle scores.
You always respond in valid JSON format with no additional text."""),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    response_text = response.content
    
    if "```json" in response_text:
        json_str = response_text.split("```json")[1].split("```")[0].strip()
    else:
        json_str = response_text.strip()
        
    evaluation = json.loads(json_str)
    
    principles = ["Independent", "Negotiable", "Valuable", "Estimable", "Small", "Testable"]
    calculated_score = sum(evaluation[p]["score"] for p in principles)
    
    if evaluation["overall"]["score"] != calculated_score:
        evaluation["overall"]["score"] = calculated_score
        current_summary = evaluation["overall"]["summary"]
        score_pattern = r"Current INVEST Score: \d+/30"
        updated_summary = re.sub(score_pattern, f"Current INVEST Score: {calculated_score}/30", current_summary)
        evaluation["overall"]["summary"] = updated_summary
    
    return {"input": user_story_dict, "evaluation": evaluation}

# MCP Server Tool
@mcp.tool(description="Evaluates user stories against INVEST principles.")
async def evaluate_user_story_mcp(user_story: Dict[str, Any], format_output: Optional[bool] = False) -> Dict[str, Any]:
    logger.debug(f"Server processing user story: {user_story}")
    result = evaluate_story(user_story)
    logger.debug(f"Server generated result: {result}")
    return result

# FastAPI endpoint with Pydantic model
@app.post("/evaluate_story")
async def evaluate_story_endpoint(user_story: UserStoryInput = Body(...)):
    try:
        logger.debug(f"Received request body at endpoint: {user_story.dict()}")
        user_story_dict = user_story.dict()
        
        # Simulate MCP request
        mcp_request = {
            "tool": "evaluate_user_story_mcp",
            "params": {"user_story": user_story_dict, "format_output": False}
        }
        
        # Process using MCP tool
        response = await evaluate_user_story_mcp(**mcp_request["params"])
        logger.debug(f"Endpoint response: {response}")
        return response
    except json.JSONDecodeError:
        logger.error("Invalid JSON in request body")
        return {"error": "Invalid JSON in request body"}
    except Exception as e:
        logger.error(f"Endpoint error: {str(e)}")
        return {"error": str(e)}

# Function to run the FastAPI server in a separate thread
def run_server():
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="info")
    server = uvicorn.Server(config)
    server.run()

# Client runner with multi-line input for acceptance criteria
async def run_client():
    logger.info("Running client...")
    print("Enter your user story details:")

    title = input("Title: ").strip()
    description = input("Description: ").strip()
    
    print("Enter acceptance criteria (one per line, press Enter twice to finish):")
    acceptance_criteria = []
    while True:
        line = input().strip()
        if line == "":
            if acceptance_criteria:  # Ensure at least one criterion is entered before finishing
                break
            else:
                print("Please enter at least one acceptance criterion.")
                continue
        acceptance_criteria.append(line)
    
    additional_information = input("Additional Information (optional, press Enter to skip): ").strip() or None
    
    user_story = {
        "title": title,
        "description": description,
        "acceptance_criteria": acceptance_criteria,
        "additional_information": additional_information
    }
    
    logger.debug(f"Constructed user story: {user_story}")
    print("\n📤 Your User Story:")
    print(json.dumps(user_story, indent=2))
    
    try:
        async with httpx.AsyncClient() as client:
            # Wait for server to start
            await asyncio.sleep(2)
            url = "http://127.0.0.1:8000/evaluate_story"
            logger.debug(f"Sending request to: {url}")
            response = await client.post(url, json=user_story)
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Client received response: {result}")
            print("Evaluation Result:")
            print(json.dumps(result, indent=2))
    except json.JSONDecodeError:
        print("Error: Invalid JSON input")
        logger.error(f"Invalid JSON input: {user_story}")
    except httpx.RequestError as e:
        print(f"Error connecting to server: {str(e)}. Is the server running?")
        logger.error(f"Connection error: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")
        logger.error(f"Client error: {str(e)}")

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