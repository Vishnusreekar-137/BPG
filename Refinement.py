import os
import re
import json
import traceback
from dotenv import load_dotenv
from langchain_together import ChatTogether
from langchain.schema import SystemMessage, HumanMessage
from nofast import generate_and_save_bpg  # Use your updated BPG script/module

load_dotenv()

class UserStoryInvestAnalyzer:
    def __init__(self):
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY not set in .env file")
        self.chat = ChatTogether(
            model="meta-llama/Llama-3-70b-chat-hf",
            together_api_key=api_key,
            temperature=0.2,
        )

    def analyze_user_story(self, user_story):
        try:
            if not isinstance(user_story, str):
                user_story = json.dumps(user_story)

            prompt = self.create_prompt(user_story)
            response = self.chat.invoke(prompt)
            content = response.content.strip()
            clean_json = self.sanitize_json(content)
            result = json.loads(clean_json)
            self.validate(result)
            return result
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            traceback.print_exc()
            return self.fallback_result()

    def create_prompt(self, user_story):
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

    def sanitize_json(self, text):
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group()
        raise ValueError("No valid JSON block found in response")

    def validate(self, result):
        required = ["OriginalUserStory", "ImprovedUserStory", "Independent", "Negotiable",
                    "Valuable", "Estimable", "Small", "Testable", "overall"]
        for field in required:
            if field not in result:
                raise ValueError(f"Missing field: {field}")

    def fallback_result(self):
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

def preprocess_input(user_input):
    try:
        return json.loads(user_input)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON input: {e}")

if __name__ == "__main__":
    print("📥 Paste your UserStory JSON:")
    user_input = input()
    analyzer = UserStoryInvestAnalyzer()

    try:
        user_story = preprocess_input(user_input)
        result = analyzer.analyze_user_story(user_story)

        print("✅ INVEST Analysis Complete:")
        print(json.dumps(result, indent=2))

        with open("invest_analysis.json", "w") as f:
            f.write(json.dumps(result, indent=2))

        # Trigger BPG generation
        print("\n⚙️ Generating Business Process Guide from ImprovedUserStory...")
        improved_json = json.dumps({"UserStory": result["ImprovedUserStory"]})
        pdf_path, word_path = generate_and_save_bpg(improved_json)
        print(f"📄 PDF saved: {pdf_path}")
        print(f"📝 Word saved: {word_path}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()
