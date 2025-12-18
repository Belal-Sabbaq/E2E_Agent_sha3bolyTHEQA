import json
import openai
import time

class LLMBrain:
    def __init__(self, model="qwen3:4b", mode="Local (Ollama)", api_key=None):
        self.model = model
        self.mode = mode
        self.api_key = api_key
        self.metrics = [] 
        if self.mode == "API (OpenAI)":
            openai.api_key = self.api_key

    import time
from openai import OpenAI

class LLMBrain:
    def __init__(self, model="gpt-4o-mini", mode="Local (Ollama)", api_key=None):
        self.model = model
        self.mode = mode
        self.api_key = api_key
        self.metrics = []

        if self.mode == "API (OpenAI)":
            if not api_key:
                raise ValueError("OpenAI API key is required for API mode")
            self.client = OpenAI(api_key=api_key)

    def chat(self, system_prompt, user_prompt, response_format="text", temperature=0.1):
        start_time = time.time()
        tokens_used = 0
        model_used = self.model

        try:
            # ===============================
            # LOCAL OLLAMA
            # ===============================
            if self.mode == "Local (Ollama)":
                import ollama
                print("🔧 Sending request to Ollama...")
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    format="json" if response_format == "json" else None,
                    options={"temperature": temperature}
                )
                content = response["message"]["content"]

            # ===============================
            # OPENAI API (NEW SDK)
            # ===============================
            elif self.mode == "API (OpenAI)":
                print("🌐 Sending request to OpenAI API...")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature
                )

                content = response.choices[0].message.content
                tokens_used = response.usage.total_tokens

            else:
                raise ValueError("Invalid LLM mode")

            elapsed = time.time() - start_time

            # ===============================
            # METRICS
            # ===============================
            self.metrics.append({
                "model": model_used,
                "mode": self.mode,
                "time_sec": round(elapsed, 3),
                "tokens": tokens_used
            })

            return content

        except Exception as e:
            elapsed = time.time() - start_time
            self.metrics.append({
                "model": model_used,
                "mode": self.mode,
                "time_sec": round(elapsed, 3),
                "tokens": 0,
                "error": str(e)
            })
            raise e



    # --------------------------------------------------
    # TEST PLAN GENERATION
    # --------------------------------------------------
    def generate_test_plan(self, scraped_data, memory_context=None):
        elements = scraped_data.get("elements", [])
        clean_dom = scraped_data.get("cleaned_dom", "")[:10000]

        memory_prompt = ""
        if memory_context:
            if memory_context.get("avoid"):
                memory_prompt += "\n\n🚫 THINGS TO AVOID:\n" + "\n".join(memory_context["avoid"])
            if memory_context.get("emulate"):
                memory_prompt += "\n\n✅ PATTERNS TO FOLLOW:\n" + "\n".join(memory_context["emulate"])

        system_prompt = f"""
You are an expert QA Automation Engineer.

Generate 3–5 Playwright test cases.

STRICT OUTPUT RULES:
- Return ONLY a valid JSON LIST
- No markdown
- No explanations
- No wrapping keys

FORMAT:
[
  {{
    "name": "...",
    "description": "...",
    "missing_data": [],
    "requires_auth": false
  }}
]

{memory_prompt}
"""

        user_prompt = f"""
Page Title: {scraped_data.get('title')}
Elements: {json.dumps(elements[:50])}
DOM: {clean_dom}
"""

        print("🧠 Brain is thinking...")

        try:
            content = self.chat(system_prompt, user_prompt, response_format="json")

            if not content or not isinstance(content, str):
                raise ValueError("Empty LLM response")

            if "```" in content:
                content = content.replace("```json", "").replace("```", "").strip()

            parsed_json = json.loads(content)

            if not isinstance(parsed_json, list):
                raise ValueError("LLM did not return a JSON list")

            safe_plan = []
            for item in parsed_json:
                if isinstance(item, dict):
                    safe_plan.append({
                        "name": item.get("name", "Unnamed Test"),
                        "description": item.get("description", ""),
                        "missing_data": item.get("missing_data", []),
                        "requires_auth": item.get("requires_auth", False)
                    })

            return safe_plan

        except Exception as e:
            print("❌ LLM ERROR:", e)
            return [{
                "name": "Error Generating Plan",
                "description": str(e),
                "missing_data": [],
                "is_error": True
            }]

    # --------------------------------------------------
    # CODE GENERATION
    # --------------------------------------------------
    def generate_playwright_code(self, test_case, scraped_data):
        elements = scraped_data.get("elements", [])
        user_data = test_case.get("user_data", {})

        system_prompt = """
You are an expert Playwright Automation Engineer (Python).

Write a complete standalone script.

RULES:
- Use: from playwright.sync_api import sync_playwright
- Use resilient locators
- Save storage state to auth.json
"""

        user_prompt = f"""
Test Name: {test_case['name']}
Description: {test_case['description']}
URL: {scraped_data.get('url')}
Elements: {json.dumps(elements[:50])}
USER_DATA: {json.dumps(user_data)}
"""

        try:
            code = self.chat(system_prompt, user_prompt)
            if "```" in code:
                code = code.replace("```python", "").replace("```", "")
            return code.strip()
        except Exception as e:
            return f"# Error generating code: {e}"

    # --------------------------------------------------
    # SELF-HEALING
    # --------------------------------------------------
    def fix_generated_code(self, broken_code, error_log):
        system_prompt = """
You are a Python Playwright debugger.
Fix the script based on the error trace.
Return ONLY Python code.
"""
        user_prompt = f"""
BROKEN CODE:
{broken_code}

ERROR TRACE:
{error_log}
"""
        try:
            code = self.chat(system_prompt, user_prompt)
            if "```" in code:
                code = code.replace("```python", "").replace("```", "")
            return code.strip()
        except Exception as e:
            return f"# Fix failed: {e}"

    # --------------------------------------------------
    # PLAN REFINEMENT
    # --------------------------------------------------
    def refine_test_plan(self, current_plan, user_feedback, scraped_data):
        elements = scraped_data.get("elements", [])

        system_prompt = """
Modify the existing test plan based on user feedback.
Return ONLY a JSON list.
"""

        user_prompt = f"""
Current Plan: {json.dumps(current_plan)}
Feedback: {user_feedback}
Elements: {json.dumps(elements[:50])}
"""

        try:
            content = self.chat(system_prompt, user_prompt, response_format="json", temperature=0.2)
            parsed = json.loads(content)
            return parsed if isinstance(parsed, list) else []
        except Exception as e:
            return [{"name": "Error Refining Plan", "description": str(e), "is_error": True}]
    # --------------------------------------------------
    # METRICS / OBSERVABILITY
    # --------------------------------------------------

    def get_metrics_summary(self):
        if not self.metrics:
            return {
                "total_calls": 0,
                "total_tokens": 0,
                "average_time": 0,
                "errors": 0,
                "per_model": {}
            }

        total_time = sum(m["time"] for m in self.metrics)
        total_tokens = sum(m.get("tokens", 0) for m in self.metrics)
        errors = sum(1 for m in self.metrics if "error" in m)

        model_breakdown = {}
        for m in self.metrics:
            key = f"{m['mode']} / {m['model']}"
            if key not in model_breakdown:
                model_breakdown[key] = {"count": 0, "total_time": 0, "total_tokens": 0, "errors": 0}
            model_breakdown[key]["count"] += 1
            model_breakdown[key]["total_time"] += m["time"]
            model_breakdown[key]["total_tokens"] += m.get("tokens", 0)
            if "error" in m:
                model_breakdown[key]["errors"] += 1

        return {
            "total_calls": len(self.metrics),
            "total_tokens": total_tokens,
            "average_time": total_time / len(self.metrics),
            "errors": errors,
            "per_model": model_breakdown
        }

    def reset_metrics(self):
        self.metrics.clear()

