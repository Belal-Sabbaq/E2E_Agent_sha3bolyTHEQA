import json
import time
import requests
import re


class LLMBrain:
    # --------------------------------------------------
    # INIT
    # --------------------------------------------------
    def __init__(self, model="llama-3.1-8b-instant", mode="Local (Ollama)", api_key=None):
        self.model = model
        self.mode = mode
        self.api_key = api_key
        self.metrics = []

        if self.mode == "API (Groq)" and not api_key:
            raise ValueError("Groq API key is required for API (Groq) mode")

    # --------------------------------------------------
    # INTERNAL: SAFE JSON EXTRACTION
    # --------------------------------------------------
    def _extract_json_array(self, text: str) -> str:
        """
        Extract the first JSON array from LLM output.
        Guards against explanations or extra text.
        """
        match = re.search(r"\[\s*{.*}\s*\]", text, re.DOTALL)
        if not match:
            raise ValueError("No valid JSON array found in LLM output")
        return match.group(0)

    # --------------------------------------------------
    # CORE CHAT METHOD
    # --------------------------------------------------
    def chat(self, system_prompt, user_prompt, response_format="text", temperature=0.1):
        start_time = time.time()
        tokens_used = 0
        model_used = self.model

        print("🔥 FINAL MODEL SENT TO LLM:", repr(self.model))

        try:
            # ===============================
            # LOCAL (OLLAMA)
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
                    options={"temperature": float(temperature)}
                )
                content = response["message"]["content"]

            # ===============================
            # GROQ API
            # ===============================
            elif self.mode == "API (Groq)":
                print("🌐 Sending request to Groq API...")
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": float(temperature),
                    "max_tokens": 1024
                }

                response = requests.post(url, headers=headers, json=payload)

                if response.status_code != 200:
                    print("❌ STATUS:", response.status_code)
                    print("❌ BODY:", response.text)
                    raise RuntimeError("Groq request failed")

                data = response.json()
                content = data["choices"][0]["message"]["content"]
                tokens_used = data.get("usage", {}).get("total_tokens", 0)

            else:
                raise ValueError(f"Unsupported mode: {self.mode}")

            # ===============================
            # METRICS (SUCCESS)
            # ===============================
            elapsed = time.time() - start_time
            self.metrics.append({
                "model": model_used,
                "mode": self.mode,
                "time": elapsed,
                "tokens": tokens_used
            })

            return content

        except Exception as e:
            elapsed = time.time() - start_time
            self.metrics.append({
                "model": model_used,
                "mode": self.mode,
                "time": elapsed,
                "tokens": 0,
                "error": str(e)
            })
            print("❌ LLM ERROR:", e)
            raise

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
- Return ONLY a valid JSON array
- The response MUST start with '[' and end with ']'
- No markdown
- No explanations
- No trailing commas

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

        print("🧠 Generating test plan...")

        try:
            content = self.chat(system_prompt, user_prompt)

            if not content or not isinstance(content, str):
                raise ValueError("Empty LLM response")

            content = content.replace("```json", "").replace("```", "").strip()

            json_text = self._extract_json_array(content)
            parsed_json = json.loads(json_text)

            if not isinstance(parsed_json, list):
                raise ValueError("LLM did not return a JSON list")

            safe_plan = []
            seen = set()

            for item in parsed_json:
                if isinstance(item, dict):
                    name = item.get("name", "").strip()
                    if name and name not in seen:
                        seen.add(name)
                        safe_plan.append({
                            "name": name,
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
    # PLAYWRIGHT CODE GENERATION
    # --------------------------------------------------
    def generate_playwright_code(self, test_case, scraped_data):
        elements = scraped_data.get("elements", [])
        user_data = test_case.get("user_data", {})

        system_prompt = """
You are an expert Playwright Automation Engineer (Python).

CRITICAL RULES:
- Output ONLY valid Python code
- No explanations
- No markdown
- No backticks
- Code must be directly executable

TECHNICAL REQUIREMENTS:
- Use: from playwright.sync_api import sync_playwright
- Define a main() function
- Launch Chromium
- Open the provided URL
- Perform the test
- Close the browser properly
"""

        user_prompt = f"""
TEST_NAME: {test_case['name']}
DESCRIPTION: {test_case['description']}
URL: {scraped_data.get('url')}
ELEMENTS: {json.dumps(elements[:50])}
USER_DATA: {json.dumps(user_data)}
"""

        try:
            code = self.chat(system_prompt, user_prompt)

            code = code.replace("```python", "").replace("```", "").strip()

            # HARD PYTHON VALIDATION
            compile(code, "<generated_playwright_test>", "exec")

            return code

        except Exception as e:
            return (
                "# GENERATED CODE REJECTED\n"
                f"# Reason: {e}\n"
            )

    # --------------------------------------------------
    # METRICS SUMMARY
    # --------------------------------------------------
    def get_metrics_summary(self):
        if not self.metrics:
            return {
                "total_calls": 0,
                "total_tokens": 0,
                "average_time": 0.0,
                "errors": 0,
                "per_model": {}
            }

        total_time = sum(m["time"] for m in self.metrics)
        total_tokens = sum(m.get("tokens", 0) for m in self.metrics)
        errors = sum(1 for m in self.metrics if "error" in m)

        per_model = {}
        for m in self.metrics:
            key = f"{m['mode']} / {m['model']}"
            stats = per_model.setdefault(
                key, {"count": 0, "total_time": 0.0, "total_tokens": 0, "errors": 0}
            )
            stats["count"] += 1
            stats["total_time"] += m["time"]
            stats["total_tokens"] += m.get("tokens", 0)
            if "error" in m:
                stats["errors"] += 1

        return {
            "total_calls": len(self.metrics),
            "total_tokens": total_tokens,
            "average_time": total_time / len(self.metrics),
            "errors": errors,
            "per_model": per_model
        }

    # --------------------------------------------------
    # RESET METRICS
    # --------------------------------------------------
    def reset_metrics(self):
        self.metrics.clear()
