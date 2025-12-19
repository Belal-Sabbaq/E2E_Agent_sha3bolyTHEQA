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

    def _extract_json_object(self, text: str) -> str:
        """
        Extract first JSON object from text.
        """
        match = re.search(r"\{\s*\".*\"\s*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No valid JSON object found in LLM output")
        return match.group(0)

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
            content = content.replace("```json", "").replace("```", "").strip()

            json_text = self._extract_json_array(content)
            parsed_json = json.loads(json_text)

            safe_plan = []
            seen = set()
            for item in parsed_json:
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
    # -------------------------------------------------
    def generate_playwright_code(self, test_case, scraped_data):
        """
        Template-based codegen with retry-safe JSON spec generation.
        """
        url = scraped_data.get("url") or scraped_data.get("final_url") or ""
        elements = scraped_data.get("elements", [])
        user_data = test_case.get("user_data", {})

        test_name = test_case.get("name", "unnamed_test").strip()
        safe_test_name = (
            test_name.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("'", "")
            .replace('"', "")
        )

        # --------------------------------------------------
        # JSON SPEC PROMPTS
        # --------------------------------------------------
        base_system_prompt = """
    You are a strict JSON generator.

    Output ONLY a valid JSON object.
    No explanations, no markdown, no comments.

    SCHEMA (MUST FOLLOW EXACTLY):
    {
    "selector": "<string or empty>",
    "assertion": {
        "type": "visible | has_text | title_is",
        "value": "<string or empty>"
    }
    }

    RULES:
    - Use double quotes only
    - No trailing commas
    - Start with '{' and end with '}'
    - If test is about page title:
    selector = ""
    assertion.type = "title_is"
    """

        retry_system_prompt = """
    You MUST output valid JSON.

    DO NOT explain.
    DO NOT add text.
    DO NOT format.

    ONLY THIS JSON OBJECT:
    {
    "selector": "",
    "assertion": {
        "type": "title_is",
        "value": ""
    }
    }

    Replace values correctly based on the test.
    """

        user_prompt = f"""
    TEST_NAME: {test_name}
    DESCRIPTION: {test_case.get("description", "")}
    URL: {url}

    KNOWN_ELEMENTS:
    {json.dumps(elements[:50])}

    USER_DATA:
    {json.dumps(user_data)}
    """

        spec = None
        raw_output = ""

        # --------------------------------------------------
        # Attempt 1
        # --------------------------------------------------
        try:
            raw_output = self.chat(
                base_system_prompt,
                user_prompt,
                temperature=0
            ).strip()

            raw_output = raw_output.replace("```json", "").replace("```", "").strip()
            spec_json_text = self._extract_json_object(raw_output)
            spec = json.loads(spec_json_text)

        except Exception:
            # --------------------------------------------------
            # Attempt 2 (retry with extreme constraints)
            # --------------------------------------------------
            try:
                raw_output = self.chat(
                    retry_system_prompt,
                    user_prompt,
                    temperature=0
                ).strip()

                raw_output = raw_output.replace("```json", "").replace("```", "").strip()
                spec_json_text = self._extract_json_object(raw_output)
                spec = json.loads(spec_json_text)

            except Exception as e:
                return (
                    "# GENERATED CODE REJECTED\n"
                    "# Reason: could not build valid spec JSON after retry\n"
                    f"# Raw LLM output:\n# {raw_output}\n"
                    f"# Error: {e}\n"
                )

        # --------------------------------------------------
        # Validate spec
        # --------------------------------------------------
        selector = (spec.get("selector") or "").strip()
        assertion = spec.get("assertion") or {}
        assertion_type = assertion.get("type", "visible")
        assertion_value = (assertion.get("value") or "").strip()

        if not url:
            return "# GENERATED CODE REJECTED\n# Reason: missing URL\n"

        if assertion_type not in {"visible", "has_text", "title_is"}:
            assertion_type = "visible"

        if assertion_type != "title_is" and not selector:
            return (
                "# GENERATED CODE REJECTED\n"
                "# Reason: selector missing for non-title assertion\n"
            )

        # --------------------------------------------------
        # Render FIXED template (unchanged)
        # --------------------------------------------------
        template = f'''\
    import os
    import json
    from datetime import datetime
    from playwright.sync_api import sync_playwright

    TEST_NAME = "{safe_test_name}"
    URL = {json.dumps(url)}

    def save_step(page, step_name):
        base = os.path.join("artifacts", TEST_NAME)
        os.makedirs(base, exist_ok=True)
        path = os.path.join(base, step_name + ".png")
        page.screenshot(path=path, full_page=True)

    def write_result(status, error_msg=None):
        base = os.path.join("artifacts", TEST_NAME)
        os.makedirs(base, exist_ok=True)
        result = {{
            "test_name": TEST_NAME,
            "status": status,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "error": error_msg
        }}
        with open(os.path.join(base, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    def main():
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()

            try:
                save_step(page, "step_01_before_goto")
                page.goto(URL, wait_until="domcontentloaded")
                save_step(page, "step_02_after_goto")

                assertion_type = {json.dumps(assertion_type)}
                assertion_value = {json.dumps(assertion_value)}
                selector = {json.dumps(selector)}

                if assertion_type == "title_is":
                    title = page.title()
                    save_step(page, "step_03_title_captured")
                    if title != assertion_value:
                        save_step(page, "step_error")
                        raise AssertionError(
                            f"Title mismatch: got '{{title}}', expected '{{assertion_value}}'"
                        )
                    save_step(page, "step_04_assert_pass")
                else:
                    loc = page.locator(selector)
                    loc.wait_for(state="visible", timeout=5000)
                    save_step(page, "step_03_element_visible")

                    if assertion_type == "has_text":
                        txt = loc.first.inner_text().strip()
                        save_step(page, "step_04_text_captured")
                        if assertion_value not in txt:
                            save_step(page, "step_error")
                            raise AssertionError(
                                f"Text mismatch: got '{{txt}}', expected to contain '{{assertion_value}}'"
                            )
                        save_step(page, "step_05_assert_pass")
                    else:
                        if not loc.first.is_visible():
                            save_step(page, "step_error")
                            raise AssertionError("Element not visible")
                        save_step(page, "step_04_assert_pass")

                write_result("PASS")
            except Exception as e:
                write_result("FAIL", str(e))
                raise
            finally:
                browser.close()

    if __name__ == "__main__":
        main()
    '''

        try:
            compile(template, "<template_playwright_test>", "exec")
        except SyntaxError as e:
            return (
                "# GENERATED CODE REJECTED\n"
                f"# Reason: template compile failed ({e})\n"
            )

        return template

    