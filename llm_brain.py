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
<<<<<<< HEAD
            content = self.chat(system_prompt, user_prompt)
            content = content.replace("```json", "").replace("```", "").strip()

            json_text = self._extract_json_array(content)
            parsed_json = json.loads(json_text)

=======
            response = ollama.chat(
                model=self.model, 
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                format='json',
                options={'temperature': 0.1}
            )
            
            content = response['message']['content']
            print(f"DEBUG: Raw LLM Output: {content}") # Keep this for debugging
            
            parsed_json = json.loads(content)
            
            # --- FIX: ROBUST LIST EXTRACTION ---
            final_plan = []
            
            # Case 1: It's already a list (Perfect)
            if isinstance(parsed_json, list):
                final_plan = parsed_json
                
# Case 2: It's a dict (single test object or wrapped response)
            elif isinstance(parsed_json, dict):
                # If it looks like a single test object, wrap it
                if parsed_json.get("name") or parsed_json.get("description"):
                    final_plan = [parsed_json]
                    print("DEBUG: Parsed single test object; wrapped into list")
                else:
                    # Prefer lists of dicts (i.e., actual test arrays)
                    found_list = False
                    for key, value in parsed_json.items():
                        if isinstance(value, list) and value and isinstance(value[0], dict):
                            final_plan = value
                            found_list = True
                            print(f"DEBUG: Extracted plan from key: '{key}' (list of dicts)")
                            break

                    if not found_list:
                        # Fallback: first list we find (but warn if it's a list of primitives)
                        for key, value in parsed_json.items():
                            if isinstance(value, list):
                                final_plan = value
                                found_list = True
                                print(f"DEBUG: Extracted plan from key: '{key}' (fallback list)")
                                break

                    if not found_list:
                        print("Error: JSON is a dict but contains no suitable lists.")
                        return [{"name": "Error", "description": "LLM returned invalid format (Dict with no list).", "is_error": True}]
            
            
            # --- VALIDATION LOOP ---
            # Ensure every item has the required fields
>>>>>>> 7b067e5d67bddb45dde3639fbd8d31a7f6460ee7
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
<<<<<<< HEAD

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

=======
        
        system_prompt = """
        You are an expert Playwright Automation Engineer (Python).
        Your task: Write a complete, standalone Python script.

        STRICT RULES:
        1. IMPORTS: You MUST use exactly this import line:
           from playwright.sync_api import sync_playwright
        2. VIDEO: The script MUST create a browser context with video recording enabled
           - Use: context = browser.new_context(record_video_dir="test_videos", record_video_size={"width":1280,"height":720})
           - Use: page = context.new_page()
           - After the test completes, close the page (page.close()) then copy the finalized video file
             into the test artifacts folder (artifacts/<TEST_NAME>/video.webm) using shutil.copy2(page.video.path(), dst).
        3. LOCATORS: Use resilient locators: page.get_by_role(), page.get_by_placeholder().
        4. DATA: Use the provided USER_DATA values.
        5. STRUCTURE:
           - Define a run() function.
           - Inside run(), use 'with sync_playwright() as p:'.
           - Launch browser with headless=False.
           - Create a context as specified above, and call 'page.set_default_timeout(5000)'.
           - At the end, ensure context.close() and browser.close() are called.
           - Call run() at the end under 'if __name__ == "__main__":'.
        6. STATE MANAGEMENT (CRITICAL):
           - If the test performed a login step and cookies should be preserved, save storage with:
             context.storage_state(path='auth.json')
        7. OUTPUT: Return ONLY the Python code (no markdown or explanations). Ensure code compiles.
        """
        
>>>>>>> 7b067e5d67bddb45dde3639fbd8d31a7f6460ee7
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

<<<<<<< HEAD
    
=======
        print("🧠 Brain is refining the plan...")
        try:
            response = ollama.chat(
                model=self.model, 
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                format='json',
                options={'temperature': 0.2}
            )
            
            content = response['message']['content']
            parsed_json = json.loads(content)
            
            # (Reuse your robust list extraction logic here)
            final_plan = []
            if isinstance(parsed_json, list):
                final_plan = parsed_json
            elif isinstance(parsed_json, dict):
                for val in parsed_json.values():
                    if isinstance(val, list):
                        final_plan = val
                        break
            
            # Validation Loop
            safe_plan = []
            for item in final_plan:
                safe_item = {
                    "name": item.get("name", "Unnamed Test"),
                    "description": item.get("description", ""),
                    "missing_data": item.get("missing_data", []),
                    "requires_auth": item.get("requires_auth", False)
                }
                safe_plan.append(safe_item)
                
            return safe_plan

        except Exception as e:
            return [{"name": "Error Refining Plan", "description": str(e), "is_error": True}]

    # Salma: Multipage Explorartion
    def classify_navigation_candidate(
        self,
        page_snapshot: dict,
        candidate: dict,
        *,
        journey_hint: str | None = None,
        strict: bool = True,
    ) -> dict:
        """Classify whether a navigation candidate should be followed.

        Supports multiple site types by allowing a journey_hint, e.g.:
        - "signup" / "onboarding" (multi-step wizards)
        - "ecommerce" (browse -> product -> add to cart -> cart -> checkout)

        Returns a dict:
        {
          "follow": bool,
          "confidence": float,
          "category": "service_flow"|"out_of_scope"|"blocker"|"unclear",
          "reason": str,
          "suggested_phase": str
        }
        """

        title = (page_snapshot or {}).get("title", "")
        url = (page_snapshot or {}).get("url", "")
        cleaned_dom = (page_snapshot or {}).get("cleaned_dom", "")

        # Keep DOM very small; this is a classification task.
        dom_snippet = (cleaned_dom or "")[:3000]

        system_prompt = f"""
You are an expert QA assistant deciding whether to follow a navigation action during fully-automatic main-path exploration.

Your job:
- Decide if the candidate is part of the MAIN SERVICE FLOW.
- Skip out-of-scope informational pages (e.g. About, Blog, Careers, Contact, Terms, Privacy, Help, FAQ).

This explorer supports two common journeys:
1) SIGNUP / ONBOARDING WIZARD:
   - Follow steps like Sign up / Register / Next / Continue / Submit / Finish.
   - Skip unrelated navigation like About.
2) ECOMMERCE MAIN PATH:
   - Prefer a single end-to-end purchase flow:
     Browse products -> open a product -> add to cart -> go to cart -> checkout.
   - Skip unrelated informational navigation.
   - Note: "Add to cart" may not change URL; it's still in-scope.

STRICT MODE: {"ON" if strict else "OFF"}
If STRICT MODE is ON and you are unsure, return follow=false and category="unclear".

OUTPUT RULES:
1) Output MUST be valid JSON object (not a list, not markdown).
2) Schema:
   {{
     "follow": true|false,
     "confidence": 0.0-1.0,
     "category": "service_flow"|"out_of_scope"|"blocker"|"unclear",
     "reason": "...",
     "suggested_phase": "signup"|"browse"|"product"|"cart"|"checkout"|"unknown"
   }}
3) Be conservative about following nav/footer links.
"""

        user_prompt = f"""
JOURNEY_HINT: {journey_hint or "auto"}

CURRENT_PAGE:
- title: {title}
- url: {url}

CANDIDATE:
{json.dumps(candidate, ensure_ascii=False)}

DOM_SNIPPET:
{dom_snippet}
"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                format="json",
                options={"temperature": 0.1},
            )
            content = response["message"]["content"]
            parsed = json.loads(content)

            # Minimal validation + defaults.
            follow = bool(parsed.get("follow", False))
            confidence = parsed.get("confidence", 0.0)
            try:
                confidence = float(confidence)
            except Exception:
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))

            category = parsed.get("category", "unclear")
            if category not in {"service_flow", "out_of_scope", "blocker", "unclear"}:
                category = "unclear"

            suggested_phase = parsed.get("suggested_phase", "unknown")
            if suggested_phase not in {"signup", "browse", "product", "cart", "checkout", "unknown"}:
                suggested_phase = "unknown"

            reason = parsed.get("reason", "")
            if not isinstance(reason, str):
                reason = ""

            # Enforce strict-mode conservatism.
            if strict and category == "unclear":
                follow = False

            return {
                "follow": follow,
                "confidence": confidence,
                "category": category,
                "reason": reason,
                "suggested_phase": suggested_phase,
            }

        except Exception as e:
            return {
                "follow": False,
                "confidence": 0.0,
                "category": "unclear",
                "reason": f"classifier_error: {str(e)}",
                "suggested_phase": "unknown",
            }
>>>>>>> 7b067e5d67bddb45dde3639fbd8d31a7f6460ee7
