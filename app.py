import streamlit as st
import asyncio
import os
import subprocess
import sys
import nest_asyncio

from browser_manager import BrowserManager
from llm_brain import LLMBrain
from ui_utils import config_model_selector

# -------------------------------------------------
# CONFIGURATION SECTION
# -------------------------------------------------
config_model_selector()  # This sets model_name, model_mode, and api_key
def load_test_artifacts(test_name: str):
    base = os.path.join("artifacts", test_name)
    result_path = os.path.join(base, "result.json")

    result = None
    screenshots = []

    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)

    if os.path.exists(base):
        screenshots = sorted(
            [
                os.path.join(base, f)
                for f in os.listdir(base)
                if f.endswith(".png")
            ]
        )

    return result, screenshots

# Then safely use:
api_key = st.session_state.get("api_key")
mode = st.session_state.get("model_mode")

if mode == "API (Groq)" and not api_key:
    st.error("❌ Please enter your Groq API key in the sidebar.")
    st.stop()

nest_asyncio.apply()
st.set_page_config(layout="wide", page_title="AI QA Agent")

# -------------------------------------------------
# SESSION STATE INIT
# -------------------------------------------------
if "browser_manager" not in st.session_state:
    st.session_state.browser_manager = BrowserManager()
if "scraped_data" not in st.session_state:
    st.session_state.scraped_data = None
if "test_plan" not in st.session_state:
    st.session_state.test_plan = []
if "accepted_tests" not in st.session_state:
    st.session_state.accepted_tests = []
if "plan_locked" not in st.session_state:
    st.session_state.plan_locked = False
if "current_step" not in st.session_state:
    st.session_state.current_step = "design"

# LLMBrain: Always use latest selector
if "brain" not in st.session_state:
    st.session_state.brain = LLMBrain(
        model=st.session_state.get("model_name", "qwen3:4b"),
        mode=st.session_state.get("model_mode", "API (Groq)"),
        api_key=st.session_state.get("api_key")
    )

print("Groq API Key:", st.session_state.get("api_key"))
# DEBUG: Show model in use
st.sidebar.markdown(f"**🧠 Model:** `{st.session_state.get('model_name')}`")
st.sidebar.markdown(f"**⚙️ Mode:** `{st.session_state.get('model_mode')}`")

with st.sidebar:
    st.header("System Metrics")

    brain = st.session_state.brain
    metrics = brain.get_metrics_summary()

    st.metric("LLM Calls", metrics["total_calls"])
    st.metric("Total Tokens", metrics["total_tokens"])
    st.metric("Avg Response Time", f"{metrics['average_time']:.2f}s")
    st.metric("Errors", metrics["errors"])

    st.markdown("#### Per Model Stats")
    for model_key, stats in metrics["per_model"].items():
        st.markdown(f"**{model_key}**")
        st.write(f"Requests: {stats['count']}")
        st.write(f"Tokens: {stats['total_tokens']}")
        st.write(f"Avg Time: {stats['total_time'] / stats['count']:.2f}s")
        st.write(f"Errors: {stats['errors']}")
        st.markdown("---")

    if st.button("🔄 Reset Metrics"):
        brain.reset_metrics()
        st.success("Metrics reset.")
        st.rerun()

# -------------------------------------------------
# UI LAYOUT & FLOW
# -------------------------------------------------
st.title("🤖 Human-in-the-Loop QA Agent")

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("1. Explore a Web Page")
    url = st.text_input("Enter URL", "https://saucedemo.com")
    if st.button("🌐 Explore URL"):
        with st.spinner("Navigating..."):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                manager = st.session_state.browser_manager

                data = loop.run_until_complete(manager.explore_url(url))
                screenshot = loop.run_until_complete(manager.capture_screenshot())

                st.session_state.scraped_data = data
                st.session_state.current_screenshot = screenshot
                st.success(f"Loaded: {data.get('title')}")
            except Exception as e:
                st.error(f"Navigation failed: {e}")

    if st.session_state.scraped_data:
        st.json(st.session_state.scraped_data.get("elements", []))

with col2:
    st.subheader("Agent Screenshot")
    if st.session_state.get("current_screenshot"):
        st.image(st.session_state.current_screenshot)
    else:
        st.info("Run exploration to see screenshot.")

# -------------------------------------------------
# STEP 2: TEST PLAN GENERATION
# -------------------------------------------------
if st.session_state.scraped_data and not st.session_state.plan_locked:
    st.divider()
    st.subheader("2. Generate Test Plan")

    if not st.session_state.test_plan and not st.session_state.accepted_tests:
        if st.button("🤖 Generate Initial Plan"):
            with st.spinner("AI is thinking..."):
                plan = st.session_state.brain.generate_test_plan(st.session_state.scraped_data)
                valid_plan = [p for p in plan if not p.get("is_error")]
                if valid_plan:
                    st.session_state.test_plan = valid_plan
                    st.rerun()
                else:
                    st.error("No valid tests generated.")

# -------------------------------------------------
# STEP 3: APPROVAL
# -------------------------------------------------
if st.session_state.test_plan or st.session_state.accepted_tests:
    st.divider()
    st.subheader("3. Approve or Refine Tests")

    if st.session_state.test_plan:
        for i, test in enumerate(st.session_state.test_plan):
            with st.expander(f"🧪 {test['name']}"):
                st.write(test["description"])
                if st.button("✅ Accept", key=f"acc_{i}"):
                    st.session_state.accepted_tests.append(test)
                    st.session_state.test_plan.pop(i)
                    st.rerun()
                if st.button("🗑 Remove", key=f"rem_{i}"):
                    st.session_state.test_plan.pop(i)
                    st.rerun()

    if st.session_state.accepted_tests and st.button("🚀 Lock Plan and Enter Data"):
        st.session_state.plan_locked = True
        st.session_state.test_plan = st.session_state.accepted_tests.copy()
        st.rerun()
        st.session_state.current_step = "data"


def is_valid_python(code: str) -> bool:
    try:
        compile(code, "<generated_test>", "exec")
        return True
    except SyntaxError:
        return False

# -------------------------------------------------
# STEP 4: DATA ENTRY
# -------------------------------------------------
if st.session_state.plan_locked:
    # Find tests that actually need user input
    tests_needing_data = [
        t for t in st.session_state.test_plan
        if t.get("missing_data")
    ]

    # If NO test needs data → skip step 4 entirely
    if not tests_needing_data:
        st.session_state.approved_tests = st.session_state.test_plan
        st.session_state.current_step = "code"
        st.rerun()

    st.divider()
    st.subheader("4. Fill Required Test Data")

    with st.form("data_entry_form"):
        approved_tests = []

        for i, test in enumerate(st.session_state.test_plan):
            st.markdown(f"**Test {i+1}: {test['name']}**")

            user_data = {}

            for field in test.get("missing_data", []):
                label = field.replace("_", " ").title()
                val = st.text_input(label, key=f"data_{i}_{field}")
                if val.strip():
                    user_data[field] = val.strip()

            test["user_data"] = user_data
            approved_tests.append(test)
            st.markdown("---")

        if st.form_submit_button("✅ Submit Test Data"):
            st.session_state.approved_tests = approved_tests
            st.session_state.current_step = "code"
            st.rerun()

# -------------------------------------------------
# STEP 5: CODE GENERATION
# -------------------------------------------------
if st.session_state.get("current_step") == "code":
    st.divider()
    st.subheader("5. Generate Python Code")

    if "generated_code_map" not in st.session_state:
        st.session_state.generated_code_map = {}

    if st.button("🧠 Generate Scripts"):
        with st.spinner("Generating Playwright Scripts..."):
            for test in st.session_state.approved_tests:
                code = st.session_state.brain.generate_playwright_code(test, st.session_state.scraped_data)
                st.session_state.generated_code_map[test["name"]] = code

                # Save to file
                safe_name = test["name"].lower().replace(" ", "_").replace("/", "_")
                path = f"generated_tests/test_{safe_name}.py"
                os.makedirs("generated_tests", exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(code)
            st.success("Scripts saved!")
            st.session_state.current_step = "verify"
            st.rerun()

# -------------------------------------------------
# STEP 6: EXECUTION
# -------------------------------------------------


if st.session_state.get("current_step") == "verify":
    st.divider()
    st.subheader("6. Run Tests")

    files = os.listdir("generated_tests") if os.path.exists("generated_tests") else []
    py_files = [f for f in files if f.endswith(".py")]

    for test_file in py_files:
        test_name = test_file.replace("test_", "").replace(".py", "")

        with st.expander(f"▶️ {test_file}", expanded=False):
            if st.button("Run", key=f"run_{test_file}"):
                with st.spinner("Executing test..."):
                    try:
                        result = subprocess.run(
                            [sys.executable, f"generated_tests/{test_file}"],
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                    except Exception as e:
                        st.error(f"Execution error: {e}")
                        continue

                # -----------------------------
                # Load artifacts AFTER run
                # -----------------------------
                result_json, screenshots = load_test_artifacts(test_name)

                if result_json:
                    status = result_json.get("status")
                    timestamp = result_json.get("timestamp")
                    error_msg = result_json.get("error")

                    if status == "PASS":
                        st.success("✅ TEST PASSED")
                    else:
                        st.error("❌ TEST FAILED")

                    st.caption(f"🕒 {timestamp}")

                    if error_msg:
                        st.code(error_msg, language="text")
                else:
                    st.warning("No result.json found")

                # -----------------------------
                # Screenshots timeline
                # -----------------------------
                if screenshots:
                    st.markdown("### 📸 Execution Timeline")
                    for img in screenshots:
                        st.image(img, caption=os.path.basename(img))
                else:
                    st.info("No screenshots found")

                # Raw logs
                if result.stdout:
                    st.markdown("### 📄 STDOUT")
                    st.code(result.stdout)

                if result.stderr:
                    st.markdown("### ⚠️ STDERR")
                    st.code(result.stderr)

# -------------------------------------------------
# RESET / CLEANUP
# -------------------------------------------------
if st.button("🔄 Reset Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

if st.button("🧹 Close Browser"):
    asyncio.run(st.session_state.browser_manager.close())
