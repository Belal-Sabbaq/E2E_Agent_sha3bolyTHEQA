import sys
import asyncio
import re
import json
import os
import time
# Salma: Multipage Explorartion
from urllib.parse import urljoin, urlparse, urldefrag
from playwright.async_api import async_playwright
# --- FIX FOR WINDOWS & PLAYWRIGHT ---
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
# ------------------------------------

class BrowserManager:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

    # Salma: Multipage Explorartion
    async def _extract_interactive_elements(self):
        """Extract basic interactive elements for UI/LLM consumption."""
        if not self.page:
            return []

        # This JS is superior to Regex because it checks bounding box visibility.
        return await self.page.evaluate('''() => {
            const interactables = [];
            const selector = 'button, a, input, select, textarea, [role="button"], [role="link"], [role="checkbox"]';

            document.querySelectorAll(selector).forEach((el, index) => {
                const rect = el.getBoundingClientRect();
                const isVisible = rect.width > 0 && rect.height > 0 && window.getComputedStyle(el).visibility !== 'hidden';

                if (isVisible) {
                    interactables.push({
                        id: index,
                        tag: el.tagName.toLowerCase(),
                        text: (el.innerText || '').slice(0, 50).replace(/\\n/g, ' ').trim() || el.getAttribute('placeholder') || 'No Text',
                        role: el.getAttribute('role') || el.tagName.toLowerCase(),
                        attributes: {
                            type: el.getAttribute('type'),
                            name: el.getAttribute('name'),
                            id_attr: el.getAttribute('id'),
                            class: el.getAttribute('class'),
                            placeholder: el.getAttribute('placeholder')
                        }
                    });
                }
            });

            return interactables;
        }''')

    # Salma: Multipage Explorartion
    async def snapshot_current_page(self, *, include_raw_html: bool = False):
        """Capture a normalized snapshot of the current page.

        This is the shared building block for both single-page exploration (goto)
        and future multi-step exploration (click-based navigation).

        Returns a dict compatible with the existing app: title, url, cleaned_dom, elements.
        If include_raw_html is True, also returns raw_html.
        """
        if not self.page:
            await self.start_browser()

        await self.page.wait_for_load_state("domcontentloaded")

        raw_html = await self.page.content()
        cleaned_html = await self.clean_dom(raw_html)
        elements = await self._extract_interactive_elements()
        title = await self.page.title()

        snapshot = {
            "title": title,
            "url": self.page.url,
            "cleaned_dom": cleaned_html,
            "elements": elements,
        }
        if include_raw_html:
            snapshot["raw_html"] = raw_html
        return snapshot

    # Salma: Multipage Explorartion
    async def extract_navigation_candidates(self, *, max_candidates: int = 10):
        """Extract clickable navigation candidates (links/buttons).

        This is used by the automatic multi-page explorer to decide what to follow.
        It does NOT perform any clicking.

        Returns a list of dicts with stable, LLM-friendly fields:
        - tag, role, name (best-effort), href (if any), text
        - attributes (id/name/class/data-testid/aria-label)
        """
        if not self.page:
            await self.start_browser()

        # Extract in the browser so we can reliably compute visibility and href.
        candidates = await self.page.evaluate('''() => {
            const out = [];
            const selector = 'a[href], button, [role="link"], [role="button"], input[type="submit"], input[type="button"]';

            const norm = (s) => (s || '').replace(/\s+/g, ' ').trim();

            const getBestName = (el) => {
                const text = norm(el.innerText);
                const aria = norm(el.getAttribute('aria-label'));
                const title = norm(el.getAttribute('title'));
                const placeholder = norm(el.getAttribute('placeholder'));
                return text || aria || title || placeholder || '';
            };

            const isVisible = (el) => {
                const rect = el.getBoundingClientRect();
                if (!rect || rect.width <= 0 || rect.height <= 0) return false;
                const style = window.getComputedStyle(el);
                if (!style) return false;
                if (style.visibility === 'hidden' || style.display === 'none') return false;
                return true;
            };

            document.querySelectorAll(selector).forEach((el, index) => {
                if (!isVisible(el)) return;

                const tag = (el.tagName || '').toLowerCase();
                const role = el.getAttribute('role') || tag;
                const name = getBestName(el);
                const href = tag === 'a' ? (el.getAttribute('href') || '') : '';

                // Skip completely unlabeled candidates to reduce noise.
                if (!name && !href) return;

                out.push({
                    id: index,
                    tag,
                    role,
                    name,
                    text: norm(el.innerText),
                    href,
                    attributes: {
                        id_attr: el.getAttribute('id'),
                        name_attr: el.getAttribute('name'),
                        class: el.getAttribute('class'),
                        aria_label: el.getAttribute('aria-label'),
                        data_testid: el.getAttribute('data-testid')
                    }
                });
            });

            return out;
        }''')

        # Deduplicate in Python (common in nav menus with repeated items).
        print(f"       [EXTRACT] Raw candidates from JS: {len(candidates or [])} items")
        if candidates:
            for i, c in enumerate(candidates[:3]):
                print(f"         - {c.get('tag')}: {c.get('name', 'N/A')[:40]}")
        seen = set()
        unique = []
        for c in candidates or []:
            key = (
                (c.get("tag") or ""),
                (c.get("role") or ""),
                (c.get("name") or ""),
                (c.get("href") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(c)
            if max_candidates and len(unique) >= max_candidates:
                break

        print(f"       [EXTRACT] After dedup: {len(unique)} unique candidates")
        return unique

    # Salma: Multipage Explorartion
    async def detect_required_user_input(self, *, max_fields: int = 20):
        """Detect whether the current page appears to require user input to proceed.

        Returns a list of field descriptors (possibly empty). This is a heuristic.
        The main-path explorer uses this to stop at a checkpoint page rather than
        trying to auto-fill data.
        """
        if not self.page:
            await self.start_browser()

        print(f"       [INPUT] Detecting required user input fields...")

        fields = await self.page.evaluate('''() => {
            const out = [];
            const selector = 'input, select, textarea';
            const norm = (s) => (s || '').replace(/\s+/g, ' ').trim();

            const isVisible = (el) => {
                const rect = el.getBoundingClientRect();
                if (!rect || rect.width <= 0 || rect.height <= 0) return false;
                const style = window.getComputedStyle(el);
                if (!style) return false;
                if (style.visibility === 'hidden' || style.display === 'none') return false;
                return true;
            };

            const looksLikeNewsletter = (text) => {
                const t = norm(text).toLowerCase();
                return t.includes('subscribe') || t.includes('subscription') || t.includes('newsletter');
            };

            const getLabelText = (el) => {
                const id = el.getAttribute('id');
                if (id) {
                    const lbl = document.querySelector(`label[for="${CSS.escape(id)}"]`);
                    if (lbl) return norm(lbl.innerText);
                }
                // Try nearest label/field container.
                const parentLabel = el.closest('label');
                if (parentLabel) return norm(parentLabel.innerText);
                return '';
            };

            document.querySelectorAll(selector).forEach((el) => {
                if (!isVisible(el)) return;
                const rect = el.getBoundingClientRect();
                // Heuristic: footer subscribe widgets are often at the bottom; don't treat as a blocking checkpoint.
                if (rect && rect.top > (window.innerHeight * 0.75)) return;

                const tag = (el.tagName || '').toLowerCase();
                const type = norm(el.getAttribute('type')) || (tag === 'select' ? 'select' : tag);

                // Ignore hidden inputs.
                if (tag === 'input' && (type === 'hidden')) return;

                const required = el.hasAttribute('required') || el.getAttribute('aria-required') === 'true';
                const value = norm(el.value);
                const placeholder = norm(el.getAttribute('placeholder'));
                const name = norm(el.getAttribute('name'));
                const idAttr = norm(el.getAttribute('id'));
                const ariaLabel = norm(el.getAttribute('aria-label'));
                const label = getLabelText(el);

                // Ignore newsletter/subscribe inputs; they're required sometimes but not part of the main flow.
                if (
                    looksLikeNewsletter(idAttr) ||
                    looksLikeNewsletter(name) ||
                    looksLikeNewsletter(placeholder) ||
                    looksLikeNewsletter(ariaLabel) ||
                    looksLikeNewsletter(label)
                ) {
                    return;
                }

                // We consider it "needs input" if ANY input field is empty.
                // This catches login pages that don't explicitly mark inputs as required.
                // We'll report ANY visible input field as needing attention.
                if (!value) {
                    out.push({
                        tag,
                        type,
                        required,
                        name_attr: name,
                        id_attr: idAttr,
                        placeholder,
                        aria_label: ariaLabel,
                        label
                    });
                }
            });

            return out;
        }''')

        if not fields:
            print(f"       [INPUT] ✅ No empty input fields found")
            return []
        print(f"       [INPUT] ✅ Found {len(fields)} empty input field(s) needing data: {[f.get('name_attr', f.get('id_attr', 'unknown')) for f in fields[:3]]}")
        return fields[:max_fields]

    # Salma: Multipage Explorartion
    def _make_multipage_run_dir(self, base_dir: str = "artifacts/multipage_exploration") -> str:
        """Create and return a unique run directory for persisted exploration artifacts."""
        os.makedirs(base_dir, exist_ok=True)
        run_id = str(int(time.time_ns()))
        run_dir = os.path.join(base_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    # Salma: Multipage Explorartion
    def _persist_json(self, path: str, payload: dict) -> None:
        """Persist a JSON payload safely (best-effort)."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("DEBUG: Failed to persist JSON:", path, e)

    # Salma: Multipage Explorartion
    def _canonicalize_url(self, url: str) -> str:
        """Canonicalize URL for loop detection.

        - Always strips fragment.
        - Keeps query params (important for wizards and some ecommerce flows).
        """
        if not url:
            return url
        clean, _frag = urldefrag(url)
        return clean

    # Salma: Multipage Explorartion
    def _score_candidate(self, candidate: dict) -> int:
        """Heuristic score for picking a single main-path action."""
        name = (candidate.get('name') or '').lower()
        href = (candidate.get('href') or '').lower()
        role = (candidate.get('role') or '').lower()
        tag = (candidate.get('tag') or '').lower()

        score = 0

        # Strong flow buttons.
        if any(k in name for k in ['next', 'continue', 'finish', 'submit', 'proceed']):
            score += 100

        # Signup/onboarding.
        if any(k in name for k in ['sign up', 'signup', 'register', 'create account', 'get started', 'start']):
            score += 70

        # Ecommerce.
        if any(k in name for k in ['add to cart', 'add-to-cart', 'add']):
            score += 60
        if 'cart' in name or '/cart' in href:
            score += 55
        if any(k in name for k in ['checkout', 'place order']):
            score += 65
        if any(k in href for k in ['/product', '/products', '/category', '/shop']):
            score += 25

        # Prefer buttons for step progression.
        if role == 'button' or tag == 'button':
            score += 10

        # Penalize obvious informational links.
        if any(k in name for k in ['about', 'blog', 'career', 'contact', 'privacy', 'terms', 'help', 'faq']):
            score -= 100

        return score

    # Salma: Multipage Explorartion
    async def _navigate_via_candidate(self, current_url: str, candidate: dict) -> bool:
        """Attempt to navigate using a candidate (href goto or click).

        Returns True if an action was executed without throwing.
        """
        if not self.page:
            await self.start_browser()

        href = (candidate.get('href') or '').strip()
        name = (candidate.get('name') or '').strip()
        role = (candidate.get('role') or '').strip().lower()
        tag = (candidate.get('tag') or '').strip().lower()

        if href:
            href_l = href.lower()
            # Many ecommerce sites use javascript links for cart actions (e.g. href="javascript:void(0)").
            # These must be clicked instead of navigated.
            if not (
                href_l.startswith('javascript:')
                or href_l == '#'
                or href_l.startswith('#')
                or href_l.startswith('mailto:')
                or href_l.startswith('tel:')
            ):
                target = urljoin(current_url, href)
                await self.page.goto(target)
                return True

        # Click-based navigation (wizards and buttons).
        # Use role if it is meaningful; otherwise infer.
        inferred_role = None
        if role in {'button', 'link'}:
            inferred_role = role
        else:
            # Anchors should be clicked as links.
            inferred_role = 'link' if tag == 'a' or role == 'a' else 'button'

        locator = self.page.get_by_role(inferred_role, name=name)
        await locator.first.click()
        return True

    # Salma: Multipage Explorartion
    async def explore_main_path(
        self,
        start_url: str,
        *,
        brain,
        max_steps: int = 5,
        max_candidates: int = 10,
        journey_hint: str | None = None,
        strict: bool = True,
        stop_on_input_required: bool = True,
        persist: bool = False,
        persist_dir: str | None = None,
    ) -> dict:
        """Fully automatic, main-path multi-page exploration.

        - Reuses a single Playwright page/session.
        - At each step, selects ONE candidate action to follow.
        - Stops at an input-required checkpoint (if stop_on_input_required=True).

        The `brain` must provide a `classify_navigation_candidate(page_snapshot, candidate, ...)` method
        (LLMBrain does).
        """
        if not self.page:
            await self.start_browser()

        history: list[dict] = []
        visited: set[str] = set()

        stop_reason = None
        stop_details = None

        run_dir = None
        if persist:
            try:
                run_dir = persist_dir or self._make_multipage_run_dir()
            except Exception as e:
                # Best-effort: persistence must never break exploration.
                run_dir = None
                print("DEBUG: Failed to create persistence directory:", e)

        current_url = start_url
        print(f"🚀 [MULTIPAGE] Starting exploration at {start_url}")

        completed_all_steps = True
        for step_index in range(max_steps):
            print(f"\n📍 [STEP {step_index}] Starting step (max_steps={max_steps})")
            print(f"   Current URL: {current_url}")
            try:
                print(f"   [GOTO] Navigating to {current_url}...")
                await self.page.goto(current_url)
                print(f"   [GOTO] ✅ Navigation complete")
            except Exception as e:
                print(f"   [GOTO] ❌ Navigation failed: {e}")
                stop_reason = 'navigation_error'
                stop_details = str(e)
                completed_all_steps = False
                break

            print(f"   [SNAPSHOT] Capturing page snapshot...")
            snapshot = await self.snapshot_current_page(include_raw_html=False)
            print(f"   [SNAPSHOT] ✅ Title: {snapshot.get('title')}")
            snapshot_meta = {
                'step_index': step_index,
                'decision': None,
            }

            canonical = self._canonicalize_url(snapshot.get('url', ''))
            if canonical in visited:
                stop_reason = 'loop_detected'
                stop_details = canonical
                history.append({**snapshot, '_meta': snapshot_meta})
                completed_all_steps = False
                break
            visited.add(canonical)

            # Checkpoint behavior: stop on required input.
            print(f"   [CHECKPOINT] Checking for empty input fields that need to be filled...")
            required_fields = await self.detect_required_user_input()
            print(f"   [CHECKPOINT] Found {len(required_fields)} empty input field(s). stop_on_input_required={stop_on_input_required}")
            if stop_on_input_required and required_fields:
                print(f"   [CHECKPOINT] ✅ STOPPING - User input is required to proceed")
                snapshot_meta['stop_reason'] = 'requires_input'
                snapshot_meta['required_fields'] = required_fields
                history.append({**snapshot, '_meta': snapshot_meta})
                stop_reason = 'requires_input'
                stop_details = required_fields
                completed_all_steps = False
                break

            # Extract candidates and classify.
            print(f"   [CANDIDATES] Extracting navigation candidates (max={max_candidates})...")
            candidates = await self.extract_navigation_candidates(max_candidates=max_candidates)
            print(f"   [CANDIDATES] ✅ Found {len(candidates)} candidates")

            scored = []
            print(f"   [CLASSIFY] Starting classification of {len(candidates)} candidates...")
            for i, cand in enumerate(candidates):
                cand_name = cand.get('name', 'N/A')
                print(f"     [CLASSIFY] Candidate {i+1}/{len(candidates)}: {cand_name}...")
                decision = await brain.classify_navigation_candidate(
                    snapshot,
                    cand,
                    journey_hint=journey_hint,
                    strict=strict,
                )
                follow = decision.get('follow')
                category = decision.get('category', 'unknown')
                print(f"     [CLASSIFY] ✅ Result: follow={follow}, category={category}")
                if decision.get('follow') is True:
                    score = self._score_candidate(cand)
                    print(f"     [CLASSIFY]   → Scoring: {score}")
                    scored.append((score, cand, decision))

            print(f"   [CLASSIFY] ✅ Classification complete. Scored {len(scored)} candidates to follow")
            if not scored:
                snapshot_meta['stop_reason'] = 'no_follow_action'
                history.append({**snapshot, '_meta': snapshot_meta})
                stop_reason = 'no_follow_action'
                stop_details = None
                completed_all_steps = False
                break

            scored.sort(key=lambda t: t[0], reverse=True)
            best_score, best_candidate, best_decision = scored[0]
            best_name = best_candidate.get('name', 'N/A')
            print(f"   [SELECT] Best candidate (score={best_score}): {best_name}")
            snapshot_meta['decision'] = {
                'candidate': best_candidate,
                'classifier': best_decision,
                'score': best_score,
            }
            history.append({**snapshot, '_meta': snapshot_meta})

            # Try to navigate and determine progress.
            before_url = snapshot.get('url', '')
            before_dom = snapshot.get('cleaned_dom', '')
            print(f"   [NAVIGATE] Executing action: {best_name}...")
            try:
                print(f"   [NAVIGATE] Clicking/navigating...")
                await self._navigate_via_candidate(before_url or current_url, best_candidate)
                print(f"   [NAVIGATE] Action executed, waiting for page load...")
                # Allow both navigations and AJAX-type changes.
                await self.page.wait_for_load_state('domcontentloaded')
                print(f"   [NAVIGATE] ✅ Page loaded")
            except Exception as e:
                print(f"   [NAVIGATE] ❌ Action failed: {e}")
                stop_reason = 'action_failed'
                stop_details = str(e)
                completed_all_steps = False
                break

            print(f"   [PROGRESS] Taking after-snapshot to check progress...")
            after_snapshot = await self.snapshot_current_page(include_raw_html=False)
            after_url = after_snapshot.get('url', '')
            after_dom = after_snapshot.get('cleaned_dom', '')

            url_changed = self._canonicalize_url(after_url) != self._canonicalize_url(before_url)
            dom_changed = (after_dom or '') != (before_dom or '')
            print(f"   [PROGRESS] URL changed: {url_changed}, DOM changed: {dom_changed}")

            if not url_changed and not dom_changed:
                # No observable progress; stop at the current page.
                print(f"   [PROGRESS] ❌ No progress detected (no URL or DOM change)")
                stop_reason = 'no_progress'
                stop_details = best_candidate
                completed_all_steps = False
                break

            # Continue from the new URL (even if dom change only, keep current URL).
            current_url = after_url or before_url or current_url
            print(f"   [PROGRESS] ✅ Step {step_index} complete. Next URL: {current_url}")
            print(f"   Waiting before next step...")
            await asyncio.sleep(0.5)  # Small delay to prevent overwhelming browser

            # Persist step snapshot (best-effort) without screenshot bytes.
            if run_dir:
                step_payload = {
                    "step_index": step_index,
                    "url": snapshot.get("url"),
                    "title": snapshot.get("title"),
                    "meta": snapshot_meta,
                    # Keep content trimmed to avoid huge files.
                    "cleaned_dom": (snapshot.get("cleaned_dom") or "")[:5000],
                    "elements": (snapshot.get("elements") or [])[:200],
                }
                self._persist_json(os.path.join(run_dir, f"step_{step_index:03d}.json"), step_payload)

        if completed_all_steps and stop_reason is None:
            stop_reason = 'max_steps_reached'
            print(f"\n🏁 [MULTIPAGE] Exploration complete. Reached max steps.")
        else:
            print(f"\n🏁 [MULTIPAGE] Exploration stopped. Reason: {stop_reason}")

        print(f"📊 [MULTIPAGE] Summary: visited {len(history)} pages")
        checkpoint = history[-1] if history else None
        result = {
            'history': history,
            'checkpoint': checkpoint,
            'stop_reason': stop_reason,
            'stop_details': stop_details,
            'persist_dir': run_dir,
        }

        if run_dir:
            summary = {
                "start_url": start_url,
                "max_steps": max_steps,
                "visited": len(history),
                "stop_reason": stop_reason,
                "stop_details": stop_details,
                "checkpoint_url": (checkpoint or {}).get("url") if isinstance(checkpoint, dict) else None,
            }
            self._persist_json(os.path.join(run_dir, "summary.json"), summary)

        return result

    async def start_browser(self, auth_file="aauth.json"):
        """
        Starts the browser. 
        If 'auth.json' exists, it loads cookies/storage to restore session.
        """
        video_dir = "test_videos"
        os.makedirs(video_dir, exist_ok=True)
        
        if self.playwright is None:
            self.playwright = await async_playwright().start()
        
        if self.browser is None:
            # Headless=False is REQUIRED for the "Visual Context" grading pillar
            self.browser = await self.playwright.chromium.launch(headless=False)
         
        if self.context is None:
            # Configure context to record video for every page created in this context
            context_kwargs = {
                "viewport": {"width": 1280, "height": 720},
                # Playwright will save per-page videos into this directory
                "record_video_dir": video_dir,
                "record_video_size": {"width": 1280, "height": 720}
            }

            if os.path.exists(auth_file):
                print(f"Loading session from {auth_file}")
                context_kwargs["storage_state"] = auth_file

            # Pass the kwargs through so Playwright actually records video
            self.context = await self.browser.new_context(**context_kwargs)
            print(f"DEBUG: Browser context created with video recording enabled -> dir={context_kwargs.get('record_video_dir')} size={context_kwargs.get('record_video_size')}")

        
        if self.page is None:
            self.page = await self.context.new_page()

    async def clean_dom(self, raw_html: str) -> str:
        """
        Cleans HTML to save tokens before sending to the LLM.
        (Adapted from your colleague's approach)
        """
        # Remove script and style tags
        cleaner = re.sub(r"<script[\s\S]*?</script>", "", raw_html)
        cleaner = re.sub(r"<style[\s\S]*?</style>", "", cleaner)
        # Remove comments
        cleaner = re.sub(r"", "", cleaner)
        # Remove meta/link tags
        cleaner = re.sub(r"<(meta|link|base|br|hr)\s*[^>]*?/?>", "", cleaner)
        # Collapse whitespace
        cleaner = re.sub(r"\s+", " ", cleaner).strip()
        return cleaner[:15000]  # Limit context window

    async def explore_url(self, url):
        """
        Navigates to the URL and returns BOTH:
        1. structured_elements (for the UI/Human to see)
        2. cleaned_html (for the LLM to read)
        """
        if not self.page:
            await self.start_browser()

        try:
            print(f"Navigating to {url}...")
            await self.page.goto(url)
            return await self.snapshot_current_page(include_raw_html=False)

        except Exception as e:
            return {"error": str(e)}

    async def capture_screenshot(self):
        """Returns screenshot bytes for Streamlit display."""
        if self.page:
            try:
                return await self.page.screenshot()
            except:
                return None
        return None

    async def save_storage_state(self, path="auth.json"):
        """Call this after a successful login test to save cookies."""
        if self.context:
            await self.context.storage_state(path=path)
            print(f"Session saved to {path}")

    async def list_recorded_videos(self):
        """Return list of recorded video files (most recent first)."""
        video_dir = "test_videos"
        if not os.path.exists(video_dir):
            return []
        files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.lower().endswith((".webm", ".mp4"))]
        files = sorted(files, key=lambda p: os.path.getmtime(p), reverse=True)
        print("DEBUG: Found recorded videos:", files)
        return files

    async def copy_latest_video_to_artifacts(self, test_name: str, src_video_path: str = None):
        """Copy the most recent video into the test artifacts folder (returns dst or None).

        If src_video_path is provided, copy that file. Otherwise pick the latest from video dir.
        The destination file will include a timestamp to avoid caching issues in the UI.
        """
        # Determine source
        if src_video_path and os.path.exists(src_video_path):
            latest = src_video_path
        else:
            videos = await self.list_recorded_videos()
            if not videos:
                print("DEBUG: No recorded videos to copy")
                return None
            latest = videos[0]

        # Prepare destination with timestamp to avoid client caching showing an old file
        dest_dir = os.path.join("artifacts", test_name)
        os.makedirs(dest_dir, exist_ok=True)
        ext = os.path.splitext(latest)[1] or ".webm"
        # use nanosecond timestamp for maximum uniqueness to avoid caching issues
        timestamp = int(time.time_ns())
        dst = os.path.join(dest_dir, f"video_{timestamp}{ext}")
        try:
            import shutil
            shutil.copy2(latest, dst)
            print(f"DEBUG: Copied video {latest} to {dst}")
            return dst
        except Exception as e:
            print("DEBUG: Failed to copy video:", e)
            return None

    async def close(self):
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()