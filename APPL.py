"""APPL. (Prototype) — Adaptive Personalized Learning Platform (Streamlit)."""

from __future__ import annotations

import base64
import dataclasses
import difflib
import hashlib
import json
import os
import random
import re
import textwrap
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st


st.set_page_config(
    page_title="APPL.",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
  :root { color-scheme: dark; }
  html, body, .stApp { background: #0A0A0A !important; color: #F2F2F2 !important; }
  * { font-family: ui-serif, Georgia, "Times New Roman", Garamond, serif; }
  a { color: #F2F2F2 !important; text-decoration: underline; }
  section[data-testid="stSidebar"] { background: #0A0A0A !important; border-right: 1px solid rgba(255,255,255,0.10); }
  [data-testid="stHeader"] { background: rgba(0,0,0,0.0) !important; }
  .block-container { padding-top: 1.2rem; }

  .card {
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 18px;
    background: rgba(255,255,255,0.03);
  }
  .title { font-size: 44px; font-weight: 700; letter-spacing: -0.02em; margin: 0; }
  .muted { opacity: 0.85; }

  /* Make inputs and buttons monochrome */
  button[kind="primary"], button[kind="secondary"] {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
  }
  button[kind="primary"] { background: #F2F2F2 !important; color: #0A0A0A !important; }
  button[kind="secondary"] { background: rgba(255,255,255,0.05) !important; color: #F2F2F2 !important; }
  [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-testid="stSelectbox"] div, [data-testid="stFileUploader"] section {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
  }

  /* Force Streamlit alerts (success/error/warning/info) to monochrome */
  div[data-testid="stAlert"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    color: #F2F2F2 !important;
  }
  div[data-testid="stAlert"] svg { color: #F2F2F2 !important; }

  /* Progress bar monochrome */
  [data-testid="stProgress"] > div > div {
    background: rgba(255,255,255,0.25) !important;
  }
</style>
""",
    unsafe_allow_html=True,
)


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("page", "landing")  # landing/auth/profile/dashboard/tutor/advanced
    ss.setdefault("user_db", {})
    ss.setdefault("is_authed", False)
    ss.setdefault("username", "")
    ss.setdefault("name", "")
    ss.setdefault("education_level", "")
    ss.setdefault("target_subject", "")
    ss.setdefault("topic", "")
    ss.setdefault("web_material", "")
    ss.setdefault("uploads", [])
    ss.setdefault("difficulty", 2)
    ss.setdefault("last_lesson", "")
    ss.setdefault("visual_prompt", "")
    ss.setdefault("active_question", None)
    ss.setdefault("answer_history", [])
    ss.setdefault("notes_log", [])
    ss.setdefault("handwriting_last_score", None)
    ss.setdefault("handwriting_last_feedback", "")


_init_state()


@dataclasses.dataclass
class LLMConfig:
    provider: str  # "mock" or "gemini"
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.2


class TutorLLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._genai = None
        self._model = None
        if cfg.provider == "gemini":
            try:
                import google.generativeai as genai  # type: ignore
                api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
                if not api_key:
                    self.cfg.provider = "mock"
                else:
                    genai.configure(api_key=api_key)
                    self._genai = genai
                    self._model = genai.GenerativeModel(cfg.model_name)
            except Exception:
                self.cfg.provider = "mock"

    @staticmethod
    def _clean_text(s: str) -> str:
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def _level_style(education_level: str, difficulty: int) -> Tuple[str, str]:
        edu = (education_level or "").lower()
        if any(k in edu for k in ["school", "high school", "middle", "secondary"]):
            base_tone = "simple and friendly"
        elif any(k in edu for k in ["b.tech", "btech", "undergrad", "college", "bachelor"]):
            base_tone = "technical but approachable"
        elif any(k in edu for k in ["phd", "research", "postdoc", "masters", "m.tech", "mtech"]):
            base_tone = "precise and research-oriented"
        else:
            base_tone = "clear and helpful"

        depth_map = {
            1: "very basic overview with analogies",
            2: "basic explanation with 1-2 examples",
            3: "standard depth with key definitions and a worked example",
            4: "advanced depth with assumptions, edge cases, and intuition",
            5: "deep dive with technical nuance and common misconceptions",
        }
        return base_tone, depth_map.get(int(difficulty), depth_map[2])

    def _mock_summarize(
        self,
        topic: str,
        education_level: str,
        difficulty: int,
        grounded_text: str,
        target_subject: str,
    ) -> str:
        tone, depth = self._level_style(education_level, difficulty)
        snippet = grounded_text.strip()
        if snippet:
            snippet = snippet[:450].strip()
            snippet = self._clean_text(snippet)
            grounding = f"Grounded notes (from web): {snippet}"
        else:
            grounding = (
                "Grounded notes: (No web material fetched yet. "
                "Tip: search the topic on the Dashboard to ground the tutor.)"
            )
        lesson = f"""
        Topic: {topic}  |  Subject: {target_subject or "General"}  |  Level: {education_level or "Not set"}

        Teaching style: {tone}; Depth: {depth}

        1) Big picture
        - Explain what {topic} is and why it matters.

        2) Key idea (in one sentence)
        - {topic}: define it in a way suitable for {education_level or "your level"}.

        3) Key terms (mini glossary)
        - Term A: short meaning
        - Term B: short meaning
        - Term C: short meaning

        4) Example / intuition
        - Provide a concrete example relevant to {target_subject or "the subject"}.

        5) Check yourself
        - A quick question will appear below to test understanding.

        {grounding}
        """
        return textwrap.dedent(lesson).strip()

    def _mock_visual_prompt(self, topic: str, education_level: str) -> str:
        """A placeholder prompt that would be sent to an image/diagram generator."""
        level_hint = education_level or "student"
        return (
            f"Generate a clean educational diagram for '{topic}' tailored to a {level_hint}. "
            "Use labeled parts, minimal clutter, and a simple color palette. "
            "Include 1 real-world analogy annotation."
        )

    def _seed(self, *parts: str) -> int:
        """Stable seed so the same topic produces consistent questions."""
        blob = "||".join([p.strip().lower() for p in parts if p is not None])
        return int(hashlib.sha256(blob.encode("utf-8")).hexdigest(), 16) % (2**31 - 1)

    def _mock_question(
        self,
        topic: str,
        education_level: str,
        difficulty: int,
        grounded_text: str,
    ) -> Dict[str, Any]:
        rnd = random.Random(self._seed(topic, education_level, str(difficulty)))
        templates = {
            1: [
                ("Which option best describes {topic}?", "definition"),
                ("Why do people study {topic}?", "purpose"),
            ],
            2: [
                ("Which statement about {topic} is most accurate?", "concept"),
                ("Pick the best everyday example of {topic}.", "example"),
            ],
            3: [
                ("Which choice is a correct implication of {topic}?", "implication"),
                ("Which condition is most related to {topic}?", "condition"),
            ],
            4: [
                ("Which is the best explanation of a common misconception about {topic}?", "misconception"),
                ("Which trade-off is most associated with {topic}?", "tradeoff"),
            ],
            5: [
                ("Which subtle edge case is most likely to break naive reasoning about {topic}?", "edge_case"),
                ("Which statement best captures the technical nuance of {topic}?", "nuance"),
            ],
        }
        q_text, kind = rnd.choice(templates.get(int(difficulty), templates[2]))
        q_text = q_text.format(topic=topic)
        words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", grounded_text or "")
        keywords = [w for w in words if w.lower() not in {"the", "and", "that", "with", "from", "this"}]
        keywords = keywords[:20]

        def kw(i: int, default: str) -> str:
            return keywords[i] if i < len(keywords) else default

        correct = ""
        distractors: List[str] = []
        if kind in {"definition", "concept", "nuance"}:
            correct = f"A structured idea describing '{topic}' using clear definitions and examples."
            distractors = [
                f"A random fact list about {kw(0,'unrelated')} with no connection to '{topic}'.",
                f"A pure opinion about '{topic}' without definitions.",
                "A definition of a different concept entirely.",
            ]
        elif kind in {"purpose"}:
            correct = f"To understand and apply '{topic}' to solve problems more effectively."
            distractors = [
                f"Only to memorize {kw(1,'terms')} without understanding.",
                "Because it has no real-world use (false).",
                "Only because exams require it (too narrow).",
            ]
        elif kind in {"example"}:
            correct = f"An example that shows the core idea of '{topic}' in a simple setting."
            distractors = [
                f"A case study about {kw(2,'another topic')} that does not illustrate '{topic}'.",
                "An example with missing information that cannot be evaluated.",
                "A scenario that contradicts the definition of the topic.",
            ]
        elif kind in {"implication"}:
            correct = f"If you understand '{topic}', you can predict outcomes in related problems."
            distractors = [
                "It guarantees perfect results in all situations (too strong).",
                "It is unrelated to any other concept (unlikely).",
                "It only applies in one exact scenario (too restrictive).",
            ]
        elif kind in {"condition"}:
            correct = f"A key condition is that the assumptions behind '{topic}' are satisfied."
            distractors = [
                f"That {kw(3,'noise')} is always zero (rare).",
                "That nothing changes over time (not always required).",
                "That all variables are identical (not generally true).",
            ]
        elif kind in {"misconception"}:
            correct = f"A common mistake is confusing '{topic}' with a similar-looking concept."
            distractors = [
                "Believing the topic has no definition at all.",
                "Assuming the topic is only a formula, not an idea.",
                "Thinking the topic is purely subjective.",
            ]
        else:  # edge_case / tradeoff
            correct = f"Edge cases appear when hidden assumptions about '{topic}' do not hold."
            distractors = [
                "Edge cases never exist if you study enough (false).",
                "All edge cases are identical (false).",
                "Edge cases are only caused by arithmetic mistakes (too narrow).",
            ]

        options = [correct] + distractors
        rnd.shuffle(options)
        answer_idx = options.index(correct)

        explanation = (
            "We pick the option that stays aligned with the core definition and does not overclaim. "
            "In this prototype, questions are constrained to keep grading reliable."
        )

        return {
            "question": q_text,
            "options": options,
            "answer_idx": answer_idx,
            "explanation": explanation,
            "difficulty": int(difficulty),
        }

    def summarize(
        self,
        *,
        topic: str,
        education_level: str,
        difficulty: int,
        grounded_text: str,
        target_subject: str,
    ) -> str:
        topic = (topic or "").strip()
        if not topic:
            return "No topic selected yet."

        if self.cfg.provider == "gemini" and self._model is not None:
            prompt = f"""
You are a careful tutor. If you are unsure about facts, say so.

Student profile:
- Education level: {education_level}
- Target subject: {target_subject}
- Difficulty (1=basic, 5=deep): {difficulty}

Task:
Teach the concept: {topic}

Use this grounded material (quote small parts if helpful; do not invent details beyond it):
{grounded_text[:2000]}

Format:
1) Big picture (2-3 lines)
2) Key definitions (bullets)
3) Example (short)
4) Common misconception (1 bullet)
5) Mini-check question (1 question)
"""
            try:
                resp = self._model.generate_content(prompt)
                return (getattr(resp, "text", "") or "").strip() or self._mock_summarize(
                    topic, education_level, difficulty, grounded_text, target_subject
                )
            except Exception:
                return self._mock_summarize(topic, education_level, difficulty, grounded_text, target_subject)

        return self._mock_summarize(topic, education_level, difficulty, grounded_text, target_subject)

    def visual_prompt(self, *, topic: str, education_level: str) -> str:
        if not topic:
            return ""
        return self._mock_visual_prompt(topic, education_level)

    def question(
        self,
        *,
        topic: str,
        education_level: str,
        difficulty: int,
        grounded_text: str,
    ) -> Dict[str, Any]:
        if not topic:
            return {
                "question": "Pick a topic first.",
                "options": ["Go to Dashboard and search a topic.", "Stay here"],
                "answer_idx": 0,
                "explanation": "Topics live in the Dashboard in this prototype.",
                "difficulty": 1,
            }
        return self._mock_question(topic, education_level, difficulty, grounded_text)


def _llm_from_sidebar() -> TutorLLM:
    provider = st.session_state.get("llm_provider", "mock")
    provider = provider if provider in {"mock", "gemini"} else "mock"
    model_name = st.session_state.get("llm_model_name", "gemini-1.5-flash")

    cfg = LLMConfig(provider=provider, model_name=model_name, temperature=0.2)
    return TutorLLM(cfg)


def fetch_wikipedia_summary(topic: str, timeout_s: int = 12) -> Tuple[str, str]:
    topic = (topic or "").strip()
    if not topic:
        return "", ""

    title = urllib.parse.quote(topic.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "NeuroLearnPrototype/1.0 (Streamlit; educational prototype)",
            "Accept": "application/json",
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            extract = (data.get("extract") or "").strip()
            page_url = (data.get("content_urls", {}).get("desktop", {}).get("page") or "").strip()
            return extract, page_url
    except Exception:
        return "", ""


def goto(page: str) -> None:
    st.session_state.page = page


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


def normalize(s: str) -> str:
    """Simple normalization to compare user answers (used in handwriting mock)."""
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def add_note(line: str) -> None:
    line = (line or "").strip()
    if line:
        st.session_state.notes_log.append(line)


def compile_notes_markdown() -> str:
    ss = st.session_state
    lines: List[str] = []
    lines.append("# APPL. — AI Notes (Prototype)")
    lines.append("")
    lines.append("## Student Profile")
    lines.append(f"- Name: {ss.get('name','')}")
    lines.append(f"- Username: {ss.get('username','')}")
    lines.append(f"- Education level: {ss.get('education_level','')}")
    lines.append(f"- Target subject: {ss.get('target_subject','')}")
    lines.append("")
    lines.append("## Current Topic")
    lines.append(f"- Topic: {ss.get('topic','')}")
    lines.append("")
    if ss.get("web_material"):
        lines.append("## Grounded Web Material (Summary)")
        lines.append(ss.get("web_material", ""))
        lines.append("")
    lines.append("## Lessons / Tutor Notes")
    if ss.get("notes_log"):
        for i, n in enumerate(ss["notes_log"], 1):
            lines.append(f"{i}. {n}")
    else:
        lines.append("_No notes yet. Generate a lesson in the Tutor._")
    lines.append("")
    lines.append("## Practice History")
    if ss.get("answer_history"):
        for item in ss["answer_history"]:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(item.get("ts", time.time())))
            lines.append(
                f"- [{ts}] Difficulty {item.get('difficulty')} | "
                f"{'Correct' if item.get('correct') else 'Wrong'} | "
                f"Q: {item.get('q')} | "
                f"Your choice: {item.get('chosen')}"
            )
    else:
        lines.append("_No questions answered yet._")
    lines.append("")
    lines.append("## Uploads (Prototype)")
    if ss.get("uploads"):
        for u in ss["uploads"]:
            lines.append(f"- {u.get('name')} ({u.get('type')}, {u.get('size')} bytes, sha={u.get('sha')[:10]}...)")
    else:
        lines.append("_No uploads yet._")
    lines.append("")
    return "\n".join(lines)


with st.sidebar:
    st.markdown("### APPL.")
    page = st.session_state.page
    st.progress({"landing": 0.08, "auth": 0.22, "profile": 0.38, "dashboard": 0.60, "tutor": 0.82, "advanced": 0.95}.get(page, 0.1))
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Home", use_container_width=True):
            goto("landing")
        if st.button("Dashboard", use_container_width=True, disabled=not st.session_state.is_authed):
            goto("dashboard")
    with c2:
        if st.button("Tutor", use_container_width=True, disabled=not st.session_state.is_authed):
            goto("tutor")
        if st.button("Advanced", use_container_width=True, disabled=not st.session_state.is_authed):
            goto("advanced")

    st.session_state.llm_provider = st.selectbox("AI provider", ["mock", "gemini"], help="Gemini uses GOOGLE_API_KEY; mock is deterministic.")
    st.session_state.llm_model_name = st.text_input("Model", value=st.session_state.get("llm_model_name", "gemini-1.5-flash"))

    if st.button("Reset session", type="secondary", use_container_width=True):
        st.session_state.clear()
        _init_state()
        st.rerun()


def page_landing() -> None:
    st.markdown(
        """
<div class="card">
  <p class="title">APPL.</p>
  <p class="muted">An adaptive personalized learning prototype — grounded search, tutoring, and testing.</p>
  <p class="muted">Sign in, pick your level, choose a topic, then learn + practice in an adaptive loop.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        if st.button("Get started", type="primary", use_container_width=True):
            goto("auth")


# ------------------------------------
# 8) PAGE: AUTH (MOCK SIGN IN / SIGN UP)
# ------------------------------------


def page_auth() -> None:
    st.markdown("### Sign in / Sign up")
    st.caption("Prototype auth: stored in memory for this session only.")

    tab_in, tab_up = st.tabs(["Sign In", "Sign Up"])

    with tab_up:
        st.markdown("#### Create account")
        new_user = st.text_input("Username", key="signup_user")
        new_pass = st.text_input("Password", type="password", key="signup_pass")
        if st.button("Create account", use_container_width=True, key="signup_btn"):
            new_user = (new_user or "").strip()
            if not new_user or not new_pass:
                st.error("Please enter a username and password.")
            elif new_user in st.session_state.user_db:
                st.error("That username already exists.")
            else:
                st.session_state.user_db[new_user] = {
                    "password_hash": _hash_password(new_pass),
                    "created_at": time.time(),
                }
                st.success("Account created. Now sign in.")

    with tab_in:
        st.markdown("#### Welcome back")
        user = st.text_input("Username", key="signin_user")
        pw = st.text_input("Password", type="password", key="signin_pass")
        if st.button("Sign in", type="primary", use_container_width=True, key="signin_btn"):
            user = (user or "").strip()
            rec = st.session_state.user_db.get(user)
            if not rec:
                st.error("User not found. Please sign up first.")
            elif rec.get("password_hash") != _hash_password(pw):
                st.error("Incorrect password.")
            else:
                st.session_state.is_authed = True
                st.session_state.username = user
                st.success("Signed in successfully.")
                goto("profile")
                st.rerun()


# ------------------------------------
# 9) PAGE: PROFILE SETUP
# ------------------------------------


def page_profile() -> None:
    st.markdown("### Profile Setup")
    st.caption("These fields personalize explanations and adaptive difficulty.")

    left, right = st.columns([1.2, 1])
    with left:
        name = st.text_input("Your name", value=st.session_state.get("name", ""))
        edu_options = [
            "",
            "School (Middle School)",
            "School (High School)",
            "College (Undergraduate / B.Tech)",
            "Masters (M.Tech / MS)",
            "PhD / Research",
        ]
        cur_edu = st.session_state.get("education_level", "")
        edu = st.selectbox("Education level", options=edu_options, index=(edu_options.index(cur_edu) if cur_edu in edu_options else 0))
        subject = st.text_input(
            "Target subject (e.g., Physics, Biology, Data Structures, Thermodynamics)",
            value=st.session_state.get("target_subject", ""),
        )

        st.session_state.name = name.strip()
        st.session_state.education_level = edu
        st.session_state.target_subject = subject.strip()

        if st.button("Save profile", type="primary", use_container_width=True):
            if not st.session_state.name or not st.session_state.education_level:
                st.error("Please provide your name and education level.")
            else:
                add_note(f"Profile created for {st.session_state.name} ({st.session_state.education_level}).")
                goto("dashboard")
                st.rerun()

    with right:
        st.markdown(
            """
<div class="card">
  <p class="muted"><b>Personalization</b></p>
  <p class="muted">- Education level changes explanation depth.</p>
  <p class="muted">- Correct answers increase difficulty; wrong answers decrease it.</p>
  <p class="muted">- Web summary grounds the tutor to reduce made-up details.</p>
</div>
""",
            unsafe_allow_html=True,
        )


# ------------------------------------
# 10) PAGE: DASHBOARD (UPLOAD + SEARCH)
# ------------------------------------


def _store_upload(file) -> None:
    raw = file.getvalue()
    sha = hashlib.sha256(raw).hexdigest()
    b64 = base64.b64encode(raw).decode("ascii")
    st.session_state.uploads.append(
        {
            "name": file.name,
            "type": file.type or "application/octet-stream",
            "size": len(raw),
            "sha": sha,
            "bytes_b64": b64,
        }
    )


def page_dashboard() -> None:
    st.markdown("### Dashboard")
    st.caption("Upload notes or search a topic online to ground the tutor.")

    st.markdown(
        f"""
<div class="card">
  <p class="muted">Hello <b>{st.session_state.get("name") or "student"}</b> • Level: <b>{st.session_state.get("education_level") or "Not set"}</b> • Subject: <b>{st.session_state.get("target_subject") or "General"}</b></p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")
    up_col, search_col = st.columns([1.1, 1])

    with up_col:
        st.markdown("#### Upload notes (PDF) or homework image")
        uploads = st.file_uploader(
            "Upload files",
            type=["pdf", "png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            help="PDF parsing / OCR is mocked (standard library only).",
        )
        if uploads:
            for f in uploads:
                raw = f.getvalue()
                sha = hashlib.sha256(raw).hexdigest()
                if any(u.get("sha") == sha for u in st.session_state.uploads):
                    continue
                _store_upload(f)
            st.success(f"Stored {len(uploads)} file(s) in your session.")

        if st.session_state.uploads:
            st.markdown("##### Your uploads (session)")
            for idx, u in enumerate(st.session_state.uploads):
                with st.expander(f"{u['name']} • {u['type']} • {u['size']} bytes", expanded=False):
                    if u["type"].startswith("image/"):
                        img_bytes = base64.b64decode(u["bytes_b64"].encode("ascii"))
                        st.image(img_bytes, caption=u["name"], use_container_width=True)
                    else:
                        st.info(
                            "PDF parsing isn't enabled here (standard library only). The file is stored for future parsing."
                        )
                    if st.button("Remove this upload", key=f"rm_upload_{idx}"):
                        st.session_state.uploads.pop(idx)
                        st.rerun()

    with search_col:
        st.markdown("#### Online search (grounding)")
        topic = st.text_input("Topic to fetch (e.g., Thermodynamics)", value=st.session_state.get("topic", ""))
        st.session_state.topic = topic.strip()
        if st.button("Fetch lesson from the web", type="primary", use_container_width=True):
            if not st.session_state.topic:
                st.error("Please enter a topic.")
            else:
                with st.spinner("Fetching summary..."):
                    summary, url = fetch_wikipedia_summary(st.session_state.topic)
                if summary:
                    st.session_state.web_material = summary
                    add_note(f"Fetched web summary for topic: {st.session_state.topic}")
                    if url:
                        st.caption(f"Source: {url}")
                    st.success("Web material fetched and saved.")
                else:
                    st.warning("Could not fetch material. Try a different topic name.")

        if st.session_state.web_material:
            st.markdown("##### Saved web summary")
            st.write(st.session_state.web_material)

        st.write("")
        if st.button("Go to Adaptive Tutor", use_container_width=True):
            goto("tutor")
            st.rerun()


# ------------------------------------
# 11) PAGE: TUTOR (TEACH + ADAPTIVE LOOP)
# ------------------------------------


def page_tutor() -> None:
    st.markdown("### Adaptive Tutor")
    st.caption("Teach → visualize → test → adapt difficulty based on your answers.")

    llm = _llm_from_sidebar()
    ss = st.session_state

    top_left, top_right = st.columns([1.25, 0.85])
    with top_left:
        st.markdown("#### Concept teaching")
        if not ss.topic:
            st.warning("No topic selected yet. Go to Dashboard and search a topic first.")
            if st.button("Open Dashboard", type="primary"):
                goto("dashboard")
                st.rerun()
            return

        ss.difficulty = st.slider("Current difficulty", min_value=1, max_value=5, value=int(ss.difficulty))

        if st.button("Teach me this topic", type="primary", use_container_width=True):
            lesson = llm.summarize(
                topic=ss.topic,
                education_level=ss.education_level,
                difficulty=int(ss.difficulty),
                grounded_text=ss.web_material,
                target_subject=ss.target_subject,
            )
            ss.last_lesson = lesson
            add_note(f"Lesson generated for '{ss.topic}' at difficulty {ss.difficulty}.")

            ss.visual_prompt = llm.visual_prompt(topic=ss.topic, education_level=ss.education_level)
            add_note("Created a diagram prompt for visual learning.")

        if ss.last_lesson:
            st.markdown("##### Your lesson")
            st.write(ss.last_lesson)

    with top_right:
        st.markdown("#### Visual learning (placeholder)")
        if ss.visual_prompt:
            st.info(ss.visual_prompt)
        else:
            st.info("Click **Teach me this topic** to generate a diagram prompt.")

        st.markdown("#### Adaptive testing loop")

        if st.button("Generate a question", use_container_width=True):
            ss.active_question = llm.question(
                topic=ss.topic,
                education_level=ss.education_level,
                difficulty=int(ss.difficulty),
                grounded_text=ss.web_material,
            )
            st.rerun()

        q = ss.active_question
        if q:
            st.markdown("##### Question")
            st.write(q["question"])

            choice = st.radio("Choose one answer", options=q["options"], key="mcq_choice")

            if st.button("Submit answer", type="primary", use_container_width=True):
                chosen_idx = q["options"].index(choice)
                correct = chosen_idx == int(q["answer_idx"])

                ss.answer_history.append(
                    {
                        "q": q["question"],
                        "chosen": choice,
                        "correct": bool(correct),
                        "difficulty": int(ss.difficulty),
                        "ts": time.time(),
                    }
                )

                if correct:
                    st.success("Correct! Nice work — leveling up.")
                    add_note(f"Answered correctly at difficulty {ss.difficulty}. Moving to a deeper level.")
                    ss.difficulty = clamp_int(int(ss.difficulty) + 1, 1, 5)
                    add_note(f"Difficulty increased to {ss.difficulty}.")
                else:
                    st.error("Not quite. Let’s simplify and try again.")
                    add_note(f"Answered incorrectly at difficulty {ss.difficulty}. Simplifying the explanation.")
                    ss.difficulty = clamp_int(int(ss.difficulty) - 1, 1, 5)
                    add_note(f"Difficulty decreased to {ss.difficulty}.")

                st.markdown("**Explanation**")
                st.write(q.get("explanation", ""))

                if not correct:
                    ss.active_question = llm.question(
                        topic=ss.topic,
                        education_level=ss.education_level,
                        difficulty=int(ss.difficulty),
                        grounded_text=ss.web_material,
                    )
                else:
                    ss.active_question = llm.question(
                        topic=ss.topic,
                        education_level=ss.education_level,
                        difficulty=int(ss.difficulty),
                        grounded_text=ss.web_material,
                    )
                st.rerun()

        st.write("")
        st.markdown("#### Quick actions")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Back to Dashboard", use_container_width=True):
                goto("dashboard")
                st.rerun()
        with c2:
            if st.button("Advanced features", use_container_width=True):
                goto("advanced")
                st.rerun()


# ------------------------------------
# 12) PAGE: ADVANCED (DOWNLOAD NOTES + HANDWRITING)
# ------------------------------------


def page_advanced() -> None:
    st.markdown("### Advanced Features")
    st.caption("Prototype implementations: downloadable notes and mocked handwriting grading.")

    left, right = st.columns([1.05, 0.95])

    with left:
        st.markdown("#### Note generator")
        notes_md = compile_notes_markdown()
        st.download_button(
            label="Download AI Notes (Markdown)",
            data=notes_md.encode("utf-8"),
            file_name="neurolearn_ai_notes.md",
            mime="text/markdown",
            use_container_width=True,
        )

        with st.expander("Preview notes", expanded=False):
            st.markdown(notes_md)

    with right:
        st.markdown("#### Handwriting analysis (mock grading)")
        st.write(
            "Upload an image of your written answer. OCR isn't included here, so use the typed answer for grading."
        )

        img = st.file_uploader("Upload handwriting image", type=["png", "jpg", "jpeg", "webp"], key="hw_img")
        typed = st.text_area(
            "Type your answer (optional, used for grading in this prototype)",
            placeholder="Example: The answer is ...",
            key="hw_typed",
        )

        if img is not None:
            st.image(img.getvalue(), caption=img.name, use_container_width=True)

        q = st.session_state.get("active_question")
        expected = ""
        if q and isinstance(q, dict) and q.get("options"):
            expected = q["options"][int(q["answer_idx"])]

        if st.button("Grade my answer", type="primary", use_container_width=True):
            if not expected:
                st.warning("No active question found. Go to Tutor and generate a question first.")
            else:
                a = normalize(typed)
                b = normalize(expected)
                if not a:
                    score = 0.0
                else:
                    score = difflib.SequenceMatcher(a=a, b=b).ratio()

                score_pct = int(round(score * 100))
                st.session_state.handwriting_last_score = score_pct

                if score_pct >= 85:
                    fb = "Great! Your answer closely matches the expected concept."
                elif score_pct >= 60:
                    fb = "Good attempt. You’re close—review the key idea and try to be more precise."
                else:
                    fb = "Needs improvement. Re-read the lesson and try answering in simpler, clearer steps."

                st.session_state.handwriting_last_feedback = fb
                add_note(f"Handwriting analysis score: {score_pct}% (prototype, based on typed answer similarity).")

        if st.session_state.handwriting_last_score is not None:
            st.markdown("##### Result")
            st.write(f"Score: **{st.session_state.handwriting_last_score}%**")
            st.write(st.session_state.handwriting_last_feedback)

        st.write("")
        if st.button("Return to Tutor", use_container_width=True):
            goto("tutor")
            st.rerun()


def enforce_access() -> None:
    ss = st.session_state

    if not ss.is_authed and ss.page not in {"landing", "auth"}:
        ss.page = "auth"

    profile_ok = bool(ss.get("name")) and bool(ss.get("education_level"))
    if ss.is_authed and not profile_ok and ss.page not in {"landing", "auth", "profile"}:
        ss.page = "profile"


def main() -> None:
    enforce_access()

    col_a, col_b = st.columns([1.2, 0.8])
    with col_a:
        st.markdown("### APPL.")
    with col_b:
        if st.session_state.get("is_authed"):
            st.caption(f"Signed in as **{st.session_state.get('username','')}**")

    page = st.session_state.page
    if page == "landing":
        page_landing()
    elif page == "auth":
        page_auth()
    elif page == "profile":
        page_profile()
    elif page == "dashboard":
        page_dashboard()
    elif page == "tutor":
        page_tutor()
    elif page == "advanced":
        page_advanced()
    else:
        st.session_state.page = "landing"
        page_landing()


if __name__ == "__main__":
    main()
