import json
import re
import time
from pathlib import Path

import fitz
import streamlit as st
from google import genai

PDF_PATH = Path("nism_mutual_fund.pdf")
CHUNK_SIZE = 1500
BATCH_SIZE = 5  # questions per request


@st.cache_data(show_spinner="Reading PDF...")
def load_chunks(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text("text") + "\n"
    text = re.sub(r"\s+", " ", text).strip()
    return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]


def find_chunks(chunks, topic, top_k=24):
    words = [w for w in re.findall(r"\w+", topic.lower()) if len(w) > 2]
    scored = [(sum(c.lower().count(w) for w in words), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for s, c in scored[:top_k] if s > 0]


def valid(q):
    try:
        return (
            q["question"].strip()
            and set(q["options"].keys()) == set("ABCD")
            and all(str(q["options"][k]).strip() for k in "ABCD")
            and q["correct_answer"] in list("ABCD")
            and q["explanation"].strip()
        )
    except Exception:
        return False


def make_batch(api_key, model, context, topic, difficulty, n, avoid):
    client = genai.Client(api_key=api_key)
    avoid_text = "\n".join("- " + a for a in avoid[-30:])
    prompt = f"""Create {n} different {difficulty} multiple-choice questions on "{topic}"
using ONLY the text below. Each question must be about a different fact.
Do NOT repeat these existing questions:
{avoid_text}

Reply as JSON: {{"questions": [{{"question": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "correct_answer": "A", "explanation": "..."}}]}}

TEXT:
{context}"""
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config={"response_mime_type": "application/json", "temperature": 0.4},
    )
    raw = re.sub(r"^```json|^```|```$", "", resp.text.strip()).strip()
    data = json.loads(raw)
    items = data["questions"] if isinstance(data, dict) else data
    return [q for q in items if valid(q)]


def call_with_retry(*args):
    for attempt in range(4):
        try:
            return make_batch(*args)
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < 3:
                time.sleep(5 * (attempt + 1))
            else:
                raise


st.title("📘 NISM Mutual Fund Practice Agent")
st.caption("Unofficial practice questions. Always verify with the official NISM book.")

api_key = st.sidebar.text_input("Gemini API key", type="password")
model = st.sidebar.text_input("Model", "gemini-3.8-flash")

if not PDF_PATH.exists():
    st.error(f"Put your PDF in the same folder and name it: {PDF_PATH}")
    st.stop()

chunks = load_chunks(str(PDF_PATH))
st.sidebar.write(f"PDF loaded: {len(chunks)} sections")

topic = st.text_input("Topic", "NAV")
difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
total = st.number_input("Number of questions", min_value=5, max_value=50, value=50, step=5)

if st.button("Generate questions"):
    if not api_key:
        st.error("Enter your Gemini API key in the left sidebar.")
    else:
        found = find_chunks(chunks, topic)
        if not found:
            st.error("Topic not found in the PDF. Try another word.")
        else:
            questions, seen = [], []
            batches = (int(total) + BATCH_SIZE - 1) // BATCH_SIZE
            bar = st.progress(0.0, text="Starting...")
            try:
                for i in range(batches):
                    # use different parts of the PDF for each batch
                    start = (i * 2) % len(found)
                    part = [found[(start + j) % len(found)] for j in range(2)]
                    new = call_with_retry(
                        api_key, model, "\n\n".join(part), topic, difficulty, BATCH_SIZE, seen
                    )
                    for q in new:
                        if q["question"] not in seen:
                            seen.append(q["question"])
                            questions.append(q)
                    bar.progress((i + 1) / batches, text=f"{len(questions)} questions ready...")
                    time.sleep(2)
            except Exception as e:
                st.warning(f"Stopped early: {e}")
            bar.empty()
            st.session_state.questions = questions[: int(total)]
            st.session_state.submitted = False
            for key in [k for k in st.session_state if str(k).startswith("q_")]:
                del st.session_state[key]

questions = st.session_state.get("questions", [])
if questions:
    st.subheader(f"Quiz: {len(questions)} questions")
    with st.form("quiz"):
        for i, q in enumerate(questions):
            st.markdown(f"**Q{i + 1}. {q['question']}**")
            st.radio(
                "Your answer",
                list("ABCD"),
                index=None,
                format_func=lambda k, q=q: f"{k}. {q['options'][k]}",
                key=f"q_{i}",
                label_visibility="collapsed",
            )
        if st.form_submit_button("Submit answers"):
            st.session_state.submitted = True

    if st.session_state.get("submitted"):
        score = 0
        for i, q in enumerate(questions):
            if st.session_state.get(f"q_{i}") == q["correct_answer"]:
                score += 1
        st.success(f"Score: {score} / {len(questions)}  ({score / len(questions) * 100:.0f}%)")
        for i, q in enumerate(questions):
            mine = st.session_state.get(f"q_{i}")
            ok = mine == q["correct_answer"]
            st.markdown(
                f"{'✅' if ok else '❌'} **Q{i + 1}.** Your answer: {mine or 'none'} | "
                f"Correct: {q['correct_answer']}"
            )
            st.caption(q["explanation"])