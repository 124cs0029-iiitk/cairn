import time
from datetime import datetime

import streamlit as st

from core import db
from core.citations import build_citation
from core.concepts import load_concept_graph
from core.config import get_settings
from core.history import finish_quiz, list_quizzes, load_review, save_answer, start_quiz
from core.ingest import load_manifest
from core.learner import asked_ids, assessment_report, load_model, log_attempt, save_model
from core.llm import LLMError, chat
from core.questions import grade_answer, load_bank
from core.quiz import MODES, QuizSession, build_quiz
from core.retrieval import load_chunks
from core.ui import learner_sidebar, page_header, render_citation, source_link

st.set_page_config(page_title="Quiz - Cairn", page_icon="🪨", layout="wide")
page_header("Quiz", "Questions on the topics you choose, checked against the course material, with feedback you can trace to its source.")

settings = get_settings()
kb = settings.kb_dir
if not (kb / "concepts.json").exists() or not (kb / "questions.json").exists():
    st.warning("The question bank has not been built yet. Run `python scripts/build_concepts.py` and `python scripts/build_questions.py`.")
    st.stop()


@st.cache_data(ttl=60, show_spinner=False)
def load_resources():
    concepts, _ = load_concept_graph(kb / "concepts.json")
    bank = [q for q in load_bank(kb / "questions.json") if q.verified]
    chunks = {c.id: c for c in load_chunks(kb / "chunks.jsonl")}
    _, sources = load_manifest(settings.root)
    return concepts, bank, chunks, {s.id: s for s in sources}


concepts, bank, chunks, sources = load_resources()
concept_of = {c.id: c for c in concepts}
bank_by_id = {q.id: q for q in bank}
if not bank:
    st.warning("There are no verified questions yet.")
    st.stop()

learner_id = learner_sidebar()
if not learner_id:
    st.info("Enter your name in the sidebar to start. Your progress is saved under that name.")
    st.stop()

with db.session() as conn:
    db.get_or_create_learner(conn, learner_id, st.session_state["learner_name"])
    model = load_model(conn, learner_id, concepts)
    asked = asked_ids(conn, learner_id)
    summaries = list_quizzes(conn, learner_id)

st.session_state.setdefault("quiz_view", "Take a quiz")
st.radio("View", ["Take a quiz", "History"], horizontal=True, key="quiz_view", label_visibility="collapsed")
key = f"quiz_{learner_id}"
quiz: QuizSession | None = st.session_state.get(key)


def citation_for(question):
    chunk = chunks[question.chunk_id]
    return build_citation(chunk, sources[chunk.source_id]), sources[chunk.source_id]


def open_review(quiz_id: str) -> None:
    st.session_state["review_id"] = quiz_id
    st.session_state["quiz_view"] = "History"


def describe(s) -> str:
    when = datetime.fromtimestamp(s.created_at).strftime("%d %b %H:%M")
    state = f"{s.correct} of {s.total} correct" + (f", {s.skipped} skipped" if s.skipped else "")
    return f"{when} | {MODES.get(s.mode, s.mode)} | {state}" + ("" if s.finished else " | unfinished")


# ---- history and review -------------------------------------------------------------------------
def show_review(review) -> None:
    s = review.summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score", f"{s.correct} of {s.total}")
    c2.metric("Answered", s.answered)
    c3.metric("Skipped", s.skipped if s.finished else "-")
    c4.metric("Accuracy of answered", f"{100 * s.correct / s.answered:.0f}%" if s.answered else "-")
    if not s.finished:
        st.caption("This quiz was not finished.")
    for item in review.items:
        q = item.question
        if q is None:
            st.caption(f"Question {item.position + 1} is no longer in the question bank.")
            continue
        status = "Skipped" if not item.answered else "Correct" if item.correct else "Incorrect"
        with st.expander(f"Question {item.position + 1}: {status} | {q.stem}"):
            if q.qtype == "mcq":
                chosen = int(item.response) if item.answered and item.response not in (None, "") else None
                for i, option in enumerate(q.options):
                    notes = [text for flag, text in ((i == q.answer_index, "correct answer"), (i == chosen, "your answer")) if flag]
                    st.markdown(f"{'ABCD'[i]}. {option}" + (f"  *({', '.join(notes)})*" if notes else ""))
            else:
                st.markdown(f"**Your answer:** {item.response or 'No answer'}")
                st.markdown(f"**Model answer:** {q.answer}")
            explanation = item.feedback or q.explanation
            if explanation:
                st.markdown(f"**Explanation:** {explanation}")
            source_link(citation_for(q)[0], key=f"review_{s.id}_{item.position}")


if st.session_state["quiz_view"] == "History":
    if not summaries:
        st.info("No quizzes yet. Take one and it will appear here, with every answer and explanation.")
        st.stop()
    labels = {s.id: describe(s) for s in summaries}
    ids = list(labels)
    wanted = st.session_state.get("review_id")
    chosen = st.selectbox("Choose a quiz to review", ids, index=ids.index(wanted) if wanted in ids else 0,
                          format_func=labels.get)
    with db.session() as conn:
        review = load_review(conn, chosen, bank_by_id)
    show_review(review)
    st.stop()

# ---- set up a quiz ------------------------------------------------------------------------------
if quiz is None:
    if not asked:
        st.info("You are new here, so a diagnostic quiz is the best start: about 10 questions across all topics, "
                "so the tutor can see where you stand.")
    order = ["diagnostic", "practice", "exam"] if not asked else ["practice", "exam", "diagnostic"]
    mode = st.radio("Quiz type", order, format_func=MODES.get, horizontal=True)
    topics = sorted({c.topic for c in concepts})
    chosen_topics = st.multiselect("Topics to cover", topics, default=topics)
    count = st.slider("Number of questions", 3, 15, 8) if mode != "diagnostic" else 10
    if st.button("Start", type="primary"):
        questions = build_quiz(mode, model, bank, concepts, count, time.time(), chosen_topics, asked)
        if questions:
            with db.session() as conn:
                quiz_id = start_quiz(conn, learner_id, mode, questions, time.time())
            st.session_state[key] = QuizSession(questions, immediate_feedback=mode != "exam", quiz_id=quiz_id)
            st.rerun()
        else:
            st.error("There are no unseen questions left for these topics. Choose more topics.")
    st.stop()

total = len(quiz.questions)

# ---- finished: score and report -----------------------------------------------------------------
if quiz.finished:
    report = assessment_report(quiz.results, model, time.time())
    st.subheader("Your results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Score", f"{quiz.n_correct} of {total}")
    c2.metric("Answered", quiz.n_answered)
    c3.metric("Skipped", total - quiz.n_answered)
    for topic, (right, count_) in report.by_topic.items():
        st.progress(right / count_, text=f"{topic}: {right} of {count_}")
    if report.weak_concepts:
        st.markdown("**Concepts to work on:** " + ", ".join(report.weak_concepts))
    if report.misconceptions:
        st.markdown("**Likely misconceptions**")
        for m in report.misconceptions:
            st.warning(f"**{m['concept']}**: for \"{m['question']}\" you chose \"{m['chose']}\"; the correct answer is \"{m['correct']}\".")
    if report.review_next:
        st.markdown("**Review next:** " + ", ".join(report.review_next))
    left, right = st.columns(2)
    left.button("Review every answer", type="primary", on_click=open_review, args=(quiz.quiz_id,))
    if right.button("Start another quiz"):
        del st.session_state[key]
        st.rerun()
    st.stop()

# ---- ready to finish? ---------------------------------------------------------------------------
if quiz.at_review:
    st.subheader("Ready to finish?")
    st.write(f"You have answered {quiz.n_answered} of {total} questions.")
    if quiz.unanswered:
        st.warning("Skipped: question " + ", ".join(str(i + 1) for i in quiz.unanswered)
                   + ". Skipped questions do not count against you, but you can still answer them.")
        if st.button("Go to the first skipped question"):
            quiz.go_to(quiz.unanswered[0])
            st.rerun()
    back, done = st.columns(2)
    if back.button("Previous"):
        quiz.previous()
        st.rerun()
    if done.button("Finish quiz", type="primary"):
        with db.session() as conn:
            finish_quiz(conn, quiz.quiz_id, time.time())
        quiz.finish()
        st.rerun()
    st.stop()

# ---- one question -------------------------------------------------------------------------------
i = quiz.index
q = quiz.current
answer = quiz.answer_at(i)
concept = concept_of.get(q.concept_id)
st.progress(quiz.n_answered / total, text=f"Question {i + 1} of {total}  |  {quiz.n_answered} answered")
st.caption(f"Topic: {concept.topic if concept else '-'}  |  Concept: {concept.name if concept else q.concept_id}  |  "
           f"Difficulty {q.difficulty} of 5  |  {'Multiple choice' if q.qtype == 'mcq' else 'Short answer'}")
st.subheader(q.stem)

widget_key = f"ans_{quiz.quiz_id}_{q.id}"
if answer is not None and widget_key not in st.session_state:
    st.session_state[widget_key] = answer.response            # coming back to a question already answered

# One form per question: typed text and the button click reach the app together (no Ctrl+Enter needed).
with st.form(f"question_{quiz.quiz_id}_{i}", border=False):
    if q.qtype == "mcq":
        response = st.radio("Choose one answer", range(4), index=None, key=widget_key, disabled=answer is not None,
                            format_func=lambda n: f"{'ABCD'[n]}. {q.options[n]}")
        ready = response is not None
    else:
        response = st.text_area("Your answer", key=widget_key, disabled=answer is not None, height=110)
        ready = bool(response.strip())
    if answer is not None:
        if quiz.immediate_feedback:
            (st.success if answer.grade.correct else st.error)("Correct" if answer.grade.correct else "Not quite")
            st.markdown(answer.grade.feedback)
            citation, source = citation_for(q)
            render_citation(citation, source)
        else:
            st.info("Your answer is recorded. You will see the results when you finish.")
    col_prev, col_skip, col_main, _ = st.columns([1, 1, 1.4, 4])
    go_previous = col_prev.form_submit_button("Previous", disabled=i == 0)
    go_skip = col_skip.form_submit_button("Skip", disabled=answer is not None)
    go_main = col_main.form_submit_button(
        "Submit answer" if answer is None else "Review and finish" if i == total - 1 else "Next", type="primary")

if go_previous:
    quiz.previous()
    st.rerun()
elif go_skip:
    quiz.skip()
    st.rerun()
elif go_main:
    if answer is not None:
        quiz.next()
        st.rerun()
    elif not ready:
        st.warning("Choose an answer first, or press Skip." if q.qtype == "mcq"
                   else "Write your answer first, or press Skip.")
    else:
        try:
            with st.spinner("Checking your answer..."):
                grade = grade_answer(q, response, chunks[q.chunk_id], chat)
        except (LLMError, ValueError) as exc:
            st.error(f"Could not check the answer right now: {exc}")
            st.stop()
        now = time.time()
        model.observe_answer(q, grade.correct, now)
        with db.session() as conn:
            save_model(conn, learner_id, model)
            log_attempt(conn, learner_id, q, grade.correct, response, quiz.quiz_id, now)
            save_answer(conn, quiz.quiz_id, i, q, response, grade, now)
        quiz.submit(response, grade)
        st.rerun()