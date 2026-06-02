"""
Streamlit demo for the school-bullying RAG.

Layout: a question box (with example chips), the generated answer, and an
expandable "behind the scenes" panel showing the retrieved chunks with
their scores and metadata. Designed for recruiters and the curious — the
goal is to show HOW the system works, not just THAT it works.

Lightweight by design: dense retrieval only (no reranker, no hybrid) so
that the demo stays responsive on a free Hugging Face Space. The full
project on GitHub documents and measures the alternative configurations.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from generate import answer as rag_answer

# Local copies of the project modules, sitting next to this file.
from retrieve import Retriever

# --- Configuration -----------------------------------------------------------
GITHUB_URL = "https://github.com/YOUR_HANDLE/YOUR_REPO"  # adjust before deploy
MAX_REQUESTS_PER_SESSION = 5  # rate limit to protect the API budget
DEFAULT_K = 5

EXAMPLE_QUESTIONS = [
    "Combien d'enfants sont touchés par le harcèlement scolaire chaque année ?",
    "La loi cible-t-elle aussi le harcèlement venant des adultes ?",
    "Qu'a dit le ministre Blanquer sur la formation des personnels ?",
    "Quels sont les arguments des détracteurs de la loi ?",
    # An out-of-corpus example, to showcase the guardrail
    "Quel était le budget de l'Éducation Nationale en 2019 ?",
]


# --- Resource loading (cached so it runs once per Space restart) -------------
@st.cache_resource
def get_retriever() -> Retriever:
    """Load the dense retriever once and keep it in memory."""
    return Retriever()


def format_date(iso: str) -> str:
    return f"{iso[:4]}-{iso[4:6]}-{iso[6:8]}" if len(iso) == 8 else iso


# --- Session state ----------------------------------------------------------
def init_session() -> None:
    if "request_count" not in st.session_state:
        st.session_state.request_count = 0
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = ""


def pick_example(q: str) -> None:
    """Callback for example buttons: fills the input on next rerun."""
    st.session_state.pending_question = q


# --- UI ---------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="RAG — débats Assemblée nationale", page_icon="📜", layout="wide"
    )
    init_session()

    st.title("📜 RAG sur les débats parlementaires")
    st.markdown(
        "**Démo interactive** d'un système RAG construit sur les débats de "
        "l'Assemblée nationale concernant la **loi contre le harcèlement "
        "scolaire** (loi n°2022-299). "
        "Le système répond à partir de 3 séances (~1800 passages indexés) "
        "en citant ses sources, et refuse explicitement quand l'information "
        "n'est pas dans le corpus."
    )
    st.markdown(f"📂 [Code complet et documentation sur GitHub]({GITHUB_URL})")
    st.divider()

    # Example questions as buttons
    st.markdown(
        "**Exemples de questions** _(la dernière est volontairement hors-corpus, pour montrer le garde-fou)_ :"
    )
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, ex in zip(cols, EXAMPLE_QUESTIONS):
        col.button(
            ex[:45] + ("…" if len(ex) > 45 else ""),
            key=f"ex_{hash(ex)}",
            on_click=pick_example,
            args=(ex,),
            use_container_width=True,
        )

    # Question input — uses the pending example if any
    question = st.text_input(
        "Votre question :",
        value=st.session_state.pending_question,
        placeholder="Ex. : Comment la loi définit-elle le harcèlement scolaire ?",
    )
    # Reset the pending state so editing works normally afterwards
    st.session_state.pending_question = ""

    if not st.button("Interroger le RAG", type="primary"):
        return
    if not question.strip():
        st.warning("Saisissez une question.")
        return

    # Soft rate limit per session
    if st.session_state.request_count >= MAX_REQUESTS_PER_SESSION:
        st.error(
            f"Limite de {MAX_REQUESTS_PER_SESSION} requêtes par session "
            "atteinte (protection du budget API de la démo). "
            "Rechargez la page pour reprendre, ou clonez le repo pour "
            "tourner l'app sans limite."
        )
        return

    # Run the actual RAG turn
    retriever = get_retriever()
    with st.spinner("Recherche puis génération…"):
        try:
            generated, hits = rag_answer(
                question,
                k=DEFAULT_K,
                exclude_procedural=True,
                retriever=retriever,
            )
        except Exception as e:
            st.error(f"Erreur pendant la génération : {e}")
            return
    st.session_state.request_count += 1

    # --- Answer ------------------------------------------------------------
    st.markdown("### Réponse")
    st.markdown(generated)

    # --- Behind-the-scenes panel ------------------------------------------
    with st.expander(
        f"🔍 Voir les {len(hits)} passages utilisés " f"(coulisses du RAG)",
        expanded=False,
    ):
        st.caption(
            "Ces passages sont récupérés par similarité sémantique "
            "(embeddings `multilingual-e5-base`, distance cosinus) avant "
            "d'être donnés au modèle de génération. "
            "Le système ne voit que ces passages — d'où sa fidélité "
            "(pas d'hallucination) et ses refus quand la réponse n'y est pas."
        )
        for h in hits:
            who = h.speaker or "—"
            if h.role:
                who += f" ({h.role})"
            st.markdown(
                f"**#{h.rank} · score {h.score:.3f}**  \n"
                f"_{format_date(h.date_iso)} — {who} — {h.agenda_item}_"
            )
            st.markdown(f"> {h.text}")
            st.markdown("")  # spacer

    # --- Footer ------------------------------------------------------------
    st.caption(
        f"Requêtes utilisées cette session : "
        f"{st.session_state.request_count}/{MAX_REQUESTS_PER_SESSION}"
    )


if __name__ == "__main__":
    main()
