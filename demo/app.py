"""
Streamlit demo for the school-bullying RAG.

Layout: a question box (with example chips), the generated answer, and an
expandable "behind the scenes" panel showing the retrieved chunks with
their scores and metadata. Designed for recruiters and the curious — the
goal is to show HOW the system works, not just THAT it works.

Lightweight by design: dense retrieval only (no reranker, no hybrid) so
that the demo stays responsive on a free Hugging Face Space. The full
project on GitHub documents and measures the alternative configurations.

The Chroma index is NOT versioned (too large for plain Git on the Space).
Instead, the demo ships with `chunks.jsonl` and rebuilds the index on the
first startup of the Space, then reuses it for subsequent sessions as
long as the container persists. Worst-case extra wait on a cold start:
a few minutes for embedding 1838 chunks on CPU.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

# Local copies of the project modules, sitting next to this file.
from src.retrieve import Retriever
from src.generate import answer as rag_answer
from src.embed_and_index import main as build_index, DB_PATH, COLLECTION

# --- Configuration -----------------------------------------------------------
GITHUB_URL = "https://github.com/YMALOT/RAG-assemblee"
MAX_REQUESTS_PER_SESSION = 5
DEFAULT_K = 5
CHUNKS_PATH = "src/chunks.jsonl"

EXAMPLE_QUESTIONS = [
    "Combien d'enfants sont touchés par le harcèlement scolaire chaque année ?",
    "La loi cible-t-elle aussi le harcèlement venant des adultes ?",
    "Qu'a dit le ministre Blanquer sur la formation des personnels ?",
    "Quels sont les arguments des détracteurs de la loi ?",
    "Quel était le budget de l'Éducation Nationale en 2019 ?",
]


# --- Index bootstrap --------------------------------------------------------
def index_ready() -> bool:
    """Heuristic: the index is considered ready if the Chroma directory
    exists and the named collection holds vectors."""
    if not Path(DB_PATH).exists():
        return False
    try:
        import chromadb
        client = chromadb.PersistentClient(path=DB_PATH)
        coll = client.get_collection(COLLECTION)
        return coll.count() > 0
    except Exception:
        return False


@st.cache_resource
def get_retriever() -> Retriever:
    """Build the index on first call if missing, then load the retriever.
    Cached as a resource: the cost is paid once per Space lifecycle."""
    if not index_ready():
        with st.spinner(
            "Première utilisation depuis le déploiement : indexation du "
            "corpus en cours (calcul des embeddings sur CPU, ~3-5 minutes). "
            "Les visites suivantes seront immédiates."
        ):
            build_index(CHUNKS_PATH)
    return Retriever()


# --- Helpers ----------------------------------------------------------------
def format_date(iso: str) -> str:
    return f"{iso[:4]}-{iso[4:6]}-{iso[6:8]}" if len(iso) == 8 else iso


def init_session() -> None:
    if "request_count" not in st.session_state:
        st.session_state.request_count = 0
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = ""


def pick_example(q: str) -> None:
    st.session_state.pending_question = q


# --- UI ---------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="RAG — débats Assemblée nationale",
                       page_icon="📜", layout="wide")
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

    # Force-load (and possibly build) the retriever before the UI accepts
    # input. This makes the cold-start cost explicit and one-shot.
    retriever = get_retriever()

    st.markdown("**Exemples de questions** _(la dernière est volontairement "
                "hors-corpus, pour montrer le garde-fou)_ :")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, ex in zip(cols, EXAMPLE_QUESTIONS):
        col.button(ex[:45] + ("…" if len(ex) > 45 else ""),
                   key=f"ex_{hash(ex)}",
                   on_click=pick_example, args=(ex,),
                   use_container_width=True)

    question = st.text_input(
        "Votre question :",
        value=st.session_state.pending_question,
        placeholder="Ex. : Comment la loi définit-elle le harcèlement scolaire ?",
    )
    st.session_state.pending_question = ""

    if not st.button("Interroger le RAG", type="primary"):
        return
    if not question.strip():
        st.warning("Saisissez une question.")
        return

    if st.session_state.request_count >= MAX_REQUESTS_PER_SESSION:
        st.error(
            f"Limite de {MAX_REQUESTS_PER_SESSION} requêtes par session "
            "atteinte (protection du budget API de la démo). "
            "Rechargez la page pour reprendre, ou clonez le repo pour "
            "tourner l'app sans limite."
        )
        return

    with st.spinner("Recherche puis génération…"):
        try:
            generated, hits = rag_answer(
                question, k=DEFAULT_K, exclude_procedural=True,
                retriever=retriever,
            )
        except Exception as e:
            st.error(f"Erreur pendant la génération : {e}")
            return
    st.session_state.request_count += 1

    st.markdown("### Réponse")
    st.markdown(generated)

    with st.expander(f"🔍 Voir les {len(hits)} passages utilisés "
                     f"(coulisses du RAG)", expanded=False):
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
            st.markdown("")

    st.caption(
        f"Requêtes utilisées cette session : "
        f"{st.session_state.request_count}/{MAX_REQUESTS_PER_SESSION}"
    )


if __name__ == "__main__":
    main()