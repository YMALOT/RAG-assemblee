"""
Ingest National Assembly "comptes rendus" (syceron XML) into a flat list of
structured interventions enriched with per-session metadata. This is the corpus
builder for the RAG project.

Corpus: PPL "Combattre le harcèlement scolaire" (loi n° 2022-299), 3 sessions:
  - CRSANR5L15S2022O1N080.xml  (1 Dec 2021,  first reading)
  - CRSANR5L15S2022O1N154.xml  (10 Feb 2022, new reading, post-CMP failure)
  - CRSANR5L15S2022O1N165.xml  (24 Feb 2022, final reading)

Code and identifiers in English; French is kept only for the XML schema tag
names imposed by the source and for the corpus text itself.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

NAMESPACE = {"an": "http://schemas.assemblee-nationale.fr/referentiel"}

# code_style values seen in the data. "NORMAL" = actual speech;
# "Info Italiques" / "Signature droite" = procedural stage directions
# (session open/close, amendment vote results, applause). Kept but flagged.
PROCEDURAL_STYLES = {"Info Italiques", "Signature droite"}


@dataclass
class SessionMeta:
    """Metadata describing one sitting (séance), shared by all its interventions."""

    uid: str  # e.g. "CRSANR5L15S2022O1N080" — unique session id
    date_iso: str  # sortable, e.g. "20211201"
    date_label: str  # human-readable, e.g. "mercredi 01 décembre 2021"
    session: str  # e.g. "Session ordinaire 2021-2022"
    legislature: str  # e.g. "15"


@dataclass
class Intervention:
    """A single contiguous turn of speech (or a procedural note)."""

    speaker: str | None  # e.g. "M. Erwan Balanant"; None for stage directions
    role: str | None  # the orateur's <qualite>, e.g. "rapporteur..."
    text: str  # flattened plain text of the turn
    agenda_item: str  # title of the nearest titled <point> (ordre du jour)
    is_procedural: bool  # True for vote results, applause, session open/close
    syceron_id: str | None  # id_syceron, for citation / traceability
    session_uid: str  # links back to the SessionMeta it belongs to
    date_iso: str  # duplicated here for convenient filtering on chunks


def _local(tag: str) -> str:
    return tag.split("}")[-1]


def _flatten(element: ET.Element | None) -> str:
    """All text inside an element, nested tags included, whitespace-normalised.

    Empty tags such as <br/> yield no text via itertext(), which can glue two
    sentences together (e.g. "...l'école.Certes..."). We restore a space after
    a sentence-ending punctuation immediately followed by a capital letter or an
    opening quote — the signature of such a removed line break.
    """
    if element is None:
        return ""
    text = " ".join("".join(element.itertext()).split())
    # Insert a missing space after .!?… when glued to a capital / opening quote,
    # but only when preceded by >=2 lowercase letters (a real word ending), so
    # initial-based acronyms like "U.S.A." are left intact.
    text = re.sub(r'([a-zà-öø-ÿ]{2}[.!?…])([«"A-ZÀ-ÖØ-Þ])', r"\1 \2", text)
    # Same glue, but when a closing quote » (with optional space) sits between
    # the punctuation and the next capital, e.g. '...mentale. »Mais...' from a
    # removed <br/> after a quoted sentence.
    text = re.sub(r'([.!?…]\s*»)([«"A-ZÀ-ÖØ-Þ])', r"\1 \2", text)
    return text


def _speaker_of(paragraph: ET.Element) -> tuple[str | None, str | None]:
    orateur = paragraph.find("an:orateurs/an:orateur", NAMESPACE)
    if orateur is None:
        return None, None
    name = (orateur.findtext("an:nom", default="", namespaces=NAMESPACE) or "").strip()
    role = (
        orateur.findtext("an:qualite", default="", namespaces=NAMESPACE) or ""
    ).strip()
    return (name or None), (role or None)


def _read_metadata(root: ET.Element) -> SessionMeta:
    meta = root.find(".//an:metadonnees", NAMESPACE)
    date_raw = meta.findtext("an:dateSeance", default="", namespaces=NAMESPACE) or ""
    return SessionMeta(
        uid=root.findtext(".//an:uid", default="", namespaces=NAMESPACE) or "",
        date_iso=date_raw[:8],  # YYYYMMDD from the timestamp
        date_label=meta.findtext("an:dateSeanceJour", default="", namespaces=NAMESPACE)
        or "",
        session=meta.findtext("an:session", default="", namespaces=NAMESPACE) or "",
        legislature=meta.findtext("an:legislature", default="", namespaces=NAMESPACE)
        or "",
    )


def parse_session(xml_path: str | Path) -> tuple[SessionMeta, list[Intervention]]:
    """Parse one syceron XML file into its metadata + de-duplicated interventions.

    Each <paragraphe> is visited once (document order). Its agenda item is the
    title of the nearest *titled* enclosing <point> (points are nested and some
    have no <texte>, so we climb until one does).
    """
    root = ET.parse(xml_path).getroot()
    meta = _read_metadata(root)
    parent = {child: par for par in root.iter() for child in par}

    def nearest_agenda_title(node: ET.Element) -> str:
        cur = parent.get(node)
        while cur is not None:
            if _local(cur.tag) == "point":
                title = _flatten(cur.find("an:texte", NAMESPACE))
                if title:
                    return title
            cur = parent.get(cur)
        return ""

    content = root.find(".//an:contenu", NAMESPACE)
    interventions: list[Intervention] = []
    if content is not None:
        for para in content.iterfind(".//an:paragraphe", NAMESPACE):
            text = _flatten(para.find("an:texte", NAMESPACE))
            if not text:
                continue
            name, role = _speaker_of(para)
            interventions.append(
                Intervention(
                    speaker=name,
                    role=role,
                    text=text,
                    agenda_item=nearest_agenda_title(para),
                    is_procedural=para.attrib.get("code_style", "")
                    in PROCEDURAL_STYLES,
                    syceron_id=para.attrib.get("id_syceron"),
                    session_uid=meta.uid,
                    date_iso=meta.date_iso,
                )
            )
    return meta, interventions


def build_corpus(xml_paths: list[str | Path]) -> list[Intervention]:
    """Parse several sessions and concatenate their interventions in date order."""
    parsed = [parse_session(p) for p in xml_paths]
    parsed.sort(key=lambda mp: mp[0].date_iso)  # chronological
    corpus: list[Intervention] = []
    for meta, items in parsed:
        speeches = sum(1 for i in items if not i.is_procedural)
        print(
            f"  {meta.date_label:32} {meta.uid}  "
            f"({len(items)} interventions, {speeches} speeches)"
        )
        corpus.extend(items)
    return corpus


def save_corpus(corpus: list[Intervention], out_path: str | Path) -> None:
    """Persist the corpus as JSON Lines (one intervention per line)."""
    out_path = Path(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        for item in corpus:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    print(f"\nWrote {len(corpus)} interventions to {out_path}")


if __name__ == "__main__":
    import sys

    # In your repo, list all three session files here:
    SESSIONS = [
        "CRSANR5L15S2022O1N080.xml",  # 1 Dec 2021  — first reading
        "CRSANR5L15S2022O1N154.xml",  # 10 Feb 2022 — new reading
        "CRSANR5L15S2022O1N165.xml",  # 24 Feb 2022 — final reading
    ]
    # Fall back to whatever files are passed on the command line / available.
    data_dir = Path("../data/corpus_15e_legislature/corpus")
    paths = sys.argv[1:] or [
        data_dir.joinpath(p) for p in SESSIONS if data_dir.joinpath(p).exists()
    ]

    print(f"Building corpus from {len(paths)} session(s):")
    corpus = build_corpus(paths)
    save_corpus(corpus, "corpus.jsonl")
