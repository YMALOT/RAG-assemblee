# Looking for relevent sittings (séances)

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

NAMESPACE = {"an": "http://schemas.assemblee-nationale.fr/referentiel"}

data_dir = Path("../data/corpus_15e_legislature")
source_dir = data_dir.joinpath("debats_15e/xml/compteRendu")
corpus_dir = data_dir.joinpath("corpus")
corpus_dir.mkdir(exist_ok=True, parents=True)

KEYWORDS = [
    "harcèlement scolaire",
    "harcelement à l'école",
]  # broad first; narrow to "harcèlement scolaire" if noisy


def full_text(element):
    """Flatten all text within an element, including child tags (exposant, italique...)."""
    return "".join(element.itertext())


def main():
    matched = []
    print("source_dir")
    for i, xml_file in enumerate(source_dir.rglob("*.xml")):
        try:
            root = ET.parse(xml_file).getroot()
        except ET.ParseError:
            print(f"  /!\\ Unreadable XML, skipped: {xml_file.name}")
            continue

        # Only scan the table of contents (sommaire) for filtering
        toc = root.find(".//an:sommaire", NAMESPACE)
        if toc is None:
            continue
        headings = [full_text(e) for e in toc.iterfind(".//an:intitule", NAMESPACE)]
        blob = " ".join(headings).lower()

        if any(keyword in blob for keyword in KEYWORDS):
            date = root.findtext(
                ".//an:dateSeanceJour", default="?", namespaces=NAMESPACE
            )
            matched.append((xml_file, date))
            print(f"Matched: {xml_file.name}  —  {date}")
            shutil.copy(xml_file, corpus_dir / xml_file.name)

            for heading in toc.iterfind(".//an:intitule", NAMESPACE):
                text = full_text(heading)
                if any(k in text.lower() for k in KEYWORDS):
                    print(f"{date:35} | {text.strip()}")

    print(f"\n{len(matched)} sessions copied to {corpus_dir}")


if __name__ == "__main__":
    main()
