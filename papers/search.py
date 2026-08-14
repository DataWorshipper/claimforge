from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import quote
import xml.etree.ElementTree as ET
import requests


OPENALEX_EMAIL = "mandalabhiraj04@gmail.com"
OPENALEX_BASE = "https://api.openalex.org/works"
ARXIV_BASE = "http://export.arxiv.org/api/query"
ARXIV_NS = "{http://www.w3.org/2005/Atom}"


@dataclass
class Paper:
    id: str
    source: str
    title: str
    abstract: str
    authors: List[str]
    year: Optional[int]
    url: str
    cited_by_count: Optional[int] = None
    referenced_works: List[str] = field(default_factory=list)


def clean(text):
    return " ".join((text or "").split())


def search_arxiv(query, max_results=10) -> List[Paper]:
    url = (
        f"{ARXIV_BASE}"
        f"?search_query=all:{quote(query)}"
        f"&start=0&max_results={max_results}"
    )

    response = requests.get(url, timeout=15)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    papers = []

    for entry in root.findall(f"{ARXIV_NS}entry"):
        raw_id = entry.findtext(f"{ARXIV_NS}id") or ""
        arxiv_id = raw_id.rsplit("/", 1)[-1]

        title = clean(
            entry.findtext(f"{ARXIV_NS}title")
        )

        abstract = clean(
            entry.findtext(f"{ARXIV_NS}summary")
        )

        published = (
            entry.findtext(f"{ARXIV_NS}published")
            or ""
        )

        year = (
            int(published[:4])
            if published[:4].isdigit()
            else None
        )

        authors = [
            clean(
                author.findtext(
                    f"{ARXIV_NS}name"
                )
            )
            for author in entry.findall(
                f"{ARXIV_NS}author"
            )
        ]

        papers.append(
            Paper(
                id=f"arxiv:{arxiv_id}",
                source="arxiv",
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
                url=raw_id,
            )
        )

    return papers


def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return ""

    positions = {}

    for word, idxs in inverted_index.items():
        for idx in idxs:
            positions[idx] = word

    return " ".join(
        positions[i]
        for i in sorted(positions)
    )


def openalex_id(full_id):
    return full_id.rsplit("/", 1)[-1]


def work_to_paper(work) -> Paper:
    authors = [
        a["author"]["display_name"]
        for a in work.get("authorships", [])
    ]

    referenced = [
        openalex_id(r)
        for r in work.get("referenced_works", [])
    ]

    return Paper(
        id=f"openalex:{openalex_id(work['id'])}",
        source="openalex",
        title=clean(work.get("title")),
        abstract=reconstruct_abstract(
            work.get("abstract_inverted_index")
        ),
        authors=authors,
        year=work.get("publication_year"),
        url=work.get("id", ""),
        cited_by_count=work.get("cited_by_count"),
        referenced_works=referenced,
    )


def search_openalex(query, max_results=10) -> List[Paper]:
    params = {
        "search": query,
        "per_page": max_results,
        "mailto": OPENALEX_EMAIL,
    }

    response = requests.get(
        OPENALEX_BASE,
        params=params,
        timeout=15,
    )

    response.raise_for_status()

    return [
        work_to_paper(w)
        for w in response.json().get("results", [])
    ]


def search_papers(query, max_results=10) -> List[Paper]:
    results = (
        search_arxiv(
            query,
            max_results=max_results,
        )
        + search_openalex(
            query,
            max_results=max_results,
        )
    )

    seen = set()
    deduped = []

    for paper in results:
        key = paper.title.lower().strip()

        if key in seen:
            continue

        seen.add(key)
        deduped.append(paper)

    return deduped


def get_openalex_work(openalex_id) -> Paper:
    clean_id = openalex_id.replace(
        "openalex:",
        "",
    )

    url = f"{OPENALEX_BASE}/{clean_id}"

    response = requests.get(
        url,
        params={"mailto": OPENALEX_EMAIL},
        timeout=15,
    )

    response.raise_for_status()

    return work_to_paper(response.json())


def get_citations(
    openalex_id,
    max_results=10,
) -> List[Paper]:

    clean_id = openalex_id.replace(
        "openalex:",
        "",
    )

    params = {
        "filter": f"cites:{clean_id}",
        "per_page": max_results,
        "mailto": OPENALEX_EMAIL,
    }

    response = requests.get(
        OPENALEX_BASE,
        params=params,
        timeout=15,
    )

    response.raise_for_status()

    return [
        work_to_paper(w)
        for w in response.json().get("results", [])
    ]


def get_references(
    openalex_id,
    max_results=25,
) -> List[Paper]:

    work = get_openalex_work(openalex_id)

    ids = work.referenced_works[:max_results]

    if not ids:
        return []

    filter_value = (
        "openalex_id:"
        + "|".join(ids)
    )

    params = {
        "filter": filter_value,
        "per_page": len(ids),
        "mailto": OPENALEX_EMAIL,
    }

    response = requests.get(
        OPENALEX_BASE,
        params=params,
        timeout=15,
    )

    response.raise_for_status()

    return [
        work_to_paper(w)
        for w in response.json().get("results", [])
    ]