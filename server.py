import sys
from mcp.server.fastmcp import FastMCP
from engine import Session
from papers.search import (
    search_papers as raw_search_papers,
    get_citations as raw_get_citations,
    get_references as raw_get_references,
)
from experiment.spec import ExperimentSpec, Variant, Probe
from experiment.probes import run_probe


claim = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "Does SMOTE improve F1 more than class-weighting on imbalanced data?"
)

session = Session(claim)
mcp = FastMCP("research-agent")


def format_result(result) -> str:
    lines = [
        f"Probe: {result.spec.probe.value}"
    ]

    for r in result.per_dataset:
        delta = r.delta[result.spec.primary_metric]
        lines.append(
            f"  {r.dataset}: "
            f"delta({result.spec.primary_metric})={delta:+.3f}, "
            f"supported={r.supported}"
        )

    lines.append(
        f"Overall: {result.support_count}/{result.total_count} "
        "datasets support this comparison."
    )

    return "\n".join(lines)


@mcp.tool(description="Read the claim you are investigating.")
def read_claim() -> str:
    return session.read_claim()


@mcp.tool(
    description=(
        "Search arXiv and OpenAlex for papers matching a query. "
        "Returns up to 5 results with id, year, title."
    )
)
def search_papers(agent: str, query: str) -> str:
    papers = raw_search_papers(
        query,
        max_results=5,
    )

    session.log_papers(agent, papers)

    if not papers:
        return "No papers found."

    return "\n".join(
        f"{p.id} ({p.year}): {p.title}"
        for p in papers
    )


@mcp.tool(
    description=(
        "Get papers that cite the given paper id "
        "(must be an 'openalex:W...' id)."
    )
)
def get_citations(
    agent: str,
    paper_id: str,
) -> str:
    papers = raw_get_citations(
        paper_id,
        max_results=5,
    )

    session.log_papers(agent, papers)

    if not papers:
        return "No citing papers found."

    return "\n".join(
        f"{p.id} ({p.year}): {p.title}"
        for p in papers
    )


@mcp.tool(
    description=(
        "Get papers referenced by the given paper id "
        "(must be an 'openalex:W...' id)."
    )
)
def get_references(
    agent: str,
    paper_id: str,
) -> str:
    papers = raw_get_references(
        paper_id,
        max_results=10,
    )

    session.log_papers(agent, papers)

    if not papers:
        return "No references found."

    return "\n".join(
        f"{p.id} ({p.year}): {p.title}"
        for p in papers
    )


@mcp.tool(
    description="Formally cite a paper you are relying on as evidence."
)
def cite(
    agent: str,
    paper_id: str,
    title: str,
    reason: str,
) -> str:
    return session.cite(
        agent,
        paper_id,
        title,
        reason,
    )


@mcp.tool(
    description=(
        "Run a probe experiment on real data. probe is one of: direct_ab, "
        "strengthen_baseline, leakage_check, metric_decompose, seed_variance, "
        "boundary_sweep. datasets is a comma-separated list of dataset names "
        "(breast_cancer, credit_g, pima_diabetes, synthetic_classification, "
        "diabetes_regression, california_housing, synthetic_regression). "
        "model_a/model_b are the two configurations being compared. "
        "technique_b/class_weight_a/class_weight_b/sweep_param/sweep_values "
        "are optional, pass an empty string to omit them."
    )
)
def run_experiment(
    agent: str,
    probe: str,
    datasets: str,
    model_a: str,
    model_b: str,
    technique_b: str = "",
    class_weight_a: str = "",
    class_weight_b: str = "",
    primary_metric: str = "f1",
    seeds: int = 5,
    sweep_param: str = "",
    sweep_values: str = "",
) -> str:

    try:
        spec = ExperimentSpec(
            probe=Probe(probe),
            datasets=[
                d.strip()
                for d in datasets.split(",")
                if d.strip()
            ],
            variant_a=Variant(
                model=model_a,
                class_weight=class_weight_a or None,
            ),
            variant_b=Variant(
                model=model_b,
                class_weight=class_weight_b or None,
                technique=technique_b or None,
            ),
            primary_metric=primary_metric,
            seeds=seeds,
            sweep_param=sweep_param or None,
            sweep_values=(
                [
                    float(v)
                    for v in sweep_values.split(",")
                ]
                if sweep_values
                else None
            ),
        )

        result = run_probe(spec)

    except Exception as error:
        return f"Error running experiment: {error}"

    session.log_probe(
        agent,
        spec,
        result,
    )

    return format_result(result)


@mcp.tool(
    description="Radio a message to your partner."
)
def send_message(
    agent: str,
    text: str,
) -> str:
    return session.send_message(
        agent,
        text,
    )


@mcp.tool(
    description="Read the messages your partner sent you."
)
def read_messages(agent: str) -> str:
    return session.read_messages(agent)


@mcp.tool(
    description="Write a short note in your private notebook."
)
def remember(
    agent: str,
    note: str,
) -> str:
    return session.remember(
        agent,
        note,
    )


@mcp.tool(
    description="Read back the notes in your private notebook."
)
def recall(agent: str) -> str:
    return session.recall(agent)


@mcp.tool(
    description=(
        "File your final report on the claim. verdict must be one of: "
        "supported, contested, refuted, inconclusive, scope_limited. "
        "key_papers is a comma-separated list of paper ids you relied on."
    )
)
def file_report(
    agent: str,
    verdict: str,
    summary: str,
    key_papers: str,
) -> str:
    return session.file_report(
        agent,
        verdict,
        summary,
        key_papers,
    )


@mcp.tool(
    description="Check whether both agents have filed their final reports yet."
)
def status() -> str:
    return session.status()


@mcp.tool(
    description=(
        "Get the full final report: both agents' verdicts, "
        "citations, and probes run. Only call after status is complete."
    )
)
def final_report() -> str:
    report = session.final_report()

    lines = [
        f"CLAIM: {report['claim']}",
        "",
    ]

    for role in ("proposer", "skeptic"):
        r = report[role]

        lines.append(
            f"--- {role.upper()} ---"
        )

        if r:
            lines.append(
                f"Verdict: {r['verdict']}"
            )
            lines.append(
                f"Summary: {r['summary']}"
            )
            lines.append(
                f"Key papers: {r['key_papers']}"
            )
        else:
            lines.append(
                "No report filed."
            )

        lines.append("")

    lines.append("--- CITATIONS ---")

    for c in report["citations"]:
        lines.append(
            f"[{c['agent']}] "
            f"{c['paper_id']} - "
            f"{c['title']} "
            f"({c['reason']})"
        )

    lines.append("")
    lines.append("--- PROBES RUN ---")

    for p in report["probes_run"]:
        lines.append(
            f"[{p['agent']}] "
            f"{p['probe']} -> "
            f"{p['support']} datasets supported"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()