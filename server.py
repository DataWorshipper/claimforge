import argparse
import sys
from mcp.server.fastmcp import FastMCP
from engine import Session, summarize_scope
from papers.search import (
    search_papers as raw_search_papers,
    get_citations as raw_get_citations,
    get_references as raw_get_references,
)
from experiment.spec import ExperimentSpec, Variant, Probe
from experiment.probes import run_probe
from experiment.datasets import (
    register_csv_dataset,
    list_datasets as get_dataset_catalog,
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "claim",
    nargs="?",
    default="Does SMOTE improve F1 more than class-weighting on imbalanced data?",
)
parser.add_argument("--csv")
parser.add_argument("--target")
parser.add_argument("--task-type", choices=["classification", "regression"])
parser.add_argument("--positive-label")
args = parser.parse_args()

claim = args.claim
session = Session(claim)
mcp = FastMCP("research-agent")

if args.csv:
    if not args.target or not args.task_type:
        raise SystemExit("--csv requires --target and --task-type")

    try:
        registered = register_csv_dataset(
            args.csv,
            args.target,
            args.task_type,
            args.positive_label,
        )
    except FileNotFoundError:
        raise SystemExit(f"--csv file not found: {args.csv}")
    except Exception as error:
        raise SystemExit(f"Failed to load --csv {args.csv}: {error}")

    print(
        f"Registered user_data: {registered.X.shape[0]} rows, "
        f"{registered.X.shape[1]} features",
        file=sys.stderr,
    )


def format_result(result) -> str:
    lines = [
        f"Probe: {result.spec.probe.value}"
    ]

    for r in result.per_dataset:
        delta = r.delta[result.spec.primary_metric]
        ratio_note = (
            f", minority_ratio={r.minority_ratio:.2f}"
            if r.minority_ratio is not None
            else ""
        )
        lines.append(
            f"  {r.dataset} (n={r.n_samples}, features={r.n_features}{ratio_note}): "
            f"delta({result.spec.primary_metric})={delta:+.3f}, "
            f"supported={r.supported}"
        )

    lines.append(
        f"Overall: {result.support_count}/{result.total_count} "
        "datasets support this comparison."
    )

    lines.append(
        f"Tested scope: {summarize_scope(result.per_dataset)}. "
        "This result only tells you what happens in this scope - if it "
        "doesn't match the claim's real-world use case, say so and consider "
        "verdict='scope_limited'."
    )

    if result.notes:
        lines.append(f"Notes: {result.notes}")

    return "\n".join(lines)


@mcp.tool(description="Read the claim you are investigating.")
def read_claim() -> str:
    return session.read_claim()


@mcp.tool(
    description=(
        "List every dataset name usable in run_experiment's datasets field, "
        "including any user-provided data (marked USER-PROVIDED). Call this "
        "before choosing datasets to check whether real user data is available "
        "beyond the standard portfolio - if it is, prefer testing against it, "
        "since 'does this hold for data like the user's' is a stronger answer "
        "than 'does this hold on generic benchmark datasets'."
    )
)
def list_datasets() -> str:
    entries = get_dataset_catalog()
    lines = []

    for e in entries:
        if e["user_provided"]:
            ratio_note = (
                f", minority_ratio={e['minority_ratio']:.2f}"
                if e["minority_ratio"] is not None
                else ""
            )
            lines.append(
                f"{e['name']} ({e['task_type']}, USER-PROVIDED): "
                f"n={e['n_samples']}, features={e['n_features']}{ratio_note}"
            )
        else:
            lines.append(f"{e['name']} ({e['task_type']}): {', '.join(e['tags'])}")

    return "\n".join(lines)


def format_papers(papers):
    lines = []

    for p in papers:
        lines.append(f"{p.id} ({p.year}): {p.title}")

        if p.abstract:
            snippet = p.abstract[:220]

            if len(p.abstract) > 220:
                snippet += "..."

            lines.append(f"  {snippet}")

    return "\n".join(lines)


@mcp.tool(
    description=(
        "Search arXiv and OpenAlex for papers matching a query. "
        "Returns up to 5 results with id, year, title, and a short "
        "abstract snippet so you can judge relevance before citing."
    )
)
def search_papers(agent: str, query: str) -> str:
    try:
        papers = raw_search_papers(
            query,
            max_results=5,
        )
    except Exception as error:
        return f"Error searching papers: {error}"

    session.log_papers(agent, papers)

    if not papers:
        return "No papers found."

    return format_papers(papers)


@mcp.tool(
    description=(
        "Get papers that cite the given paper id. Only works on "
        "'openalex:W...' ids - arxiv: ids are not supported here."
    )
)
def get_citations(
    agent: str,
    paper_id: str,
) -> str:
    if not paper_id.startswith("openalex:"):
        return (
            f"Error: get_citations only works with openalex: ids "
            f"(e.g. 'openalex:W12345'), not '{paper_id}'. Use a "
            f"paper id from an openalex search result instead."
        )

    try:
        papers = raw_get_citations(
            paper_id,
            max_results=5,
        )
    except Exception as error:
        return f"Error fetching citations for {paper_id}: {error}"

    session.log_papers(agent, papers)

    if not papers:
        return "No citing papers found."

    return format_papers(papers)


@mcp.tool(
    description=(
        "Get papers referenced by the given paper id. Only works on "
        "'openalex:W...' ids - arxiv: ids are not supported here."
    )
)
def get_references(
    agent: str,
    paper_id: str,
) -> str:
    if not paper_id.startswith("openalex:"):
        return (
            f"Error: get_references only works with openalex: ids "
            f"(e.g. 'openalex:W12345'), not '{paper_id}'. Use a "
            f"paper id from an openalex search result instead."
        )

    try:
        papers = raw_get_references(
            paper_id,
            max_results=10,
        )
    except Exception as error:
        return f"Error fetching references for {paper_id}: {error}"

    session.log_papers(agent, papers)

    if not papers:
        return "No references found."

    return format_papers(papers)


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
        "boundary_sweep. datasets is a comma-separated list from: breast_cancer, "
        "credit_g, pima_diabetes, synthetic_classification, diabetes_regression, "
        "california_housing, synthetic_regression - call list_datasets first to "
        "check whether 'user_data' (real user-provided data) is also available; "
        "if it is, include it. "
        "model_a and model_b must each be EXACTLY one bare model name: "
        "logistic_regression, random_forest, gradient_boosting, xgboost, catboost "
        "(classification) or linear_regression, random_forest, gradient_boosting, "
        "xgboost, catboost (regression). Do NOT put hyperparameters, class weights, "
        "or techniques inside the model name string - those are separate parameters. "
        "IMPORTANT: unlike datasets, model_a/model_b/technique_b/class_weight_a/"
        "class_weight_b/primary_metric do NOT accept comma-separated lists - each is "
        "exactly one value. To compare three models, call run_experiment three "
        "separate times, once per pair you want to compare. "
        "class_weight_a/class_weight_b are optional, the only meaningful value is "
        "'balanced'. technique_b is optional, lowercase only: smote or adasyn. "
        "primary_metric decides which delta counts as supported: f1, precision, "
        "recall, or auc for classification; rmse, mae, or r2 for regression. "
        "Do not use 'f1_score' or 'accuracy' - they are not valid metric names. "
        "sweep_param/sweep_values are only used by boundary_sweep - datasets must "
        "then be exactly ONE parametrized dataset (synthetic_classification or "
        "synthetic_regression), and sweep_param is one of its numeric kwargs: for "
        "synthetic_classification that's imbalance_ratio, n_samples, or n_features; "
        "for synthetic_regression it's n_samples, n_features, or noise. "
        "sweep_values is a comma-separated list of numbers to try, e.g. "
        "'0.3,0.2,0.1,0.05,0.02'. IMPORTANT: if direct_ab (or strengthen_baseline) "
        "comes back mixed - supported on some datasets, not others - that is usually "
        "more interesting to pin down with boundary_sweep than to just report as "
        "'inconclusive'. Sweeping imbalance_ratio or n_features tells you the exact "
        "regime where the effect appears or disappears, which is a stronger, more "
        "useful finding than a plain yes/no. "
        "Worked example, to test whether SMOTE beats a class-weighted logistic "
        "regression baseline, set these SEPARATE tool arguments (not one string): "
        "probe is direct_ab. datasets is 'credit_g,breast_cancer,pima_diabetes'. "
        "model_a is 'logistic_regression'. class_weight_a is 'balanced'. "
        "model_b is 'logistic_regression'. technique_b is 'smote'. "
        "primary_metric is 'f1'. A follow-up boundary_sweep on the same variants "
        "would set: probe=boundary_sweep, datasets='synthetic_classification', "
        "sweep_param='imbalance_ratio', sweep_values='0.3,0.2,0.1,0.05,0.02'."
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

    def norm(text):
        return text.strip().lower()

    single_value_fields = {
        "model_a": model_a,
        "model_b": model_b,
        "technique_b": technique_b,
        "class_weight_a": class_weight_a,
        "class_weight_b": class_weight_b,
        "primary_metric": primary_metric,
    }

    for field_name, value in single_value_fields.items():
        if "," in value:
            return (
                f"Error: {field_name} must be a single value, not a list "
                f"(you passed '{value}'). Only datasets accepts a comma-separated "
                f"list. To compare more than two configurations, call "
                f"run_experiment again, once per pair."
            )

    try:
        spec = ExperimentSpec(
            probe=Probe(norm(probe)),
            datasets=[
                d.strip()
                for d in datasets.split(",")
                if d.strip()
            ],
            variant_a=Variant(
                model=norm(model_a),
                class_weight=norm(class_weight_a) or None,
            ),
            variant_b=Variant(
                model=norm(model_b),
                class_weight=norm(class_weight_b) or None,
                technique=norm(technique_b) or None,
            ),
            primary_metric=norm(primary_metric),
            seeds=seeds,
            sweep_param=sweep_param.strip() or None,
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
    description="Internal: check whether a specific agent has filed their report yet."
)
def has_filed(agent: str) -> str:
    return "true" if session.has_filed(agent) else "false"


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
            f"{p['support']} datasets supported "
            f"(scope: {p['scope']})"
        )

        for b in p["breakdown"]:
            delta_str = (
                f"{b['delta']:+.3f}"
                if b["delta"] is not None
                else "n/a"
            )
            lines.append(
                f"    {b['dataset']}: delta={delta_str}, supported={b['supported']}"
            )

        if p.get("notes"):
            lines.append(f"    Notes: {p['notes']}")

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()