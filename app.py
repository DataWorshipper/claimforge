import glob
import json
import os
import time

import pandas as pd
import streamlit as st
from pyvis.network import Network

LOG_DIR = "logs"

VERDICT_BOX = {
    "supported": st.success,
    "refuted": st.error,
    "contested": st.warning,
    "inconclusive": st.info,
    "scope_limited": st.info,
}

ROLE_AVATAR = {
    "proposer": "🧑‍🔬",
    "skeptic": "🕵️",
}


def load_trace_paths():
    return sorted(
        glob.glob(os.path.join(LOG_DIR, "*.json")),
        key=os.path.getmtime,
        reverse=True,
    )


def load_trace(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def render_verdicts(report):
    cols = st.columns(2)

    for col, role in zip(cols, ("proposer", "skeptic")):
        with col:
            st.subheader(role.upper())
            r = report.get(role) if report else None

            if not r:
                st.write("No report filed.")
                continue

            box = VERDICT_BOX.get(r["verdict"], st.info)
            box(f"**{r['verdict'].upper()}**")
            st.write(r["summary"])

            if r.get("key_papers"):
                st.caption(f"Key papers: {r['key_papers']}")


def render_turn(event):
    avatar = ROLE_AVATAR.get(event["role"])

    with st.chat_message(event["role"], avatar=avatar):
        st.markdown(f"**{event['role'].upper()}** · turn {event['turn']}")

        if event["said"]:
            st.write(event["said"])

        for action in event["actions"]:
            arg_preview = ", ".join(f"{k}={v}" for k, v in action["args"].items())

            with st.expander(f"🔧 {action['tool']}({arg_preview})"):
                st.code(action["result"])


def render_transcript(trace):
    events = trace["events"]

    col1, col2 = st.columns([1, 2])
    replay = col1.toggle("▶ Replay as it happened")
    pace = col2.slider("Seconds between turns", 0.2, 3.0, 1.0, 0.1, disabled=not replay)

    if replay:
        st.caption("Playing back the conversation - the rest of the page will finish once this ends.")

        for event in events:
            render_turn(event)
            time.sleep(pace)
    else:
        for event in events:
            render_turn(event)


def render_probes(report):
    rows = []

    for p in report.get("probes_run", []):
        for b in p["breakdown"]:
            rows.append({
                "agent": p["agent"],
                "probe": p["probe"],
                "dataset": b["dataset"],
                "delta": b["delta"],
                "supported": b["supported"],
            })

    if not rows:
        st.write("No probes were run in this investigation.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    sweep_df = df[df["probe"] == "boundary_sweep"]

    if not sweep_df.empty:
        st.caption("Boundary sweep deltas")
        st.bar_chart(sweep_df.set_index("dataset")["delta"])

    for p in report.get("probes_run", []):
        if p.get("notes"):
            st.caption(f"**{p['probe']}** ({p['scope']}): {p['notes']}")


def build_citation_graph(report):
    paper_index = report.get("paper_index", {})
    edges = report.get("citation_edges", [])
    cited_ids = {c["paper_id"] for c in report.get("citations", [])}

    node_ids = set(paper_index.keys())
    for e in edges:
        node_ids.add(e["from"])
        node_ids.add(e["to"])

    net = Network(
        height="620px",
        width="100%",
        bgcolor="#111111",
        font_color="white",
        directed=True,
        cdn_resources="in_line",
    )
    net.barnes_hut()

    for pid in node_ids:
        info = paper_index.get(pid, {})
        title = info.get("title") or pid
        year = info.get("year", "?")
        source = info.get("source", "unknown")
        cited_by = info.get("cited_by_count") or 0

        size = 12 + min(cited_by, 500) ** 0.5
        color = "#f39c12" if source == "arxiv" else "#3498db"

        if pid in cited_ids:
            color = "#2ecc71"

        label = title if len(title) <= 40 else title[:37] + "..."
        hover = f"{title}\n({year}, {source})\ncited_by={cited_by}\n{pid}"

        net.add_node(pid, label=label, title=hover, color=color, size=size)

    for e in edges:
        net.add_edge(e["from"], e["to"])

    return net


def render_citation_graph(report):
    paper_index = report.get("paper_index", {})

    if not paper_index:
        st.write("No papers were touched in this investigation.")
        return

    st.caption(
        "🟢 formally cited as evidence · 🟠 arXiv · 🔵 OpenAlex · "
        "bubble size = citation count · drag to explore"
    )

    net = build_citation_graph(report)
    html = net.generate_html()
    st.components.v1.html(html, height=640, scrolling=True)


def main():
    st.set_page_config(page_title="ClaimForge", layout="wide")
    st.title("ClaimForge")

    paths = load_trace_paths()

    if not paths:
        st.info("No runs yet. Run `python orchestrator.py \"<claim>\"` first.")
        return

    labels = []
    for path in paths:
        trace = load_trace(path)
        claim = trace["meta"].get("scenario", "")[:60]
        marker = "" if trace.get("final_report") else "  ⚠️ no graph/verdict data"
        labels.append(f"{claim}  —  {os.path.basename(path)}{marker}")

    choice = st.sidebar.selectbox(
        "Choose a run",
        options=range(len(paths)),
        format_func=lambda i: labels[i],
    )

    trace = load_trace(paths[choice])
    meta = trace["meta"]
    report = trace.get("final_report")

    st.caption(meta.get("scenario", ""))

    metric_cols = st.columns(4)
    metric_cols[0].metric("Result", meta.get("result", "?"))
    metric_cols[1].metric("Turns", meta.get("turns", "?"))
    metric_cols[2].metric("Tokens", meta.get("total_tokens", "?"))
    metric_cols[3].metric("Model", meta.get("model", "?"))

    if report:
        render_verdicts(report)
    else:
        st.error(
            "⚠️ This run has no verdict/citation-graph data - it was saved before that "
            "feature existed. Run a brand new investigation (`python orchestrator.py "
            "\"<claim>\"`) and pick the new one from the sidebar."
        )

    tab_transcript, tab_probes, tab_graph = st.tabs(
        ["Transcript", "Probes", "Citation graph"]
    )

    with tab_transcript:
        render_transcript(trace)

    with tab_probes:
        if report:
            render_probes(report)
        else:
            st.write("No structured probe data for this run - see the warning above.")

    with tab_graph:
        if report:
            render_citation_graph(report)
        else:
            st.write("No citation graph for this run - see the warning above.")


if __name__ == "__main__":
    main()
