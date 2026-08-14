from experiment.spec import Verdict


def summarize_scope(per_dataset):
    if not per_dataset:
        return ""

    task_types = sorted({r.task_type for r in per_dataset})
    samples = [r.n_samples for r in per_dataset]
    features = [r.n_features for r in per_dataset]
    ratios = [r.minority_ratio for r in per_dataset if r.minority_ratio is not None]
    tags = sorted({t for r in per_dataset for t in r.tags})

    parts = [
        f"task_type={'/'.join(task_types)}",
        f"n_samples={min(samples)}-{max(samples)}",
        f"n_features={min(features)}-{max(features)}",
    ]

    if ratios:
        parts.append(f"minority_ratio={min(ratios):.2f}-{max(ratios):.2f}")

    if tags:
        parts.append(f"domains={','.join(tags)}")

    return ", ".join(parts)


class Session:
    def __init__(self, claim):
        self.claim = claim
        self.messages = []
        self.read_counts = {}
        self.notebooks = {}
        self.papers_seen = []
        self.citations = []
        self.probe_log = []
        self.reports = {"proposer": None, "skeptic": None}

    def read_claim(self):
        return self.claim

    def send_message(self, sender, text):
        self.messages.append({
            "sender": sender,
            "text": text
        })
        return "Message sent."

    def read_messages(self, receiver):
        seen = self.read_counts.get(receiver, 0)
        new = self.messages[seen:]
        self.read_counts[receiver] = len(self.messages)

        incoming = [
            f"{m['sender']}: {m['text']}"
            for m in new
            if m["sender"] != receiver
        ]

        if not incoming:
            return "No new messages."

        return "\n".join(incoming)

    def remember(self, agent, note):
        self.notebooks.setdefault(agent, []).append(note)
        return "Noted."

    def recall(self, agent):
        notes = self.notebooks.setdefault(agent, [])

        if not notes:
            return "Notebook empty."

        return "\n".join(notes)

    def log_papers(self, agent, papers):
        for p in papers:
            self.papers_seen.append({
                "agent": agent,
                "id": p.id,
                "title": p.title
            })

    def cite(self, agent, paper_id, title, reason):
        self.citations.append({
            "agent": agent,
            "paper_id": paper_id,
            "title": title,
            "reason": reason
        })
        return "Cited."

    def log_probe(self, agent, spec, result):
        self.probe_log.append({
            "agent": agent,
            "probe": spec.probe.value,
            "support": f"{result.support_count}/{result.total_count}",
            "scope": summarize_scope(result.per_dataset),
            "notes": result.notes,
            "breakdown": [
                {
                    "dataset": r.dataset,
                    "delta": r.delta.get(spec.primary_metric),
                    "supported": r.supported,
                }
                for r in result.per_dataset
            ],
        })

    def file_report(
        self,
        agent,
        verdict,
        summary,
        key_papers
    ):
        if agent not in self.reports:
            return f"Unknown agent {agent}"

        try:
            Verdict(verdict)
        except ValueError:
            return (
                f"Invalid verdict '{verdict}', "
                f"choose from {[v.value for v in Verdict]}"
            )

        self.reports[agent] = {
            "verdict": verdict,
            "summary": summary,
            "key_papers": key_papers
        }

        return "Report filed."

    def status(self):
        if (
            self.reports["proposer"]
            and self.reports["skeptic"]
        ):
            return "complete"

        return "investigating"

    def has_filed(self, agent):
        return self.reports.get(agent) is not None

    def final_report(self):
        return {
            "claim": self.claim,
            "proposer": self.reports["proposer"],
            "skeptic": self.reports["skeptic"],
            "citations": self.citations,
            "papers_seen": self.papers_seen,
            "probes_run": self.probe_log,
        }