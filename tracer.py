import json
import os
import time


class Tracer:
    def __init__(self, scenario, model):
        self.meta = {"scenario": scenario, "model": model, "started": time.time()}
        self.events = []

    def log(self, turn, role, said, actions, tokens, seconds, status):
        self.events.append({
            "turn": turn,
            "role": role,
            "said": said,
            "actions": [{"tool": name, "args": args, "result": result} for name, args, result in actions],
            "tokens": tokens,
            "seconds": round(seconds, 2),
            "status": status,
        })

    def finish(self, result, turns):
        self.meta["result"] = result
        self.meta["turns"] = turns
        self.meta["total_tokens"] = sum(event["tokens"] for event in self.events)
        os.makedirs("logs", exist_ok=True)
        path = os.path.join("logs", f"run_{int(self.meta['started'])}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"meta": self.meta, "events": self.events}, f, indent=2)
        return path