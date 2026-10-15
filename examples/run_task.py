"""Retrieve MVEU evidence and generate one program using a configured API."""

import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", choices=("query_50", "query_100"), default="query_50")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    settings = {name: os.getenv(name, "").strip() for name in
                ("OPENAI_API_KEY", "OPENAI_API_BASE_URL", "GENERATION_MODEL")}
    missing = [name for name, value in settings.items() if not value]
    if missing:
        parser.error("Set environment variables: " + ", ".join(missing))
    directory = ROOT / "data" / args.collection
    tasks = read_rows(directory / "tasks.jsonl")
    records = read_rows(directory / "mveus.jsonl")
    if not 0 <= args.task_index < len(tasks):
        parser.error("Task index outside collection")
    task = tasks[args.task_index]
    from cryptoslice.agentic_mveu import MVEUToolkit, retrieve_from_mveu_kb_agent
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    toolkit = MVEUToolkit(records)
    if args.check_only:
        print(f"Ready: {len(tasks)} tasks, {len(toolkit.records)} evidence units. No API calls made.")
        return
    output = ROOT / "outputs" / args.collection / str(args.task_index)
    if output.exists():
        parser.error(f"Output directory already exists: {output}; preserve or move the previous run first")
    print("Calling the configured API for retrieval and generation; charges may apply.")
    retrieved = retrieve_from_mveu_kb_agent(
        query_id=task["task_id"], question=task["prompt"], records=records,
        model=settings["GENERATION_MODEL"], api_base_url=settings["OPENAI_API_BASE_URL"],
        top_k=5, max_steps=6,
    ).result.model_dump(mode="json")
    evidence = [item["payload"]["evidence"] for item in retrieved["items"]
                if isinstance(item.get("payload", {}).get("evidence"), dict)]
    if not evidence:
        raise RuntimeError("No evidence selected; generation was not attempted")
    llm = ChatOpenAI(model=settings["GENERATION_MODEL"],
                     base_url=settings["OPENAI_API_BASE_URL"],
                     api_key=settings["OPENAI_API_KEY"], temperature=0,
                     timeout=120, max_retries=2)
    response = llm.invoke([
        SystemMessage(content="Generate a complete Python program for the supplied task using the supplied evidence. Return only the program, optionally in one Python code fence. The program must follow the task's output contract."),
        HumanMessage(content=json.dumps({"task": task, "evidence": evidence}, ensure_ascii=False)),
    ])
    if not isinstance(response.content, str):
        raise RuntimeError("Expected a text response from the configured model")
    raw = response.content.strip()
    match = re.fullmatch(r"```(?:python)?\s*\n(.*?)\n```", raw, flags=re.DOTALL)
    code = match.group(1) if match else raw
    output.mkdir(parents=True, exist_ok=False)
    (output / "retrieval.json").write_text(json.dumps(retrieved, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "response.txt").write_text(raw, encoding="utf-8")
    (output / "solution.py").write_text(code + "\n", encoding="utf-8")
    print(f"Saved to {output}. Generated code was not executed.")


if __name__ == "__main__":
    main()
