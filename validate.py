"""Validate the dataset files and their cross-record references."""

from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path):
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        require(isinstance(row, dict), f"{path.name}:{number}: expected an object")
        rows.append(row)
    return rows


def index(rows, key, label):
    require(all(isinstance(row.get(key), str) and row[key] for row in rows),
            f"{label}: missing or invalid {key}")
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows), f"{label}: duplicate {key}")
    return result


def validate_collection(root, size):
    data = {name: read_rows(root / f"{name}.jsonl")
            for name in ("tasks", "queries", "mveus", "gold", "sources")}
    for name, rows in data.items():
        require(len(rows) == size, f"{root.name}/{name}: expected {size} records")
    tasks = index(data["tasks"], "task_id", "tasks")
    queries = index(data["queries"], "query_id", "queries")
    gold = index(data["gold"], "task_id", "gold")
    sources = index(data["sources"], "paper_id", "sources")
    require(tasks.keys() == queries.keys() == gold.keys(), "Task/query/gold IDs differ")
    for task in tasks.values():
        require(isinstance(task.get("prompt"), str) and task["prompt"].strip(), "Empty task prompt")
        require(isinstance(task.get("allowed_backends"), list), "Missing allowed backends")
        contract = task.get("output_contract", {})
        require(contract.get("format") == "json", "Unsupported output format")
        require(isinstance(contract.get("required_fields"), list), "Missing output fields")
        require(not {"gold_answer", "gold_mveu_targets", "eval"}.intersection(task),
                "Evaluator fields present in a public task")
    require(all(isinstance(q.get("question"), str) and q["question"].strip()
                for q in queries.values()), "Empty retrieval question")
    evidence = []
    paper_by_unit = {}
    for record in data["mveus"]:
        item = record.get("evidence")
        metadata = record.get("eval")
        require(isinstance(item, dict) and isinstance(metadata, dict), "Invalid MVEU blocks")
        require({"mveu_id", "scope", "template", "template_slots", "source_evidence"} <= item.keys(),
                "Missing MVEU fields")
        require(metadata.get("paper_id") in sources, "MVEU paper not found")
        require(isinstance(item["source_evidence"], list), "Invalid evidence sources")
        evidence.append(item)
        paper_by_unit[item["mveu_id"]] = metadata["paper_id"]
    units = index(evidence, "mveu_id", "mveus")
    for row in gold.values():
        require(row.get("paper_id") in sources, "Gold paper not found")
        require("gold_answer" in row, "Missing reference answer")
        targets = row.get("gold_mveu_targets")
        require(isinstance(targets, list) and targets, "Missing gold targets")
        for target in targets:
            unit_id = target.get("mveu_id")
            require(unit_id in units, f"Unknown target: {unit_id}")
            require(paper_by_unit[unit_id] == row["paper_id"], "Gold/evidence paper mismatch")
    for row in sources.values():
        require(row.get("url") == "https://doi.org/" + row.get("doi", ""), "Invalid DOI URL")
        digest = row.get("pdf_sha256", "")
        require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "Invalid PDF checksum")
    print(f"{root.name}: {len(tasks)} tasks, {len(units)} MVEUs, {len(sources)} papers; links OK")
    return set(sources)


def main():
    manifest = ROOT / "checksums.sha256"
    covered = set()
    for line in manifest.read_text(encoding="ascii").splitlines():
        expected, name = line.split("  ", 1)
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT), "Checksum path outside release")
        require(path.is_file(), f"Missing file: {name}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"Checksum mismatch: {name}")
        covered.add(name)
    expected_files = {p.relative_to(ROOT).as_posix() for p in (ROOT / "data").rglob("*.jsonl")}
    require(expected_files <= covered, "Unchecksummed data file")
    paper_sets = [validate_collection(ROOT / "data" / f"query_{size}", size) for size in (50, 100)]
    shared = len(paper_sets[0] & paper_sets[1])
    require(shared == 27, f"Unexpected source overlap: {shared}")
    print(f"Source overlap: {shared} papers. All checks passed.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"Validation failed: {error}", file=sys.stderr)
        sys.exit(1)
