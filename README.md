# CryptoSlice

This repository provides cryptanalysis tasks paired with reference answers and
Minimal Verifiable Evidence Units (MVEUs) from research papers. The two collections
contain 50 and 100 tasks, respectively, with one paper and one MVEU per task.
They share 27 source papers and should be treated as separate benchmark
configurations, not disjoint training and test splits.

## Contents

```text
data/
  query_50/
  query_100/
src/cryptoslice/      MVEU closure checks and tool-based retrieval
examples/run_task.py  API-based retrieval and program generation
DATA_FORMAT.md
checksums.sha256
validate.py
```

Each collection has five files:

| File | Contents |
| --- | --- |
| `tasks.jsonl` | Task prompts, allowed backends, and output contracts |
| `queries.jsonl` | Retrieval queries derived from the task prompts |
| `mveus.jsonl` | Evidence units and their source and construction metadata |
| `gold.jsonl` | Reference answers and target evidence identifiers |
| `sources.jsonl` | Paper identifiers, DOI links, and source PDF checksums |

Task prompts are in Chinese, with English technical terms. Evidence records
contain English descriptions. Original task identifiers and data values are
preserved. See [DATA_FORMAT.md](DATA_FORMAT.md) for field definitions.

## Use

Python 3.11 or later is sufficient to read and validate the data. No additional
packages or API credentials are needed.

From this directory, check the files and their identifier links:

```sh
python validate.py
```

Read a task and the evidence collection:

```python
import json
from pathlib import Path

root = Path("data/query_50")
tasks = [json.loads(line) for line in (root / "tasks.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
records = [json.loads(line) for line in (root / "mveus.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
evidence = [record["evidence"] for record in records]
print(tasks[0]["prompt"])
print(evidence[0]["mveu_id"])
```

Use the public task and retrieved `evidence` as generator input. Load `gold.jsonl`
separately for evaluation; do not use gold target IDs to select generator context.
The `eval` block in an MVEU record is metadata, not generator input.

Paper DOI links are listed in `sources.jsonl`. The PDF checksums identify the
source copies used to prepare the collection; another publisher version may
have a different checksum.

## Run a task

Create a Python 3.11+ virtual environment and install the dependencies (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The example retrieves evidence from the selected collection and asks a model to
produce a Python program for one task. It uses a paid API: supply your own API
key, endpoint, and model name. The endpoint must support OpenAI-compatible chat
completions and the model must support tool calling. Calls send the task and
selected evidence to your configured provider and may incur charges. Model
availability, pricing, and outputs depend on that provider.

In PowerShell, configure the service for the current session:

```powershell
$env:OPENAI_API_KEY = "YOUR_API_KEY"
$env:OPENAI_API_BASE_URL = "https://YOUR_PROVIDER/v1"
$env:GENERATION_MODEL = "YOUR_TOOL_CALLING_MODEL"
python examples/run_task.py --check-only
python examples/run_task.py --collection query_50 --task-index 0
```

`--check-only` validates local inputs and configuration without contacting the
service. The run writes retrieved evidence, the model response, and `solution.py`
under `outputs/`. Generated code is not executed automatically; review it and
run it in an isolated environment. The example does not load reference answers.
Use `--collection query_100` to select the larger collection.

Keep API keys in environment variables, not in source files or uploaded logs.
