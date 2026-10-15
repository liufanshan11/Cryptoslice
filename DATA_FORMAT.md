# Data Format

All data files use UTF-8 JSON Lines: each nonempty line is one JSON object.
Join records by identifiers rather than line position. The collections contain
50 and 100 records in each of their five files.

## Tasks and queries

`tasks.jsonl` contains `task_id`, `prompt`, `allowed_backends`, and
`output_contract`. The contract specifies the output format and required JSON
fields. `queries.jsonl` contains `query_id` and `question`; `query_id` matches
the corresponding `task_id`.

## Evidence

Each `mveus.jsonl` record has two blocks:

- `evidence`: the evidence object available to retrieval and generation.
- `eval`: recorded metadata, including paper identity, construction provenance,
  closure flags, and size statistics.

The `evidence` block contains `mveu_id`, `template`, `scope`, `template_slots`,
`source_evidence`, and modeling fields such as `variables`, `objective`,
`round_function`, `sbox_table`, and `state_layout`. Fields may be empty or null
when they do not apply. `source_evidence` identifies source facts and locations
and gives their descriptions. Recorded closure flags are supplied annotations;
the file validator checks structure and links, not their semantic correctness.

`eval.paper_id` joins to `sources.jsonl`. Construction provenance can contain
identifiers from an earlier source collection; these are provenance labels,
not paths or instructions for loading generator input.

## Reference answers

`gold.jsonl` contains `task_id`, `paper_id`, `gold_answer`,
`gold_mveu_targets`, `source_provenance`, `verification`, and
`expected_solver_status`. Each target's `mveu_id` refers to an evidence unit
in the same collection. The reference answer may be a structured object
rather than a single number. Verification metadata describes the recorded
answer-checking context; it is not an execution log.

## Sources

`sources.jsonl` contains `paper_id`, `doi`, `url`, `pdf_name`, `pdf_sha256`,
and `rank`. The rank preserves the source manifest order. DOI identifiers are
recovered from the supplied PDF filenames. The checksum is SHA-256 of the
original PDF bytes.

## Integrity checks

`checksums.sha256` records the files in this release. `validate.py` verifies
these checksums, required fields, record counts, unique identifiers, task/query
alignment, and gold-to-evidence and paper links. It does not call a model,
execute generated programs, or score task answers.
