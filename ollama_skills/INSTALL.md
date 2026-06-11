# Installing the Ollama PK/PE curation skills

This `ollama_skills/` bundle is the **flat, self-contained** variant of the
PK/PE curation suite, tuned for smaller open models (e.g. Qwen via Ollama) run
through Claude Code. Each curation pipeline is its **own** top-level skill with
its **own** scripts and resources — there is no shared `curation-common/`.

## Install
Copy every folder in here into your project's `.claude/skills/`:

```bash
cp -R ollama_skills/* <your-project>/.claude/skills/
```

You then have 11 skills: the `pk-pe-curation` router plus 10 standalone
curation skills.

## Use
- **Guided (router):** ask to "curate PK/PE data from paper <pmid>" — the
  `pk-pe-curation` router prepares + routes, then triggers the matching pipeline
  skills one at a time.
- **Direct (recommended for the smallest models):** trigger a single pipeline
  skill, e.g. "use pk-individual-curation to curate paper <pmid>", once you know
  which table type you have. This keeps the full procedure in front of the model.

## Dependencies
Pipelines that convert HTML tables or prepare papers need BeautifulSoup:
`pip install -r <skill>/scripts/requirements.txt` (provides `beautifulsoup4`).
`verify_provenance.py` and the `clean_*_rows.py` cleanup scripts are Python-3
stdlib only.

## Regenerating
Do not hand-edit this folder. It is generated from `skills/pk-pe-curation/` by:

```bash
python scripts/build_ollama_skills.py
```
