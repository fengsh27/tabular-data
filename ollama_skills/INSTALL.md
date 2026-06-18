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

You then have 11 skills: the `pk-pe-prepare` front door, plus 10 standalone
curation skills. There is no router skill — routing/orchestration is the Claude
bundle's job and is too much for small open models; here you prepare the paper,
then trigger the right pipeline skill yourself.

## Use
- **Prepare first (any path):** trigger `pk-pe-prepare` (or run its
  `scripts/prepare_paper.py`) on a raw `.html` **or** `.xml` paper to produce
  `./.paper_assets/<pmid>/`.
- **Then curate (one pipeline at a time):** once you know which data the paper
  has, trigger the matching pipeline skill, e.g. "use pk-individual-curation to
  curate paper <pmid>", pointing it at the prepared `./.paper_assets/<pmid>/`
  files. Triggering a single skill keeps its full procedure in front of the
  model — the reliable path for the smallest models.

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
