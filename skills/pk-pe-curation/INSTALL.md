# Installing the `pk-pe-curation` skill

This is a **single Claude Code skill** that bundles the whole PK/PE curation suite
(the orchestrator, the shared library, and ten curation pipelines as internal
sub-procedures). Installing it is one folder copy.

## Install

Copy the entire `pk-pe-curation/` directory into your project's skills folder:

```bash
mkdir -p <your-project>/.claude/skills
cp -r pk-pe-curation <your-project>/.claude/skills/
```

Result:

```
<your-project>/.claude/skills/pk-pe-curation/
├── SKILL.md            ← the one auto-triggered skill
├── curation-common/    ← shared scripts + prompts (keep alongside!)
└── pipelines/          ← the 10 pipelines + route + prepare-paper
```

**Copy the whole folder.** The pipelines reference `curation-common/` and each
other by paths relative to `pk-pe-curation/`; removing or splitting out any
sub-folder breaks them.

Personal (all projects) instead of per-project: copy into `~/.claude/skills/`.

## Requirements

The bundled scripts need **Python 3** and **`beautifulsoup4`**:

```bash
pip install beautifulsoup4
```

(`requirements.txt` is in `curation-common/scripts/`.) No other runtime is needed —
everything else is prose the model follows.

## Use

Once installed, Claude Code auto-discovers the skill from its `SKILL.md`
`description` and triggers it when you ask for PK/PE curation, e.g.:

> "Curate the PK data in `paper.html`."

The skill prepares the paper, decides which pipelines apply, and (after confirming)
runs them. See `SKILL.md` for the full flow. To point it at a local model, run
Claude Code against your Ollama server — the prompts are model-agnostic and the
bundled scripts run identically regardless of model.

## Notes

- Intermediate state is written to git-ignored scratch dirs in **your project root**
  (`./.paper_assets/`, `./.pk_pe_route_scratch/`, `./.{pipeline}_scratch/`), never
  inside the skill folder.
- Scope of `prepare-paper` is **HTML** (PMC / Wiley / Elsevier); JATS/PMC XML is not
  yet supported.
