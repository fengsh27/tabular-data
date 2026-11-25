"""One parser for every schema described in `schema_specs.SPECS`.

`SchemaParser(spec)` exposes the same surface as `_common` / `_common_summary`
(`COLS`, `NUMERIC_COLS`, `_TEXT_COLS`, `_VOCAB`, `_VALUE_RE`, `_fits`,
`parse_csv`, `write_csv`, ...), so `run_simple_prompt.py`, `validity.py` and
the cascade's arms can treat it as a module. The algorithms are those of
`_common_summary.py` - fence/reasoning stripping, header mapping through
aliases, a scored realignment for a row that is short or long a field
(`_realign`), a one-or-two-move repair for a count-correct row whose cells
sit in the wrong columns (`_reshift`), the headerless block fallback and the
"report every unparsable line as dropped" rule - parametrised by the spec's
column roles instead of module constants. `_common_summary.py` itself is left
untouched so the validated pk_summary path cannot regress from a change here.

Cell cleaning is `_common.clean`: every spelling of "missing" - including the
"N/A" the production workflows write - becomes "", which is what the two
hand-written schemas and the benchmark scorers expect.
"""

from __future__ import annotations

import csv
import io
import itertools
import re

from _common import clean, squash, strip_fences, strip_reasoning  # noqa: F401
from schema_specs import SchemaSpec

_VALUE_RE = re.compile(r"^[<>~≈≤≥]?\s*-?\d+(?:\.\d+)?")
_WORDS_RE = re.compile(r"[A-Za-z]{2,}")
_MAX_GAP = 3
_HARD_MISFIT = -2
_MAX_MOVES = 2


class SchemaParser:
    def __init__(self, spec: SchemaSpec):
        self.spec = spec
        self.COLS = list(spec.cols)
        self.ALIASES = dict(spec.aliases)
        self.NUMERIC_COLS = set(spec.numeric_cols)
        self._TEXT_COLS = tuple(spec.text_cols)
        self._REQUIRED = tuple(spec.required_cols)
        self._VOCAB = {k: set(v) for k, v in spec.vocab.items()}
        self._VALUE_RE = _VALUE_RE
        self.__name__ = f"schema:{spec.name}"

    # --- per-cell plausibility -------------------------------------------------

    def _fits(self, name: str, cell: str) -> int:
        """How plausible `cell` is under canonical column `name` (see _common._fits)."""
        c = (cell or "").strip()
        if name in self.NUMERIC_COLS:
            if not c:
                return 0
            return 2 if self._VALUE_RE.match(c) else -3
        if name in self._VOCAB:
            if not c:
                return 0
            return 2 if c.lower() in self._VOCAB[name] else -1
        if name == "Patient ID":
            if not c:
                return -1 if name in self._REQUIRED else 0
            return -2 if len(c.split()) > 2 else 1  # ids are short: "1", "Case 5"
        if name in self._REQUIRED:
            if not c:
                return -1
            return -2 if self._VALUE_RE.match(c) else 0
        if name in self._TEXT_COLS:
            return -2 if self._VALUE_RE.match(c) else 0
        return 0

    def _fits_strict(self, name: str, cell: str) -> int:
        """`_fits` for the `_reshift` search: a cell that starts with a digit and
        goes on in words is text, a label outside the vocabulary is neutral.
        See _common_summary._fits_strict."""
        c = (cell or "").strip()
        if self._VALUE_RE.match(c) and _WORDS_RE.search(c):
            if name in self.NUMERIC_COLS:
                return -3
            if name in self._TEXT_COLS or name in self._REQUIRED:
                return 0
        if name in self._VOCAB and c and c.lower() not in self._VOCAB[name]:
            return 0
        return self._fits(name, cell)

    # --- row repairs -----------------------------------------------------------

    def _map_headers(self, row):
        mapped = [self.ALIASES.get(squash(h)) for h in row]
        known = sum(1 for name in mapped if name)
        need = min(self.spec.min_header_hits, max(2, len(self.COLS) - 1))
        if known < need or known < len(row) / 2:
            return None
        return mapped

    def _realign(self, row, header_names):
        gap = len(row) - len(header_names)
        if gap == 0 or abs(gap) > _MAX_GAP:
            return None
        seen = {}
        if gap < 0:
            missing = -gap
            for combo in itertools.combinations_with_replacement(range(len(row) + 1), missing):
                cand = list(row)
                for pos in reversed(combo):
                    cand.insert(pos, "")
                cand = tuple(cand)
                run_edits = sum(
                    1 for pos in combo
                    if (pos > 0 and not (row[pos - 1] or "").strip())
                    or (pos < len(row) and not (row[pos] or "").strip())
                )
                if cand not in seen or seen[cand][1] < run_edits:
                    seen[cand] = (sum(self._fits(n, c) for n, c in zip(header_names, cand)), run_edits)
        else:
            for combo in itertools.combinations(range(len(row)), gap):
                drop = set(combo)
                cand = tuple(c for i, c in enumerate(row) if i not in drop)
                run_edits = sum(1 for i in combo if not (row[i] or "").strip())
                if cand not in seen or seen[cand][1] < run_edits:
                    seen[cand] = (sum(self._fits(n, c) for n, c in zip(header_names, cand)), run_edits)
        ranked = sorted(seen.items(), key=lambda kv: kv[1], reverse=True)
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return None
        return list(ranked[0][0])

    def _row_consistency(self, cells, names) -> int:
        rec = dict(zip(names, cells))
        lo = (rec.get("Lower bound") or rec.get("Lower limit") or "").strip()
        hi = (rec.get("Upper bound") or rec.get("High limit") or "").strip()
        return -2 if bool(lo) != bool(hi) else 0

    @staticmethod
    def _moves(cells):
        original = tuple(cells)
        for e, cell in enumerate(cells):
            if (cell or "").strip():
                continue
            rest = list(cells[:e]) + list(cells[e + 1:])
            for p in range(len(rest) + 1):
                cand = tuple(rest[:p] + [""] + rest[p:])
                if cand != original:
                    yield cand, abs(p - e)

    def _reshift(self, row, names):
        if len(row) != len(names):
            return None

        def score(cells):
            per_cell = [self._fits_strict(n, c) for n, c in zip(names, cells)]
            return sum(per_cell) + self._row_consistency(cells, names), min(per_cell)

        base_total, base_min = score(row)
        if base_min > _HARD_MISFIT:
            return None
        original = tuple(row)
        seen = {}
        frontier = [(original, 0, 0)]
        for depth in range(1, _MAX_MOVES + 1):
            next_frontier = []
            for cells, _, displaced in frontier:
                for cand, moved in self._moves(cells):
                    key = (depth, displaced + moved)
                    if cand == original or (cand in seen and seen[cand][2:] <= key):
                        continue
                    total, lowest = score(cand)
                    seen[cand] = (total, lowest, depth, displaced + moved)
                    next_frontier.append((cand, depth, displaced + moved))
            frontier = next_frontier
        if not seen:
            return None
        ranked = sorted(seen.items(), key=lambda kv: (-kv[1][0], kv[1][2], kv[1][3]))
        best, (best_total, best_min, best_moves, best_displaced) = ranked[0]
        if best_total <= base_total or best_min <= _HARD_MISFIT:
            return None
        if len(ranked) > 1 and ranked[1][1][0] == best_total \
                and ranked[1][1][2:] == (best_moves, best_displaced):
            return None
        return list(best)

    # --- parsing ---------------------------------------------------------------

    def _pop_trailing_empties(self, row, width=None):
        width = len(self.COLS) if width is None else width
        row = list(row)
        while len(row) > width and not row[-1].strip():
            row.pop()
        return row

    def parse_csv(self, text: str, pmid: str = "", dropped=None):
        """Model output -> list of dicts on the canonical schema. `pmid` is
        accepted for signature parity with `_common.parse_csv`."""
        text = strip_fences(strip_reasoning(text or ""))
        lines = [ln for ln in text.splitlines() if ln.strip()]

        header, start = None, 0
        for i, line in enumerate(lines):
            if "," not in line:
                continue
            try:
                row = next(csv.reader([line]))
            except Exception:
                continue
            mapped = self._map_headers(row)
            if mapped:
                header, start = mapped, i + 1
                break
        if header is None:
            return self._parse_headerless(lines, dropped=dropped)

        body_rows = [self._pop_trailing_empties(r)
                     for r in csv.reader(io.StringIO("\n".join(lines[start:])))
                     if r and any(c.strip() for c in r)]

        if len(header) != len(self.COLS):
            full = sum(1 for r in body_rows if len(r) in (len(self.COLS), len(self.COLS) + 1))
            if full >= max(2, len(body_rows) // 2):
                header = list(self.COLS)

        out = []
        for row in body_rows:
            if self._map_headers(row):  # a repeated header block
                continue
            row = self._pop_trailing_empties(row, len(header))
            if len(row) != len(header):
                fixed = self._realign(row, header)
                if fixed is None:
                    if dropped is not None:
                        dropped.append(row)
                    continue
                row = fixed
            else:
                fixed = self._reshift(row, header)
                if fixed is not None:
                    row = fixed
            rec = {c: "" for c in self.COLS}
            for name, cell in zip(header, row):
                if name:
                    rec[name] = clean(cell)
            if any(rec.values()):
                out.append(rec)
        return out

    def _parse_headerless(self, lines, dropped=None):
        parsed = []
        for line in lines:
            if "," not in line:
                continue
            try:
                row = next(csv.reader([line]))
            except Exception:
                continue
            parsed.append(self._pop_trailing_empties(row))

        exact = sum(1 for r in parsed if len(r) == len(self.COLS))
        if exact < min(2, len(parsed)) or exact == 0:
            if dropped is not None:
                dropped.extend(parsed)
            return []

        rows = []
        for row in parsed:
            if len(row) != len(self.COLS):
                fixed = self._realign(row, self.COLS)
                if fixed is None:
                    if dropped is not None:
                        dropped.append(row)
                    continue
                row = fixed
            else:
                fixed = self._reshift(row, self.COLS)
                if fixed is not None:
                    row = fixed
            rec = {c: clean(cell) for c, cell in zip(self.COLS, row)}
            if any(rec.values()):
                rows.append(rec)
        return rows

    def write_csv(self, path, rows, cols=None):
        cols = list(cols or self.COLS)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})
