"""Markdown + JSON result tables with confidence intervals and significance.

A report states, in its first line, which labels the numbers come from (the
"label provenance" line). Only ``label_source="hand"`` reports are headline
results; anything else is rendered with a DIAGNOSTIC warning.

Statistics (see ``dishcovery.eval.stats``):

* every cell is the mean over queries ± half the width of the 95% percentile
  bootstrap CI (JSON keeps the exact lower/upper bounds);
* ``†`` marks a system whose per-query values differ significantly from the
  named baseline: two-sided paired randomization test, Holm–Bonferroni
  corrected across the systems compared within each metric;
* ``judged@k`` metrics are pool-coverage diagnostics and are never tested.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Mapping, Optional, Sequence

from .metrics import DEFAULT_METRICS, EvalResult, Run, evaluate_run, per_class
from .stats import bootstrap_ci, holm_bonferroni, paired_randomization_test, paired_ttest

DAGGER = "†"


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _is_diagnostic(metric: str) -> bool:
    return metric.startswith("judged")


def _pretty_metric(metric: str) -> str:
    name, _, k = metric.partition("@")
    pretty = {"ndcg": "NDCG", "mrr": "MRR", "recall": "Recall", "precision": "P",
              "judged": "judged", "dcg": "DCG"}.get(name, name)
    return f"{pretty}@{k}" if k else pretty


def provenance_line(qrels: Mapping[str, Mapping[str, int]], label_source: str = "hand",
                    results: Optional[Mapping[str, EvalResult]] = None,
                    primary_metric: str = "ndcg@10", judged_metric: str = "judged@10") -> str:
    """One-line statement of where the labels come from and how complete they are.

    e.g. ``"NDCG@10 on 50 hand-labelled queries, 1,234 judged pairs (grades
    0/1/2/3: 41%/27%/19%/13%); judged@10 = 0.87 (bm25), 0.81 (hybrid)."``
    """
    judged_q = [q for q, j in qrels.items() if j]
    n_pairs = sum(len(qrels[q]) for q in judged_q)
    hist = [0, 0, 0, 0]
    for q in judged_q:
        for g in qrels[q].values():
            hist[min(max(int(g), 0), 3)] += 1
    dist = "/".join(f"{100.0 * h / n_pairs:.0f}%" for h in hist) if n_pairs else "n/a"
    noun = {"hand": "hand-labelled", "weak": "weakly-labelled (heuristic)",
            "clicks": "click-labelled (simulated PBM clicks, IPW)"}.get(label_source,
                                                                        f"{label_source}-labelled")
    n_eval = len(judged_q)
    skipped_note = ""
    if results:
        first = next(iter(results.values()))
        if primary_metric in first.n_queries:
            n_eval = first.n_queries[primary_metric]
            n_skip = len(first.skipped.get(primary_metric, []))
            if n_skip:
                skipped_note = f" ({n_skip} without any relevant judged doc skipped)"
    line = (f"{_pretty_metric(primary_metric)} on {n_eval} {noun} queries{skipped_note}, "
            f"{_fmt_int(n_pairs)} judged pairs (grades 0/1/2/3: {dist})")
    if results:
        parts = []
        for sys_name, res in results.items():
            v = res.means.get(judged_metric)
            if v is not None and not math.isnan(v):
                parts.append(f"{v:.2f} ({sys_name})")
        if parts:
            line += f"; {judged_metric} = " + ", ".join(parts)
    line += "."
    if label_source != "hand":
        line = (f"DIAGNOSTIC ONLY — labels are '{label_source}', not hand judgments; "
                f"do not report as headline results. " + line)
    return line


def build_report(results: Mapping[str, EvalResult], baseline: Optional[str] = None,
                 metrics: Optional[Sequence[str]] = None,
                 qid_to_class: Optional[Mapping[str, str]] = None,
                 qrels: Optional[Mapping[str, Mapping[str, int]]] = None,
                 label_source: str = "hand", class_metric: Optional[str] = None,
                 n_resamples: int = 10000, alpha: float = 0.05, seed: int = 0,
                 title: str = "Evaluation") -> dict:
    """Assemble a JSON-serialisable report from per-system ``EvalResult``s.

    Args:
        results: ``system -> EvalResult`` (all against the same qrels).
        baseline: system the † significance marker is relative to.
        metrics: metrics to tabulate (default: those of the first result).
        qid_to_class: enables the per-query-class breakdown table.
        qrels: the judgments, for the provenance line.
        label_source: ``hand`` | ``weak`` | ``clicks``.
        class_metric: metric of the per-class table (default: first metric).
    """
    if not results:
        raise ValueError("no results")
    if baseline is not None and baseline not in results:
        raise ValueError(f"baseline {baseline!r} not among systems {list(results)}")
    systems = list(results)
    if baseline is not None:
        systems = [baseline] + [s for s in systems if s != baseline]
    metrics = list(metrics or next(iter(results.values())).metrics)

    table: Dict[str, Dict[str, dict]] = {s: {} for s in systems}
    for m in metrics:
        raw_p: Dict[str, float] = {}
        for s in systems:
            vals = results[s].per_query[m]
            ci = bootstrap_ci(list(vals.values()), n_resamples=n_resamples, alpha=alpha, seed=seed)
            cell = {"mean": ci.mean, "ci_lo": ci.lo, "ci_hi": ci.hi, "n": len(vals),
                    "skipped": len(results[s].skipped.get(m, []))}
            if baseline is not None and s != baseline and not _is_diagnostic(m):
                base = results[baseline].per_query[m]
                common = sorted(set(base) & set(vals))
                a = [vals[q] for q in common]
                b = [base[q] for q in common]
                cell["n_paired"] = len(common)
                cell["delta"] = (sum(a) - sum(b)) / len(common) if common else float("nan")
                raw_p[s] = paired_randomization_test(a, b, n=n_resamples, seed=seed) if common else float("nan")
                cell["p_randomization"] = raw_p[s]
                try:
                    cell["p_ttest"] = paired_ttest(a, b).pvalue if len(common) > 1 else float("nan")
                except ImportError:  # scipy missing: the randomization test is what we report
                    cell["p_ttest"] = float("nan")
            table[s][m] = cell
        adjusted = holm_bonferroni(raw_p) if raw_p else {}
        for s, p_adj in adjusted.items():
            table[s][m]["p_holm"] = p_adj
            table[s][m]["significant"] = bool(not math.isnan(p_adj) and p_adj < alpha)

    report: dict = {
        "title": title,
        "label_source": label_source,
        "provenance": provenance_line(qrels, label_source, {s: results[s] for s in systems},
                                      primary_metric=metrics[0],
                                      judged_metric=next((m for m in metrics if _is_diagnostic(m)),
                                                         "judged@10"))
        if qrels is not None else f"labels: {label_source}",
        "baseline": baseline,
        "systems": systems,
        "metrics": metrics,
        "alpha": alpha,
        "n_resamples": n_resamples,
        "seed": seed,
        "ci": f"{100 * (1 - alpha):.0f}% percentile bootstrap over queries",
        "test": "two-sided paired randomization (sign-flip) test",
        "correction": "Holm-Bonferroni across systems vs baseline, per metric",
        "rel_threshold": next(iter(results.values())).rel_threshold,
        "table": table,
        "skipped": {s: {m: results[s].skipped.get(m, []) for m in metrics} for s in systems},
        "missing_in_run": {s: results[s].missing_in_run for s in systems},
    }
    if qid_to_class:
        cm = class_metric or metrics[0]
        breakdown: Dict[str, Dict[str, dict]] = {}
        for s in systems:
            pq = results[s].per_query[cm]
            per = per_class(results[s], qid_to_class)
            for c, row in per.items():
                n = sum(1 for q in pq if qid_to_class.get(q, "unknown") == c)
                breakdown.setdefault(c, {})[s] = {"mean": row[cm], "n": n}
        report["per_class"] = {"metric": cm, "classes": _order_classes(breakdown)}
    return report


def _order_classes(breakdown: Dict[str, Dict[str, dict]]) -> Dict[str, Dict[str, dict]]:
    try:
        from dishcovery.labels.queries import QUERY_CLASSES
    except ImportError:  # pragma: no cover
        QUERY_CLASSES = ()
    order = [c for c in QUERY_CLASSES if c in breakdown] + sorted(c for c in breakdown
                                                                  if c not in QUERY_CLASSES)
    return {c: breakdown[c] for c in order}


def _fmt_cell(cell: dict, digits: int = 3) -> str:
    mean = cell.get("mean")
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "–"
    hw = (cell["ci_hi"] - cell["ci_lo"]) / 2.0
    s = f"{mean:.{digits}f} ± {hw:.{digits}f}"
    if cell.get("significant"):
        s += f" {DAGGER}"
    return s


def render_markdown(report: dict, digits: int = 3) -> str:
    """Render a report built by :func:`build_report` as Markdown."""
    metrics: List[str] = report["metrics"]
    systems: List[str] = report["systems"]
    baseline = report.get("baseline")
    lines = [f"## {report['title']}", "", f"**Label provenance:** {report['provenance']}", ""]
    lines.append("| System | " + " | ".join(metrics) + " |")
    lines.append("|:--|" + "--:|" * len(metrics))
    for s in systems:
        name = f"{s} *(baseline)*" if s == baseline else s
        cells = [_fmt_cell(report["table"][s][m], digits) for m in metrics]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    note = (f"Mean over queries ± half-width of the {report['ci']} CI "
            f"({_fmt_int(report['n_resamples'])} resamples). Relevant = grade ≥ "
            f"{report['rel_threshold']} for MRR/recall/precision; NDCG uses gains 2^g − 1 with the "
            f"ideal ranking over all judged docs and unjudged docs = 0.")
    if baseline is not None:
        note += (f" {DAGGER} = significantly different from **{baseline}** "
                 f"({report['test']}, {_fmt_int(report['n_resamples'])} sign flips or exact; "
                 f"{report['correction']}; α = {report['alpha']}). judged@k is a pool-coverage "
                 f"diagnostic and is not tested.")
    lines.append(note)
    skipped = {m: report["skipped"][systems[0]][m] for m in metrics if report["skipped"][systems[0]][m]}
    if skipped:
        lines.append("")
        lines.append("Queries skipped (metric undefined — no judged relevant doc): " + "; ".join(
            f"{m}: {len(q)} ({', '.join(q[:8])}{', …' if len(q) > 8 else ''})" for m, q in skipped.items()))
    if "per_class" in report:
        pc = report["per_class"]
        lines += ["", f"### Per query class — {pc['metric']}", ""]
        lines.append("| Class | n | " + " | ".join(systems) + " |")
        lines.append("|:--|--:|" + "--:|" * len(systems))
        for c, row in pc["classes"].items():
            n = max((row[s]["n"] for s in systems if s in row), default=0)
            vals = []
            for s in systems:
                v = row.get(s, {}).get("mean")
                vals.append("–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{digits}f}")
            lines.append(f"| {c} | {n} | " + " | ".join(vals) + " |")
        lines.append("")
        lines.append("Per-class means only: with ~5 queries per class these are indicative and untested.")
    return "\n".join(lines) + "\n"


def _json_clean(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {str(k): _json_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_clean(v) for v in x]
    return x


def write_report(report: dict, md_path: str, json_path: Optional[str] = None) -> None:
    """Write the Markdown rendering and (optionally) the same data as strict JSON."""
    for p in (md_path, json_path):
        if p and os.path.dirname(p):
            os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(report))
    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(_json_clean(report), f, indent=2, ensure_ascii=False)
            f.write("\n")


def evaluate_and_report(runs: Mapping[str, Run], qrels: Mapping[str, Mapping[str, int]],
                        baseline: Optional[str] = None,
                        metrics: Sequence[str] = DEFAULT_METRICS, ks: Sequence[int] = (10,),
                        qid_to_class: Optional[Mapping[str, str]] = None,
                        label_source: str = "hand", rel_threshold: int = 2,
                        md_path: Optional[str] = None, json_path: Optional[str] = None,
                        **kwargs) -> dict:
    """Convenience: evaluate every system's run, build the report, optionally write it.

    ``kwargs`` go to :func:`build_report` (``n_resamples``, ``alpha``, ``seed``,
    ``title``, ``class_metric``).
    """
    results = {s: evaluate_run(r, qrels, ks=ks, metrics=metrics, rel_threshold=rel_threshold,
                               label_source=label_source) for s, r in runs.items()}
    report = build_report(results, baseline=baseline, qid_to_class=qid_to_class, qrels=qrels,
                          label_source=label_source, **kwargs)
    if md_path:
        write_report(report, md_path, json_path)
    return report
