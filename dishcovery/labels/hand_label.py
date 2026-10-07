"""Hand-labelling tools for the evaluation set (qrels ``source = "hand"``).

Two workflows produce the same JSONL qrels:

* :func:`run_labeler` — interactive terminal tool. Shows the guidelines summary
  once, then each pooled (query, doc) pair with everything needed to apply the
  rubric (distance, open at query time, inside delivery zone, ...). Keys:
  ``0``–``3`` grade, ``s`` skip, ``b`` back, ``q`` quit. Every grade is
  appended and fsync'ed immediately; restarting skips pairs already in the
  output file, so a session can be interrupted at any point. Going back and
  re-grading appends a new record — the latest record wins on load.
* :func:`export_csv` / :func:`import_csv` — the same information as a
  spreadsheet with an empty ``grade`` column.

Documents are shown in a per-query shuffled order (seeded) so the annotator
cannot infer which system or rank surfaced them.
"""
from __future__ import annotations

import csv
import os
import random
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

from dishcovery.eval.io import append_qrel, make_qrel, read_jsonl
from dishcovery.labels.pooling import load_pool
from dishcovery.labels.queries import EvalQuery, load_queries, parse_time
from dishcovery.labels.weak import can_deliver, haversine_km, is_open_at

GUIDELINES_SUMMARY = """\
Grade each result for the query, as if you were the user at that place and time
(full rubric: data/eval/LABELING_GUIDELINES.md).

  3  Exactly what was asked for AND usable now: open at the query time and
     inside the delivery zone; satisfies every stated constraint (diet, price).
  2  Clearly relevant / good substitute (same dish style, right cuisine,
     restaurant that serves the dish), or a 3 with a minor issue.
  1  Marginal: related but not what was asked (generic pizza for "domino's"),
     or relevant but closed at the query time.
  0  Irrelevant, violates a hard constraint (non-veg for "vegan", wrong
     dietary rule), or cannot be served (outside the delivery zone).

Keys: 0-3 grade · s skip · b back · q quit (progress is saved after every grade)
"""

KEY_HELP = "[0-3] grade  [s] skip  [b] back  [q] quit"


def _fmt_bool(x: Optional[bool]) -> str:
    return "unknown" if x is None else ("yes" if x else "no")


def doc_context(doc: Mapping[str, Any], query: EvalQuery) -> Dict[str, Any]:
    """Derived facts the rubric needs: distance, open now, deliverable."""
    dist = None
    if "lat" in doc and "lon" in doc:
        dist = haversine_km(query.lat, query.lon, float(doc["lat"]), float(doc["lon"]))
    dow, h, m = parse_time(query.time)
    return {
        "distance_km": None if dist is None else round(dist, 2),
        "open_at_query_time": is_open_at(doc, dow, h, m),
        "in_delivery_zone": can_deliver(doc, query.lat, query.lon),
    }


def _short(text: str, n: int = 160) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def describe_doc(doc: Optional[Mapping[str, Any]], query: EvalQuery, doc_id: str) -> List[str]:
    """Human-readable lines for one candidate."""
    if doc is None:
        return [f"  {doc_id}: (document not found in docs lookup)"]
    c = doc_context(doc, query)
    dist = "?" if c["distance_km"] is None else f"{c['distance_km']:.1f} km"
    price = doc.get("price")
    price_s = f"${float(price):.2f}" if isinstance(price, (int, float)) and price else "-"
    lines = [
        f"  {doc.get('name', '?')}   [{doc.get('type', '?')}]   id={doc_id}",
        f"  restaurant: {doc.get('restaurant_name', '')}"
        + (f"  (chain: {doc['chain']})" if doc.get("chain") else ""),
        f"  cuisines: {', '.join(doc.get('cuisines') or []) or '-'}   diet: "
        f"{', '.join(doc.get('diet') or []) or '-'}",
        f"  price: {price_s}  tier {'$' * int(doc.get('price_tier') or 0) or '-'}   rating "
        f"{float(doc.get('rating') or 0):.1f} ({int(doc.get('review_count') or 0)} reviews)",
        f"  distance: {dist}   open at {query.time}: {_fmt_bool(c['open_at_query_time'])}   "
        f"in delivery zone: {_fmt_bool(c['in_delivery_zone'])}",
    ]
    desc = doc.get("description") or doc.get("reviews") or ""
    if desc:
        lines.append(f"  {_short(desc)}")
    return lines


def _judged_pairs(out_path: str) -> Set[Tuple[str, str]]:
    if not os.path.exists(out_path):
        return set()
    return {(str(r["qid"]), str(r["doc_id"])) for r in read_jsonl(out_path)
            if r.get("source") == "hand"}


def _ordered_items(pool: Mapping[str, Sequence[str]], queries: Sequence[EvalQuery],
                   shuffle_seed: Optional[int]) -> List[Tuple[str, str]]:
    order = [q.qid for q in queries if q.qid in pool]
    order += sorted(q for q in pool if q not in set(order))
    items: List[Tuple[str, str]] = []
    for qid in order:
        docs = list(pool[qid])
        if shuffle_seed is not None:
            random.Random(f"{shuffle_seed}:{qid}").shuffle(docs)
        items += [(qid, d) for d in docs]
    return items


def run_labeler(pool_path: str, queries_path: str, docs_lookup: Mapping[str, Mapping[str, Any]],
                out_path: str, annotator: Optional[str] = None, shuffle_seed: Optional[int] = 0,
                input_fn: Callable[[str], str] = input,
                print_fn: Callable[..., None] = print) -> Dict[str, int]:
    """Interactive labelling session (see module docstring).

    Args:
        pool_path: pool JSONL from :func:`dishcovery.labels.pooling.save_pool`.
        queries_path: evaluation queries JSONL.
        docs_lookup: ``doc_id -> doc`` (any Mapping, e.g. a docs.jsonl-backed dict).
        out_path: qrels JSONL to append ``source="hand"`` records to.
        annotator: stored on every record (defaults to ``$USER``).
        shuffle_seed: per-query display shuffle seed (``None`` keeps pool order).
        input_fn / print_fn: injectable for tests.

    Returns:
        Session counts ``{"labelled", "skipped", "remaining", "total"}``.
    """
    annotator = annotator or os.environ.get("USER") or "anonymous"
    pool = load_pool(pool_path)
    queries = load_queries(queries_path)
    qmap = {q.qid: q for q in queries}
    judged = _judged_pairs(out_path)
    items = [it for it in _ordered_items(pool, queries, shuffle_seed) if it not in judged]
    total = len(items)
    print_fn(GUIDELINES_SUMMARY)
    print_fn(f"{total} unjudged pairs in pool ({len(judged)} already judged in {out_path}).")
    labelled = skipped = 0
    history: List[int] = []
    shown_qid: Optional[str] = None
    i = 0
    while i < len(items):
        qid, doc_id = items[i]
        q = qmap.get(qid)
        if q is None:  # pool entry for an unknown query: cannot show context
            i += 1
            continue
        if qid != shown_qid:
            print_fn("")
            print_fn("=" * 72)
            print_fn(f"QUERY {qid}: \"{q.text}\"   class={q.query_class}   metro={q.metro}   "
                     f"time={q.time}   at ({q.lat:.4f}, {q.lon:.4f})")
            if q.notes:
                print_fn(f"notes: {q.notes}")
            print_fn("=" * 72)
            shown_qid = qid
        print_fn("")
        print_fn(f"[{i + 1}/{total}]")
        for line in describe_doc(docs_lookup.get(doc_id), q, doc_id):
            print_fn(line)
        try:
            key = input_fn(f"{KEY_HELP} > ").strip().lower()
        except EOFError:
            break
        if key in ("0", "1", "2", "3"):
            append_qrel(out_path, make_qrel(qid, doc_id, int(key), "hand", annotator))
            labelled += 1
            history.append(i)
            i += 1
        elif key == "s":
            skipped += 1
            history.append(i)
            i += 1
        elif key == "b":
            if history:
                i = history.pop()
                shown_qid = None
            else:
                print_fn("(nothing to go back to)")
        elif key == "q":
            break
        else:
            print_fn(f"unrecognised key {key!r}; {KEY_HELP}")
    remaining = len([it for it in items if it not in _judged_pairs(out_path)])
    print_fn(f"\nsession: {labelled} labelled, {skipped} skipped, {remaining} remaining.")
    return {"labelled": labelled, "skipped": skipped, "remaining": remaining, "total": total}


CSV_COLUMNS = ["qid", "query", "query_class", "metro", "time", "doc_id", "name", "type",
               "restaurant", "chain", "cuisines", "diet", "price", "price_tier", "rating",
               "review_count", "distance_km", "open_at_query_time", "in_delivery_zone",
               "description", "grade", "annotator", "comment"]


def export_csv(pool: Mapping[str, Sequence[str]], queries: Union[str, Sequence[EvalQuery]],
               docs_lookup: Mapping[str, Mapping[str, Any]], out_path: str,
               existing_qrels: Optional[Mapping[str, Mapping[str, int]]] = None,
               shuffle_seed: Optional[int] = 0) -> int:
    """Write the pool as a spreadsheet with an empty ``grade`` column.

    Already-judged pairs (``existing_qrels``) are left out. Returns the number
    of rows written.
    """
    qs = load_queries(queries) if isinstance(queries, str) else list(queries)
    qmap = {q.qid: q for q in qs}
    existing_qrels = existing_qrels or {}
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for qid, doc_id in _ordered_items(pool, qs, shuffle_seed):
            q = qmap.get(qid)
            if q is None or doc_id in existing_qrels.get(qid, {}):
                continue
            doc = docs_lookup.get(doc_id) or {}
            c = doc_context(doc, q) if doc else {"distance_km": None, "open_at_query_time": None,
                                                 "in_delivery_zone": None}
            w.writerow({
                "qid": qid, "query": q.text, "query_class": q.query_class, "metro": q.metro,
                "time": q.time, "doc_id": doc_id, "name": doc.get("name", ""),
                "type": doc.get("type", ""), "restaurant": doc.get("restaurant_name", ""),
                "chain": doc.get("chain", ""), "cuisines": ", ".join(doc.get("cuisines") or []),
                "diet": ", ".join(doc.get("diet") or []), "price": doc.get("price", ""),
                "price_tier": doc.get("price_tier", ""), "rating": doc.get("rating", ""),
                "review_count": doc.get("review_count", ""),
                "distance_km": "" if c["distance_km"] is None else c["distance_km"],
                "open_at_query_time": _fmt_bool(c["open_at_query_time"]),
                "in_delivery_zone": _fmt_bool(c["in_delivery_zone"]),
                "description": _short(doc.get("description") or doc.get("reviews") or "", 300),
                "grade": "", "annotator": "", "comment": "",
            })
            n += 1
    return n


def import_csv(path: str, annotator: Optional[str] = None) -> List[dict]:
    """Read a filled-in spreadsheet back as ``source="hand"`` qrel records.

    Rows with an empty or invalid grade are ignored. The row's ``annotator``
    column wins over the ``annotator`` argument.
    """
    out: List[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            g = (row.get("grade") or "").strip()
            if g not in ("0", "1", "2", "3"):
                continue
            who = (row.get("annotator") or "").strip() or annotator
            out.append(make_qrel(row["qid"], row["doc_id"], int(g), "hand", who))
    return out
