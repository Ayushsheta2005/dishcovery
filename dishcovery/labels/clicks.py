"""Simulated user clicks under the position-based model (qrels ``source = "clicks"``).

Position-based model (PBM; Craswell et al. 2008, Joachims et al. 2017)::

    P(click at rank r) = P(examined at r) * P(attractive | grade)
                       = (1 / r) ** eta   * attract[grade]
    P(order | click)   = p_order_given_click[grade]

``grade`` is a *latent* relevance supplied by a grader callable — typically the
weak labeler (``labels.weak``) — so clicks are a noisy, position-biased view of
it. The simulated log feeds two consumers:

* **CTR features** (``ctr_table``): Beta-smoothed click/order rates per
  ``(query_norm, doc)`` shrunk toward a doc-level prior. NB: when training a
  ranker on these, compute features from sessions disjoint from the ones that
  produce its labels, or the model learns the label.
* **IPW click labels** (``ipw_click_labels``): the inverse-propensity-weighted
  click rate ``sum(c / p_r) / n`` is an unbiased estimate of ``attract[grade]``
  under the PBM; mapping it back through thresholds yields 0–3 training labels.

Everything is deterministic given ``seed``.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from dishcovery.eval.io import read_jsonl, write_jsonl
from dishcovery.qu.text import normalize

DEFAULT_ATTRACT = (0.05, 0.25, 0.6, 0.95)
DEFAULT_P_ORDER = (0.0, 0.1, 0.3, 0.5)
CLICK_ANNOTATOR = "pbm_ipw_v1"
_T0 = 1_767_225_600.0  # 2026-01-01T00:00:00Z: synthetic log start


@dataclass
class Impression:
    """One result page shown to one user, with the clicks/orders it got."""
    session_id: str
    qid: str
    query_norm: str
    ts: float                  # unix seconds (synthetic)
    doc_ids: List[str]         # in displayed order (rank 1 first)
    clicks: List[int]          # 0/1 per position
    orders: List[int]          # 0/1 per position (only possible where clicked)
    shuffled: bool = False     # displayed order randomised (intervention session)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Impression":
        return cls(**d)


def examination_prob(rank: int, eta: float = 1.0) -> float:
    """PBM examination propensity at 1-based ``rank``: ``(1/rank) ** eta``."""
    return (1.0 / rank) ** eta


def _get(q: Any, key: str, default: Any = None) -> Any:
    if isinstance(q, Mapping):
        return q.get(key, default)
    return getattr(q, key, default)


def simulate(queries: Sequence[Any], ranker: Callable[[Any], List[str]],
             grader: Callable[[Any, str], int], n_sessions: int, topn: int = 10,
             eta: float = 1.0, attract: Sequence[float] = DEFAULT_ATTRACT,
             p_order_given_click: Sequence[float] = DEFAULT_P_ORDER,
             noise_shuffle: float = 0.0, seed: int = 0,
             weights: Optional[Sequence[float]] = None) -> List[Impression]:
    """Simulate ``n_sessions`` search sessions under the PBM.

    Args:
        queries: query objects/dicts with ``qid`` and ``text`` (a query log is
            already Zipf-sampled, so sessions draw queries uniformly unless
            ``weights`` are given).
        ranker: ``query -> ranked doc ids`` (the production/logging policy);
            called once per distinct ``qid`` and cached.
        grader: ``(query, doc_id) -> latent grade 0..3``; cached per pair.
        n_sessions: number of impressions to generate.
        topn: page size; only the top ``topn`` results can be examined.
        eta: position-bias severity (0 = no bias).
        attract: click probability of an examined doc per grade.
        p_order_given_click: order probability per grade, given a click.
        noise_shuffle: fraction of sessions whose top-``topn`` is shown in a
            uniformly random order (a RandTop-n intervention; makes position
            independent of relevance in those sessions).
        seed: RNG seed.
        weights: optional per-query sampling weights.
    """
    if len(attract) != 4 or len(p_order_given_click) != 4:
        raise ValueError("attract and p_order_given_click need one value per grade 0..3")
    if not queries:
        return []
    rng = random.Random(seed)
    rankings: Dict[str, List[str]] = {}
    grades: Dict[Tuple[str, str], int] = {}
    exam = [examination_prob(r, eta) for r in range(1, topn + 1)]
    cum = None
    if weights is not None:
        cum, acc = [], 0.0
        for w in weights:
            acc += float(w)
            cum.append(acc)
    out: List[Impression] = []
    ts = _T0
    for s in range(n_sessions):
        q = rng.choices(queries, cum_weights=cum)[0] if cum else queries[rng.randrange(len(queries))]
        qid = str(_get(q, "qid"))
        if qid not in rankings:
            rankings[qid] = list(ranker(q))[:topn]
        docs = list(rankings[qid])
        shuffled = noise_shuffle > 0 and rng.random() < noise_shuffle
        if shuffled:
            rng.shuffle(docs)
        clicks, orders = [], []
        for r, d in enumerate(docs):
            key = (qid, d)
            if key not in grades:
                grades[key] = max(0, min(3, int(grader(q, d))))
            g = grades[key]
            c = 1 if rng.random() < exam[r] * attract[g] else 0
            o = 1 if c and rng.random() < p_order_given_click[g] else 0
            clicks.append(c)
            orders.append(o)
        ts += rng.expovariate(1.0 / 30.0)  # ~30 s between sessions
        out.append(Impression(session_id=f"s{s:08d}", qid=qid,
                              query_norm=normalize(str(_get(q, "text", ""))), ts=round(ts, 3),
                              doc_ids=docs, clicks=clicks, orders=orders, shuffled=shuffled))
    return out


# ---------------------------------------------------------------------------
# CTR / CVR table
# ---------------------------------------------------------------------------

@dataclass
class CtrTable:
    """Smoothed engagement rates for ranking features.

    Smoothing (empirical-Bayes style, Beta prior with strength ``alpha + beta``):

    * doc prior:  ``ctr_doc = (clicks_d + alpha) / (impressions_d + alpha + beta)``
    * pair:       ``ctr = (clicks + m * ctr_doc) / (impressions + m)``, ``m = alpha + beta``
    * conversion: same scheme with clicks as trials and orders as successes.
    * ``coec`` (clicks over expected clicks, only when ``eta`` is given):
      ``(clicks + alpha) / (sum of examination propensities + alpha + beta)`` —
      a position-debiased CTR.
    """
    pairs: Dict[Tuple[str, str], Dict[str, float]]
    docs: Dict[str, Dict[str, float]]
    alpha: float
    beta: float
    global_ctr: float
    global_cvr: float

    def lookup(self, query_norm: str, doc_id: str) -> Dict[str, float]:
        """Pair stats, falling back to the doc prior, then the global prior."""
        p = self.pairs.get((query_norm, doc_id))
        if p is not None:
            return p
        d = self.docs.get(doc_id)
        if d is not None:
            return {"impressions": 0, "clicks": 0, "orders": 0,
                    "ctr_smoothed": d["ctr_smoothed"], "cvr_smoothed": d["cvr_smoothed"]}
        return {"impressions": 0, "clicks": 0, "orders": 0,
                "ctr_smoothed": self.global_ctr, "cvr_smoothed": self.global_cvr}

    def to_records(self) -> List[dict]:
        recs = [{"kind": "pair", "query_norm": q, "doc_id": d, **v}
                for (q, d), v in sorted(self.pairs.items())]
        recs += [{"kind": "doc", "doc_id": d, **v} for d, v in sorted(self.docs.items())]
        return recs


def ctr_table(impressions: Sequence[Impression], alpha: float = 1.0, beta: float = 20.0,
              eta: Optional[float] = None) -> CtrTable:
    """Aggregate impressions into smoothed per-(query_norm, doc) CTR/CVR plus
    doc-level priors (see :class:`CtrTable`)."""
    pair_raw: Dict[Tuple[str, str], List[float]] = {}
    doc_raw: Dict[str, List[float]] = {}
    for imp in impressions:
        for r, d in enumerate(imp.doc_ids):
            e = examination_prob(r + 1, eta) if eta is not None else 0.0
            for agg, key in ((pair_raw, (imp.query_norm, d)), (doc_raw, d)):
                row = agg.setdefault(key, [0.0, 0.0, 0.0, 0.0])  # impr, clicks, orders, exam
                row[0] += 1
                row[1] += imp.clicks[r]
                row[2] += imp.orders[r]
                row[3] += e
    m = alpha + beta
    prior_ctr = alpha / m
    docs: Dict[str, Dict[str, float]] = {}
    for d, (n, c, o, e) in doc_raw.items():
        docs[d] = {"impressions": int(n), "clicks": int(c), "orders": int(o),
                   "ctr_smoothed": (c + alpha) / (n + m), "cvr_smoothed": (o + alpha) / (c + m)}
        if eta is not None:
            docs[d]["coec"] = (c + alpha) / (e + m)
    pairs: Dict[Tuple[str, str], Dict[str, float]] = {}
    for (q, d), (n, c, o, e) in pair_raw.items():
        dp = docs[d]
        row = {"impressions": int(n), "clicks": int(c), "orders": int(o),
               "ctr_smoothed": (c + m * dp["ctr_smoothed"]) / (n + m),
               "cvr_smoothed": (o + m * dp["cvr_smoothed"]) / (c + m)}
        if eta is not None:
            row["coec"] = (c + m * dp["coec"]) / (e + m)
        pairs[(q, d)] = row
    tot_n = sum(v[0] for v in doc_raw.values())
    tot_c = sum(v[1] for v in doc_raw.values())
    tot_o = sum(v[2] for v in doc_raw.values())
    g_ctr = (tot_c + alpha) / (tot_n + m) if tot_n else prior_ctr
    g_cvr = (tot_o + alpha) / (tot_c + m) if tot_c else prior_ctr
    return CtrTable(pairs=pairs, docs=docs, alpha=alpha, beta=beta, global_ctr=g_ctr, global_cvr=g_cvr)


# ---------------------------------------------------------------------------
# IPW click labels
# ---------------------------------------------------------------------------

def default_thresholds(attract: Sequence[float] = DEFAULT_ATTRACT) -> Tuple[float, ...]:
    """Grade boundaries at the midpoints between consecutive attractiveness levels."""
    return tuple((attract[i] + attract[i + 1]) / 2.0 for i in range(len(attract) - 1))


def ipw_click_labels(impressions: Sequence[Impression], eta: float = 1.0,
                     thresholds: Optional[Sequence[float]] = None,
                     min_propensity: float = 0.05, min_impressions: int = 1
                     ) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Inverse-propensity-weighted relevance estimate per ``(qid, doc_id)``.

    ``ipw = (1/n) * sum_i click_i / max(p(rank_i), min_propensity)`` clipped to
    [0, 1], where ``p(r) = (1/r)**eta`` must match the bias the log was
    generated (or estimated) with. Under the PBM ``E[ipw] = attract[grade]``.
    Propensity clipping trades a little bias for much lower variance at deep ranks.

    Returns:
        ``(qid, doc_id) -> {"ipw", "impressions", "clicks", "orders", "grade"}``
        where ``grade = #thresholds below ipw`` (default thresholds: midpoints of
        ``DEFAULT_ATTRACT`` = 0.15 / 0.425 / 0.775). Pairs with fewer than
        ``min_impressions`` impressions are omitted.
    """
    th = sorted(thresholds if thresholds is not None else default_thresholds())
    acc: Dict[Tuple[str, str], List[float]] = {}
    for imp in impressions:
        for r, d in enumerate(imp.doc_ids):
            p = max(examination_prob(r + 1, eta), min_propensity)
            row = acc.setdefault((imp.qid, d), [0.0, 0.0, 0.0, 0.0])  # n, w-clicks, clicks, orders
            row[0] += 1
            row[1] += imp.clicks[r] / p
            row[2] += imp.clicks[r]
            row[3] += imp.orders[r]
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for key, (n, wc, c, o) in acc.items():
        if n < min_impressions:
            continue
        est = min(1.0, max(0.0, wc / n))
        out[key] = {"ipw": est, "impressions": int(n), "clicks": int(c), "orders": int(o),
                    "grade": sum(1 for t in th if est >= t)}
    return out


def click_qrels(labels: Mapping[Tuple[str, str], Mapping[str, float]],
                annotator: str = CLICK_ANNOTATOR) -> List[dict]:
    """Qrel records (``source = "clicks"``) from :func:`ipw_click_labels` output."""
    return [{"qid": q, "doc_id": d, "grade": int(v["grade"]), "source": "clicks", "annotator": annotator}
            for (q, d), v in sorted(labels.items())]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_impressions(path: str, impressions: Sequence[Impression]) -> int:
    return write_jsonl(path, (i.to_dict() for i in impressions))


def load_impressions(path: str) -> List[Impression]:
    return [Impression.from_dict(r) for r in read_jsonl(path)]
