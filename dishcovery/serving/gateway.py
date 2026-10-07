"""HTTP gateway (aiohttp) in front of the broker.

    GET  /search?q=...&lat=..&lon=..&time=tue+13:00&k=20&system=full[&debug=1]
    POST /search                 JSON body with the same fields (+ "filters")
    GET  /autocomplete?q=chi&k=8 proxied to the FST suggest service
    POST /index[?refresh=true]   documents (JSON object or array); routed by restaurant to shards
    POST /delete                 {"ids": [...]}
    GET  /metrics                Prometheus text; /stats JSON; /health
    GET  /                       minimal search page for demos
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, Optional

import aiohttp
import orjson
from aiohttp import web

from dishcovery.config import Config, load_config
from dishcovery.serving.broker import SYSTEMS, Broker, SearchParams

_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Dishcovery</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font:15px/1.4 system-ui,sans-serif;max-width:860px;margin:24px auto;padding:0 16px;color:#222}
input,select{font:inherit;padding:6px}#q{width:50%}.r{padding:8px 0;border-bottom:1px solid #eee}
.m{color:#666;font-size:13px}.t{font-size:12px;color:#888}</style></head><body>
<h2>Dishcovery</h2>
<form id=f><input id=q placeholder="chicken biryani, cheap vegan near me open now…" autofocus>
<input id=lat value="39.9526" size=8><input id=lon value="-75.1652" size=9>
<select id=sys>%SYSTEMS%</select> <button>Search</button></form>
<div id=u class=m></div><div id=out></div>
<script>
const $=id=>document.getElementById(id);
$('f').onsubmit=async e=>{e.preventDefault();
 const p=new URLSearchParams({q:$('q').value,lat:$('lat').value,lon:$('lon').value,system:$('sys').value});
 const r=await (await fetch('/search?'+p)).json();
 const u=r.understanding||{};
 $('u').textContent=`intent=${u.intent} corrected="${u.corrected}" filters=${JSON.stringify(u.filters||{})} `+
   `· ${Object.entries(r.timings_ms||{}).map(([k,v])=>k+'='+v+'ms').join(' ')}`;
 $('out').innerHTML=(r.results||[]).map(x=>`<div class=r><b>${x.name}</b> · ${x.type} · ${x.restaurant_name}
  <div class=m>${(x.cuisines||[]).join(', ')} · ${'$'.repeat(x.price_tier||1)} · ★${x.rating} ·
  ${x.distance_km} km · ${x.open_now?'open':'closed'}</div><div class=t>${JSON.stringify(x.paths)}</div></div>`).join('');};
</script></body></html>"""


def _float(v: Optional[str]) -> Optional[float]:
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


def _params(d: Dict[str, Any]) -> SearchParams:
    return SearchParams(q=str(d.get("q", "")), lat=_float(d.get("lat")), lon=_float(d.get("lon")),
                        when=d.get("time") or d.get("when"), k=int(d.get("k", 20)), system=d.get("system", "full"),
                        debug=str(d.get("debug", "")).lower() in ("1", "true", "yes"),
                        use_cache=str(d.get("nocache", "")).lower() not in ("1", "true", "yes"),
                        filter_mode=d.get("filter_mode"), ef=int(d["ef"]) if d.get("ef") else None,
                        filters=d.get("filters") or {},
                        hedge=None if d.get("hedge") in (None, "") else str(d["hedge"]).lower() in ("1", "true"))


def _json(data: Any, status: int = 200) -> web.Response:
    return web.Response(body=orjson.dumps(data), status=status, content_type="application/json")


def load_components(cfg: Config):
    """QU artifacts, encoder, LTR model and behaviour store, whichever exist."""
    from dishcovery.embed.encoders import load_encoder
    from dishcovery.ranking.behavior import BehaviorStore
    from dishcovery.ranking.ltr import LTRModel

    qu = None
    if os.path.exists(os.path.join(cfg.qu_dir, "lexicon", "vocab.tsv")):
        from dishcovery.qu.pipeline import QueryUnderstanding
        qu = QueryUnderstanding.load(cfg.qu_dir)
    enc = load_encoder(cfg["embedding"]["model"], cfg["embedding"]["dim"], cfg["embedding"].get("device"))
    model_path = cfg.path("ltr", "model.pkl")
    ltr = LTRModel.load(model_path) if os.path.exists(model_path) else None
    beh = BehaviorStore.maybe_load(cfg.path("clicks", "history.jsonl"))
    return qu, enc, ltr, beh


def create_app(cfg: Config, broker: Optional[Broker] = None) -> web.Application:
    app = web.Application(client_max_size=256 * 1024 * 1024)

    async def on_startup(app_):
        if broker is None:
            qu, enc, ltr, beh = load_components(cfg)
            app_["broker"] = await Broker(cfg, qu, enc, ltr, beh).start()
        else:
            app_["broker"] = await broker.start()
        app_["http"] = aiohttp.ClientSession()

    async def on_cleanup(app_):
        await app_["broker"].close()
        await app_["http"].close()

    async def search_get(req: web.Request):
        p = _params(dict(req.query))
        if p.system not in SYSTEMS:
            return _json({"error": f"unknown system {p.system}", "systems": list(SYSTEMS)}, 400)
        return _json(await req.app["broker"].search(p))

    async def search_post(req: web.Request):
        p = _params(orjson.loads(await req.read()))
        return _json(await req.app["broker"].search(p))

    async def autocomplete(req: web.Request):
        q = req.query.get("q", "")
        k = req.query.get("k", "8")
        try:
            async with req.app["http"].get(f"{cfg.suggest_url}/suggest", params={"q": q, "k": k, "fuzzy": "1"},
                                           timeout=aiohttp.ClientTimeout(total=1.0)) as r:
                return web.Response(body=await r.read(), content_type="application/json")
        except Exception as e:
            return _json({"error": f"suggest service unavailable: {e}"}, 503)

    async def index(req: web.Request):
        body = orjson.loads(await req.read())
        docs = body if isinstance(body, list) else [body]
        refresh = req.query.get("refresh") == "true"
        return _json(await req.app["broker"].index_docs(docs, refresh=refresh))

    async def delete(req: web.Request):
        body = orjson.loads(await req.read())
        return _json(await req.app["broker"].delete_docs(body["ids"]))

    async def metrics(req: web.Request):
        return web.Response(text=req.app["broker"].metrics.prometheus(), content_type="text/plain")

    async def stats(req: web.Request):
        return _json(req.app["broker"].describe())

    async def health(req: web.Request):
        return _json({"ok": True, "generations": list(req.app["broker"].generations())})

    async def page(req: web.Request):
        opts = "".join(f"<option{' selected' if s == 'full' else ''}>{s}</option>" for s in SYSTEMS)
        return web.Response(text=_PAGE.replace("%SYSTEMS%", opts), content_type="text/html")

    app.router.add_get("/", page)
    app.router.add_get("/search", search_get)
    app.router.add_post("/search", search_post)
    app.router.add_get("/autocomplete", autocomplete)
    app.router.add_post("/index", index)
    app.router.add_post("/delete", delete)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/stats", stats)
    app.router.add_get("/health", health)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def _run_worker(config_path: Optional[str], port: int, reuse_port: bool) -> None:
    cfg = load_config(config_path)
    try:
        import uvloop  # type: ignore
        uvloop.install()
    except ImportError:
        pass
    web.run_app(create_app(cfg), host="0.0.0.0", port=port, access_log=None, reuse_port=reuse_port,
                print=None)


def serve(config_path: Optional[str] = None, port: Optional[int] = None, workers: Optional[int] = None) -> None:
    """Run the gateway; ``workers`` > 1 forks processes sharing the port (SO_REUSEPORT,
    kernel load-balanced on Linux) so QU, fusion and LTR scale past one core."""
    cfg = load_config(config_path)
    port = port or cfg["serving"]["gateway_port"]
    workers = workers or int(cfg["serving"].get("gateway_workers", 1))
    if workers <= 1:
        _run_worker(config_path, port, False)
        return
    import multiprocessing as mp

    procs = [mp.Process(target=_run_worker, args=(config_path, port, True), daemon=False) for _ in range(workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
