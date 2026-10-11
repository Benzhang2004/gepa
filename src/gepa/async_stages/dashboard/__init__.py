"""A live dashboard for a run of the asynchronous engine.

The engine writes ``async_status.json`` beside its event log every few seconds (``Engine(status_every=...)``): how
full each place is, what runs in each body, what crossed each edge and at what rate over the last minute, what
ended where, and what each resource has left. This package serves a page that draws that snapshot as the
pipeline figure and keeps a history of it, and can replay a finished run from ``async_events.jsonl``.

Serve a run directory and open the printed address (over ssh, forward the port first)::

    python -m gepa.async_stages.dashboard runs/cbl/oa-async-k16-cap16 --port 8765

The page itself is static (``index.html`` and ``app.js``); any web server that exposes ``status.json`` and
``events.jsonl`` beside them works.
"""

from gepa.async_stages.dashboard.server import serve

__all__ = ["serve"]
