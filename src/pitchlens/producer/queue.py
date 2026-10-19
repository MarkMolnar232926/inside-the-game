"""Approval queue: what the producer sees, and the only path to air.

Lifecycle: ``pending`` -> ``approved`` (goes on air) | ``rejected`` | ``expired`` (stat went stale) |
``superseded`` (a newer proposal for the same player+metric replaced it). Approved proposals are
handed to the on-air channel once; the queue keeps a full history for the post-match log.
"""

from __future__ import annotations

import time

from .analyst import Proposal


class ApprovalQueue:
    def __init__(self, max_pending: int = 4):
        self.items: dict[str, Proposal] = {}
        self.max_pending = max_pending

    # ------------------------------------------------------------------ views
    def pending(self) -> list[Proposal]:
        return sorted((p for p in self.items.values() if p.status == "pending"), key=lambda p: (-p.score, p.t))

    def history(self) -> list[Proposal]:
        return sorted((p for p in self.items.values() if p.status != "pending"), key=lambda p: -(p.decided_wall or 0))

    def get(self, pid: str) -> Proposal:
        if pid not in self.items:
            raise KeyError(pid)
        return self.items[pid]

    # ------------------------------------------------------------------ changes
    def add(self, p: Proposal) -> list[Proposal]:
        """Add a proposal; returns proposals whose status changed as a side effect."""
        changed: list[Proposal] = []
        for q in self.pending():
            if q.player.id == p.player.id and q.metric == p.metric:
                q.status, q.decided_wall = "superseded", time.time()
                p.supersedes = q.id
                changed.append(q)
        self.items[p.id] = p
        pend = self.pending()
        for q in pend[self.max_pending:]:  # never more than a handful waiting: drop the weakest
            q.status, q.decided_wall = "expired", time.time()
            changed.append(q)
        return changed

    def expire(self, t_match: float) -> list[Proposal]:
        out = []
        for q in self.pending():
            if t_match > q.expires_t:
                q.status, q.decided_wall = "expired", time.time()
                out.append(q)
        return out

    def decide(self, pid: str, approve: bool, headline: str | None = None, why: str | None = None) -> Proposal:
        p = self.get(pid)
        if p.status != "pending":
            raise ValueError(f"{pid} is {p.status}, not pending")
        if headline is not None or why is not None:
            p.headline = headline if headline is not None else p.headline
            p.why = why if why is not None else p.why
            p.edited = True
        p.status = "approved" if approve else "rejected"
        p.decided_wall = time.time()
        return p
