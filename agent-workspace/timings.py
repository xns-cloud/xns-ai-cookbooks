#!/usr/bin/env python3
"""
Wall-clock timings for the workspace operations this recipe performs.

The recipe argues that reads are cheap enough to do repeatedly. This
module is how a reader checks that claim on their own gateway instead of
taking it from the README: every write, read, list and presign is timed,
and the run finishes by printing what each one cost in wall time, bytes
and derived throughput.

Deliberately not here:

  * No results file. The table goes to stdout and nowhere else. A recipe
    that scatters artifacts on a reader's disk is harder to trust.
  * No cost column, and no comparison against anyone's published rate.
    This repository carries no dollar figures by construction. A rate in
    a scraped corpus cannot be retracted, and an estimated bill is a
    claim about someone else's pricing that goes stale without notice.
    Timings are measurements; bills are not.

Usage — the recipe wires this up for you. Standalone:

    from timings import Timings
    t = Timings()
    with t.record("put_object", key="work/brief.md") as m:
        m.bytes = len(payload)
        s3.put_object(...)
    print(t.table())
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Measurement:
    """One timed operation. `bytes` is set by the caller inside the block,
    because only the caller knows how much moved."""

    op: str
    key: str | None = None
    bytes: int | None = None
    seconds: float = 0.0
    failed: bool = False


def _human_bytes(n: int | None) -> str:
    if n is None:
        return "-"
    if n < 1024:
        return f"{n} B"
    for unit in ("KiB", "MiB", "GiB"):
        n /= 1024.0
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}"
    return f"{n:.1f} GiB"


def _human_rate(nbytes: int | None, seconds: float) -> str:
    # Throughput is only meaningful when bytes actually moved. A list or a
    # presign moves none, and dividing zero by a duration invites a reader
    # to compare a number that means nothing.
    if not nbytes or seconds <= 0:
        return "-"
    return f"{_human_bytes(int(nbytes / seconds))}/s"


@dataclass
class Timings:
    """Collects measurements and renders them as one aligned table."""

    rows: list[Measurement] = field(default_factory=list)

    @contextmanager
    def record(self, op: str, key: str | None = None):
        m = Measurement(op=op, key=key)
        start = time.perf_counter()
        try:
            yield m
        except Exception:
            m.failed = True
            raise
        finally:
            m.seconds = time.perf_counter() - start
            self.rows.append(m)

    def table(self) -> str:
        if not self.rows:
            return "no operations recorded"

        header = ("operation", "key", "bytes", "wall", "throughput")
        body = [
            (
                r.op + (" (failed)" if r.failed else ""),
                r.key or "-",
                _human_bytes(r.bytes),
                f"{r.seconds * 1000:.0f} ms",
                _human_rate(r.bytes, r.seconds),
            )
            for r in self.rows
        ]

        moved = sum(r.bytes or 0 for r in self.rows)
        total = sum(r.seconds for r in self.rows)
        footer = (
            f"{len(self.rows)} operations",
            "-",
            _human_bytes(moved) if moved else "-",
            f"{total * 1000:.0f} ms",
            _human_rate(moved, total),
        )

        widths = [
            max(len(row[i]) for row in (header, *body, footer))
            for i in range(len(header))
        ]

        def line(cells, fill=" "):
            return fill.join(
                # keys read left-to-right; numbers read right-to-left
                c.ljust(widths[i]) if i < 2 else c.rjust(widths[i])
                for i, c in enumerate(cells)
            ).rstrip()

        rule = "  ".join("-" * w for w in widths)
        out = ["", "storage operations, this run", "", line(header, "  "), rule]
        out += [line(b, "  ") for b in body]
        out += [rule, line(footer, "  ")]
        out += [
            "",
            "Wall time is the full client-side round trip: request signing, network,"
            " gateway and disk.",
            "Small objects are dominated by per-request overhead, so their throughput"
            " figure describes",
            "the round trip rather than your bandwidth. Compare like with like, and"
            " re-run before",
            "concluding anything from a single sample.",
        ]
        return "\n".join(out)
