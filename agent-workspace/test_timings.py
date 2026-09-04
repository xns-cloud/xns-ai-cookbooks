#!/usr/bin/env python3
"""
Offline tests for the timing harness. No gateway, no network, no LLM.

These cover the formatting decisions that are easy to get wrong in a way
nobody notices: a throughput figure printed for an operation that moved
no bytes, a failed call silently credited with a transfer, or a total row
that does not add up.

    python test_timings.py
"""

import sys

from timings import Timings, _human_bytes, _human_rate


def test_human_bytes():
    assert _human_bytes(None) == "-"
    assert _human_bytes(0) == "0 B"
    assert _human_bytes(41) == "41 B"
    assert _human_bytes(1782) == "1.7 KiB"
    assert _human_bytes(5 * 1024 * 1024) == "5.0 MiB"
    assert _human_bytes(3 * 1024**3) == "3.0 GiB"
    print("ok  byte formatting")


def test_rate_needs_bytes():
    # The point of the harness is honest numbers. An operation that moved
    # nothing has no throughput, and zero-over-a-duration is not it.
    assert _human_rate(None, 1.0) == "-"
    assert _human_rate(0, 1.0) == "-"
    assert _human_rate(1024, 0) == "-"
    assert _human_rate(1024, 1.0) == "1.0 KiB/s"
    print("ok  throughput only where bytes moved")


def test_records_and_totals():
    t = Timings()
    with t.record("put_object", "work/brief.md") as m:
        m.bytes = 1000
    with t.record("get_object", "work/brief.md") as m:
        m.bytes = 1000
    with t.record("list_objects_v2", "work/"):
        pass

    assert len(t.rows) == 3
    assert [r.op for r in t.rows] == ["put_object", "get_object", "list_objects_v2"]
    assert all(r.seconds > 0 for r in t.rows), "every row needs a duration"
    assert t.rows[2].bytes is None, "a list moves no object bytes"

    table = t.table()
    assert "3 operations" in table
    assert "2.0 KiB" in table, "total row should sum the bytes"
    print("ok  rows recorded and totalled")


def test_failure_is_kept_and_reraised():
    t = Timings()
    try:
        with t.record("put_object", "work/brief.md"):
            raise RuntimeError("gateway said no")
    except RuntimeError:
        pass
    else:
        raise AssertionError("record() must not swallow the exception")

    assert len(t.rows) == 1, "a failed call is still a round trip worth timing"
    assert t.rows[0].failed is True
    assert t.rows[0].bytes is None
    assert "(failed)" in t.table()
    assert t.rows[0].seconds > 0
    print("ok  failed call timed, marked, and re-raised")


def test_empty_table_says_so():
    assert Timings().table() == "no operations recorded"
    print("ok  empty run says so instead of printing a header")


def test_table_columns_align():
    t = Timings()
    with t.record("put_object", "a/very/long/key/that/stretches/the/column.md") as m:
        m.bytes = 12345
    with t.record("get", "k"):
        pass

    lines = t.table().splitlines()
    rules = [ln for ln in lines if set(ln) <= {"-", " "} and "-" in ln]
    assert rules, "expected a rule line"
    widths = [len(seg) for seg in rules[0].split("  ")]

    # The rule spans the widest cell in each column, so every header and
    # data line must start its second column at the same offset.
    offset = widths[0] + 2
    start = lines.index(rules[0]) - 1  # the header sits directly above
    for ln in lines[start:]:
        if not ln or ln in rules:
            continue
        if ln.startswith(("Wall time", "Small objects", "the round trip", "concluding")):
            break
        assert ln[:widths[0]].rstrip() == ln[:offset].strip(), f"column 1 overflows: {ln!r}"
        assert ln[widths[0]:offset] == "  ", f"column 2 misaligned: {ln!r}"
    print("ok  columns align")


def main():
    for fn in (
        test_human_bytes,
        test_rate_needs_bytes,
        test_records_and_totals,
        test_failure_is_kept_and_reraised,
        test_empty_table_says_so,
        test_table_columns_align,
    ):
        fn()
    print("\nall timing tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
