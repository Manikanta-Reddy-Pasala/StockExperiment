"""Capital simulator — equal-weight 200K across 53 nifty50 symbols.

Reads each case _summary.md, allocates capital/N per symbol, computes
INR P&L per symbol from sum_pct, prints final capital + winners/losers.
"""
import os
import sys

CAPITAL = 200_000


def parse_summary(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        in_table = False
        for line in f:
            if line.startswith("| Symbol"):
                in_table = True
                continue
            if in_table and line.startswith("|"):
                cells = [c.strip() for c in line.split("|")[1:-1]]
                if len(cells) < 12 or not cells[0] or cells[0].startswith("---"):
                    continue
                try:
                    sum_pct = float(cells[11].replace("%", ""))
                    legs = int(cells[3])
                    out.append((cells[0], sum_pct, legs))
                except ValueError:
                    pass
    return out


def simulate(case_dir, label):
    path = os.path.join(case_dir, "_summary.md")
    rows = parse_summary(path)
    n = len(rows)
    if n == 0:
        print(f"{label}: no data\n")
        return
    alloc = CAPITAL / n
    pnls = [(s, alloc * p / 100, l, p) for s, p, l in rows]
    total = sum(p for _, p, _, _ in pnls)
    final = CAPITAL + total
    winners = [(s, p, l, pct) for s, p, l, pct in pnls if p > 0]
    losers = [(s, p, l, pct) for s, p, l, pct in pnls if p < 0]
    flat = [(s, p, l, pct) for s, p, l, pct in pnls if p == 0]

    print(f"=== {label} ({n} symbols, {alloc:,.0f} INR/sym) ===")
    print(
        f"  start: {CAPITAL:,} INR -> final: {final:,.0f} INR | "
        f"P&L: {total:+,.0f} INR | ROI: {total/CAPITAL*100:+.2f}%"
    )
    print(f"  winners: {len(winners)} | losers: {len(losers)} | flat: {len(flat)}")
    if winners:
        print("  TOP WINNERS:")
        for s, p, l, pct in sorted(winners, key=lambda x: -x[1])[:10]:
            print(f"    + {s:<14}  {p:+10,.0f} INR  ({pct:+6.1f}%, {l} legs)")
    if losers:
        print("  WORST LOSERS:")
        for s, p, l, pct in sorted(losers, key=lambda x: x[1])[:10]:
            print(f"    - {s:<14}  {p:+10,.0f} INR  ({pct:+6.1f}%, {l} legs)")
    print()


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "/app/exports/backtests"
    for case in sys.argv[2:] or ["buy_plain", "sell_plain"]:
        simulate(os.path.join(base, case), case)
