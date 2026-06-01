# Liquid 100 Momentum (`momentum_pseudo_n100_adv`)

**Status:** LIVE  
Monthly (1st trading day) rotation, single position (rank-1, RET1), 30-trading-day return rank, uptrend (>200d SMA) + ≤₹3K filter. Universe rebuilt yearly at a FIXED mid-May anchor. + DAILY from-entry ATR×3.0 hard stop (entry − 3×ATR(14)).

**Universe:** Top-100 by 20d ADV from N500 (yearly-PIT rebuild)

Backtest window: **2021-03-01 → 2026-05-29** (full ~5.2-year cycle: 2021 bull, 2022 correction, 2023-24 bull, 2025 chop, 2026 recovery). Recent clean-data window also reported: **2025-03-01 → 2026-05-29**.

## Trade rules

| When | Rule |
|---|---|
| **Rebalance** | 1st trading day of month (mid-month available as opt-in, default OFF). |
| **Universe & filters** | Top-100 by 20d ADV from PIT N500 (yearly fixed mid-May anchor) minus Smallcap-250; close > 200d SMA; price ≤ ₹3000. |
| **Entry** | BUY rank-1 by 30-day return (single position, max 1). |
| **Exit** | Rotate: SELL when the held name is no longer rank-1 (RETAIN=1). |
| **Source** | Live: niftyindices.com `ind_nifty500list.csv` + `ind_niftysmallcap250list.csv` → yearly_universes.json. Backtest: PIT `n500_membership.csv` (xlsx-verified). Prices: Fyers daily OHLCV. |

## Results (net of costs)

| Metric | Value |
|---|---|
| Final NAV (₹10L start) | ₹20,239,089 |
| Total return | +1923.9% |
| CAGR (annualized) | +77.4% |
| Max drawdown | 43.8% |
| Calmar | 1.77 |
| Trades | 50 (37W / 13L) · 74% win |

## Year-by-year breakdown

| Year | Return % | Intra-yr DD % |
|---|---:|---:|
| 2021 | +55.3% | 31.6% |
| 2022 | -6.2% | 27.1% |
| 2023 | +137.8% | 19.7% |
| 2024 | +45.2% | 43.3% |
| 2025 | +114.7% | 26.5% |
| 2026 | +74.2% | 12.1% |

## Note

⚠️ ADV-ranked pseudo-N100 (not the real index) — selects already-liquid/hot names, an OPTIMISTIC upper bound vs the real-index sibling. NOW with a from-entry ATR×3.0 hard stop (2026-06-02, backtest-validated both windows): full-cycle 2021-03→2026-05 +77.4% CAGR / 43.8% DD / Calmar 1.77 / 74% win; recent 2025-03→2026-05 +209% CAGR / 16% DD. The stop is a FIXED level at entry−3×ATR (cuts genuine breakdowns, winners run to rotation); shared helper tools.shared.stops used by backtest + live --stop-check (no drift). DD is now on the stricter DAILY-MTM (intraday-low) basis — not comparable to the prior rebal-snapshot DD; the stop's gain is the within-basis delta (50.1→43.8). ADV-selection bias remains by design.

**Open position at window end:** NSE:ADANIGREEN-EQ qty 13717 entry ₹1290.7 on ? (unrealized +0)

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
