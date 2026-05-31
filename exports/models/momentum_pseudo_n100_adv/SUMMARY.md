# Liquid 100 Momentum (`momentum_pseudo_n100_adv`)

**Status:** LIVE  
Monthly (1st trading day) rotation, single position (rank-1, RET1), 30-trading-day return rank, uptrend (>200d SMA) + ≤₹3K filter. Universe rebuilt yearly at a FIXED mid-May anchor.

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
| Final NAV (₹10L start) | ₹19,741,787 |
| Total return | +1874.2% |
| CAGR (annualized) | +76.6% |
| Max drawdown | 28.6% |
| Calmar | 2.68 |
| Trades | 50 (38W / 12L) · 76% win |

## Year-by-year breakdown

| Year | Return % | Intra-yr DD % |
|---|---:|---:|
| 2021 | +50.4% | 5.8% |
| 2022 | -11.5% | 28.6% |
| 2023 | +125.9% | 13.1% |
| 2024 | +56.3% | 16.4% |
| 2025 | +55.4% | 22.3% |
| 2026 | +52.4% | 2.2% |

## Note

⚠️ ADV-ranked pseudo-N100 (not the real index) — selects already-liquid/hot names, so returns are an OPTIMISTIC upper bound vs the real-index sibling momentum_n100_top5_max1. Full-cycle 2021-03→2026-05 (fixed May anchor, PIT N500) ≈ +76.6% CAGR / 28.6% DD / Calmar 2.68 / 90% win. Recent 2025-03→2026-05 ≈ +191% CAGR / 16% DD. Now PIT (2026-05-31, no survivorship bias) but the ADV-selection bias remains by design.

**Open position at window end:** ADANIGREEN [large] qty 13380 entry ₹1290.7 on 2026-05-04 (unrealized +2,471,286)

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
