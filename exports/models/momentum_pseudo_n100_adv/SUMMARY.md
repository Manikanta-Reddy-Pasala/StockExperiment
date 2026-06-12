# Liquid 100 Momentum (`momentum_pseudo_n100_adv`)

**Status:** LIVE  
Monthly (1st trading day) rotation, single position (rank-1, RET1), 30-trading-day return rank, uptrend (>200d SMA) + ≤₹3K filter. Universe rebuilt yearly at a FIXED mid-May anchor. + DAILY from-entry ATR×3.0 hard stop (entry − 3×ATR(14)).

**Universe:** Top-100 by 20d ADV from N500 (yearly-PIT rebuild)

Backtest window: **2021-03-01 → 2026-05-31** (emerging → 2026-06-10; full ~5.2-year cycle: 2021 bull, 2022 correction, 2023-24 bull, 2025 chop, 2026 recovery). 3-yr window also reported in notes: **2023-05-15 → 2026-05-12**. REALISM CONVENTION (2026-06-13): all figures are **net of real Fyers CNC charges, with next-open fills** (decide on bar d's close, fill at bar d+1's open) and PIT universes.

## Trade rules

| When | Rule |
|---|---|
| **Rebalance** | 1st trading day of month (mid-month available as opt-in, default OFF). |
| **Universe & filters** | Top-100 by 20d ADV from PIT N500 (yearly fixed mid-May anchor) minus Smallcap-250; close > 200d SMA; price ≤ ₹3000. |
| **Entry** | BUY rank-1 by 30-day return (single position, max 1). |
| **Exit** | Rotate: SELL when the held name is no longer rank-1 (RETAIN=1). |
| **Source** | Live: niftyindices.com `ind_nifty500list.csv` + `ind_niftysmallcap250list.csv` → yearly_universes.json. Backtest: PIT `n500_membership.csv` (xlsx-verified). Prices: Fyers daily OHLCV. |

## Results (net of charges, next-open fills)

| Metric | Value |
|---|---|
| Final NAV (₹10L start) | ₹1,875,741 |
| Total return | +87.6% |
| CAGR (annualized) | +12.7% |
| Max drawdown | 59.4% |
| Calmar | 0.21 |
| Trades | 49 (26W / 23L) · 53% win |
| Total charges (real Fyers CNC, deducted) | ₹110,268 |

## Year-by-year breakdown

| Year | Return % | Intra-yr DD % |
|---|---:|---:|
| 2021 | -2.0% | 31.2% |
| 2022 | -17.4% | 34.5% |
| 2023 | +85.7% | 23.7% |
| 2024 | -23.2% | 46.3% |
| 2025 | -2.8% | 43.4% |
| 2026 | +55.5% | 14.6% |

## Note

⚠️⚠️ COLLAPSED UNDER PIT TREATMENT (2026-06-13 realism regen, net of charges + next-open fills + PIT smallcap-250 snapshots): full-cycle 2021-03→2026-05 +12.7% CAGR / 59.4% DD / Calmar 0.21 / 53% win; 3-yr 2023-05→2026-05 +23.9% / 59.4% DD / Calmar 0.40. The previously-published +77.4% was substantially SURVIVORSHIP-BIASED: the old smallcap-exclusion applied TODAY's Smallcap-250 list to every historical year, which silently kept names that were smallcap THEN but grew large (the multibaggers the model rode). Diagnostic isolation: realism alone (next-open fills + charges) = 77.4→66.5%; the PIT smallcap fix = 66.5→12.7%. The 'drop smallcaps, +2pp free' sweep finding is INVALIDATED. Model pending strategy-level review (possibly drop the smallcap exclusion entirely or re-validate). ADV-ranked pseudo-N100 (not the real index) — ADV-selection bias remains by design. From-entry ATR×3.0 hard stop unchanged (tools.shared.stops, backtest + live --stop-check, no drift).

**Open position at window end:** NSE:ADANIGREEN-EQ qty 1271 entry ₹1291.0 on ? (unrealized +0)

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
