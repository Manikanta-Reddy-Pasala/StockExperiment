# Liquid 100 Momentum (`momentum_pseudo_n100_adv`)

**Status:** LIVE  
Monthly (1st trading day) rotation, single position (rank-1, RET1), 30-trading-day return rank, ≤₹3K filter (200d-SMA gate off). No smallcap exclusion (nosml). Universe rebuilt yearly at a FIXED mid-May anchor. + DAILY from-entry ATR×3.0 hard stop (entry − 3×ATR(14)).

**Universe:** Top-100 by 20d ADV from N500 (yearly-PIT rebuild)

Backtest window: **2021-03-01 → 2026-06-12** (₹10L capital; full ~5.3-year cycle: 2021 bull, 2022 correction, 2023-24 bull, 2025 chop, 2026 recovery). 3-yr window also reported in notes: **2023-05-15 → 2026-05-12**. REALISM CONVENTION (2026-06-13): all figures are **net of real Fyers CNC charges, with next-open fills** (decide on bar d's close, fill at bar d+1's open) and PIT universes.

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
| Final NAV (₹10L start) | ₹14,340,569 |
| Total return | +1334.1% |
| CAGR (annualized) | +65.6% |
| Max drawdown | 44.9% |
| Calmar | 1.46 |
| Trades | 52 (38W / 14L) · 73% win |
| Total charges (real Fyers CNC, deducted) | ₹413,998 |

## Year-by-year breakdown

| Year | Return % | Intra-yr DD % |
|---|---:|---:|
| 2021 | +43.2% | 15.7% |
| 2022 | -3.3% | 36.4% |
| 2023 | +129.3% | 23.5% |
| 2024 | +39.9% | 44.9% |
| 2025 | +107.1% | 28.4% |
| 2026 | +45.1% | 17.9% |

## Note

2026-06-13 'nosml' rework — Smallcap-250 exclusion DROPPED (EXCLUDE_SMALLCAP=False, backtest + live in parity). Net of charges + next-open fills + PIT N500: full-cycle 2021-03→2026-06 +65.6% CAGR / 44.9% DD / Calmar 1.46 / 52 trades / 73.1% win; 3-yr 2023-06→2026-06 +109.3% / 44.9% DD / Calmar 2.43 / 80% win; since Mar-2025 +158.8% / 25.8% DD / Calmar 6.15. Per-year net: 2021 +43 / 2022 −3 / 2023 +129 / 2024 +40 / 2025 +107 / 2026 +45 (every year positive bar a flat 2022). Walk-forward-validated: stitched OOS 2023→2026 +60.3% CAGR / Calmar 1.34 vs the old smallcap-excluded config +23.8% / 0.51, beating every fold (adversarially re-verified). The old exclusion was survivorship-biased — applying TODAY's Smallcap-250 list to every historical year deleted the ADV-rising midcap winners the model rides (collapsed full-cycle CAGR to ~13% under PIT); the prior published +77.4% was that bias. ADV-ranked pseudo-N100 (not the real index) — ADV-selection bias remains by design. HIGH-DD sleeve (~45% full) — size accordingly in the blend. From-entry ATR×3.0 hard stop unchanged (tools.shared.stops, backtest + live --stop-check, no drift).

**Open position at window end:** NSE:HFCL-EQ qty 83443 entry ₹180.0 on ? (unrealized +0)

---
*Auto-generated from summary.json by tools/analysis/refresh_export_docs.py — do not hand-edit.*
