# finnifty_ic_otm2_w150_lots5

Safe-tight FinNifty monthly Iron Condor — promoted to live 2026-05-22.

## Strategy

- **Underlying:** FINNIFTY (Nifty Financial Services Index)
- **Setup:** Iron Condor monthly expiry
  - SELL OTM 2% CE + OTM 2% PE (body — near-ATM, high credit, liquid)
  - BUY wings ±150 points further OTM (defined risk, wings still in liquid band)
- **Position size:** 5 lots
- **Stop:** 3× entry credit OR hold to expiry
- **Capital:** ₹200,000
- **Window:** 2023-05-15 .. 2026-05-15
- **Slippage:** realistic tiered (1× at ATM → 15× at >6% OTM)
- **Lot history applied:** 40 (pre Sep-2024) → 65 (Sep-2024) → 60 (2026)

## Final Result

- **Started with:** ₹200,000
- **Ended with:** ₹1,906,153
- **Total profit:** ₹1,706,153
- **Total return:** +853.1 %
- **3-yr compounded CAGR:** **+112.0 %**
- **Avg/yr (arithmetic):** +213.3 %
- **Trades:** 36
- **Win rate:** 77.8 %
- **Months tracked:** 33
- **Avg/mo:** +25.85 %
- **Max drawdown:** -32.7 %
- **Calmar (CAGR / \|DD\|):** 3.43
- **Max single-trade loss:** **17.5 % of capital** (vs 48-58% on wider variants)

## Yearly

| Year | Trades | Wins | WR | P&L | ROI on ₹2L |
|---|---:|---:|---:|---:|---:|
| 2023 (May→Dec) | 8 | 7 | 87.5 % | ₹355,046 | +177.52 % |
| 2024 | 12 | 9 | 75.0 % | ₹490,719 | +245.36 % |
| 2025 | 12 | 9 | 75.0 % | ₹434,677 | +217.34 % |
| 2026 (Jan→May) | 4 | 3 | 75.0 % | ₹425,710 | +212.86 % |

Notable: returns are **stable** across years (177-245%). Older OTM4/W300 variant swings 171→324% — this one is more predictable.

## Exit reasons

| Reason | Count | Avg P&L | Total P&L |
|---|---:|---:|---:|
| EXPIRY (held to monthly settlement) | 31 | +₹73,736 | +₹2,285,807 |
| SL (3× credit stop triggered) | 5 | -₹115,931 | -₹579,655 |

**86 % expiry rate** — strategy collects time decay almost every cycle. The 5 stops were sharp directional moves that breached the 2% OTM short strike.

<!-- MARGIN-BLOCK-START -->
## Margin (SPAN+exposure approx)

Approximation calibrated to live Sensibull basket 2026-05-23 ±2 %. See `compute_ic_margin` in `sweep.py` for the formula (SPAN 2.9 % of short notional + 0.5 % exposure − long-wing credit).

| Metric | Value |
|---|---:|
| Avg margin / trade | ₹319,162 |
| Peak margin / trade | ₹572,081 |
| Configured capital | ₹200,000 |
| Capital / avg-margin ratio | 0.63× |

> ⚠️ **Margin required exceeds configured capital.** Avg margin ₹319,162 > capital ₹200,000. The backtest assumed 5 lots could always be opened on ₹200k capital — but the live broker will block trades when funds are insufficient. Two ways to fix the gap:
> 1. **Increase capital** to ≥ ₹629,289 (≈ 1.1× peak margin) so every trade has headroom.
> 2. **Reduce lots** to keep avg margin ≤ ~80 % of capital.

<!-- MARGIN-BLOCK-END -->

## Files in this folder

| File | Description |
|---|---|
| `SUMMARY.md` | This document |
| `COMPARISON.md` | Side-by-side vs older OTM4/W300 variant |
| `trades.csv` | One row per IC trade (36 rows). Includes entry/exit/strikes/per-leg prices/P&L/drawdown |
| `orders.csv` | One row per leg ORDER (288 rows = 36 trades × 4 legs × 2 phases). Ready for broker reconciliation |
| `monthly.csv` | Monthly P&L roll-up |

## `orders.csv` schema

```
trade_idx, entry_date, exit_date, expiry, spot_at_entry, lot_size, lots, qty_per_leg,
exit_reason, phase (ENTRY|EXIT), leg (ce_short|pe_short|wce_long|wpe_long),
action (BUY|SELL), strike, price, qty, value_inr
```

`value_inr` sign convention: BUY = positive cash-out, SELL = negative (cash-in). Net of all 4 ENTRY rows = credit collected; net of all 4 EXIT rows = debit paid. Per-trade P&L = entry-net − exit-net.

## Reproducibility

```bash
# Reproduce on the VM
ssh root@77.42.45.12 "docker exec trading_system_app bash -c \
  'cd /app && python3 -m tools.models.finnifty_ic_otm4_w300_lots5.sweep \
   --realistic-slip --filter FN_IC_OTM2_w150_lots5'"

# Re-generate this export folder
ssh root@77.42.45.12 "docker exec trading_system_app python3 /tmp/gen_exports_otm2.py"
```

## Live deployment

- Runtime MODEL_NAME: `finnifty_ic_otm2_w150_lots5`
- Code folder (legacy name kept for cron stability): `tools/models/finnifty_ic_otm4_w300_lots5/`
- Executor: `tools/live/fyers_executor_options.py` (depth gate + LIMIT-walk for wings)
- Cron: signal 09:25, execute 09:32, monitor 14:30 (all IST)
- Live commit: `d44c1185` on `origin/main`
