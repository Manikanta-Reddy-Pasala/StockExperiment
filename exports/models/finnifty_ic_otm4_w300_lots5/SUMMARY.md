# FINNIFTY monthly Iron Condor — 3-entry-week comparison

**Window:** 2023-05-15 → 2026-05-15 (3 yr)
**Live params (OTM 4 / W 300 / 5 lots / 3× SL / 1 % slippage)** — runtime
MODEL_NAME `finnifty_ic_otm4_w300_lots5`. Capital ₹200,000.

## ⚠️ Critical methodology — all 4 legs must have volume on entry day

The backtest now **requires every one of the 4 IC legs to have non-zero
volume on the chosen entry day**. Without this, the backtest happily
"executed" against historical close prices on contracts that nobody
actually traded — purely a mark-to-market fantasy.

Within the chosen entry-week, the backtest walks Mon → Tue → Wed → Thu
→ Fri until a day where ALL 4 legs (CE short, PE short, CE wing, PE
wing) had at least 1 contract traded. If no weekday in the week
qualifies, that monthly cycle is skipped entirely. **Basket is atomic —
partial fill on an IC leaves uncapped risk on the remaining naked
short.**

## 🏆 Results — honest, tradeable-only

| Entry week | Trades | WR % | Total return | CAGR | Total ₹ P&L |
|---|---:|---:|---:|---:|---:|
| Week 1 (first viable day of week 1) | 10 | 20.0 | **-105.9 %** | n/a (blown out) | -₹211,727 |
| **Week 2 (first viable day of week 2)** ⭐ | **15** | **66.7** | **+20.6 %** | **+6.5 %/yr** | **+₹41,230** |
| Week 3 (first viable day of week 3) | 27 | 66.7 | -27.3 % | -10.1 %/yr | -₹54,634 |

**Only Week 2 is profitable.** +20.6 % over 3 years on ₹2L capital = +₹41k.
~ 1.05× FD return at zero credit risk (or 1.5× vs FD 7% on net basis).

## 🛡️ Liquidity profile

| Entry week | Zero-vol days % | Risky-fill days % | Median ₹/trade | Median trades/day |
|---|---:|---:|---:|---:|
| Week 1 | 4.4 % | 5.6 % | n/a (pre-UDiFF) | n/a |
| Week 2 | 2.1 % | 5.7 % | ₹4,184 | 41 |
| Week 3 | 0.1 % | 16.6 % | ₹3,180 | 1,168 |

Zero-vol % dropped dramatically vs the prior fantasy-fill version
(previously 22-33%). Confirms the entry-day gate is doing its job.

## 🔄 Reality vs. prior backtest

| Metric | OLD (fantasy fills allowed) | NEW (require 4-leg entry volume) |
|---|---:|---:|
| Week 1 trades | 35 | **10** |
| Week 1 return | +801 % | **-106 %** |
| Week 2 trades | 35 | **15** |
| Week 2 return | +1,034 % | **+20.6 %** |
| Week 3 trades | 36 | **27** |
| Week 3 return | +863 % | **-27 %** |

The old "10× capital" headline numbers ARE fantasy. ~57-72 % of
historical IC entries had at least one wing with 0 traded contracts on
the close price the backtest used. When you remove those, only
Week 2 produces a slight positive edge.

## 🎯 Decision

**This IC strategy at OTM 4 / W 300 / 5 lots is barely viable.**
+20.6 % over 3 years on ₹2L is +₹41k = ~₹13k/year ≈ matches FD return
with significantly more risk and operational complexity.

Possible paths forward:

1. **Pivot back to equity momentum** (`momentum_n100_top5_max1` +87 % CAGR
   walk-forward validated — already live).
2. **Try different IC geometry** — narrower wings (W 150, W 200) closer
   to ATM may have more tradeable wing volume. Re-run with
   `entry_week=2` + new geometry to check.
3. **Try weekly options instead of monthly** — far more volume on
   near-expiry contracts. Different code path needed.

## 🗂️ Files in this folder

| File | Description |
|---|---|
| `week<N>_trades.csv` | Per-IC-trade ledger (only trades that passed entry-volume gate) |
| `week<N>_daily_volumes.csv` | Per-leg per-held-day liquidity audit |
| `entry_weeks_summary.json` | Roll-up |
| `SUMMARY.md` | This document |

## Inspection commands

```bash
# Show all trade entries that survived the gate (only days where all 4 legs had volume)
awk -F',' 'NR==1 || $9>0' week2_daily_volumes.csv | head -20

# How many cycles were SKIPPED because no weekday in week 2 had all 4 legs?
# (Total monthly cycles in 3yr ≈ 36; week 2 placed 15 trades → 21 cycles skipped)

# Per-trade fill share — high risky days
awk -F',' 'NR>1 && $15>0.10 {print $1, $5, $6, $15}' week2_daily_volumes.csv
```

## Reproduce

```bash
ssh root@77.42.45.12 "docker exec trading_system_app python3 -m \
  tools.models.finnifty_ic_otm4_w300_lots5.run_entry_weeks"
```

## Recommendation

**Don't deploy this IC at ₹2L capital based on these honest numbers.**
The depth-gate in live executor (`tools/live/option_depth_check.py`)
will protect against fantasy fills at signal time, but with only 15
viable cycles per 3 years and +6.5 %/yr CAGR — it's not worth the
operational overhead vs `momentum_n100_top5_max1`.
