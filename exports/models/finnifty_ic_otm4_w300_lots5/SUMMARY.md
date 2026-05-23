# FINNIFTY monthly Iron Condor — 3-entry-week comparison

**Window:** 2023-05-15 → 2026-05-15 (3 yr)
**Live params (OTM 4 / W 300 / 5 lots / 3× SL / 1 % slippage)** — current
runtime MODEL_NAME still `finnifty_ic_otm4_w300_lots5`. Capital ₹200,000.

Backtest now **records** per-leg daily volume + num_trades +
traded_value_inr + our_share_of_traded — no rejection on liquidity, only
observation. Caller decides per-trade if fill is realistic.

## 🏆 Returns ranking — Week 2 wins

| Entry week | Trades | WR % | Total return | CAGR | Total ₹ P&L |
|---|---:|---:|---:|---:|---:|
| Week 1 (first weekday of week 1) | 35 | 57.1 | +801.3 % | +101 %/yr | ₹1,602,644 |
| **Week 2 (first weekday of week 2)** ⭐ | 35 | **77.1** | **+1033.6 %** | **+118 %/yr** | **₹2,067,115** |
| Week 3 (first weekday of week 3) | 36 | 75.0 | +863.3 % | +106 %/yr | ₹1,726,659 |

**Best for returns: Week 2.** +1,034 % over 3 years, 77 % WR, ₹20.67 lakh on ₹2L = **10.3× capital**.

## 🛡️ Liquidity ranking — Week 3 wins

| Entry week | Zero-vol days % | Risky-fill days % | Median ₹/trade | Median trades/day |
|---|---:|---:|---:|---:|
| Week 1 | 33.4 % | 22.9 % | ₹7,355 | 497 |
| Week 2 | 22.4 % | 20.2 % | ₹5,448 | 843 |
| **Week 3** ⭐ | **10.8 %** | **14.2 %** | ₹3,432 | **1,351** |

Definitions:
- **Zero-vol days %** — % of held-leg-days where that leg traded 0 contracts (no fill possible)
- **Risky-fill days %** — % of held-leg-days where our intended ₹ order > 10 % of day's traded value (won't fill cleanly)
- **Median ₹/trade** — typical fill size on that contract that day (post-Jul-2024 only; pre-UDiFF lacks trade count)
- **Median trades/day** — distinct executions on that leg per day (proxy for book depth)

## 🎯 Trade-off & recommendation

| If you optimize for… | Pick |
|---|---|
| **Maximum return** | Week 2 entry — +1,034 % but 20.2 % of days might not fill cleanly |
| **Maximum fill confidence** | Week 3 entry — +863 % AND only 10.8 % zero-vol + 14.2 % risky-fill |
| Compromise | Week 2 + skip any cycle where week-2 entry day has zero-vol on any leg (manual gate at signal time) |

**My pick: Week 2.** The +170 percentage-point lift over Week 3 (1034 % vs 863 %) is worth the moderately higher fill risk, especially since the depth-gate in live executor (`tools/live/option_depth_check.py`) already aborts baskets when any leg is too thin at signal time. Backtest "risky" days = days where intraday fill would have been worse than close — but live executor wouldn't have placed those orders at all.

## 🗂️ Files in this folder

| File | Description |
|---|---|
| `week1_trades.csv` | 35 IC trades, entered week 1 of monthly cycle |
| `week2_trades.csv` | 35 IC trades, entered week 2 |
| `week3_trades.csv` | 36 IC trades, entered week 3 |
| `week<N>_daily_volumes.csv` | Per-leg per-held-day liquidity (volume, num_trades, notional_lakh, traded_value_inr, avg_trade_inr, our_share_of_traded) |
| `entry_weeks_summary.json` | Roll-up of all 3 variants |
| `SUMMARY.md` | This document |

## 📊 Trade distribution by year

| Year | Week 1 | Week 2 | Week 3 |
|---|---:|---:|---:|
| 2023 (May-Dec) | 8 | 7 | 8 |
| 2024 | 12 | 12 | 12 |
| 2025 | 12 | 12 | 12 |
| 2026 (Jan-May) | 3 | 4 | 4 |

## How to spot-check fill safety per trade

```bash
# Show all leg-days where trade_idx=5 (week 2) had our order > 10% of traded value
awk -F',' 'NR==1 || ($1==5 && $15>0.10)' week2_daily_volumes.csv

# Show all zero-volume leg-days across all trades for week 2
awk -F',' 'NR==1 || $9==0' week2_daily_volumes.csv

# Count cycles with at least one zero-vol leg-day per entry-week
for w in 1 2 3; do
  echo -n "Week $w cycles with ≥1 zero-vol leg-day: "
  awk -F',' 'NR>1 && $9==0 {print $1}' week${w}_daily_volumes.csv | sort -u | wc -l
done
```

## How to reproduce

```bash
ssh root@77.42.45.12 "docker exec trading_system_app python3 -m \
  tools.models.finnifty_ic_otm4_w300_lots5.run_entry_weeks"
```

Outputs to `/app/exports/models/finnifty_ic_otm4_w300_lots5/` inside the container.

## Live deployment

**Runtime MODEL_NAME:** `finnifty_ic_otm4_w300_lots5` (folder name preserved for cron path stability)

**Recommended live config (this analysis):** Week 2 entry — first trading day of week 2 of new monthly cycle. ~12 entries per year. Each cycle: depth-gate (`option_depth_check.gate_basket`) verifies legs are tradeable at signal time, LIMIT-walk on wings, MARKET on shorts.

If you also want maximum fill safety: gate the Week 2 entry on `our_share_of_traded < 0.10` at signal time. Skip the cycle when any leg fails. Expected ~10-20 % of cycles skipped per the backtest's risky-fill column.
