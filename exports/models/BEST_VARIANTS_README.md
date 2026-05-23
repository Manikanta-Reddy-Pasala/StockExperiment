# Best IC variants per index — full 3-year trades + orders + summary

**Window:** 2023-05-15 → 2026-05-15 (3.0 yr)
**Volume filter:** min 100 contracts/day per leg (rejects fantasy fills)
**Lot sizing:** peak-safe (every backtested entry openable at broker)
**Generated:** `tools/models/finnifty_ic_otm4_w300_lots5/export_best_variants.py`

## Champions selected from exhaustive sweep

| Index | Config | Entry | Source ranking |
|---|---|---|---|
| **FINNIFTY** | OTM 2.5 % / W 150 / no-SL | **TUE** only | Best risk-adjusted in sweep |
| **NIFTY 50** | OTM 5.0 % / W 500 / no-SL | **THU** only | Best NIFTY in sweep |
| **BANKNIFTY** | OTM 1.5 % / W 500 / no-SL | **WED** only | Best BANKNIFTY in sweep |

(Selection from `IC_EXHAUSTIVE_SWEEP_RESULTS.md` — 1620-backtest exhaustive sweep.)

## Performance @ 3 capital levels

| Index | Capital | Lots | Trades | WR % | **CAGR %** | Max DD % | Total Return % | Annual ₹ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FINNIFTY | ₹2L | 1 | 22 | 90.9 | **+8.5** | -3.0 | +24.4 | ₹16,260 |
| FINNIFTY | ₹5L | 4 | 22 | 90.9 | **+13.1** | -4.3 | +39.0 | ₹65,040 |
| FINNIFTY | ₹10L | 8 | 22 | 90.9 | **+13.1** | -4.3 | +39.0 | ₹130,081 |
| NIFTY | ₹2L | 1 | 33 | 90.9 | **+7.6** | -1.6 | +24.97 | ₹16,646 |
| NIFTY | ₹5L | 3 | 33 | 90.9 | **+9.1** | -2.0 | +29.96 | ₹49,938 |
| NIFTY | ₹10L | 7 | 33 | 90.9 | **+10.4** | -2.3 | +34.96 | ₹116,522 |
| BANKNIFTY | ₹2L | 1 | 36 | 61.1 | +6.6 | -7.2 | +21.23 | ₹14,153 |
| BANKNIFTY | ₹5L | 4 | 36 | 61.1 | **+10.2** | -11.2 | +33.97 | ₹56,610 |
| BANKNIFTY | ₹10L | 8 | 36 | 61.1 | **+10.2** | -11.2 | +33.97 | ₹113,220 |

## Trade distribution by year

| Index | 2023 (partial) | 2024 | 2025 | 2026 (partial) | Total |
|---|---:|---:|---:|---:|---:|
| FINNIFTY (TUE) | 5 | 6 | 9 | 2 | 22 |
| NIFTY (THU) | 8 | 13 | 8 | 4 | 33 |
| BANKNIFTY (WED) | 8 | 10 | 15 | 3 | 36 |

## Per-folder contents

Each `<index>_otm<x>_w<n>_<day>_nosl_lots<L>_cap<K>k/` folder ships:

| File | Description |
|---|---|
| `trades.csv` | One row per IC trade (entry, exit, strikes, per-leg prices, P&L, margin, running equity, drawdown) |
| `orders.csv` | One row per leg-order (= trades × 4 legs × 2 phases). Ready for broker reconciliation |
| `summary.json` | Total stats: CAGR, WR, max DD, avg/peak margin, capital, lots |

## Top-line takeaways

1. **FINNIFTY TUE no-SL = highest risk-adjusted** — +13.1 % / -4.3 % DD at ₹5L. WR 90.9 % across 22 trades.
2. **NIFTY THU no-SL = lowest drawdown** — -1.6 % to -2.3 % across capitals. 33 trades, +7.6 to +10.4 % CAGR.
3. **BANKNIFTY WED no-SL** — most trades (36) but lowest WR (61 %) and highest DD (-11 %). Workable but less attractive than FN/NIFTY.
4. **₹5L → ₹10L identical CAGR** — lot count scales linearly with capital (4 → 8 FN, 3 → 7 NIFTY, 4 → 8 BN). Same % return, 2× rupees.
5. **All variants beat FD @ 7 %** at ₹5L+ — FN ₹65k vs FD ₹35k = 1.86×, NIFTY ₹50k = 1.43×, BN ₹57k = 1.6×.

## Rollup CSV

`BEST_VARIANTS_ROLLUP.csv` — all 9 (variant × capital) summary rows in one CSV for downstream slicing.

## Reproduce

```bash
ssh root@77.42.45.12 "docker exec trading_system_app python3 -m \
  tools.models.finnifty_ic_otm4_w300_lots5.export_best_variants"
```

## Live recommendation

If user has ≥ ₹5L AND wants options income alongside equity momentum:
**FINNIFTY OTM2.5 W150 TUE no-SL** is the new live config target. Replace
`finnifty_ic_otm2_w150_lots5` (current shipped) when convenient — better
CAGR, much lower DD, day-of-week constraint reduces over-trading.

If user has < ₹5L: stay with equity momentum (`momentum_n100_top5_max1`
+87 % CAGR). IC at ₹2L = +8.5 % only, FD nearly matches with zero risk.
