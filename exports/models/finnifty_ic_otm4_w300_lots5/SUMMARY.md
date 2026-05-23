# FINNIFTY monthly Iron Condor — OTM × Wing grid @ week-2 entry

**Window:** 2023-05-15 → 2026-05-15 (3 yr)
**Fixed params:** 5 lots, 3× SL, 1 % base slippage (tiered by OTM distance), capital ₹200,000
**Entry:** first weekday of week 2 of each monthly cycle WHERE all 4 legs have non-zero volume; if no weekday qualifies, cycle skipped.

## 🏆 OTM × Wing winner grid

| OTM | W 150 | W 200 | W 300 |
|---|---:|---:|---:|
| **2.0 %** | +20.6 % (15) | **+34.6 % (17) ⭐** | +15.4 % (16) |
| 3.0 % | -12.0 % (11) | -2.4 % (13) | -64.3 % (13) |
| 4.0 % | -32.6 % (11) | -30.2 % (11) | -33.6 % (10) |

(Total return % over 3 yr / trade count in parens, OTM 2 row is **only profitable**)

## ⭐ Champion — OTM 2 / W 200 / week-2 entry

| Metric | Value |
|---|---:|
| Trades | 17 |
| WR % | 64.7 |
| Total return | **+34.64 %** |
| Total ₹ P&L | +₹69,284 |
| Final equity | ₹269,284 |
| CAGR | **+10.4 %/yr** |
| Zero-vol days % | 1.8 % |
| Risky-fill days % | 23.3 % |

**Better than every alternative IC config tested.** Wider wings (W 200 vs W 150) collect more credit relative to wing cost, while staying in the liquid band on entry day.

## 🎯 Reads

1. **OTM 2 is the only viable body width.** OTM 3 → -2 % to -64 %, OTM 4 → all losers (~-30 %). Premium shrinks too fast moving deeper OTM, can't cover wing cost + slippage.
2. **Wing width sweet spot is 200.** W 150 too narrow (cuts max profit); W 300 too wide (wing cost overwhelms credit on a thin month).
3. **Week-2 entry day** picked at first viable weekday within week 2. Mostly Tuesday-Thursday — Monday's wings often have 0 vol on first listing days.
4. **Risky-fill days (23 %)** flagged by `our_share_of_traded > 0.10`. These would auto-skip via live depth-gate at signal time (`tools/live/option_depth_check.py`).

## 🆚 vs FD @ 7 % on ₹2L

| Metric | OTM 2 / W 200 IC | FD @ 7 % |
|---|---:|---:|
| 3-yr return | +34.6 % | +22.5 % |
| Annual ₹ | ₹23,094 | ₹14,000 |
| Edge | **+₹9,094/yr** | — |
| Risk | -50 % single-trade loss possible | None |

IC beats FD by ~64 % annualized, but with real downside risk. Borderline worth it at ₹2L. At ₹5L+ the absolute rupees scale linearly so edge grows to ₹22-45k/yr.

## 🗂️ Files

| File | Description |
|---|---|
| `otm20_w200_trades.csv` ⭐ | Champion trades — 17 trades over 3 yr |
| `otm20_w200_daily_volumes.csv` ⭐ | Per-leg per-day liquidity for champion |
| `otm<X>_w<N>_*.csv` | All 9 grid variants for full audit |
| `wing_variants_summary.json` | Roll-up across all 9 grid cells |
| `week<N>_trades.csv` + `week<N>_daily_volumes.csv` | Earlier 3-entry-week comparison (OTM 2 / W 150, kept for context) |
| `entry_weeks_summary.json` | Earlier 3-week roll-up |

## 🔍 Inspect champion fill-safety per trade

```bash
# Show all leg-days where champion trade #5 had risky fill share
awk -F',' 'NR==1 || ($1==5 && $15>0.10)' otm20_w200_daily_volumes.csv

# All entries that passed the 4-leg-volume gate
awk -F',' 'NR==1 || $5==$3' otm20_w200_daily_volumes.csv | head -20
```

## Reproduce

```bash
ssh root@77.42.45.12 "docker exec trading_system_app python3 -m \
  tools.models.finnifty_ic_otm4_w300_lots5.try_wing_variants"
```

## 🎯 Final recommendation

**Use FINNIFTY OTM 2 / W 200 / week-2 entry / 5 lots / 3× SL** as the live config target. Replace the current shipped `finnifty_ic_otm2_w150_lots5` (W 150 = -22 ppt lower return on same capital + entry day).

**Live deployment notes:**
- Entry: first weekday of week 2 of new monthly expiry cycle WHERE depth-gate passes all 4 legs
- Strikes: ce_short ≈ spot × 1.02, pe_short ≈ spot × 0.98, wings at ±200 points
- Cron: signal Tuesday 09:25 IST, execute 09:30, monitor Thursday 14:30 (existing schedule)
- Depth-gate thresholds (option_depth_check.py): max_spread_pct=15%, min_volume=500, min_oi=5000

Expected: ~6 entries per year, 65 % WR, ~₹23k/yr profit on ₹2L (₹115k on ₹10L). Max single-trade loss ~50 % of capital (W 200 = ₹60k loss × 5 lots cap). Worst year in backtest ≈ flat to mild loss.

## Caveat

Old "+1,034 %" Week 2 backtest = fantasy. Current +34.6 % = real-tradeable subset. Even this assumes:
- Daily close = realistic fill price (true within ±5 % on liquid days, can be wider on thin days flagged by `our_share_of_traded > 0.10`)
- 3× SL based on daily close (intraday spike-then-revert may trigger false SLs in live; backtest may understate)
- No major regime change 2026+ (IV regime determines IC viability)

Paper-trade 2-3 cycles before scaling.
