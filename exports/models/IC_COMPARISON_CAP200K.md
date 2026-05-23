# Iron Condor cross-underlying comparison @ ₹2L capital

**Date:** 2026-05-23
**Window:** 2023-05-15 → 2026-05-15
**Capital:** ₹200,000
**Lot sizing:** peak-safe (every entry openable at broker)
**Volume filter:** **min_leg_volume = 100** — rejects historical leg-days where any of the 4 legs had volume < 100 contracts (purely-MTM fantasy fills that no real broker could have executed)

## ⚠️ Critical finding

When the backtest is forced to use only fills that had real volume, **every Iron Condor variant on every underlying delivers < 3 % CAGR at ₹2L**. The earlier "+112 % CAGR safe-tight" claim was driven almost entirely by fantasy fills against zero-volume option bars.

Equity momentum (`momentum_n100_top5_max1` at +87 % CAGR) beats every IC variant by 30-90×. Iron Condors are NOT a viable income strategy at ₹2L capital with realistic execution.

## Full ranking (sorted by CAGR)

| Rank | Underlying | Variant | Lots | Trades | Vol-Rej % | WR % | CAGR % | Total % | Max DD % | Avg Margin | Peak Margin |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | FINNIFTY | otm3_w300 | 1 | 28 | 22.2 | 92.9 | **+2.6** | +7.9 | -5.9 | ₹81,674 | ₹120,815 |
| 2 | FINNIFTY | otm4_w300 | 1 | 25 | 28.6 | 80.0 | +0.8 | +2.2 | -3.1 | ₹89,724 | ₹121,347 |
| 3 | FINNIFTY | otm3_w200 | 1 | 31 | 13.9 | 87.1 | +0.4 | +1.1 | -5.7 | ₹83,666 | ₹121,370 |
| 4 | FINNIFTY | otm2_w150 | 1 | 31 | 13.9 | 83.9 | +0.1 | +0.4 | -4.5 | ₹81,483 | ₹121,162 |
| 5 | NIFTY | otm5_w500 | 1 | 35 | 2.8 | 80.0 | +0.0 | +0.1 | -11.2 | ₹97,382 | ₹131,621 |
| 6 | FINNIFTY | otm2_w200 | 1 | 31 | 13.9 | 83.9 | -0.0 | -0.0 | -8.9 | ₹81,035 | ₹121,089 |
| 7 | FINNIFTY | otm4_w500 | 1 | 24 | 31.4 | 83.3 | -0.7 | -2.0 | -5.9 | ₹86,571 | ₹120,767 |
| 8 | NIFTY | otm2_w200 | 1 | 35 | 2.8 | 57.1 | -1.4 | -4.3 | -13.7 | ₹86,198 | ₹128,824 |
| 9 | BANKNIFTY | otm4_w500 | 1 | 34 | 0.0 | 67.6 | -1.5 | -4.5 | -10.4 | ₹74,245 | ₹120,281 |
| 10 | FINNIFTY | otm5_w500 | 1 | 14 | 58.8 | 71.4 | -1.6 | -4.1 | -4.4 | ₹89,662 | ₹120,894 |
| 11 | NIFTY | otm4_w500 | 1 | 36 | 0.0 | 66.7 | -1.7 | -5.1 | -19.7 | ₹96,076 | ₹132,648 |
| 12 | NIFTY | otm2_w150 | 1 | 35 | 0.0 | 57.1 | -2.2 | -6.4 | -12.7 | ₹85,645 | ₹128,128 |
| 13 | BANKNIFTY | otm2_w200 | 1 | 33 | 0.0 | 48.5 | -2.2 | -6.7 | -9.1 | ₹62,969 | ₹113,430 |
| 14 | BANKNIFTY | otm5_w500 | 1 | 30 | 3.2 | 66.7 | -2.6 | -7.5 | -9.1 | ₹80,734 | ₹120,976 |
| 15 | BANKNIFTY | otm2_w150 | 1 | 33 | 0.0 | 42.4 | -2.7 | -8.1 | -9.7 | ₹62,708 | ₹112,828 |
| 16 | BANKNIFTY | otm3_w300 | 1 | 33 | 0.0 | 54.5 | -2.9 | -8.3 | -11.4 | ₹69,437 | ₹118,171 |
| 17 | BANKNIFTY | otm3_w200 | 1 | 33 | 0.0 | 45.5 | -3.2 | -9.1 | -9.4 | ₹68,529 | ₹117,690 |
| 18 | NIFTY | otm3_w300 | 1 | 36 | 0.0 | 58.3 | -3.6 | -10.5 | -16.7 | ₹92,076 | ₹131,566 |
| 19 | BANKNIFTY | otm4_w300 | 1 | 34 | 0.0 | 32.4 | -4.1 | -11.6 | -12.1 | ₹74,874 | ₹119,913 |
| 20 | NIFTY | otm3_w200 | 1 | 36 | 0.0 | 55.6 | -5.1 | -14.6 | -17.8 | ₹90,952 | ₹131,199 |
| 21 | NIFTY | otm4_w300 | 1 | 36 | 0.0 | 61.1 | -6.4 | -18.2 | -21.5 | ₹94,754 | ₹132,292 |

## Per-underlying volume reality

The `Vol-Rej %` column is the share of historical IC entries that the volume filter rejected — i.e. how often that underlying's options were NOT actually fillable in live trading.

| Underlying | Avg Vol-Rej % across variants | Verdict |
|---|---:|---|
| **NIFTY 50** | 0.8 % | Deep liquidity. Every monthly cycle had real fills. |
| **BANKNIFTY** | 0.4 % | Same — high liquidity on all body strikes. |
| **FINNIFTY** | 24.6 % | **~1 in 4 monthly entries was a fantasy** without the volume gate. |

This is the source-of-truth for the live screenshot finding: FinNifty wings (and even body) DO have illiquid days. NIFTY 50 and BankNifty do not — they trade thick on every body strike. So why does NIFTY 50 still under-perform? Because the *body* premium relative to wing-debit is too small on NIFTY 50 — IV is structurally lower than FinNifty, so the credit-to-risk ratio is unfavourable even with thick liquidity.

## Recommendation

**Do NOT trade Iron Condors at ₹2L capital on any of these three underlyings.** The best variant (FinNifty OTM3/W300) clocks +2.6 % CAGR, max DD only -5.9 % — capital is parked, not earning. Bond funds beat it.

**For ₹2L of options income** the only path that survived earlier sweeps was equity momentum, not premium selling. See `momentum_n100_top5_max1` (+87 % CAGR / -6 % DD) — that strategy is live.

**If you raise capital to ≥ ₹6L** the original w150 / w300 IC variants at 5 lots become openable. Their backtested CAGRs (still volume-filtered) need to be re-run at the higher capital to see if margin headroom alone fixes the issue, or if the +112 % CAGR was fantasy on every level.

## How to reproduce

```bash
# On the prod VM (where the options DB lives):
docker exec trading_system_app python3 -m \
    tools.models.finnifty_ic_otm4_w300_lots5.compare_underlyings \
    --capital 200000 --min-leg-volume 100 --top 5

# Output: /tmp/ic_underlying_compare_cap200000.json
# Export folders: /app/exports/models/<underlying>_ic_<variant>_lots<L>_cap200k/
```

Drop `--min-leg-volume 0` to see the unfiltered (fantasy) numbers for comparison. JSON dump at `exports/models/ic_compare_cap200k.json`.

## Files

| Folder | What |
|---|---|
| `finnifty_ic_otm3_w300_lots1_cap200k/` | Best variant — CAGR +2.6 %, WR 92.9 %, 28 trades |
| `finnifty_ic_otm4_w300_lots1_cap200k/` | Second-best — CAGR +0.8 %, WR 80 %, 25 trades |
| `finnifty_ic_otm3_w200_lots1_cap200k/` | Third — CAGR +0.4 %, WR 87.1 %, 31 trades |
| `finnifty_ic_otm2_w150_lots1_cap200k/` | Original live config rescaled — CAGR +0.1 %, WR 83.9 % |
| `nifty_ic_otm5_w500_lots1_cap200k/` | Best NIFTY 50 — CAGR ~0 %, but full volume coverage |
| `ic_compare_cap200k.json` | Raw machine-readable dump of all 21 runs |
