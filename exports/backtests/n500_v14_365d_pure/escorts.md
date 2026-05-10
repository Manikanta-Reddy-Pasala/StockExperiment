# Escorts Kubota Ltd. (ESCORTS)

## Backtest Summary

- **Window:** 2025-10-27 09:15:00 → 2026-05-08 15:15:00 (917 bars)
- **Last close:** 3148.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 1 |
| ALERT3 | 4 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 7 / 8
- **Target hits / Stop hits / Partials:** 3 / 8 / 4
- **Avg / median % per leg:** 2.46% / -0.23%
- **Sum % (uncompounded):** 36.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 15 | 7 | 46.7% | 3 | 8 | 4 | 2.46% | 36.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 15 | 7 | 46.7% | 3 | 8 | 4 | 2.46% | 36.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 15 | 7 | 46.7% | 3 | 8 | 4 | 2.46% | 36.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 14:15:00 | 3465.90 | 3678.00 | 3678.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 09:15:00 | 3422.60 | 3673.60 | 3675.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 10:15:00 | 3600.10 | 3597.98 | 3633.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 10:15:00 | 3600.10 | 3597.98 | 3633.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 10:15:00 | 3600.10 | 3597.98 | 3633.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 15:00:00 | 3583.70 | 3644.01 | 3650.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 15:15:00 | 3404.51 | 3600.24 | 3626.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 3591.90 | 3589.65 | 3618.79 | SL hit (close>ema200) qty=0.50 sl=3589.65 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 14:15:00 | 3586.40 | 3590.58 | 3618.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 15:15:00 | 3584.90 | 3590.71 | 3618.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 11:15:00 | 3587.90 | 3590.88 | 3618.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 12:15:00 | 3407.08 | 3579.95 | 3610.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 12:15:00 | 3405.65 | 3579.95 | 3610.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 12:15:00 | 3408.51 | 3579.95 | 3610.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 3227.76 | 3518.24 | 3574.25 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 3226.41 | 3518.24 | 3574.25 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 3229.11 | 3518.24 | 3574.25 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 3263.00 | 3121.19 | 3266.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 12:30:00 | 3242.00 | 3189.87 | 3275.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 14:45:00 | 3240.60 | 3190.63 | 3274.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 14:15:00 | 3310.50 | 3201.16 | 3274.30 | SL hit (close>static) qty=1.00 sl=3287.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-28 14:15:00 | 3310.50 | 3201.16 | 3274.30 | SL hit (close>static) qty=1.00 sl=3287.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 11:00:00 | 3221.00 | 3209.66 | 3275.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 14:00:00 | 3233.80 | 3210.13 | 3274.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 3250.00 | 3210.81 | 3274.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 3307.00 | 3210.81 | 3274.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 3290.40 | 3211.60 | 3274.15 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 3290.40 | 3211.60 | 3274.15 | SL hit (close>static) qty=1.00 sl=3287.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 3290.40 | 3211.60 | 3274.15 | SL hit (close>static) qty=1.00 sl=3287.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:45:00 | 3270.60 | 3212.09 | 3274.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 10:45:00 | 3275.00 | 3213.27 | 3270.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:45:00 | 3276.60 | 3214.57 | 3270.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 3326.10 | 3218.35 | 3271.36 | SL hit (close>static) qty=1.00 sl=3325.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 3326.10 | 3218.35 | 3271.36 | SL hit (close>static) qty=1.00 sl=3325.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 3326.10 | 3218.35 | 3271.36 | SL hit (close>static) qty=1.00 sl=3325.60 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 3229.90 | 3225.48 | 3273.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-02-13 15:00:00 | 3583.70 | 2026-02-20 15:15:00 | 3404.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-13 15:00:00 | 3583.70 | 2026-02-25 09:15:00 | 3591.90 | STOP_HIT | 0.50 | -0.23% |
| SELL | retest2 | 2026-02-25 14:15:00 | 3586.40 | 2026-03-02 12:15:00 | 3407.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 15:15:00 | 3584.90 | 2026-03-02 12:15:00 | 3405.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 11:15:00 | 3587.90 | 2026-03-02 12:15:00 | 3408.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 14:15:00 | 3586.40 | 2026-03-09 09:15:00 | 3227.76 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-25 15:15:00 | 3584.90 | 2026-03-09 09:15:00 | 3226.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-26 11:15:00 | 3587.90 | 2026-03-09 09:15:00 | 3229.11 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-24 12:30:00 | 3242.00 | 2026-04-28 14:15:00 | 3310.50 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2026-04-24 14:45:00 | 3240.60 | 2026-04-28 14:15:00 | 3310.50 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2026-04-30 11:00:00 | 3221.00 | 2026-05-04 09:15:00 | 3290.40 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2026-04-30 14:00:00 | 3233.80 | 2026-05-04 09:15:00 | 3290.40 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2026-05-04 10:45:00 | 3270.60 | 2026-05-07 09:15:00 | 3326.10 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-05-06 10:45:00 | 3275.00 | 2026-05-07 09:15:00 | 3326.10 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2026-05-06 12:45:00 | 3276.60 | 2026-05-07 09:15:00 | 3326.10 | STOP_HIT | 1.00 | -1.51% |
