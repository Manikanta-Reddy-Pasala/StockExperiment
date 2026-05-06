# ASIANPAINT (ASIANPAINT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:15:00 (4996 bars)
- **Last close:** 2519.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 2 |
| ALERT3 | 3 |
| PENDING | 7 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 0 |
| ENTRY2 | 5 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 5
- **Target hits / Stop hits / Partials:** 0 / 5 / 0
- **Avg / median % per leg:** -2.13% / -1.93%
- **Sum % (uncompounded):** -10.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.13% | -10.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.13% | -10.6% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.13% | -10.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 12:15:00 | 3231.00 | 3159.07 | 3158.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-13 13:15:00 | 3242.95 | 3163.42 | 3161.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-09 09:15:00 | 3291.85 | 3297.58 | 3247.29 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-12 09:15:00 | 3249.35 | 3294.04 | 3250.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 09:15:00 | 3249.35 | 3294.04 | 3250.44 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-01-16 09:15:00 | 3298.00 | 3291.12 | 3251.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-16 10:15:00 | 3300.40 | 3291.21 | 3252.11 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-01-17 14:15:00 | 3244.50 | 3290.27 | 3253.73 | SL hit qty=1.00 sl=3244.50 alert=retest2 |
| CROSSOVER_SKIP | 2024-07-03 14:15:00 | 2924.45 | 2898.37 | 2898.29 | HTF filter: close below htf_sma |
| Cross detected — sustain check pending | 2024-09-10 11:15:00 | 3302.95 | 3130.54 | 3061.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-10 12:15:00 | 3308.50 | 3132.31 | 3062.93 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-09-11 09:15:00 | 3361.00 | 3139.45 | 3067.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 10:15:00 | 3372.10 | 3141.76 | 3069.42 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-09-19 09:15:00 | 3305.70 | 3206.04 | 3118.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 10:15:00 | 3318.00 | 3207.15 | 3119.58 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-09-24 14:15:00 | 3244.50 | 3225.25 | 3139.54 | SL hit qty=1.00 sl=3244.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-24 14:15:00 | 3244.50 | 3225.25 | 3139.54 | SL hit qty=1.00 sl=3244.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-24 14:15:00 | 3244.50 | 3225.25 | 3139.54 | SL hit qty=1.00 sl=3244.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 10:15:00 | 3148.55 | 3241.24 | 3163.04 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2025-04-16 11:15:00 | 2442.00 | 2322.15 | 2321.81 | HTF filter: close below htf_sma |
| CROSSOVER_SKIP | 2025-07-04 15:15:00 | 2424.20 | 2316.55 | 2316.38 | HTF filter: close below htf_sma |

### Cycle 2 — BUY (started 2025-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 15:15:00 | 2518.80 | 2438.26 | 2437.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 09:15:00 | 2524.60 | 2443.70 | 2440.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-09 14:15:00 | 2794.50 | 2797.80 | 2683.55 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 14:15:00 | 2756.40 | 2805.29 | 2754.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 2756.40 | 2805.29 | 2754.27 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-01-19 09:15:00 | 2765.60 | 2804.41 | 2754.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 10:15:00 | 2779.20 | 2804.15 | 2754.46 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-19 15:15:00 | 2750.90 | 2802.17 | 2754.69 | SL hit qty=1.00 sl=2750.90 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-23 10:15:00 | 2782.00 | 2780.07 | 2748.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-23 11:15:00 | 2755.60 | 2779.82 | 2748.16 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-23 12:15:00 | 2764.10 | 2779.67 | 2748.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-23 13:15:00 | 2729.00 | 2779.16 | 2748.15 | ENTRY2 sustain failed after 60m |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-01-16 10:15:00 | 3300.40 | 2024-01-17 14:15:00 | 3244.50 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-09-10 12:15:00 | 3308.50 | 2024-09-24 14:15:00 | 3244.50 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-09-11 10:15:00 | 3372.10 | 2024-09-24 14:15:00 | 3244.50 | STOP_HIT | 1.00 | -3.78% |
| BUY | retest2 | 2024-09-19 10:15:00 | 3318.00 | 2024-09-24 14:15:00 | 3244.50 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2026-01-19 10:15:00 | 2779.20 | 2026-01-19 15:15:00 | 2750.90 | STOP_HIT | 1.00 | -1.02% |
