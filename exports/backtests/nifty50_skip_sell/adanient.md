# ADANIENT (ADANIENT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:15:00 (4996 bars)
- **Last close:** 2540.30
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 4 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 2 |
| PENDING | 6 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 1 |
| ENTRY2 | 3 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** 2.10% / 5.80%
- **Sum % (uncompounded):** 8.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 0 | 4 | 0 | 2.10% | 8.4% |
| BUY @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.00% | 1.0% |
| BUY @ 3rd Alert (retest2) | 3 | 2 | 66.7% | 0 | 3 | 0 | 2.47% | 7.4% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.00% | 1.0% |
| retest2 (combined) | 3 | 2 | 66.7% | 0 | 3 | 0 | 2.47% | 7.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 09:15:00 | 2529.00 | 2480.99 | 2480.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 12:15:00 | 2541.40 | 2306.17 | 2346.09 | Break + close above crossover candle high |

### Cycle 2 — BUY (started 2023-12-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 14:15:00 | 2891.00 | 2383.34 | 2382.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 13:15:00 | 2924.20 | 2546.51 | 2472.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 14:15:00 | 2895.00 | 2908.15 | 2769.51 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-01-29 09:15:00 | 3046.70 | 2905.26 | 2778.65 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-29 10:15:00 | 3045.95 | 2906.66 | 2779.98 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 3033.35 | 3197.75 | 3076.27 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-03-13 09:15:00 | 3076.27 | 3197.75 | 3076.27 | SL hit qty=1.00 sl=3076.27 alert=retest1 |
| Cross detected — sustain check pending | 2024-03-26 09:15:00 | 3141.00 | 3141.69 | 3073.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-03-26 10:15:00 | 3130.00 | 3141.57 | 3073.80 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-03-28 10:15:00 | 3155.60 | 3139.21 | 3077.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 11:15:00 | 3156.40 | 3139.38 | 3077.55 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-04-19 09:15:00 | 2981.40 | 3167.87 | 3119.19 | SL hit qty=1.00 sl=2981.40 alert=retest2 |
| Cross detected — sustain check pending | 2024-05-21 11:15:00 | 3145.95 | 3025.97 | 3050.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 12:15:00 | 3159.85 | 3027.30 | 3051.40 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-05-22 14:15:00 | 3142.75 | 3035.18 | 3054.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-05-22 15:15:00 | 3137.00 | 3036.19 | 3054.78 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-05-23 09:15:00 | 3165.05 | 3037.47 | 3055.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 10:15:00 | 3200.05 | 3039.09 | 3056.05 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-24 14:15:00 | 3385.60 | 3074.56 | 3073.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-24 14:15:00 | 3385.60 | 3074.56 | 3073.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 14:15:00 | 3385.60 | 3074.56 | 3073.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 14:15:00 | 3410.90 | 3136.60 | 3107.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 3121.05 | 3176.00 | 3128.95 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 3121.05 | 3176.00 | 3128.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 3121.05 | 3176.00 | 3128.95 | EMA400 retest candle locked |

### Cycle 4 — BUY (started 2024-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 13:15:00 | 3128.80 | 3077.40 | 3077.34 | EMA200 above EMA400 |
| CROSSOVER_SKIP | 2025-04-24 10:15:00 | 2451.50 | 2342.97 | 2342.53 | HTF filter: close below htf_sma |
| CROSSOVER_SKIP | 2025-05-12 09:15:00 | 2387.20 | 2344.51 | 2344.45 | HTF filter: close below htf_sma |

### Cycle 5 — BUY (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 14:15:00 | 2616.50 | 2422.51 | 2422.42 | EMA200 above EMA400 |

### Cycle 6 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 2323.20 | 2101.36 | 2100.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 2394.20 | 2106.45 | 2103.03 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-01-29 10:15:00 | 3045.95 | 2024-03-13 09:15:00 | 3076.27 | STOP_HIT | 1.00 | 1.00% |
| BUY | retest2 | 2024-03-28 11:15:00 | 3156.40 | 2024-04-19 09:15:00 | 2981.40 | STOP_HIT | 1.00 | -5.54% |
| BUY | retest2 | 2024-05-21 12:15:00 | 3159.85 | 2024-05-24 14:15:00 | 3385.60 | STOP_HIT | 1.00 | 7.14% |
| BUY | retest2 | 2024-05-23 10:15:00 | 3200.05 | 2024-05-24 14:15:00 | 3385.60 | STOP_HIT | 1.00 | 5.80% |
