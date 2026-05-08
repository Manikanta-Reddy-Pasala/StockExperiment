# BAJFINANCE (BAJFINANCE)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 954.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 3 |
| ALERT3 | 3 |
| PENDING | 4 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 0 |
| ENTRY2 | 2 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 2 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 2
- **Target hits / Stop hits / Partials:** 0 / 2 / 0
- **Avg / median % per leg:** -3.25% / -3.07%
- **Sum % (uncompounded):** -6.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.25% | -6.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.25% | -6.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.25% | -6.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 881.45 | 919.61 | 919.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 873.90 | 918.43 | 919.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 915.20 | 896.46 | 906.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 915.20 | 896.46 | 906.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 915.20 | 896.46 | 906.40 | EMA400 retest candle locked (from downside) |
| CROSSOVER_SKIP | 2025-09-10 13:15:00 | 967.00 | 908.05 | 908.01 | min_gap filter: gap=0.004% < 0.010% |
| TREND_RESET | 2025-09-10 13:15:00 | 967.00 | 908.05 | 908.01 | EMA inversion without crossover edge (EMA200=908.05 EMA400=908.01) — end cycle |

### Cycle 2 — SELL (started 2026-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 15:15:00 | 975.00 | 1008.84 | 1008.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 09:15:00 | 968.45 | 1003.44 | 1006.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 955.95 | 954.92 | 974.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 10:15:00 | 969.95 | 955.52 | 974.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 969.95 | 955.52 | 974.09 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-02-05 09:15:00 | 957.30 | 956.05 | 973.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-05 10:15:00 | 961.90 | 956.11 | 973.75 | ENTRY2 sustain failed after 60m |
| CROSSOVER_SKIP | 2026-02-23 11:15:00 | 1037.60 | 983.25 | 983.17 | min_gap filter: gap=0.008% < 0.010% |
| TREND_RESET | 2026-02-23 11:15:00 | 1037.60 | 983.25 | 983.17 | EMA inversion without crossover edge (EMA200=983.25 EMA400=983.17) — end cycle |

### Cycle 3 — SELL (started 2026-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 15:15:00 | 951.70 | 984.31 | 984.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 925.45 | 983.72 | 984.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 906.60 | 885.10 | 920.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 10:15:00 | 921.70 | 885.46 | 920.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 921.70 | 885.46 | 920.65 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-09 12:15:00 | 906.10 | 887.80 | 920.30 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 13:15:00 | 893.90 | 887.86 | 920.16 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 924.55 | 888.52 | 920.02 | SL hit (close>static) qty=1.00 sl=922.95 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 902.40 | 890.56 | 919.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 903.50 | 890.69 | 919.89 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 931.20 | 893.22 | 919.35 | SL hit (close>static) qty=1.00 sl=922.95 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-16 14:15:00 | 905.70 | 894.16 | 919.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-16 15:15:00 | 906.95 | 894.29 | 919.12 | ENTRY2 sustain failed after 60m |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-04-09 13:15:00 | 893.90 | 2026-04-10 09:15:00 | 924.55 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2026-04-13 10:15:00 | 903.50 | 2026-04-16 09:15:00 | 931.20 | STOP_HIT | 1.00 | -3.07% |
