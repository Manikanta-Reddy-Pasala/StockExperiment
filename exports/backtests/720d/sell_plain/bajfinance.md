# BAJFINANCE (BAJFINANCE)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 973.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 4 |
| ALERT3 | 4 |
| PENDING | 6 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 0 |
| ENTRY2 | 4 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -1.31% / -0.28%
- **Sum % (uncompounded):** -5.23%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 1 | 25.0% | 0 | 4 | 0 | -1.31% | -5.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 1 | 25.0% | 0 | 4 | 0 | -1.31% | -5.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 4 | 1 | 25.0% | 0 | 4 | 0 | -1.31% | -5.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 09:15:00 | 684.30 | 713.59 | 713.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 10:15:00 | 676.66 | 709.20 | 711.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 11:15:00 | 679.20 | 676.70 | 688.75 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 14:15:00 | 686.63 | 676.95 | 688.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 686.63 | 676.95 | 688.70 | EMA400 retest candle locked |

### Cycle 2 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 872.05 | 911.16 | 911.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 10:15:00 | 860.50 | 904.26 | 907.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 915.20 | 896.50 | 903.25 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 915.20 | 896.50 | 903.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 915.20 | 896.50 | 903.25 | EMA400 retest candle locked |

### Cycle 3 — SELL (started 2026-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 15:15:00 | 975.00 | 1008.84 | 1008.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 09:15:00 | 968.45 | 1003.44 | 1006.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 955.95 | 954.92 | 974.31 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 10:15:00 | 969.95 | 955.52 | 974.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 969.95 | 955.52 | 974.03 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-02-05 09:15:00 | 957.30 | 956.05 | 973.76 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-02-05 10:15:00 | 961.90 | 956.11 | 973.70 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-04 09:15:00 | 948.75 | 991.10 | 987.66 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 11:15:00 | 949.05 | 990.31 | 987.30 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-03-06 09:15:00 | 955.55 | 986.25 | 985.38 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 11:15:00 | 955.50 | 985.60 | 985.06 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-03-06 15:15:00 | 951.70 | 984.31 | 984.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-06 15:15:00 | 951.70 | 984.31 | 984.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2026-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 15:15:00 | 951.70 | 984.31 | 984.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 925.45 | 983.72 | 984.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 906.60 | 885.10 | 920.63 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 10:15:00 | 921.70 | 885.46 | 920.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 921.70 | 885.46 | 920.63 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-09 12:15:00 | 906.10 | 887.80 | 920.28 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 14:15:00 | 904.00 | 888.02 | 920.07 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 924.55 | 888.52 | 920.00 | SL hit (close>static) qty=1.00 sl=922.95 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 902.40 | 890.56 | 919.96 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 11:15:00 | 903.40 | 890.81 | 919.79 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 931.20 | 893.22 | 919.34 | SL hit (close>static) qty=1.00 sl=922.95 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-16 14:15:00 | 905.70 | 894.16 | 919.17 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-16 15:15:00 | 906.95 | 894.29 | 919.11 | ENTRY2 sustain failed after 60m |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-03-04 11:15:00 | 949.05 | 2026-03-06 15:15:00 | 951.70 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2026-03-06 11:15:00 | 955.50 | 2026-03-06 15:15:00 | 951.70 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2026-04-09 14:15:00 | 904.00 | 2026-04-10 09:15:00 | 924.55 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2026-04-13 11:15:00 | 903.40 | 2026-04-16 09:15:00 | 931.20 | STOP_HIT | 1.00 | -3.08% |
