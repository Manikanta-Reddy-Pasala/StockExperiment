# Fortis Healthcare Ltd. (FORTIS)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 951.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 9 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 8 |
| PARTIAL | 1 |
| TARGET_HIT | 3 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 10
- **Target hits / Stop hits / Partials:** 1 / 10 / 1
- **Avg / median % per leg:** -0.38% / -1.48%
- **Sum % (uncompounded):** -4.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 2 | 16.7% | 1 | 10 | 1 | -0.38% | -4.6% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.88% | -8.6% |
| SELL @ 3rd Alert (retest2) | 9 | 2 | 22.2% | 1 | 7 | 1 | 0.45% | 4.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.88% | -8.6% |
| retest2 (combined) | 9 | 2 | 22.2% | 1 | 7 | 1 | 0.45% | 4.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 11:15:00 | 908.15 | 965.74 | 965.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 12:15:00 | 905.20 | 965.14 | 965.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 11:15:00 | 907.45 | 907.37 | 929.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-22 12:00:00 | 907.45 | 907.37 | 929.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 922.85 | 902.18 | 921.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:00:00 | 922.85 | 902.18 | 921.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 915.05 | 902.30 | 921.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 14:30:00 | 914.55 | 902.43 | 921.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 906.75 | 902.56 | 921.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 10:15:00 | 913.00 | 902.70 | 920.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 13:30:00 | 915.00 | 903.20 | 920.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 928.55 | 903.70 | 920.86 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 928.55 | 903.70 | 920.86 | SL hit (close>static) qty=1.00 sl=923.05 alert=retest2 |

### Cycle 2 — BUY (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 09:15:00 | 956.15 | 902.29 | 902.29 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 10:15:00 | 876.85 | 902.88 | 902.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 11:15:00 | 876.20 | 902.62 | 902.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 845.30 | 842.24 | 864.89 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 10:15:00 | 841.05 | 842.24 | 864.89 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 13:30:00 | 843.65 | 842.29 | 864.47 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 09:15:00 | 843.05 | 843.42 | 863.35 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 866.85 | 844.29 | 863.01 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 866.85 | 844.29 | 863.01 | SL hit (close>ema400) qty=1.00 sl=863.01 alert=retest1 |

### Cycle 4 — BUY (started 2026-04-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 15:15:00 | 957.00 | 876.17 | 875.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 09:15:00 | 962.50 | 903.40 | 891.15 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-01-02 14:30:00 | 914.55 | 2026-01-06 09:15:00 | 928.55 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2026-01-05 09:15:00 | 906.75 | 2026-01-06 09:15:00 | 928.55 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2026-01-05 10:15:00 | 913.00 | 2026-01-06 09:15:00 | 928.55 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-01-05 13:30:00 | 915.00 | 2026-01-06 09:15:00 | 928.55 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2026-01-08 12:45:00 | 921.35 | 2026-01-20 09:15:00 | 875.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 12:45:00 | 921.35 | 2026-01-27 12:15:00 | 829.22 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-11 10:30:00 | 921.70 | 2026-02-12 09:15:00 | 934.95 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2026-02-11 11:00:00 | 921.50 | 2026-02-12 09:15:00 | 934.95 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-02-12 14:30:00 | 923.10 | 2026-02-25 11:15:00 | 931.80 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest1 | 2026-04-08 10:15:00 | 841.05 | 2026-04-15 09:15:00 | 866.85 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest1 | 2026-04-08 13:30:00 | 843.65 | 2026-04-15 09:15:00 | 866.85 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest1 | 2026-04-13 09:15:00 | 843.05 | 2026-04-15 09:15:00 | 866.85 | STOP_HIT | 1.00 | -2.82% |
