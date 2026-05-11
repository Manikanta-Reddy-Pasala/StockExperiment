# Laurus Labs Ltd. (LAURUSLABS)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 1225.20
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
| ALERT2_SKIP | 1 |
| ALERT3 | 4 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 6 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 9
- **Target hits / Stop hits / Partials:** 0 / 9 / 0
- **Avg / median % per leg:** -2.44% / -3.18%
- **Sum % (uncompounded):** -21.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -3.79% | -18.9% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.25% | -9.7% |
| BUY @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -4.59% | -9.2% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.75% | -3.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.75% | -3.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.25% | -9.7% |
| retest2 (combined) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.03% | -12.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 12:15:00 | 976.95 | 1006.22 | 1006.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-10 13:15:00 | 966.30 | 1005.83 | 1006.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 14:15:00 | 1013.25 | 1004.63 | 1005.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 14:15:00 | 1013.25 | 1004.63 | 1005.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 1013.25 | 1004.63 | 1005.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 15:00:00 | 1013.25 | 1004.63 | 1005.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 1014.20 | 1004.73 | 1005.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:15:00 | 1001.00 | 1004.73 | 1005.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 1012.90 | 1004.99 | 1005.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 1005.50 | 1005.57 | 1005.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 11:00:00 | 1009.60 | 1005.60 | 1005.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 15:00:00 | 1011.10 | 1005.96 | 1006.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:45:00 | 1011.60 | 1006.09 | 1006.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 10:15:00 | 1017.00 | 1006.20 | 1006.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 1017.00 | 1006.20 | 1006.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 1024.10 | 1007.29 | 1006.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 11:15:00 | 1026.40 | 1027.67 | 1018.27 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 13:00:00 | 1030.60 | 1027.70 | 1018.33 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 13:45:00 | 1029.90 | 1027.82 | 1018.44 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 15:15:00 | 1031.20 | 1027.83 | 1018.49 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 997.10 | 1028.98 | 1019.82 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 997.10 | 1028.98 | 1019.82 | SL hit (close<ema400) qty=1.00 sl=1019.82 alert=retest1 |

### Cycle 3 — SELL (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 09:15:00 | 971.40 | 1014.23 | 1014.44 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2026-04-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 15:15:00 | 1059.00 | 1013.43 | 1013.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 10:15:00 | 1061.05 | 1014.34 | 1013.83 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-02-13 09:15:00 | 1005.50 | 2026-02-16 10:15:00 | 1017.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-02-13 11:00:00 | 1009.60 | 2026-02-16 10:15:00 | 1017.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2026-02-13 15:00:00 | 1011.10 | 2026-02-16 10:15:00 | 1017.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-02-16 09:45:00 | 1011.60 | 2026-02-16 10:15:00 | 1017.00 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2026-03-04 13:00:00 | 1030.60 | 2026-03-09 09:15:00 | 997.10 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest1 | 2026-03-04 13:45:00 | 1029.90 | 2026-03-09 09:15:00 | 997.10 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest1 | 2026-03-04 15:15:00 | 1031.20 | 2026-03-09 09:15:00 | 997.10 | STOP_HIT | 1.00 | -3.31% |
| BUY | retest2 | 2026-03-09 12:30:00 | 1011.10 | 2026-03-16 12:15:00 | 966.00 | STOP_HIT | 1.00 | -4.46% |
| BUY | retest2 | 2026-03-09 13:30:00 | 1013.90 | 2026-03-16 12:15:00 | 966.00 | STOP_HIT | 1.00 | -4.72% |
