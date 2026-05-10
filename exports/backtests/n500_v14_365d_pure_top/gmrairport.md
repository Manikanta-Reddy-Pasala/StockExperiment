# GMR Airports Ltd. (GMRAIRPORT)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 101.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 12 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 14 |
| PARTIAL | 0 |
| TARGET_HIT | 5 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 9
- **Target hits / Stop hits / Partials:** 5 / 9 / 0
- **Avg / median % per leg:** 2.31% / -1.30%
- **Sum % (uncompounded):** 32.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 5 | 83.3% | 5 | 1 | 0 | 7.95% | 47.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 5 | 83.3% | 5 | 1 | 0 | 7.95% | 47.7% |
| SELL (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.92% | -15.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.92% | -15.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 5 | 35.7% | 5 | 9 | 0 | 2.31% | 32.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 11:15:00 | 93.32 | 98.93 | 98.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 92.72 | 98.26 | 98.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 10:15:00 | 97.74 | 97.49 | 98.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-04 11:00:00 | 97.74 | 97.49 | 98.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 97.75 | 97.50 | 98.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 12:30:00 | 97.70 | 97.50 | 98.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 98.65 | 97.51 | 98.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 14:00:00 | 98.65 | 97.51 | 98.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 98.06 | 97.52 | 98.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 15:15:00 | 97.94 | 97.52 | 98.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 13:45:00 | 97.96 | 97.52 | 98.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 11:00:00 | 97.98 | 97.49 | 98.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 11:30:00 | 98.03 | 97.49 | 98.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 98.02 | 97.50 | 98.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 09:15:00 | 97.72 | 97.52 | 98.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 13:00:00 | 97.76 | 97.52 | 98.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 15:15:00 | 97.70 | 97.52 | 98.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 99.25 | 97.18 | 97.84 | SL hit (close>static) qty=1.00 sl=98.83 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 99.25 | 97.18 | 97.84 | SL hit (close>static) qty=1.00 sl=98.83 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 99.25 | 97.18 | 97.84 | SL hit (close>static) qty=1.00 sl=98.83 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 99.25 | 97.18 | 97.84 | SL hit (close>static) qty=1.00 sl=98.83 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 99.25 | 97.18 | 97.84 | SL hit (close>static) qty=1.00 sl=98.19 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 99.25 | 97.18 | 97.84 | SL hit (close>static) qty=1.00 sl=98.19 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 99.25 | 97.18 | 97.84 | SL hit (close>static) qty=1.00 sl=98.19 alert=retest2 |

### Cycle 2 — BUY (started 2026-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 15:15:00 | 102.10 | 98.37 | 98.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 102.77 | 98.41 | 98.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 97.97 | 99.00 | 98.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 97.97 | 99.00 | 98.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 97.97 | 99.00 | 98.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 09:30:00 | 99.17 | 98.46 | 98.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 12:15:00 | 96.73 | 98.42 | 98.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 12:15:00 | 96.73 | 98.42 | 98.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 14:15:00 | 94.95 | 98.37 | 98.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 94.67 | 92.02 | 94.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 94.67 | 92.02 | 94.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 94.67 | 92.02 | 94.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:30:00 | 93.50 | 92.61 | 94.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 98.65 | 92.81 | 94.41 | SL hit (close>static) qty=1.00 sl=95.64 alert=retest2 |

### Cycle 4 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 96.31 | 95.43 | 95.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 97.75 | 95.46 | 95.44 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 09:15:00 | 87.52 | 2025-07-09 09:15:00 | 92.98 | TARGET_HIT | 1.00 | 6.24% |
| BUY | retest2 | 2025-05-30 14:45:00 | 84.53 | 2025-07-09 09:15:00 | 92.99 | TARGET_HIT | 1.00 | 10.01% |
| BUY | retest2 | 2025-06-02 09:15:00 | 84.54 | 2025-07-09 09:15:00 | 93.23 | TARGET_HIT | 1.00 | 10.27% |
| BUY | retest2 | 2025-06-04 09:45:00 | 84.75 | 2025-07-09 09:15:00 | 92.07 | TARGET_HIT | 1.00 | 8.64% |
| BUY | retest2 | 2025-06-25 15:15:00 | 83.70 | 2025-07-18 09:15:00 | 96.27 | TARGET_HIT | 1.00 | 15.02% |
| SELL | retest2 | 2026-02-04 15:15:00 | 97.94 | 2026-02-16 09:15:00 | 99.25 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-02-05 13:45:00 | 97.96 | 2026-02-16 09:15:00 | 99.25 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-02-09 11:00:00 | 97.98 | 2026-02-16 09:15:00 | 99.25 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2026-02-09 11:30:00 | 98.03 | 2026-02-16 09:15:00 | 99.25 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-02-10 09:15:00 | 97.72 | 2026-02-16 09:15:00 | 99.25 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2026-02-10 13:00:00 | 97.76 | 2026-02-16 09:15:00 | 99.25 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2026-02-10 15:15:00 | 97.70 | 2026-02-16 09:15:00 | 99.25 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2026-03-06 09:30:00 | 99.17 | 2026-03-06 12:15:00 | 96.73 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2026-04-13 09:30:00 | 93.50 | 2026-04-15 09:15:00 | 98.65 | STOP_HIT | 1.00 | -5.51% |
