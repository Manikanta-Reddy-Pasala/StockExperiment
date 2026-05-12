# SJVN Ltd. (SJVN)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 78.69
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
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 33 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 29 |
| PARTIAL | 5 |
| TARGET_HIT | 4 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 34 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 23
- **Target hits / Stop hits / Partials:** 4 / 25 / 5
- **Avg / median % per leg:** 0.76% / -1.22%
- **Sum % (uncompounded):** 25.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 1 | 5.0% | 0 | 20 | 0 | -1.75% | -34.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 20 | 1 | 5.0% | 0 | 20 | 0 | -1.75% | -34.9% |
| SELL (all) | 14 | 10 | 71.4% | 4 | 5 | 5 | 4.34% | 60.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 10 | 71.4% | 4 | 5 | 5 | 4.34% | 60.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 34 | 11 | 32.4% | 4 | 25 | 5 | 0.76% | 25.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 14:15:00 | 104.20 | 94.69 | 94.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 105.68 | 94.89 | 94.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 97.71 | 98.23 | 96.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 10:15:00 | 97.66 | 98.23 | 96.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 96.48 | 98.20 | 96.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:00:00 | 96.48 | 98.20 | 96.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 96.00 | 98.18 | 96.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:30:00 | 95.90 | 98.18 | 96.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 96.30 | 98.15 | 96.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 15:00:00 | 96.30 | 98.15 | 96.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 96.49 | 98.13 | 96.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 96.16 | 98.13 | 96.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 11:15:00 | 97.07 | 98.08 | 96.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 10:00:00 | 97.20 | 98.02 | 96.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 10:30:00 | 97.27 | 98.01 | 96.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 10:15:00 | 97.69 | 99.68 | 98.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 12:15:00 | 96.08 | 99.41 | 98.04 | SL hit (close<static) qty=1.00 sl=96.63 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 11:15:00 | 95.15 | 98.09 | 98.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 93.86 | 97.98 | 98.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 97.83 | 95.97 | 96.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 09:15:00 | 97.83 | 95.97 | 96.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 97.83 | 95.97 | 96.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:00:00 | 97.83 | 95.97 | 96.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 98.20 | 95.99 | 96.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 11:00:00 | 98.20 | 95.99 | 96.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 97.98 | 95.44 | 96.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:45:00 | 97.90 | 95.44 | 96.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 98.58 | 95.47 | 96.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:45:00 | 98.52 | 95.47 | 96.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 78.75 | 75.68 | 79.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 11:30:00 | 78.52 | 77.82 | 80.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 14:15:00 | 80.28 | 77.85 | 80.05 | SL hit (close>static) qty=1.00 sl=79.87 alert=retest2 |

### Cycle 3 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 79.55 | 73.30 | 73.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 10:15:00 | 80.84 | 73.38 | 73.31 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-04 10:00:00 | 97.20 | 2025-06-18 12:15:00 | 96.08 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-06-04 10:30:00 | 97.27 | 2025-06-18 12:15:00 | 96.08 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-06-16 10:15:00 | 97.69 | 2025-06-18 12:15:00 | 96.08 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-06-23 11:00:00 | 97.23 | 2025-07-04 12:15:00 | 97.96 | STOP_HIT | 1.00 | 0.75% |
| BUY | retest2 | 2025-06-24 10:45:00 | 99.92 | 2025-07-04 12:15:00 | 97.96 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-06-25 09:15:00 | 100.49 | 2025-07-04 12:15:00 | 97.96 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-06-26 14:15:00 | 100.15 | 2025-07-04 12:15:00 | 97.96 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-06-27 15:15:00 | 100.11 | 2025-07-10 10:15:00 | 97.78 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-07-02 15:00:00 | 99.09 | 2025-07-10 10:15:00 | 97.78 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-07-03 09:30:00 | 98.89 | 2025-07-10 10:15:00 | 97.78 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-07-03 13:30:00 | 99.03 | 2025-07-10 12:15:00 | 97.72 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-07-04 09:45:00 | 98.92 | 2025-07-10 12:15:00 | 97.72 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-07-07 10:45:00 | 99.63 | 2025-07-10 12:15:00 | 97.72 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-07-08 09:15:00 | 100.25 | 2025-07-10 12:15:00 | 97.72 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-07-09 11:15:00 | 100.08 | 2025-07-25 09:15:00 | 97.44 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2025-07-18 13:30:00 | 99.70 | 2025-07-25 09:15:00 | 97.44 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2025-07-21 10:30:00 | 99.40 | 2025-07-25 09:15:00 | 97.44 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-07-21 11:45:00 | 99.25 | 2025-07-25 09:15:00 | 97.44 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-07-21 15:15:00 | 99.18 | 2025-07-25 09:15:00 | 97.44 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-07-23 09:45:00 | 99.14 | 2025-07-25 11:15:00 | 96.38 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2026-01-09 11:30:00 | 78.52 | 2026-01-12 14:15:00 | 80.28 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2026-01-14 10:00:00 | 78.35 | 2026-01-19 12:15:00 | 74.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 09:15:00 | 78.42 | 2026-01-19 12:15:00 | 74.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 11:45:00 | 78.29 | 2026-01-19 13:15:00 | 74.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 10:00:00 | 78.35 | 2026-01-27 09:15:00 | 70.52 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-16 09:15:00 | 78.42 | 2026-01-27 09:15:00 | 70.58 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-16 11:45:00 | 78.29 | 2026-01-27 09:15:00 | 70.46 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-13 14:00:00 | 76.78 | 2026-02-16 13:15:00 | 77.42 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2026-02-13 15:00:00 | 76.54 | 2026-02-16 13:15:00 | 77.42 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2026-02-16 09:15:00 | 75.83 | 2026-02-16 13:15:00 | 77.42 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2026-02-18 11:45:00 | 76.79 | 2026-02-24 09:15:00 | 72.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 11:45:00 | 76.79 | 2026-02-25 09:15:00 | 75.18 | STOP_HIT | 0.50 | 2.10% |
| SELL | retest2 | 2026-02-19 11:00:00 | 76.54 | 2026-03-02 09:15:00 | 72.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 11:00:00 | 76.54 | 2026-03-04 09:15:00 | 68.89 | TARGET_HIT | 0.50 | 10.00% |
