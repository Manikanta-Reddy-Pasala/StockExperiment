# SJVN Ltd. (SJVN)

## Backtest Summary

- **Window:** 2025-01-16 09:15:00 → 2026-05-08 15:15:00 (2256 bars)
- **Last close:** 78.69
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 1 |
| ALERT2_SKIP | 1 |
| ALERT3 | 24 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 28 |
| PARTIAL | 5 |
| TARGET_HIT | 6 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 33 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 19
- **Target hits / Stop hits / Partials:** 6 / 22 / 5
- **Avg / median % per leg:** 1.72% / -0.40%
- **Sum % (uncompounded):** 56.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 4 | 21.1% | 2 | 17 | 0 | -0.22% | -4.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 19 | 4 | 21.1% | 2 | 17 | 0 | -0.22% | -4.2% |
| SELL (all) | 14 | 10 | 71.4% | 4 | 5 | 5 | 4.34% | 60.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 10 | 71.4% | 4 | 5 | 5 | 4.34% | 60.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 33 | 14 | 42.4% | 6 | 22 | 5 | 1.72% | 56.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 93.25 | 97.79 | 97.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 13:15:00 | 93.03 | 97.71 | 97.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 97.83 | 95.97 | 96.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 09:15:00 | 97.83 | 95.97 | 96.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 97.83 | 95.97 | 96.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:00:00 | 97.83 | 95.97 | 96.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 98.20 | 95.99 | 96.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 11:00:00 | 98.20 | 95.99 | 96.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 97.98 | 95.44 | 96.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:45:00 | 97.90 | 95.44 | 96.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 98.58 | 95.47 | 96.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:45:00 | 98.52 | 95.47 | 96.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 78.75 | 75.68 | 79.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 11:30:00 | 78.52 | 77.82 | 80.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 14:15:00 | 80.28 | 77.85 | 80.05 | SL hit (close>static) qty=1.00 sl=79.87 alert=retest2 |

### Cycle 2 — BUY (started 2026-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 15:15:00 | 78.04 | 73.24 | 73.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 79.55 | 73.30 | 73.27 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 14:45:00 | 96.75 | 2025-05-19 09:15:00 | 106.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-13 09:30:00 | 96.14 | 2025-05-19 09:15:00 | 105.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-18 14:30:00 | 96.15 | 2025-06-19 15:15:00 | 93.06 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2025-06-20 10:00:00 | 95.76 | 2025-07-10 10:15:00 | 97.78 | STOP_HIT | 1.00 | 2.11% |
| BUY | retest2 | 2025-06-23 14:45:00 | 98.15 | 2025-07-10 10:15:00 | 97.78 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-07-04 12:00:00 | 98.17 | 2025-07-10 10:15:00 | 97.78 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-07-04 14:30:00 | 98.00 | 2025-07-11 13:15:00 | 98.04 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2025-07-08 12:30:00 | 98.16 | 2025-07-15 09:15:00 | 97.85 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-07-08 14:15:00 | 98.49 | 2025-07-15 09:15:00 | 97.85 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-07-10 09:15:00 | 99.05 | 2025-07-15 09:15:00 | 97.85 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-07-10 10:00:00 | 98.49 | 2025-07-25 09:15:00 | 97.44 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-07-11 12:45:00 | 98.75 | 2025-07-25 10:15:00 | 97.09 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-07-14 12:15:00 | 98.63 | 2025-07-25 10:15:00 | 97.09 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-07-14 14:45:00 | 98.43 | 2025-07-25 10:15:00 | 97.09 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-07-14 15:15:00 | 98.45 | 2025-07-25 11:15:00 | 96.38 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-07-15 11:00:00 | 98.29 | 2025-07-25 11:15:00 | 96.38 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-07-18 12:00:00 | 98.95 | 2025-07-25 11:15:00 | 96.38 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2025-07-24 12:00:00 | 98.38 | 2025-07-25 11:15:00 | 96.38 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-07-24 13:30:00 | 98.90 | 2025-08-01 09:15:00 | 93.16 | STOP_HIT | 1.00 | -5.80% |
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
