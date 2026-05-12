# IDBI Bank Ltd. (IDBI)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 74.79
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 26 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 23 |
| PARTIAL | 0 |
| TARGET_HIT | 5 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 18
- **Target hits / Stop hits / Partials:** 5 / 18 / 0
- **Avg / median % per leg:** 0.66% / -1.51%
- **Sum % (uncompounded):** 15.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 5 | 38.5% | 5 | 8 | 0 | 2.82% | 36.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 5 | 38.5% | 5 | 8 | 0 | 2.82% | 36.7% |
| SELL (all) | 10 | 0 | 0.0% | 0 | 10 | 0 | -2.14% | -21.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 0 | 0.0% | 0 | 10 | 0 | -2.14% | -21.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 23 | 5 | 21.7% | 5 | 18 | 0 | 0.66% | 15.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 10:15:00 | 88.54 | 92.87 | 92.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 11:15:00 | 88.10 | 92.83 | 92.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 13:15:00 | 92.80 | 92.18 | 92.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 13:15:00 | 92.80 | 92.18 | 92.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 92.80 | 92.18 | 92.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:45:00 | 93.17 | 92.18 | 92.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 97.67 | 92.24 | 92.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:30:00 | 98.00 | 92.24 | 92.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 92.53 | 92.70 | 92.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:30:00 | 92.60 | 92.70 | 92.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 92.02 | 91.27 | 91.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 92.02 | 91.27 | 91.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 93.21 | 91.29 | 91.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:00:00 | 93.21 | 91.29 | 91.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 93.17 | 91.31 | 91.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 15:00:00 | 92.58 | 91.35 | 91.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 13:45:00 | 92.50 | 91.42 | 91.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 15:15:00 | 91.80 | 91.44 | 91.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 95.47 | 91.48 | 92.00 | SL hit (close>static) qty=1.00 sl=94.40 alert=retest2 |

### Cycle 2 — BUY (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 14:15:00 | 93.30 | 92.40 | 92.39 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 09:15:00 | 89.77 | 92.38 | 92.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 11:15:00 | 88.88 | 92.15 | 92.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 13:15:00 | 92.30 | 91.97 | 92.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 13:15:00 | 92.30 | 91.97 | 92.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 92.30 | 91.97 | 92.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:45:00 | 92.24 | 91.97 | 92.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 92.32 | 91.98 | 92.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:00:00 | 92.32 | 91.98 | 92.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 92.21 | 92.12 | 92.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 10:45:00 | 91.76 | 92.12 | 92.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 93.90 | 92.05 | 92.18 | SL hit (close>static) qty=1.00 sl=93.18 alert=retest2 |

### Cycle 4 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 94.16 | 92.30 | 92.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 12:15:00 | 95.28 | 92.33 | 92.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 98.30 | 99.56 | 97.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-03 10:00:00 | 98.30 | 99.56 | 97.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 97.68 | 99.54 | 97.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:00:00 | 97.68 | 99.54 | 97.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 97.97 | 99.53 | 97.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 10:15:00 | 98.36 | 99.43 | 97.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 11:15:00 | 97.13 | 99.28 | 97.48 | SL hit (close<static) qty=1.00 sl=97.26 alert=retest2 |

### Cycle 5 — SELL (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 10:15:00 | 75.08 | 103.64 | 103.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-17 14:15:00 | 73.90 | 102.50 | 103.18 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-30 10:45:00 | 94.79 | 2025-07-31 09:15:00 | 92.40 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-09-08 15:00:00 | 92.58 | 2025-09-10 09:15:00 | 95.47 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2025-09-09 13:45:00 | 92.50 | 2025-09-10 09:15:00 | 95.47 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2025-09-09 15:15:00 | 91.80 | 2025-09-10 09:15:00 | 95.47 | STOP_HIT | 1.00 | -4.00% |
| SELL | retest2 | 2025-09-11 14:45:00 | 92.49 | 2025-09-22 14:15:00 | 93.30 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-10-08 10:45:00 | 91.76 | 2025-10-10 09:15:00 | 93.90 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-10-14 11:00:00 | 91.85 | 2025-10-15 11:15:00 | 93.33 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-10-14 15:00:00 | 91.75 | 2025-10-15 11:15:00 | 93.33 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-10-15 09:45:00 | 91.85 | 2025-10-15 11:15:00 | 93.33 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-10-16 11:30:00 | 92.84 | 2025-10-20 11:15:00 | 94.16 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-10-17 09:30:00 | 92.78 | 2025-10-20 11:15:00 | 94.16 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-12-04 10:15:00 | 98.36 | 2025-12-05 11:15:00 | 97.13 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-12-12 12:00:00 | 98.32 | 2025-12-18 09:15:00 | 96.84 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-12-16 10:00:00 | 98.27 | 2025-12-18 09:15:00 | 96.84 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-12-16 12:00:00 | 98.50 | 2025-12-18 09:15:00 | 96.84 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-12-18 12:00:00 | 98.10 | 2026-01-02 11:15:00 | 107.91 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-18 15:00:00 | 97.64 | 2026-01-02 11:15:00 | 107.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-19 09:15:00 | 98.46 | 2026-01-02 11:15:00 | 108.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-21 14:00:00 | 97.94 | 2026-01-27 10:15:00 | 96.24 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-02-01 09:15:00 | 101.26 | 2026-02-01 13:15:00 | 99.26 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-02-03 09:15:00 | 100.93 | 2026-02-04 11:15:00 | 111.02 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-03 10:15:00 | 100.77 | 2026-02-04 11:15:00 | 110.85 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-10 11:45:00 | 101.08 | 2026-03-11 12:15:00 | 99.87 | STOP_HIT | 1.00 | -1.20% |
