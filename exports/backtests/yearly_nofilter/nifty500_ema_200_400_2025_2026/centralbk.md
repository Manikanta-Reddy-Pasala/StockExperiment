# Central Bank of India (CENTRALBK)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 36.55
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 38 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 22 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 27 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 26
- **Target hits / Stop hits / Partials:** 0 / 26 / 0
- **Avg / median % per leg:** -2.12% / -1.46%
- **Sum % (uncompounded):** -55.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 0 | 0.0% | 0 | 10 | 0 | -3.19% | -31.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 0 | 0.0% | 0 | 10 | 0 | -3.19% | -31.9% |
| SELL (all) | 16 | 0 | 0.0% | 0 | 16 | 0 | -1.46% | -23.3% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.30% | -5.2% |
| SELL @ 3rd Alert (retest2) | 12 | 0 | 0.0% | 0 | 12 | 0 | -1.51% | -18.1% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.30% | -5.2% |
| retest2 (combined) | 22 | 0 | 0.0% | 0 | 22 | 0 | -2.27% | -50.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 10:15:00 | 37.67 | 37.15 | 37.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 14:15:00 | 37.82 | 37.17 | 37.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 10:15:00 | 37.05 | 37.18 | 37.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 10:15:00 | 37.05 | 37.18 | 37.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 37.05 | 37.18 | 37.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 37.05 | 37.18 | 37.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 36.87 | 37.18 | 37.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:45:00 | 36.93 | 37.18 | 37.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 38.06 | 37.17 | 37.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 38.58 | 37.20 | 37.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 14:15:00 | 36.99 | 37.27 | 37.21 | SL hit (close<static) qty=1.00 sl=37.15 alert=retest2 |

### Cycle 2 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 11:15:00 | 36.42 | 37.94 | 37.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 14:15:00 | 36.25 | 37.89 | 37.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 09:15:00 | 37.25 | 37.22 | 37.53 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-23 14:15:00 | 37.03 | 37.21 | 37.52 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 13:45:00 | 37.03 | 37.21 | 37.51 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 14:15:00 | 37.03 | 37.21 | 37.51 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 14:45:00 | 37.02 | 37.20 | 37.50 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 37.41 | 37.13 | 37.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:45:00 | 37.40 | 37.13 | 37.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 37.43 | 37.13 | 37.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 37.43 | 37.13 | 37.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 37.51 | 37.14 | 37.43 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-31 11:15:00 | 37.51 | 37.14 | 37.43 | SL hit (close>ema400) qty=1.00 sl=37.43 alert=retest1 |

### Cycle 3 — BUY (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 12:15:00 | 38.44 | 37.63 | 37.62 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2026-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 09:15:00 | 36.95 | 37.61 | 37.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 10:15:00 | 36.43 | 37.60 | 37.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 11:15:00 | 37.50 | 37.28 | 37.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 11:15:00 | 37.50 | 37.28 | 37.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 37.50 | 37.28 | 37.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:00:00 | 37.50 | 37.28 | 37.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 37.29 | 37.28 | 37.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 13:15:00 | 37.21 | 37.28 | 37.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 36.75 | 37.28 | 37.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 15:15:00 | 37.24 | 37.12 | 37.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 10:30:00 | 37.16 | 37.12 | 37.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 37.70 | 37.11 | 37.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-09 09:15:00 | 37.70 | 37.11 | 37.30 | SL hit (close>static) qty=1.00 sl=37.52 alert=retest2 |

### Cycle 5 — BUY (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 09:15:00 | 38.35 | 37.44 | 37.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 38.89 | 37.50 | 37.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 37.57 | 38.19 | 37.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 09:15:00 | 37.57 | 38.19 | 37.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 37.57 | 38.19 | 37.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:00:00 | 37.57 | 38.19 | 37.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 37.26 | 38.18 | 37.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:45:00 | 37.11 | 38.18 | 37.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 37.69 | 38.08 | 37.82 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 35.60 | 37.59 | 37.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 35.44 | 37.55 | 37.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 35.23 | 34.99 | 36.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 10:00:00 | 35.23 | 34.99 | 36.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 35.79 | 35.08 | 35.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 09:45:00 | 35.59 | 35.51 | 35.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 36.50 | 35.54 | 35.97 | SL hit (close>static) qty=1.00 sl=36.09 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-10-16 09:15:00 | 38.58 | 2025-10-17 14:15:00 | 36.99 | STOP_HIT | 1.00 | -4.12% |
| BUY | retest2 | 2025-10-20 10:30:00 | 38.45 | 2025-11-13 13:15:00 | 37.43 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2025-11-06 12:30:00 | 38.35 | 2025-11-13 13:15:00 | 37.43 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-11-06 14:00:00 | 38.38 | 2025-12-03 09:15:00 | 37.54 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-11-07 11:30:00 | 38.30 | 2025-12-03 09:15:00 | 37.54 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-11-07 12:00:00 | 38.21 | 2025-12-03 10:15:00 | 37.28 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2025-11-17 09:15:00 | 39.16 | 2025-12-03 10:15:00 | 37.28 | STOP_HIT | 1.00 | -4.80% |
| BUY | retest2 | 2025-11-21 10:00:00 | 38.23 | 2025-12-04 14:15:00 | 37.10 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2025-12-02 09:15:00 | 38.83 | 2025-12-04 14:15:00 | 37.10 | STOP_HIT | 1.00 | -4.46% |
| BUY | retest2 | 2025-12-02 10:15:00 | 38.61 | 2025-12-04 14:15:00 | 37.10 | STOP_HIT | 1.00 | -3.91% |
| SELL | retest1 | 2025-12-23 14:15:00 | 37.03 | 2025-12-31 11:15:00 | 37.51 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest1 | 2025-12-24 13:45:00 | 37.03 | 2025-12-31 11:15:00 | 37.51 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest1 | 2025-12-24 14:15:00 | 37.03 | 2025-12-31 11:15:00 | 37.51 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest1 | 2025-12-24 14:45:00 | 37.02 | 2025-12-31 11:15:00 | 37.51 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-01-08 14:15:00 | 37.39 | 2026-01-13 14:15:00 | 37.77 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2026-01-08 15:00:00 | 37.36 | 2026-01-13 14:15:00 | 37.77 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-01-09 11:00:00 | 37.33 | 2026-01-13 14:15:00 | 37.77 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2026-01-13 09:30:00 | 37.33 | 2026-01-13 14:15:00 | 37.77 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2026-01-30 13:15:00 | 37.21 | 2026-02-09 09:15:00 | 37.70 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-02-01 09:15:00 | 36.75 | 2026-02-09 09:15:00 | 37.70 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2026-02-04 15:15:00 | 37.24 | 2026-02-09 09:15:00 | 37.70 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-02-05 10:30:00 | 37.16 | 2026-02-09 09:15:00 | 37.70 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-04-24 09:45:00 | 35.59 | 2026-04-27 09:15:00 | 36.50 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2026-05-04 09:45:00 | 35.55 | 2026-05-05 15:15:00 | 36.11 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2026-05-04 13:15:00 | 35.60 | 2026-05-05 15:15:00 | 36.11 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2026-05-04 13:45:00 | 35.59 | 2026-05-05 15:15:00 | 36.11 | STOP_HIT | 1.00 | -1.46% |
