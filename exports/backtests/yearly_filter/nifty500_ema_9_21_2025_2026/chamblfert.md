# Chambal Fertilizers & Chemicals Ltd. (CHAMBLFERT)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 455.85
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 81 |
| ALERT1 | 51 |
| ALERT2 | 50 |
| ALERT2_SKIP | 25 |
| ALERT3 | 134 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 40 |
| PARTIAL | 7 |
| TARGET_HIT | 1 |
| STOP_HIT | 41 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 49 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 32
- **Target hits / Stop hits / Partials:** 1 / 41 / 7
- **Avg / median % per leg:** 0.71% / -0.56%
- **Sum % (uncompounded):** 34.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 1 | 5.3% | 0 | 19 | 0 | -0.78% | -14.8% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.08% | -0.1% |
| BUY @ 3rd Alert (retest2) | 18 | 1 | 5.6% | 0 | 18 | 0 | -0.82% | -14.7% |
| SELL (all) | 30 | 16 | 53.3% | 1 | 22 | 7 | 1.65% | 49.6% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.33% | -2.3% |
| SELL @ 3rd Alert (retest2) | 29 | 16 | 55.2% | 1 | 21 | 7 | 1.79% | 51.9% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.20% | -2.4% |
| retest2 (combined) | 47 | 17 | 36.2% | 1 | 39 | 7 | 0.79% | 37.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 09:15:00 | 553.15 | 549.69 | 549.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 11:15:00 | 559.00 | 552.21 | 550.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 557.70 | 562.63 | 558.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 13:15:00 | 557.70 | 562.63 | 558.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 557.70 | 562.63 | 558.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:00:00 | 557.70 | 562.63 | 558.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 565.35 | 563.17 | 559.34 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 545.70 | 556.87 | 557.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 15:15:00 | 544.00 | 552.45 | 555.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 550.35 | 549.49 | 552.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 13:15:00 | 550.35 | 549.49 | 552.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 13:15:00 | 550.35 | 549.49 | 552.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 14:00:00 | 550.35 | 549.49 | 552.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 550.50 | 549.69 | 552.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 14:30:00 | 552.20 | 549.69 | 552.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 550.40 | 548.88 | 550.84 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-06-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 15:15:00 | 558.30 | 553.02 | 552.43 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 13:15:00 | 550.55 | 552.66 | 552.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 15:15:00 | 549.00 | 551.77 | 552.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 09:15:00 | 552.50 | 551.91 | 552.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 552.50 | 551.91 | 552.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 552.50 | 551.91 | 552.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:30:00 | 555.10 | 551.91 | 552.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 550.55 | 551.64 | 552.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:45:00 | 552.85 | 551.64 | 552.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 548.60 | 547.91 | 549.73 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 09:15:00 | 553.65 | 550.07 | 550.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 10:15:00 | 560.30 | 552.11 | 550.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 561.80 | 562.24 | 558.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 14:00:00 | 561.80 | 562.24 | 558.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 558.35 | 562.24 | 560.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:00:00 | 558.35 | 562.24 | 560.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 560.00 | 561.79 | 560.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:30:00 | 558.35 | 561.79 | 560.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 561.45 | 561.72 | 560.23 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 10:15:00 | 553.50 | 558.57 | 559.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 11:15:00 | 552.75 | 557.40 | 558.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 14:15:00 | 556.60 | 555.84 | 557.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 14:15:00 | 556.60 | 555.84 | 557.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 556.60 | 555.84 | 557.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 15:00:00 | 556.60 | 555.84 | 557.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 558.50 | 556.37 | 557.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:15:00 | 560.75 | 556.37 | 557.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 561.05 | 557.30 | 557.85 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 11:15:00 | 561.30 | 558.82 | 558.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 13:15:00 | 562.75 | 559.77 | 558.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 15:15:00 | 559.85 | 559.93 | 559.21 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 09:15:00 | 562.00 | 559.93 | 559.21 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 561.55 | 563.16 | 561.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 561.55 | 563.16 | 561.69 | SL hit (close<ema400) qty=1.00 sl=561.69 alert=retest1 |

### Cycle 8 — SELL (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 11:15:00 | 554.70 | 560.44 | 560.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 551.10 | 557.44 | 559.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 10:15:00 | 553.00 | 552.40 | 554.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 11:00:00 | 553.00 | 552.40 | 554.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 558.35 | 553.58 | 555.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:45:00 | 560.20 | 553.58 | 555.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 561.35 | 555.13 | 555.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:00:00 | 561.35 | 555.13 | 555.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 14:15:00 | 564.00 | 556.90 | 556.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 15:15:00 | 567.00 | 558.92 | 557.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 560.95 | 561.01 | 558.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 12:00:00 | 560.95 | 561.01 | 558.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 557.65 | 561.08 | 559.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:45:00 | 558.45 | 561.08 | 559.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 556.85 | 560.23 | 559.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:45:00 | 557.30 | 560.23 | 559.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 12:15:00 | 554.45 | 558.44 | 558.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 552.05 | 556.46 | 557.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 554.10 | 553.94 | 555.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 14:30:00 | 554.50 | 553.94 | 555.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 552.80 | 554.01 | 555.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 14:15:00 | 551.85 | 553.37 | 554.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 11:45:00 | 551.05 | 553.21 | 554.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 12:45:00 | 551.55 | 552.94 | 553.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 09:45:00 | 550.55 | 544.57 | 544.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 550.70 | 545.80 | 545.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 550.70 | 545.80 | 545.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 12:15:00 | 554.10 | 548.49 | 546.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 14:15:00 | 552.20 | 553.28 | 550.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 15:00:00 | 552.20 | 553.28 | 550.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 552.25 | 553.07 | 551.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:15:00 | 548.00 | 553.07 | 551.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 552.00 | 552.86 | 551.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:30:00 | 548.30 | 552.86 | 551.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 552.95 | 552.88 | 551.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:30:00 | 551.20 | 552.88 | 551.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 551.20 | 553.69 | 552.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 551.20 | 553.69 | 552.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 547.35 | 552.42 | 552.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 547.55 | 552.42 | 552.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 548.70 | 551.44 | 551.71 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 15:15:00 | 553.00 | 551.27 | 551.27 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 10:15:00 | 550.00 | 551.10 | 551.20 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 12:15:00 | 552.90 | 551.34 | 551.28 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 14:15:00 | 550.40 | 551.23 | 551.24 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 09:15:00 | 556.10 | 552.00 | 551.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 10:15:00 | 558.95 | 553.39 | 552.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 12:15:00 | 554.60 | 557.81 | 555.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 12:15:00 | 554.60 | 557.81 | 555.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 554.60 | 557.81 | 555.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:00:00 | 554.60 | 557.81 | 555.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 556.00 | 557.44 | 555.87 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 541.85 | 553.21 | 554.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 540.25 | 550.62 | 552.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 14:15:00 | 533.00 | 531.04 | 535.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 15:00:00 | 533.00 | 531.04 | 535.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 539.95 | 533.21 | 535.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:30:00 | 545.10 | 533.21 | 535.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 537.25 | 534.01 | 535.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 13:15:00 | 534.60 | 534.86 | 535.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 529.75 | 536.18 | 536.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 14:15:00 | 507.87 | 525.55 | 530.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 14:15:00 | 503.26 | 525.55 | 530.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 532.70 | 525.74 | 529.39 | SL hit (close>ema200) qty=0.50 sl=525.74 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 15:15:00 | 533.10 | 528.49 | 527.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 09:15:00 | 543.80 | 531.55 | 529.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 09:15:00 | 563.35 | 564.65 | 556.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 12:15:00 | 557.50 | 562.45 | 557.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 557.50 | 562.45 | 557.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 13:00:00 | 557.50 | 562.45 | 557.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 558.65 | 561.69 | 557.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 13:30:00 | 558.30 | 561.69 | 557.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 561.70 | 561.69 | 557.63 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 10:15:00 | 532.40 | 551.17 | 553.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 530.60 | 540.08 | 546.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 11:15:00 | 529.70 | 526.07 | 532.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-12 11:30:00 | 529.00 | 526.07 | 532.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 536.95 | 529.02 | 532.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 15:00:00 | 536.95 | 529.02 | 532.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 535.00 | 530.22 | 532.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 09:30:00 | 530.35 | 530.30 | 532.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 11:15:00 | 533.30 | 530.07 | 529.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 11:15:00 | 533.30 | 530.07 | 529.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 15:15:00 | 534.95 | 532.49 | 531.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 12:15:00 | 558.60 | 560.11 | 554.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 12:30:00 | 560.00 | 560.11 | 554.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 559.95 | 560.20 | 556.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 557.00 | 560.20 | 556.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 560.45 | 561.22 | 559.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:30:00 | 559.00 | 561.22 | 559.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 559.25 | 560.83 | 559.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:00:00 | 559.25 | 560.83 | 559.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 559.90 | 560.64 | 559.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 14:00:00 | 559.90 | 560.64 | 559.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 557.80 | 560.07 | 559.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 14:45:00 | 558.15 | 560.07 | 559.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 554.95 | 559.05 | 558.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 548.30 | 559.05 | 558.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 550.65 | 557.37 | 557.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 12:15:00 | 545.40 | 550.67 | 553.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 547.00 | 545.70 | 549.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 547.00 | 545.70 | 549.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 545.30 | 545.62 | 549.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 548.55 | 545.62 | 549.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 545.00 | 543.10 | 545.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 545.00 | 543.10 | 545.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 543.60 | 543.20 | 544.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 544.20 | 543.20 | 544.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 561.60 | 546.88 | 546.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 567.70 | 551.04 | 548.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 568.05 | 569.23 | 563.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 11:00:00 | 568.05 | 569.23 | 563.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 558.85 | 567.16 | 563.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:45:00 | 559.40 | 567.16 | 563.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 559.75 | 565.67 | 562.85 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 09:15:00 | 557.45 | 560.77 | 561.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 12:15:00 | 553.50 | 557.82 | 559.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 544.30 | 541.14 | 546.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 10:00:00 | 544.30 | 541.14 | 546.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 546.20 | 541.98 | 545.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:00:00 | 546.20 | 541.98 | 545.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 544.20 | 542.42 | 545.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 09:30:00 | 542.55 | 544.05 | 545.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 13:15:00 | 543.60 | 544.73 | 545.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 14:30:00 | 543.40 | 544.29 | 544.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 09:45:00 | 542.80 | 543.74 | 544.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 538.80 | 539.29 | 540.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:30:00 | 539.00 | 539.29 | 540.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 543.30 | 539.91 | 540.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:00:00 | 543.30 | 539.91 | 540.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 544.70 | 540.87 | 541.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:00:00 | 544.70 | 540.87 | 541.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-16 11:15:00 | 547.25 | 542.14 | 541.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 11:15:00 | 547.25 | 542.14 | 541.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 553.75 | 545.96 | 543.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 09:15:00 | 551.00 | 552.04 | 548.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 551.00 | 552.04 | 548.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 551.00 | 552.04 | 548.86 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 10:15:00 | 546.60 | 547.91 | 547.96 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 11:15:00 | 549.40 | 548.21 | 548.09 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 14:15:00 | 541.95 | 546.84 | 547.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 10:15:00 | 540.50 | 544.47 | 546.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 12:15:00 | 537.90 | 537.07 | 539.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 13:00:00 | 537.90 | 537.07 | 539.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 510.40 | 511.22 | 515.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 513.65 | 511.22 | 515.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 512.00 | 509.87 | 512.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 516.45 | 509.87 | 512.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 517.05 | 511.31 | 512.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:45:00 | 518.05 | 511.31 | 512.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 512.90 | 511.63 | 512.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:30:00 | 512.50 | 511.63 | 512.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 513.55 | 512.01 | 512.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 12:00:00 | 513.55 | 512.01 | 512.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 517.50 | 513.11 | 513.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 12:45:00 | 518.55 | 513.11 | 513.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 13:15:00 | 516.80 | 513.85 | 513.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 520.50 | 515.18 | 514.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 15:15:00 | 515.10 | 515.16 | 514.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 09:15:00 | 521.20 | 515.16 | 514.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 514.90 | 515.73 | 514.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:00:00 | 514.90 | 515.73 | 514.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 515.10 | 515.60 | 514.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 13:30:00 | 515.65 | 515.80 | 515.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 10:15:00 | 512.25 | 515.02 | 514.94 | SL hit (close<static) qty=1.00 sl=513.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 11:15:00 | 510.60 | 514.14 | 514.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 12:15:00 | 509.10 | 513.13 | 514.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 09:15:00 | 512.45 | 512.14 | 513.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 512.45 | 512.14 | 513.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 512.45 | 512.14 | 513.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:30:00 | 514.45 | 512.14 | 513.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 510.15 | 511.74 | 512.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 11:15:00 | 509.85 | 511.74 | 512.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 15:15:00 | 509.90 | 511.30 | 512.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 13:15:00 | 484.36 | 490.60 | 496.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 13:15:00 | 484.40 | 490.60 | 496.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 14:15:00 | 484.60 | 483.31 | 488.45 | SL hit (close>ema200) qty=0.50 sl=483.31 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 488.55 | 486.05 | 485.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 12:15:00 | 497.80 | 489.48 | 487.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 09:15:00 | 490.95 | 491.85 | 489.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 09:30:00 | 490.35 | 491.85 | 489.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 489.00 | 491.28 | 489.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:00:00 | 489.00 | 491.28 | 489.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 489.25 | 490.87 | 489.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:30:00 | 487.90 | 490.87 | 489.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 487.00 | 490.10 | 489.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:00:00 | 487.00 | 490.10 | 489.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 485.85 | 489.25 | 488.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:45:00 | 484.30 | 489.25 | 488.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 15:15:00 | 487.70 | 488.63 | 488.68 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 489.30 | 488.76 | 488.73 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 12:15:00 | 487.80 | 488.55 | 488.65 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 491.75 | 488.59 | 488.52 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 13:15:00 | 486.05 | 488.23 | 488.39 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 493.40 | 489.13 | 488.75 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 485.70 | 490.03 | 490.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 12:15:00 | 484.00 | 488.82 | 489.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 13:15:00 | 482.80 | 482.78 | 485.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-31 14:00:00 | 482.80 | 482.78 | 485.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 486.60 | 482.97 | 484.80 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 14:15:00 | 488.30 | 486.13 | 485.87 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 481.45 | 485.39 | 485.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 12:15:00 | 480.20 | 483.65 | 484.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 15:15:00 | 482.70 | 482.52 | 483.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 09:15:00 | 487.95 | 482.52 | 483.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 489.50 | 483.92 | 484.39 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 10:15:00 | 488.60 | 484.86 | 484.77 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 483.40 | 484.56 | 484.65 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 12:15:00 | 485.50 | 484.75 | 484.73 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 477.05 | 483.25 | 484.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 09:15:00 | 472.00 | 479.07 | 481.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 15:15:00 | 466.95 | 465.80 | 470.23 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-12 09:30:00 | 460.00 | 464.70 | 469.33 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 467.10 | 465.79 | 468.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:30:00 | 467.80 | 465.79 | 468.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 464.65 | 465.81 | 467.74 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-13 11:15:00 | 470.70 | 467.00 | 467.96 | SL hit (close>ema400) qty=1.00 sl=467.96 alert=retest1 |

### Cycle 45 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 469.05 | 466.70 | 466.63 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 464.00 | 466.75 | 466.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 460.50 | 463.91 | 465.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 11:15:00 | 459.85 | 457.72 | 459.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 11:15:00 | 459.85 | 457.72 | 459.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 459.85 | 457.72 | 459.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:45:00 | 462.35 | 457.72 | 459.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 12:15:00 | 460.70 | 458.32 | 459.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:30:00 | 461.60 | 458.32 | 459.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 461.15 | 458.89 | 459.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 14:00:00 | 461.15 | 458.89 | 459.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 458.70 | 458.85 | 459.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 14:30:00 | 461.35 | 458.85 | 459.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 459.75 | 459.03 | 459.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:15:00 | 457.35 | 459.03 | 459.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 453.35 | 457.89 | 458.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 11:00:00 | 452.55 | 456.82 | 458.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 09:15:00 | 429.92 | 434.84 | 436.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 15:15:00 | 433.00 | 431.40 | 432.95 | SL hit (close>ema200) qty=0.50 sl=431.40 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 430.90 | 425.54 | 425.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 10:15:00 | 438.00 | 433.86 | 431.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 14:15:00 | 446.05 | 446.45 | 440.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 14:30:00 | 445.45 | 446.45 | 440.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 446.00 | 446.28 | 441.78 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 433.00 | 441.13 | 441.18 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 444.00 | 439.40 | 439.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 15:15:00 | 445.80 | 440.68 | 439.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 09:15:00 | 471.80 | 472.26 | 467.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 11:15:00 | 473.35 | 472.42 | 467.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 473.35 | 472.42 | 467.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:45:00 | 468.90 | 472.42 | 467.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 475.20 | 473.42 | 470.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 13:45:00 | 481.20 | 474.69 | 471.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 09:45:00 | 477.90 | 475.19 | 472.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:30:00 | 477.90 | 475.94 | 473.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 09:30:00 | 476.85 | 476.60 | 475.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 477.70 | 479.21 | 477.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:00:00 | 477.70 | 479.21 | 477.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 478.20 | 479.00 | 477.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 11:30:00 | 480.00 | 479.02 | 477.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 12:30:00 | 479.50 | 479.20 | 477.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 14:15:00 | 480.00 | 479.17 | 477.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 479.85 | 479.32 | 478.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 476.45 | 478.75 | 478.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:30:00 | 478.00 | 478.75 | 478.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 477.55 | 478.51 | 478.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 11:15:00 | 478.65 | 478.51 | 478.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 13:00:00 | 478.95 | 478.40 | 478.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 14:15:00 | 479.55 | 478.36 | 478.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 475.70 | 478.30 | 478.14 | SL hit (close<static) qty=1.00 sl=476.15 alert=retest2 |

### Cycle 50 — SELL (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 10:15:00 | 474.75 | 477.59 | 477.83 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 12:15:00 | 479.60 | 478.11 | 478.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 13:15:00 | 482.55 | 479.00 | 478.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 479.20 | 480.18 | 479.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 479.20 | 480.18 | 479.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 479.20 | 480.18 | 479.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:45:00 | 479.65 | 480.18 | 479.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 478.85 | 479.91 | 479.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:00:00 | 478.85 | 479.91 | 479.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 477.50 | 479.43 | 479.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:00:00 | 477.50 | 479.43 | 479.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2026-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 13:15:00 | 476.35 | 478.54 | 478.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 14:15:00 | 474.00 | 477.63 | 478.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 11:15:00 | 440.90 | 439.68 | 445.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 15:15:00 | 444.90 | 441.16 | 444.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 444.90 | 441.16 | 444.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 447.20 | 441.16 | 444.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 443.30 | 441.59 | 444.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:00:00 | 441.45 | 443.73 | 444.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 419.38 | 429.83 | 434.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 426.45 | 424.69 | 429.33 | SL hit (close>ema200) qty=0.50 sl=424.69 alert=retest2 |

### Cycle 53 — BUY (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 10:15:00 | 435.65 | 428.80 | 428.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 12:15:00 | 438.55 | 431.47 | 429.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-27 10:15:00 | 433.40 | 434.13 | 432.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-27 11:00:00 | 433.40 | 434.13 | 432.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 432.55 | 439.11 | 437.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 432.55 | 439.11 | 437.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 430.30 | 437.35 | 436.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:30:00 | 429.65 | 437.35 | 436.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 11:15:00 | 428.85 | 435.65 | 436.17 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 12:15:00 | 435.65 | 435.58 | 435.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 14:15:00 | 441.45 | 437.29 | 436.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 433.55 | 439.64 | 438.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 433.55 | 439.64 | 438.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 433.55 | 439.64 | 438.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 433.55 | 439.64 | 438.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 432.65 | 438.24 | 437.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 428.70 | 438.24 | 437.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 431.90 | 436.21 | 436.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 431.15 | 435.19 | 436.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 429.75 | 426.87 | 430.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 429.75 | 426.87 | 430.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 431.90 | 427.88 | 430.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 433.30 | 427.88 | 430.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 430.65 | 428.43 | 430.69 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 434.55 | 431.24 | 431.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 13:15:00 | 437.20 | 432.80 | 431.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 430.15 | 433.78 | 432.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 430.15 | 433.78 | 432.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 430.15 | 433.78 | 432.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 430.15 | 433.78 | 432.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 428.55 | 432.73 | 432.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:45:00 | 427.85 | 432.73 | 432.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 11:15:00 | 425.30 | 431.25 | 431.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 12:15:00 | 424.75 | 429.95 | 431.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 09:15:00 | 431.60 | 429.28 | 430.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 431.60 | 429.28 | 430.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 431.60 | 429.28 | 430.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:45:00 | 437.30 | 429.28 | 430.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 432.95 | 430.02 | 430.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:15:00 | 433.15 | 430.02 | 430.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 433.55 | 430.72 | 430.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:30:00 | 433.55 | 430.72 | 430.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 12:15:00 | 432.30 | 431.04 | 430.96 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 14:15:00 | 428.40 | 430.58 | 430.77 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 433.45 | 431.15 | 430.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 13:15:00 | 440.00 | 433.91 | 432.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 09:15:00 | 459.05 | 459.82 | 452.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-12 09:30:00 | 457.50 | 459.82 | 452.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 451.55 | 457.96 | 454.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:45:00 | 451.35 | 457.96 | 454.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 452.30 | 456.83 | 454.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 12:30:00 | 455.65 | 456.10 | 454.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 14:15:00 | 454.25 | 455.63 | 454.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 15:15:00 | 455.00 | 455.23 | 454.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 450.85 | 454.31 | 454.19 | SL hit (close<static) qty=1.00 sl=451.05 alert=retest2 |

### Cycle 62 — SELL (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 10:15:00 | 448.55 | 453.16 | 453.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 14:15:00 | 447.60 | 450.41 | 452.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 452.00 | 450.11 | 451.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 452.00 | 450.11 | 451.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 452.00 | 450.11 | 451.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 451.90 | 450.11 | 451.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 450.90 | 450.27 | 451.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 451.70 | 450.27 | 451.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 449.90 | 450.20 | 451.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:45:00 | 450.15 | 450.20 | 451.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 450.50 | 450.26 | 451.32 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2026-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 15:15:00 | 456.00 | 451.95 | 451.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 14:15:00 | 458.95 | 454.75 | 453.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 453.90 | 455.29 | 453.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 453.90 | 455.29 | 453.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 453.90 | 455.29 | 453.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:30:00 | 453.40 | 455.29 | 453.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 452.50 | 454.73 | 453.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:45:00 | 452.45 | 454.73 | 453.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 449.50 | 453.69 | 453.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 449.50 | 453.69 | 453.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 450.55 | 453.06 | 453.15 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 13:15:00 | 460.05 | 454.46 | 453.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 12:15:00 | 461.35 | 458.87 | 458.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 15:15:00 | 457.95 | 459.67 | 458.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 15:15:00 | 457.95 | 459.67 | 458.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 457.95 | 459.67 | 458.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 453.85 | 459.67 | 458.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 456.15 | 458.97 | 458.50 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 452.50 | 457.68 | 457.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 449.65 | 456.07 | 457.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 434.05 | 430.59 | 436.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 434.05 | 430.59 | 436.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 430.30 | 431.71 | 435.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 11:30:00 | 428.35 | 430.92 | 435.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 12:00:00 | 427.75 | 430.92 | 435.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 436.75 | 420.48 | 423.67 | SL hit (close>static) qty=1.00 sl=436.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 431.25 | 426.37 | 425.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 437.80 | 430.14 | 427.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 431.85 | 432.98 | 430.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 431.85 | 432.98 | 430.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 431.85 | 432.98 | 430.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 431.85 | 432.98 | 430.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 431.30 | 432.64 | 430.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 432.00 | 432.64 | 430.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 426.45 | 431.40 | 430.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:15:00 | 426.70 | 431.40 | 430.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 430.85 | 431.29 | 430.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:30:00 | 432.15 | 431.97 | 430.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 421.00 | 430.79 | 430.77 | SL hit (close<static) qty=1.00 sl=423.30 alert=retest2 |

### Cycle 68 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 420.15 | 428.66 | 429.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 11:15:00 | 418.75 | 426.68 | 428.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 413.60 | 410.15 | 416.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 15:00:00 | 413.60 | 410.15 | 416.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 416.95 | 412.12 | 416.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 416.95 | 412.12 | 416.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 422.35 | 414.17 | 416.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 422.35 | 414.17 | 416.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 426.20 | 416.58 | 417.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:00:00 | 426.20 | 416.58 | 417.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 426.20 | 418.50 | 418.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 434.75 | 425.44 | 422.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 430.95 | 433.86 | 428.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 430.95 | 433.86 | 428.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 430.95 | 433.86 | 428.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 435.00 | 430.18 | 429.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 15:15:00 | 425.60 | 429.33 | 429.29 | SL hit (close<static) qty=1.00 sl=428.10 alert=retest2 |

### Cycle 70 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 419.20 | 427.30 | 428.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 11:15:00 | 414.30 | 423.12 | 426.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 418.10 | 417.88 | 421.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 418.10 | 417.88 | 421.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 418.10 | 417.88 | 421.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:30:00 | 419.85 | 417.88 | 421.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 421.45 | 417.77 | 420.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 421.65 | 417.77 | 420.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 421.90 | 418.60 | 420.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:45:00 | 420.85 | 418.60 | 420.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 420.30 | 418.94 | 420.86 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 428.90 | 422.82 | 422.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 14:15:00 | 431.15 | 426.68 | 424.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 420.50 | 426.68 | 424.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 420.50 | 426.68 | 424.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 420.50 | 426.68 | 424.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 420.50 | 426.68 | 424.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 417.05 | 424.76 | 424.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 415.95 | 424.76 | 424.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 415.95 | 423.00 | 423.46 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-03-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 11:15:00 | 428.60 | 424.03 | 423.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-30 12:15:00 | 431.00 | 425.42 | 424.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 438.75 | 440.54 | 434.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 438.75 | 440.54 | 434.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 438.75 | 440.54 | 434.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:15:00 | 442.85 | 440.39 | 435.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 451.80 | 456.66 | 456.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-04-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 09:15:00 | 451.80 | 456.66 | 456.73 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 460.80 | 456.33 | 456.12 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 09:15:00 | 455.30 | 456.20 | 456.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-16 10:15:00 | 452.55 | 455.47 | 455.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-16 15:15:00 | 454.00 | 453.53 | 454.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-17 09:15:00 | 460.00 | 453.53 | 454.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 460.15 | 454.86 | 455.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 09:30:00 | 459.25 | 454.86 | 455.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 10:15:00 | 454.40 | 454.76 | 455.06 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2026-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 13:15:00 | 455.85 | 455.21 | 455.20 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 15:15:00 | 454.90 | 455.15 | 455.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 09:15:00 | 451.75 | 454.47 | 454.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 09:15:00 | 455.35 | 453.12 | 453.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 09:15:00 | 455.35 | 453.12 | 453.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 455.35 | 453.12 | 453.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 09:30:00 | 455.80 | 453.12 | 453.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 455.05 | 453.51 | 453.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 13:15:00 | 450.80 | 453.84 | 454.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 11:00:00 | 453.55 | 452.99 | 453.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 13:15:00 | 453.00 | 453.26 | 453.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 10:15:00 | 453.90 | 448.09 | 447.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 10:15:00 | 453.90 | 448.09 | 447.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 12:15:00 | 455.65 | 451.29 | 449.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 14:15:00 | 450.60 | 451.36 | 449.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 14:15:00 | 450.60 | 451.36 | 449.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 450.60 | 451.36 | 449.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:30:00 | 449.85 | 451.36 | 449.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 450.90 | 451.26 | 449.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 448.85 | 451.26 | 449.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 444.75 | 449.96 | 449.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 444.75 | 449.96 | 449.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 436.75 | 447.32 | 448.30 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 453.00 | 443.83 | 443.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 10:15:00 | 454.80 | 450.41 | 447.54 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 12:30:00 | 692.70 | 2025-05-13 12:15:00 | 658.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-12 12:30:00 | 692.70 | 2025-05-19 09:15:00 | 623.43 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-06-30 09:15:00 | 562.00 | 2025-07-01 09:15:00 | 561.55 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-07-09 14:15:00 | 551.85 | 2025-07-15 10:15:00 | 550.70 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2025-07-10 11:45:00 | 551.05 | 2025-07-15 10:15:00 | 550.70 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-07-10 12:45:00 | 551.55 | 2025-07-15 10:15:00 | 550.70 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2025-07-15 09:45:00 | 550.55 | 2025-07-15 10:15:00 | 550.70 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-07-30 13:15:00 | 534.60 | 2025-07-31 14:15:00 | 507.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 09:15:00 | 529.75 | 2025-07-31 14:15:00 | 503.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 13:15:00 | 534.60 | 2025-08-01 09:15:00 | 532.70 | STOP_HIT | 0.50 | 0.36% |
| SELL | retest2 | 2025-07-31 09:15:00 | 529.75 | 2025-08-01 09:15:00 | 532.70 | STOP_HIT | 0.50 | -0.56% |
| SELL | retest2 | 2025-08-13 09:30:00 | 530.35 | 2025-08-18 11:15:00 | 533.30 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-09-11 09:30:00 | 542.55 | 2025-09-16 11:15:00 | 547.25 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-09-11 13:15:00 | 543.60 | 2025-09-16 11:15:00 | 547.25 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-09-11 14:30:00 | 543.40 | 2025-09-16 11:15:00 | 547.25 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-09-12 09:45:00 | 542.80 | 2025-09-16 11:15:00 | 547.25 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-10-06 13:30:00 | 515.65 | 2025-10-07 10:15:00 | 512.25 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-10-08 11:15:00 | 509.85 | 2025-10-14 13:15:00 | 484.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-08 15:15:00 | 509.90 | 2025-10-14 13:15:00 | 484.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-08 11:15:00 | 509.85 | 2025-10-15 14:15:00 | 484.60 | STOP_HIT | 0.50 | 4.95% |
| SELL | retest2 | 2025-10-08 15:15:00 | 509.90 | 2025-10-15 14:15:00 | 484.60 | STOP_HIT | 0.50 | 4.96% |
| SELL | retest1 | 2025-11-12 09:30:00 | 460.00 | 2025-11-13 11:15:00 | 470.70 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-11-14 13:45:00 | 462.75 | 2025-11-17 09:15:00 | 473.15 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-11-24 11:00:00 | 452.55 | 2025-12-04 09:15:00 | 429.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-24 11:00:00 | 452.55 | 2025-12-05 15:15:00 | 433.00 | STOP_HIT | 0.50 | 4.32% |
| BUY | retest2 | 2025-12-29 13:45:00 | 481.20 | 2026-01-05 09:15:00 | 475.70 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-12-30 09:45:00 | 477.90 | 2026-01-05 09:15:00 | 475.70 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-12-30 10:30:00 | 477.90 | 2026-01-05 09:15:00 | 475.70 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-12-31 09:30:00 | 476.85 | 2026-01-05 09:15:00 | 475.70 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2026-01-01 11:30:00 | 480.00 | 2026-01-05 09:15:00 | 475.70 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2026-01-01 12:30:00 | 479.50 | 2026-01-05 09:15:00 | 475.70 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2026-01-01 14:15:00 | 480.00 | 2026-01-05 09:15:00 | 475.70 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2026-01-02 09:15:00 | 479.85 | 2026-01-05 10:15:00 | 474.75 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2026-01-02 11:15:00 | 478.65 | 2026-01-05 10:15:00 | 474.75 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2026-01-02 13:00:00 | 478.95 | 2026-01-05 10:15:00 | 474.75 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2026-01-02 14:15:00 | 479.55 | 2026-01-05 10:15:00 | 474.75 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-01-16 11:00:00 | 441.45 | 2026-01-21 10:15:00 | 419.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 11:00:00 | 441.45 | 2026-01-22 09:15:00 | 426.45 | STOP_HIT | 0.50 | 3.40% |
| BUY | retest2 | 2026-02-13 12:30:00 | 455.65 | 2026-02-16 09:15:00 | 450.85 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-02-13 14:15:00 | 454.25 | 2026-02-16 09:15:00 | 450.85 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2026-02-13 15:15:00 | 455.00 | 2026-02-16 09:15:00 | 450.85 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-03-06 11:30:00 | 428.35 | 2026-03-10 10:15:00 | 436.75 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2026-03-06 12:00:00 | 427.75 | 2026-03-10 10:15:00 | 436.75 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2026-03-12 11:30:00 | 432.15 | 2026-03-13 09:15:00 | 421.00 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2026-03-20 09:15:00 | 435.00 | 2026-03-20 15:15:00 | 425.60 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2026-04-02 11:15:00 | 442.85 | 2026-04-13 09:15:00 | 451.80 | STOP_HIT | 1.00 | 2.02% |
| SELL | retest2 | 2026-04-21 13:15:00 | 450.80 | 2026-04-28 10:15:00 | 453.90 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2026-04-22 11:00:00 | 453.55 | 2026-04-28 10:15:00 | 453.90 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2026-04-22 13:15:00 | 453.00 | 2026-04-28 10:15:00 | 453.90 | STOP_HIT | 1.00 | -0.20% |
