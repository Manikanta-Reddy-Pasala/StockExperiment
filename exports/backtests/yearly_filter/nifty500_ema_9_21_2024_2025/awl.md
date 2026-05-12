# AWL Agri Business Ltd. (AWL)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 206.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 150 |
| ALERT1 | 93 |
| ALERT2 | 89 |
| ALERT2_SKIP | 45 |
| ALERT3 | 243 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 98 |
| PARTIAL | 18 |
| TARGET_HIT | 7 |
| STOP_HIT | 100 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 121 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 51 / 70
- **Target hits / Stop hits / Partials:** 7 / 96 / 18
- **Avg / median % per leg:** 0.85% / -0.41%
- **Sum % (uncompounded):** 102.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 7 | 17.5% | 4 | 36 | 0 | -0.36% | -14.5% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.78% | -7.1% |
| BUY @ 3rd Alert (retest2) | 36 | 7 | 19.4% | 4 | 32 | 0 | -0.20% | -7.4% |
| SELL (all) | 81 | 44 | 54.3% | 3 | 60 | 18 | 1.44% | 116.9% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.60% | -2.6% |
| SELL @ 3rd Alert (retest2) | 80 | 44 | 55.0% | 3 | 59 | 18 | 1.49% | 119.5% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.94% | -9.7% |
| retest2 (combined) | 116 | 51 | 44.0% | 7 | 91 | 18 | 0.97% | 112.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 339.25 | 333.65 | 333.45 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 14:15:00 | 335.00 | 336.20 | 336.27 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 10:15:00 | 337.40 | 336.51 | 336.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 09:15:00 | 339.95 | 337.64 | 337.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-22 10:15:00 | 342.25 | 342.67 | 340.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-22 11:00:00 | 342.25 | 342.67 | 340.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 12:15:00 | 341.25 | 342.24 | 340.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 12:30:00 | 341.45 | 342.24 | 340.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 339.70 | 341.68 | 340.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 15:00:00 | 339.70 | 341.68 | 340.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 15:15:00 | 340.40 | 341.42 | 340.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 09:15:00 | 340.85 | 341.42 | 340.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 11:00:00 | 340.75 | 341.12 | 340.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 11:15:00 | 342.00 | 344.36 | 344.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 11:15:00 | 342.00 | 344.36 | 344.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 14:15:00 | 341.10 | 342.99 | 343.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 342.60 | 339.96 | 341.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 09:15:00 | 342.60 | 339.96 | 341.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 342.60 | 339.96 | 341.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:00:00 | 342.60 | 339.96 | 341.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 343.90 | 340.75 | 341.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 11:00:00 | 343.90 | 340.75 | 341.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 12:15:00 | 345.00 | 342.40 | 342.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-30 09:15:00 | 348.10 | 344.46 | 343.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-30 14:15:00 | 345.00 | 345.47 | 344.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 14:15:00 | 345.00 | 345.47 | 344.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 14:15:00 | 345.00 | 345.47 | 344.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 14:30:00 | 344.90 | 345.47 | 344.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 15:15:00 | 344.90 | 345.36 | 344.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 09:15:00 | 348.00 | 345.36 | 344.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-03 09:15:00 | 382.80 | 354.49 | 349.77 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 331.45 | 352.47 | 353.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 329.20 | 339.01 | 345.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 346.20 | 337.82 | 341.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 346.20 | 337.82 | 341.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 346.20 | 337.82 | 341.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 346.20 | 337.82 | 341.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 345.30 | 339.32 | 341.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 11:30:00 | 343.85 | 339.88 | 341.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-07 10:15:00 | 344.00 | 342.72 | 342.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 10:15:00 | 344.00 | 342.72 | 342.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 11:15:00 | 345.30 | 343.23 | 342.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 345.30 | 346.17 | 345.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-10 14:15:00 | 345.30 | 346.17 | 345.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 345.30 | 346.17 | 345.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:30:00 | 345.90 | 346.17 | 345.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 345.50 | 346.03 | 345.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:15:00 | 345.85 | 346.03 | 345.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 346.50 | 346.13 | 345.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:30:00 | 345.35 | 346.13 | 345.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 10:15:00 | 345.40 | 345.98 | 345.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 11:00:00 | 345.40 | 345.98 | 345.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 11:15:00 | 346.10 | 346.00 | 345.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 11:30:00 | 345.35 | 346.00 | 345.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 13:15:00 | 345.35 | 345.81 | 345.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 14:00:00 | 345.35 | 345.81 | 345.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 14:15:00 | 344.25 | 345.50 | 345.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 15:00:00 | 344.25 | 345.50 | 345.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 343.50 | 345.10 | 345.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 345.90 | 345.10 | 345.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-12 12:15:00 | 344.65 | 345.10 | 345.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 12:15:00 | 344.65 | 345.10 | 345.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-12 14:15:00 | 344.35 | 344.90 | 345.01 | Break + close below crossover candle low |

### Cycle 9 — BUY (started 2024-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 09:15:00 | 346.85 | 345.22 | 345.13 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 12:15:00 | 344.50 | 345.02 | 345.06 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-06-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 10:15:00 | 346.55 | 345.08 | 345.03 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-06-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 13:15:00 | 344.45 | 344.92 | 344.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 09:15:00 | 342.90 | 344.38 | 344.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 09:15:00 | 342.30 | 339.80 | 341.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 09:15:00 | 342.30 | 339.80 | 341.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 342.30 | 339.80 | 341.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:00:00 | 342.30 | 339.80 | 341.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 341.00 | 340.04 | 341.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 12:15:00 | 340.10 | 340.18 | 341.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 09:15:00 | 340.05 | 334.06 | 333.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 09:15:00 | 340.05 | 334.06 | 333.80 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 11:15:00 | 333.50 | 335.01 | 335.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 09:15:00 | 332.50 | 334.04 | 334.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-05 14:15:00 | 333.30 | 333.30 | 333.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-05 15:00:00 | 333.30 | 333.30 | 333.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 15 — BUY (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 09:15:00 | 341.05 | 334.81 | 334.52 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 11:15:00 | 334.90 | 337.09 | 337.29 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 10:15:00 | 339.15 | 336.55 | 336.31 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 09:15:00 | 335.15 | 336.34 | 336.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 15:15:00 | 334.15 | 335.12 | 335.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 322.80 | 321.96 | 325.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 10:00:00 | 322.80 | 321.96 | 325.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 321.00 | 321.77 | 324.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 318.70 | 324.91 | 324.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 325.60 | 323.95 | 324.31 | SL hit (close>static) qty=1.00 sl=325.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 12:15:00 | 325.10 | 324.59 | 324.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 15:15:00 | 325.75 | 324.88 | 324.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 10:15:00 | 325.40 | 325.40 | 324.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-25 10:45:00 | 325.35 | 325.40 | 324.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 11:15:00 | 325.45 | 325.41 | 325.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 10:30:00 | 327.90 | 325.71 | 325.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 09:15:00 | 328.90 | 326.41 | 325.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-02 09:15:00 | 360.69 | 350.79 | 347.86 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 09:15:00 | 373.40 | 384.07 | 384.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 09:15:00 | 356.80 | 364.90 | 371.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 10:15:00 | 357.60 | 356.25 | 362.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 11:00:00 | 357.60 | 356.25 | 362.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 12:15:00 | 362.15 | 357.31 | 361.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 12:45:00 | 364.60 | 357.31 | 361.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 13:15:00 | 361.15 | 358.08 | 361.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 13:30:00 | 362.15 | 358.08 | 361.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 362.85 | 359.03 | 361.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 15:00:00 | 362.85 | 359.03 | 361.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 15:15:00 | 362.90 | 359.81 | 361.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:15:00 | 364.20 | 359.81 | 361.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 366.65 | 361.17 | 362.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:30:00 | 366.95 | 361.17 | 362.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 10:15:00 | 365.60 | 362.06 | 362.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 13:00:00 | 363.10 | 362.70 | 362.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-21 09:15:00 | 380.75 | 363.67 | 362.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 09:15:00 | 380.75 | 363.67 | 362.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 13:15:00 | 393.55 | 375.81 | 369.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 13:15:00 | 384.45 | 385.30 | 378.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 14:15:00 | 384.25 | 385.30 | 378.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 380.90 | 384.42 | 378.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 15:00:00 | 380.90 | 384.42 | 378.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 382.85 | 383.61 | 379.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:45:00 | 384.55 | 383.61 | 379.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 380.10 | 382.54 | 379.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:30:00 | 379.50 | 382.54 | 379.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 376.70 | 381.37 | 379.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 12:45:00 | 377.45 | 381.37 | 379.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 377.60 | 380.61 | 379.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:30:00 | 384.10 | 379.52 | 378.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 14:45:00 | 378.70 | 379.15 | 378.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 09:15:00 | 378.70 | 378.85 | 378.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 09:15:00 | 377.40 | 378.56 | 378.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 09:15:00 | 377.40 | 378.56 | 378.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 14:15:00 | 373.65 | 376.84 | 377.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 365.55 | 365.47 | 368.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 365.55 | 365.47 | 368.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 365.55 | 365.47 | 368.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:45:00 | 363.65 | 365.47 | 368.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 372.65 | 365.61 | 366.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 10:00:00 | 372.65 | 365.61 | 366.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 370.70 | 366.63 | 367.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 10:45:00 | 371.50 | 366.63 | 367.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2024-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 11:15:00 | 374.55 | 368.21 | 367.97 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 13:15:00 | 369.35 | 371.07 | 371.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 14:15:00 | 366.40 | 370.14 | 370.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 14:15:00 | 360.45 | 360.45 | 363.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 15:00:00 | 360.45 | 360.45 | 363.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 362.80 | 361.03 | 363.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:30:00 | 363.75 | 361.03 | 363.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 363.20 | 361.46 | 363.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 11:15:00 | 362.50 | 361.46 | 363.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 11:45:00 | 362.45 | 361.67 | 363.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 14:30:00 | 362.50 | 362.06 | 362.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 09:45:00 | 362.00 | 362.04 | 362.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 10:15:00 | 361.55 | 361.94 | 362.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 13:00:00 | 360.90 | 361.84 | 362.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 09:15:00 | 366.00 | 361.48 | 361.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 366.00 | 361.48 | 361.22 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 10:15:00 | 358.70 | 362.18 | 362.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 11:15:00 | 355.70 | 358.38 | 360.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 10:15:00 | 349.20 | 347.75 | 351.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 10:45:00 | 348.90 | 347.75 | 351.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 348.85 | 347.97 | 351.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:30:00 | 350.40 | 347.97 | 351.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 352.50 | 348.59 | 350.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:45:00 | 351.55 | 348.59 | 350.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 351.20 | 349.11 | 350.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 11:15:00 | 350.50 | 349.11 | 350.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 12:45:00 | 350.65 | 349.93 | 350.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-24 09:15:00 | 352.70 | 351.02 | 350.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2024-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 09:15:00 | 352.70 | 351.02 | 350.84 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 14:15:00 | 349.60 | 350.85 | 350.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 15:15:00 | 349.25 | 350.53 | 350.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 11:15:00 | 346.35 | 346.12 | 347.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-26 11:30:00 | 346.30 | 346.12 | 347.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 349.05 | 346.19 | 346.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:45:00 | 349.25 | 346.19 | 346.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 347.85 | 346.52 | 347.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 11:15:00 | 347.00 | 346.52 | 347.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 12:45:00 | 346.40 | 346.61 | 346.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 15:15:00 | 347.60 | 346.15 | 346.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 15:15:00 | 347.60 | 346.15 | 346.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 09:15:00 | 352.75 | 347.47 | 346.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 09:15:00 | 344.80 | 348.35 | 347.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 09:15:00 | 344.80 | 348.35 | 347.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 344.80 | 348.35 | 347.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 10:00:00 | 344.80 | 348.35 | 347.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 342.80 | 346.61 | 347.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 13:15:00 | 339.50 | 344.49 | 345.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 333.60 | 333.05 | 336.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 09:45:00 | 335.35 | 333.05 | 336.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 336.40 | 333.72 | 336.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 336.40 | 333.72 | 336.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 335.55 | 334.09 | 336.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 12:15:00 | 334.80 | 334.09 | 336.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-08 15:15:00 | 340.00 | 336.95 | 337.23 | SL hit (close>static) qty=1.00 sl=338.85 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 342.70 | 338.10 | 337.73 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 10:15:00 | 336.45 | 337.96 | 338.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 12:15:00 | 335.35 | 337.30 | 337.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-10 15:15:00 | 337.75 | 337.15 | 337.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 15:15:00 | 337.75 | 337.15 | 337.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 337.75 | 337.15 | 337.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:15:00 | 336.70 | 337.15 | 337.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 339.00 | 337.52 | 337.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:00:00 | 339.00 | 337.52 | 337.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 336.80 | 337.37 | 337.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 11:30:00 | 335.95 | 337.30 | 337.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 11:45:00 | 336.00 | 337.00 | 337.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 11:30:00 | 335.70 | 336.74 | 336.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 13:15:00 | 335.70 | 336.67 | 336.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 336.35 | 336.42 | 336.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 15:00:00 | 336.35 | 336.42 | 336.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 334.65 | 336.00 | 336.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:30:00 | 336.00 | 336.00 | 336.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 328.60 | 328.62 | 330.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:45:00 | 325.85 | 327.66 | 329.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 13:30:00 | 325.65 | 327.27 | 328.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:00:00 | 325.70 | 327.27 | 328.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:15:00 | 319.15 | 323.92 | 326.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:15:00 | 319.20 | 323.92 | 326.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:15:00 | 318.91 | 323.92 | 326.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:15:00 | 318.91 | 323.92 | 326.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 10:15:00 | 318.80 | 318.47 | 321.89 | SL hit (close>ema200) qty=0.50 sl=318.47 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 14:15:00 | 339.40 | 323.24 | 321.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 09:15:00 | 345.95 | 339.23 | 335.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-01 18:15:00 | 342.80 | 344.42 | 340.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-01 18:45:00 | 342.80 | 344.42 | 340.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 334.40 | 342.41 | 340.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 334.40 | 342.41 | 340.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 333.00 | 340.53 | 339.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 333.00 | 340.53 | 339.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 12:15:00 | 333.10 | 337.67 | 338.25 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-11-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 13:15:00 | 340.10 | 337.50 | 337.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 14:15:00 | 343.10 | 338.62 | 337.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-06 15:15:00 | 343.20 | 343.33 | 341.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 09:15:00 | 345.95 | 343.33 | 341.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 339.15 | 343.84 | 342.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 10:00:00 | 339.15 | 343.84 | 342.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 338.00 | 342.67 | 342.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 11:00:00 | 338.00 | 342.67 | 342.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 335.95 | 341.33 | 341.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 13:15:00 | 334.60 | 339.23 | 340.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 336.50 | 331.68 | 334.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 336.50 | 331.68 | 334.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 336.50 | 331.68 | 334.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:30:00 | 336.90 | 331.68 | 334.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 335.20 | 332.38 | 334.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 11:15:00 | 334.55 | 332.38 | 334.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 09:30:00 | 332.00 | 332.30 | 333.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 15:15:00 | 317.82 | 328.59 | 330.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 09:15:00 | 330.40 | 328.95 | 330.92 | SL hit (close>ema200) qty=0.50 sl=328.95 alert=retest2 |

### Cycle 37 — BUY (started 2024-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 12:15:00 | 314.80 | 299.66 | 298.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 321.40 | 309.86 | 303.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 15:15:00 | 312.65 | 313.10 | 308.55 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 09:15:00 | 317.95 | 313.10 | 308.55 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 12:30:00 | 314.90 | 313.32 | 310.10 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 13:15:00 | 314.70 | 313.32 | 310.10 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 13:45:00 | 314.75 | 313.61 | 310.52 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 312.45 | 313.35 | 311.17 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-02 12:15:00 | 309.95 | 312.16 | 311.12 | SL hit (close<ema400) qty=1.00 sl=311.12 alert=retest1 |

### Cycle 38 — SELL (started 2024-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 10:15:00 | 311.55 | 312.14 | 312.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 14:15:00 | 310.95 | 311.84 | 312.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 10:15:00 | 311.80 | 311.67 | 311.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-06 10:15:00 | 311.80 | 311.67 | 311.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 311.80 | 311.67 | 311.87 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2024-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 12:15:00 | 312.75 | 312.08 | 312.04 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 14:15:00 | 311.30 | 312.05 | 312.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 09:15:00 | 310.80 | 311.72 | 311.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 10:15:00 | 314.75 | 309.03 | 309.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 10:15:00 | 314.75 | 309.03 | 309.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 314.75 | 309.03 | 309.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:45:00 | 312.75 | 309.03 | 309.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 310.15 | 309.26 | 309.53 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2024-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 12:15:00 | 311.65 | 309.74 | 309.72 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 14:15:00 | 307.55 | 309.37 | 309.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 15:15:00 | 306.75 | 308.85 | 309.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 09:15:00 | 307.75 | 306.72 | 307.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 09:15:00 | 307.75 | 306.72 | 307.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 307.75 | 306.72 | 307.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 308.50 | 306.72 | 307.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 306.75 | 306.73 | 307.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:30:00 | 307.60 | 306.73 | 307.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 306.50 | 306.68 | 307.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 11:30:00 | 306.80 | 306.68 | 307.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 295.70 | 297.58 | 299.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 13:00:00 | 294.55 | 296.85 | 298.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 09:45:00 | 294.65 | 295.38 | 297.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-23 10:15:00 | 304.05 | 297.12 | 297.92 | SL hit (close>static) qty=1.00 sl=300.35 alert=retest2 |

### Cycle 43 — BUY (started 2024-12-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 11:15:00 | 305.60 | 298.81 | 298.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-23 12:15:00 | 310.00 | 301.05 | 299.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-26 11:15:00 | 314.00 | 314.37 | 310.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-26 12:00:00 | 314.00 | 314.37 | 310.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 328.90 | 326.88 | 322.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 328.90 | 326.88 | 322.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 323.25 | 326.15 | 322.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:15:00 | 305.50 | 326.15 | 322.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 305.30 | 321.98 | 321.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:45:00 | 305.65 | 321.98 | 321.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2024-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 10:15:00 | 306.05 | 318.80 | 319.82 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 11:15:00 | 324.45 | 317.56 | 317.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 14:15:00 | 328.75 | 322.03 | 319.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 09:15:00 | 327.95 | 328.04 | 324.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 09:30:00 | 330.55 | 328.04 | 324.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 325.60 | 327.56 | 326.21 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 15:15:00 | 325.00 | 325.66 | 325.69 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 11:15:00 | 326.85 | 325.91 | 325.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-07 12:15:00 | 328.25 | 326.38 | 326.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 09:15:00 | 326.15 | 327.30 | 326.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 09:15:00 | 326.15 | 327.30 | 326.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 326.15 | 327.30 | 326.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:30:00 | 325.70 | 327.30 | 326.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 325.65 | 326.97 | 326.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 10:30:00 | 325.60 | 326.97 | 326.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 11:15:00 | 325.00 | 326.58 | 326.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 11:45:00 | 325.10 | 326.58 | 326.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 12:15:00 | 325.40 | 326.34 | 326.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 12:45:00 | 324.70 | 326.34 | 326.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 14:15:00 | 326.00 | 326.33 | 326.34 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 09:15:00 | 327.90 | 326.62 | 326.47 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 10:15:00 | 324.15 | 326.12 | 326.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 296.45 | 318.99 | 322.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 10:15:00 | 273.60 | 269.57 | 277.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 11:00:00 | 273.60 | 269.57 | 277.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 274.00 | 272.23 | 275.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 12:00:00 | 271.55 | 272.39 | 275.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 14:00:00 | 271.95 | 272.25 | 274.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 12:15:00 | 257.97 | 261.14 | 263.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 12:15:00 | 258.35 | 261.14 | 263.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-27 09:15:00 | 244.40 | 254.06 | 258.82 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 51 — BUY (started 2025-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 12:15:00 | 263.70 | 258.28 | 257.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 09:15:00 | 265.50 | 260.80 | 259.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 10:15:00 | 262.30 | 263.17 | 261.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 10:45:00 | 261.90 | 263.17 | 261.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 11:15:00 | 261.80 | 262.90 | 261.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 11:30:00 | 261.40 | 262.90 | 261.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 12:15:00 | 261.30 | 262.58 | 261.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 12:45:00 | 260.80 | 262.58 | 261.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 258.25 | 261.71 | 261.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:00:00 | 258.25 | 261.71 | 261.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 258.90 | 261.15 | 261.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:30:00 | 258.20 | 261.15 | 261.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 15:15:00 | 258.80 | 260.68 | 260.92 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 09:15:00 | 263.30 | 261.20 | 261.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 12:15:00 | 264.90 | 262.52 | 261.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 264.95 | 266.21 | 264.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 264.95 | 266.21 | 264.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 264.95 | 266.21 | 264.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 264.05 | 266.21 | 264.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 263.85 | 265.74 | 264.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 264.40 | 265.74 | 264.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 267.50 | 266.09 | 264.52 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2025-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 13:15:00 | 262.15 | 264.27 | 264.35 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 14:15:00 | 265.55 | 264.53 | 264.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 09:15:00 | 268.50 | 265.32 | 264.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 09:15:00 | 269.70 | 270.05 | 267.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-05 10:00:00 | 269.70 | 270.05 | 267.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 13:15:00 | 269.75 | 269.80 | 268.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 13:30:00 | 268.75 | 269.80 | 268.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 269.85 | 270.66 | 269.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 269.85 | 270.66 | 269.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 269.30 | 270.39 | 269.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:00:00 | 269.30 | 270.39 | 269.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 269.65 | 270.24 | 269.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:45:00 | 269.00 | 270.24 | 269.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 269.00 | 269.99 | 269.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:15:00 | 267.60 | 269.99 | 269.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 268.25 | 269.64 | 269.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:45:00 | 268.20 | 269.64 | 269.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 268.70 | 269.46 | 269.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:15:00 | 269.95 | 269.46 | 269.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 11:15:00 | 267.95 | 269.15 | 269.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-02-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 11:15:00 | 267.95 | 269.15 | 269.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 12:15:00 | 266.05 | 268.53 | 268.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 246.00 | 242.40 | 247.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 10:00:00 | 246.00 | 242.40 | 247.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 246.95 | 243.31 | 247.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:00:00 | 246.95 | 243.31 | 247.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 246.75 | 244.00 | 247.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:45:00 | 247.45 | 244.00 | 247.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 12:15:00 | 250.30 | 245.26 | 247.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 13:00:00 | 250.30 | 245.26 | 247.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 13:15:00 | 248.70 | 245.95 | 247.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 14:15:00 | 247.55 | 245.95 | 247.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 12:15:00 | 235.17 | 240.96 | 244.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 235.90 | 235.80 | 238.93 | SL hit (close>ema200) qty=0.50 sl=235.80 alert=retest2 |

### Cycle 57 — BUY (started 2025-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 12:15:00 | 246.35 | 241.14 | 240.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 13:15:00 | 253.35 | 243.58 | 241.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 15:15:00 | 258.90 | 259.66 | 257.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-24 09:15:00 | 254.75 | 259.66 | 257.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 258.95 | 259.51 | 257.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 10:30:00 | 260.00 | 259.54 | 257.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 11:15:00 | 260.00 | 259.54 | 257.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 09:15:00 | 251.70 | 258.08 | 258.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 251.70 | 258.08 | 258.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 10:15:00 | 246.95 | 255.85 | 257.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 13:15:00 | 242.55 | 241.39 | 245.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 14:00:00 | 242.55 | 241.39 | 245.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 243.20 | 242.06 | 244.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:45:00 | 244.85 | 242.06 | 244.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 252.16 | 242.61 | 243.33 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 254.30 | 244.95 | 244.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 12:15:00 | 256.61 | 248.73 | 246.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 259.48 | 259.76 | 257.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 09:30:00 | 258.86 | 259.76 | 257.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 257.69 | 259.46 | 257.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 257.69 | 259.46 | 257.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 255.24 | 258.62 | 257.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 13:00:00 | 255.24 | 258.62 | 257.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 252.70 | 257.43 | 256.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:00:00 | 252.70 | 257.43 | 256.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 252.77 | 256.50 | 256.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 11:15:00 | 249.71 | 253.56 | 254.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 247.79 | 246.74 | 248.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 09:15:00 | 247.79 | 246.74 | 248.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 247.79 | 246.74 | 248.46 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 252.62 | 248.73 | 248.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 14:15:00 | 255.25 | 252.03 | 250.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 13:15:00 | 254.23 | 254.55 | 252.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 13:30:00 | 254.70 | 254.55 | 252.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 257.19 | 261.73 | 260.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 257.19 | 261.73 | 260.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 255.80 | 260.54 | 259.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:00:00 | 255.80 | 260.54 | 259.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2025-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 11:15:00 | 255.50 | 259.53 | 259.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 12:15:00 | 253.50 | 257.03 | 258.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 12:15:00 | 253.39 | 252.23 | 254.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 12:15:00 | 253.39 | 252.23 | 254.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 253.39 | 252.23 | 254.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:00:00 | 253.39 | 252.23 | 254.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 251.97 | 252.17 | 254.42 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2025-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 12:15:00 | 257.40 | 255.66 | 255.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 14:15:00 | 258.35 | 256.45 | 255.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 09:15:00 | 260.65 | 263.35 | 260.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 260.65 | 263.35 | 260.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 260.65 | 263.35 | 260.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-02 09:45:00 | 260.00 | 263.35 | 260.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 264.35 | 263.55 | 261.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 11:15:00 | 264.70 | 263.55 | 261.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 12:30:00 | 264.80 | 266.71 | 266.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 14:45:00 | 264.95 | 266.07 | 265.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 253.25 | 263.74 | 264.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 253.25 | 263.74 | 264.87 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-04-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-07 15:15:00 | 270.85 | 265.82 | 265.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 09:15:00 | 272.15 | 267.08 | 265.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-09 09:15:00 | 271.80 | 272.73 | 270.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-09 09:30:00 | 273.25 | 272.73 | 270.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 15:15:00 | 271.10 | 272.16 | 270.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 09:15:00 | 275.65 | 272.16 | 270.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 279.70 | 283.55 | 283.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 09:15:00 | 279.70 | 283.55 | 283.65 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 09:15:00 | 285.35 | 283.66 | 283.49 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 275.65 | 282.00 | 282.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 272.55 | 280.11 | 281.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 278.85 | 276.82 | 279.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 278.85 | 276.82 | 279.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 278.85 | 276.82 | 279.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:00:00 | 278.85 | 276.82 | 279.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 278.50 | 277.16 | 279.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:00:00 | 278.50 | 277.16 | 279.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 282.55 | 278.24 | 279.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 09:15:00 | 275.65 | 278.90 | 279.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 12:15:00 | 261.87 | 270.91 | 275.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-02 09:15:00 | 271.00 | 268.25 | 270.35 | SL hit (close>ema200) qty=0.50 sl=268.25 alert=retest2 |

### Cycle 69 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 11:15:00 | 277.50 | 271.51 | 270.69 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 11:15:00 | 267.75 | 270.77 | 270.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 12:15:00 | 265.25 | 269.67 | 270.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-08 09:15:00 | 264.80 | 263.35 | 265.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 11:15:00 | 265.30 | 263.85 | 265.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 11:15:00 | 265.30 | 263.85 | 265.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 12:00:00 | 265.30 | 263.85 | 265.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 12:15:00 | 263.30 | 263.74 | 264.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:15:00 | 260.85 | 263.74 | 264.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 266.95 | 255.87 | 257.87 | SL hit (close>static) qty=1.00 sl=265.50 alert=retest2 |

### Cycle 71 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 263.20 | 259.80 | 259.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 263.80 | 261.77 | 260.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 262.50 | 262.68 | 261.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:45:00 | 262.25 | 262.68 | 261.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 268.50 | 270.62 | 269.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 268.50 | 270.62 | 269.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 269.30 | 270.36 | 269.41 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2025-05-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 11:15:00 | 267.40 | 268.82 | 268.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 264.60 | 267.63 | 268.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 10:15:00 | 259.60 | 258.74 | 260.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 10:15:00 | 259.60 | 258.74 | 260.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 259.60 | 258.74 | 260.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 259.60 | 258.74 | 260.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 260.50 | 259.09 | 260.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:45:00 | 259.55 | 259.09 | 260.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 260.70 | 259.41 | 260.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 13:00:00 | 260.70 | 259.41 | 260.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 259.40 | 259.41 | 260.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 12:45:00 | 258.65 | 259.86 | 260.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 14:15:00 | 258.50 | 259.73 | 260.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 15:00:00 | 258.60 | 259.50 | 260.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 09:30:00 | 258.80 | 259.08 | 259.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 11:15:00 | 272.90 | 261.87 | 260.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 11:15:00 | 272.90 | 261.87 | 260.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 14:15:00 | 278.45 | 270.65 | 267.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 15:15:00 | 267.90 | 270.10 | 267.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 15:15:00 | 267.90 | 270.10 | 267.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 267.90 | 270.10 | 267.58 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 11:15:00 | 267.80 | 268.44 | 268.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 09:15:00 | 267.00 | 268.16 | 268.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 266.70 | 264.81 | 265.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 09:15:00 | 266.70 | 264.81 | 265.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 266.70 | 264.81 | 265.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:45:00 | 267.50 | 264.81 | 265.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 267.45 | 265.34 | 265.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:45:00 | 267.85 | 265.34 | 265.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-06-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 13:15:00 | 267.60 | 266.42 | 266.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 15:15:00 | 268.00 | 266.94 | 266.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 273.25 | 274.99 | 272.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 14:00:00 | 273.25 | 274.99 | 272.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 272.60 | 274.94 | 273.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:00:00 | 272.60 | 274.94 | 273.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 271.25 | 274.20 | 272.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 271.25 | 274.20 | 272.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 267.30 | 271.64 | 272.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 265.85 | 269.40 | 270.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 14:15:00 | 264.80 | 264.73 | 266.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 15:00:00 | 264.80 | 264.73 | 266.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 264.50 | 262.51 | 263.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 15:00:00 | 264.50 | 262.51 | 263.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 263.80 | 262.77 | 263.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 09:15:00 | 261.95 | 262.77 | 263.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 12:15:00 | 258.65 | 257.78 | 257.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2025-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 12:15:00 | 258.65 | 257.78 | 257.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 13:15:00 | 259.10 | 258.05 | 257.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 258.00 | 258.50 | 258.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 09:15:00 | 258.00 | 258.50 | 258.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 258.00 | 258.50 | 258.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:30:00 | 258.35 | 258.50 | 258.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 257.30 | 258.26 | 258.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:00:00 | 257.30 | 258.26 | 258.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 257.50 | 258.11 | 258.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:30:00 | 257.40 | 258.11 | 258.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2025-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 12:15:00 | 257.20 | 257.93 | 257.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 13:15:00 | 256.55 | 257.65 | 257.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 15:15:00 | 257.80 | 257.66 | 257.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 15:15:00 | 257.80 | 257.66 | 257.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 257.80 | 257.66 | 257.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:15:00 | 259.80 | 257.66 | 257.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 262.80 | 258.69 | 258.24 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 12:15:00 | 258.20 | 259.05 | 259.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 13:15:00 | 258.10 | 258.86 | 258.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 10:15:00 | 258.60 | 258.45 | 258.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 10:15:00 | 258.60 | 258.45 | 258.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 258.60 | 258.45 | 258.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:45:00 | 258.25 | 258.45 | 258.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 257.65 | 258.29 | 258.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 13:15:00 | 257.15 | 258.16 | 258.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 09:30:00 | 257.20 | 257.87 | 258.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:15:00 | 257.15 | 257.80 | 258.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 13:45:00 | 257.20 | 257.66 | 258.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 256.90 | 256.65 | 257.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 256.90 | 256.65 | 257.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 256.70 | 256.54 | 257.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 257.10 | 256.54 | 257.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 256.70 | 256.57 | 256.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:30:00 | 257.55 | 256.57 | 256.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 256.40 | 256.54 | 256.94 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 258.20 | 257.09 | 257.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 09:15:00 | 258.20 | 257.09 | 257.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 12:15:00 | 260.60 | 258.31 | 257.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 13:15:00 | 262.50 | 262.76 | 260.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 14:00:00 | 262.50 | 262.76 | 260.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 268.45 | 269.39 | 267.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:30:00 | 268.60 | 269.39 | 267.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 267.70 | 269.05 | 267.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:00:00 | 267.70 | 269.05 | 267.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 266.75 | 268.59 | 267.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:30:00 | 267.00 | 268.59 | 267.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 267.65 | 268.40 | 267.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 15:15:00 | 269.00 | 268.03 | 267.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 12:00:00 | 269.00 | 268.24 | 267.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 12:45:00 | 268.85 | 268.51 | 267.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 14:15:00 | 263.50 | 267.63 | 267.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 14:15:00 | 263.50 | 267.63 | 267.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 09:15:00 | 261.35 | 265.68 | 266.72 | Break + close below crossover candle low |

### Cycle 83 — BUY (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 09:15:00 | 277.20 | 265.69 | 265.67 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 276.30 | 277.80 | 277.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 271.40 | 276.06 | 277.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 272.70 | 272.22 | 274.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 09:45:00 | 272.95 | 272.22 | 274.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 268.00 | 265.60 | 266.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 15:00:00 | 268.00 | 265.60 | 266.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 265.85 | 265.65 | 266.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 264.70 | 265.65 | 266.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 13:45:00 | 264.80 | 264.95 | 265.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 09:15:00 | 251.46 | 252.65 | 254.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 09:15:00 | 251.56 | 252.65 | 254.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 15:15:00 | 251.50 | 250.69 | 252.52 | SL hit (close>ema200) qty=0.50 sl=250.69 alert=retest2 |

### Cycle 85 — BUY (started 2025-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 13:15:00 | 254.60 | 253.21 | 253.21 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 09:15:00 | 251.30 | 253.09 | 253.18 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 255.00 | 253.19 | 253.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 15:15:00 | 256.00 | 253.75 | 253.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 10:15:00 | 253.80 | 253.89 | 253.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 11:00:00 | 253.80 | 253.89 | 253.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 253.75 | 253.86 | 253.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 11:45:00 | 253.80 | 253.86 | 253.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 254.00 | 253.89 | 253.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 12:30:00 | 254.00 | 253.89 | 253.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 253.00 | 253.71 | 253.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:00:00 | 253.00 | 253.71 | 253.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 252.90 | 253.55 | 253.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:30:00 | 252.70 | 253.55 | 253.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 251.95 | 253.23 | 253.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 251.55 | 252.48 | 252.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 252.55 | 251.22 | 251.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 252.55 | 251.22 | 251.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 252.55 | 251.22 | 251.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 12:00:00 | 250.40 | 251.17 | 251.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 10:15:00 | 256.05 | 252.44 | 252.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 10:15:00 | 256.05 | 252.44 | 252.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 11:15:00 | 259.05 | 253.76 | 252.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 12:15:00 | 260.20 | 260.73 | 258.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 12:30:00 | 260.40 | 260.73 | 258.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 258.75 | 260.05 | 259.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 257.85 | 260.05 | 259.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 257.85 | 259.61 | 259.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 257.85 | 259.61 | 259.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 259.15 | 259.40 | 259.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:30:00 | 258.85 | 259.40 | 259.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 260.70 | 259.66 | 259.17 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 09:15:00 | 257.40 | 258.85 | 258.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 10:15:00 | 256.65 | 258.41 | 258.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 14:15:00 | 260.00 | 258.11 | 258.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 14:15:00 | 260.00 | 258.11 | 258.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 260.00 | 258.11 | 258.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 15:00:00 | 260.00 | 258.11 | 258.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 258.80 | 258.25 | 258.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 255.75 | 258.25 | 258.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 11:15:00 | 253.05 | 251.48 | 251.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 253.05 | 251.48 | 251.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 12:15:00 | 257.40 | 252.66 | 252.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 15:15:00 | 265.75 | 265.83 | 262.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-05 09:15:00 | 263.50 | 265.83 | 262.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 262.80 | 265.22 | 262.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:30:00 | 262.45 | 265.22 | 262.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 262.70 | 264.72 | 262.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 11:15:00 | 263.50 | 264.72 | 262.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 261.65 | 264.10 | 262.56 | SL hit (close<static) qty=1.00 sl=261.80 alert=retest2 |

### Cycle 92 — SELL (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 13:15:00 | 261.30 | 262.23 | 262.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 10:15:00 | 259.60 | 261.21 | 261.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 259.90 | 259.04 | 260.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 09:30:00 | 259.60 | 259.04 | 260.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 259.60 | 259.15 | 260.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:30:00 | 260.10 | 259.15 | 260.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 262.10 | 258.33 | 259.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:00:00 | 262.10 | 258.33 | 259.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 262.05 | 259.07 | 259.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:30:00 | 263.00 | 259.07 | 259.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 259.50 | 259.45 | 259.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 15:00:00 | 258.45 | 259.26 | 259.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 261.50 | 259.75 | 259.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 261.50 | 259.75 | 259.62 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 10:15:00 | 258.60 | 259.47 | 259.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 14:15:00 | 258.00 | 258.96 | 259.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 259.05 | 258.87 | 259.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 259.05 | 258.87 | 259.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 259.05 | 258.87 | 259.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 10:45:00 | 258.45 | 258.88 | 259.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 14:15:00 | 260.70 | 259.54 | 259.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 14:15:00 | 260.70 | 259.54 | 259.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 15:15:00 | 261.50 | 259.93 | 259.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 13:15:00 | 259.00 | 260.42 | 260.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 13:15:00 | 259.00 | 260.42 | 260.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 259.00 | 260.42 | 260.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 259.00 | 260.42 | 260.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 259.00 | 260.14 | 259.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:30:00 | 259.00 | 260.14 | 259.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 259.70 | 260.02 | 259.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:00:00 | 259.70 | 260.02 | 259.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 10:15:00 | 259.20 | 259.86 | 259.86 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 11:15:00 | 260.00 | 259.88 | 259.87 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 259.60 | 259.83 | 259.85 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 09:15:00 | 261.00 | 259.92 | 259.87 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 10:15:00 | 257.10 | 259.36 | 259.62 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 13:15:00 | 262.10 | 259.68 | 259.67 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 14:15:00 | 257.00 | 259.15 | 259.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 15:15:00 | 254.30 | 258.18 | 258.96 | Break + close below crossover candle low |

### Cycle 103 — BUY (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 11:15:00 | 267.60 | 259.59 | 259.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 12:15:00 | 275.70 | 262.81 | 260.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 10:15:00 | 262.80 | 266.40 | 263.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 10:15:00 | 262.80 | 266.40 | 263.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 262.80 | 266.40 | 263.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 262.80 | 266.40 | 263.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 261.60 | 265.44 | 263.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:00:00 | 261.60 | 265.44 | 263.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 261.35 | 264.62 | 263.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:15:00 | 261.40 | 264.62 | 263.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 260.80 | 263.86 | 263.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:45:00 | 260.75 | 263.86 | 263.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2025-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 15:15:00 | 260.40 | 262.53 | 262.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 256.65 | 261.36 | 262.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 10:15:00 | 256.00 | 255.38 | 257.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 10:45:00 | 256.15 | 255.38 | 257.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 12:15:00 | 257.00 | 255.65 | 257.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:30:00 | 257.00 | 255.65 | 257.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 258.00 | 256.12 | 257.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 13:45:00 | 258.85 | 256.12 | 257.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 262.00 | 257.30 | 258.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:00:00 | 262.00 | 257.30 | 258.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-26 09:15:00 | 262.20 | 258.90 | 258.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 11:15:00 | 265.00 | 263.15 | 262.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 264.15 | 264.26 | 263.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 10:15:00 | 263.55 | 264.89 | 264.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 263.55 | 264.89 | 264.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 263.55 | 264.89 | 264.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 264.60 | 264.83 | 264.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 15:15:00 | 265.75 | 264.87 | 264.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 263.20 | 264.68 | 264.42 | SL hit (close<static) qty=1.00 sl=263.40 alert=retest2 |

### Cycle 106 — SELL (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 10:15:00 | 263.15 | 264.41 | 264.45 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 11:15:00 | 265.10 | 264.55 | 264.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 12:15:00 | 266.60 | 264.96 | 264.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 11:15:00 | 267.25 | 267.39 | 266.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-10 12:00:00 | 267.25 | 267.39 | 266.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 266.40 | 267.16 | 266.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:00:00 | 266.40 | 267.16 | 266.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 268.55 | 267.44 | 266.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:30:00 | 266.85 | 267.44 | 266.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 268.20 | 267.79 | 266.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 15:00:00 | 269.15 | 267.72 | 267.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 13:30:00 | 269.10 | 267.27 | 267.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 14:30:00 | 268.80 | 267.54 | 267.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 269.25 | 267.64 | 267.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 268.50 | 267.81 | 267.41 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-16 12:15:00 | 265.15 | 267.44 | 267.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 12:15:00 | 265.15 | 267.44 | 267.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 09:15:00 | 263.50 | 265.96 | 266.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 261.25 | 260.51 | 262.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-21 13:45:00 | 260.85 | 260.51 | 262.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 259.95 | 260.72 | 261.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:30:00 | 261.25 | 260.72 | 261.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 262.00 | 260.51 | 261.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 12:00:00 | 262.00 | 260.51 | 261.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 263.00 | 261.01 | 261.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 12:45:00 | 263.00 | 261.01 | 261.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 14:15:00 | 263.25 | 261.78 | 261.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 11:15:00 | 265.15 | 262.77 | 262.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 15:15:00 | 274.15 | 274.27 | 272.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-03 09:15:00 | 273.65 | 274.27 | 272.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 273.40 | 274.10 | 272.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 273.40 | 274.10 | 272.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 272.50 | 273.78 | 272.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:30:00 | 273.25 | 273.78 | 272.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 271.30 | 273.28 | 272.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:45:00 | 272.40 | 273.28 | 272.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 272.00 | 273.03 | 272.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 13:45:00 | 273.00 | 273.17 | 272.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 10:15:00 | 273.30 | 273.61 | 273.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 11:00:00 | 273.30 | 273.55 | 273.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 12:00:00 | 274.70 | 273.78 | 273.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 277.35 | 274.91 | 273.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:30:00 | 274.30 | 274.91 | 273.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 274.45 | 275.28 | 274.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:45:00 | 273.25 | 275.28 | 274.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 273.30 | 274.89 | 274.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:00:00 | 273.30 | 274.89 | 274.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 271.95 | 274.30 | 274.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:00:00 | 271.95 | 274.30 | 274.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 273.00 | 274.04 | 274.00 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-06 14:15:00 | 271.10 | 273.45 | 273.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2025-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 14:15:00 | 271.10 | 273.45 | 273.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 15:15:00 | 268.60 | 270.80 | 271.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 11:15:00 | 271.55 | 270.52 | 271.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 11:15:00 | 271.55 | 270.52 | 271.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 271.55 | 270.52 | 271.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:00:00 | 271.55 | 270.52 | 271.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 271.00 | 270.61 | 271.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 14:45:00 | 270.10 | 270.86 | 271.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 15:15:00 | 270.00 | 270.86 | 271.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 10:30:00 | 270.75 | 270.86 | 271.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 11:15:00 | 272.20 | 271.13 | 271.34 | SL hit (close>static) qty=1.00 sl=272.05 alert=retest2 |

### Cycle 111 — BUY (started 2025-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 13:15:00 | 273.00 | 271.70 | 271.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 14:15:00 | 274.70 | 272.30 | 271.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 11:15:00 | 272.65 | 272.68 | 272.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 11:15:00 | 272.65 | 272.68 | 272.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 272.65 | 272.68 | 272.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:30:00 | 272.15 | 272.68 | 272.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 272.50 | 272.65 | 272.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 12:45:00 | 272.70 | 272.65 | 272.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 272.30 | 272.58 | 272.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:00:00 | 272.30 | 272.58 | 272.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 272.85 | 272.63 | 272.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:30:00 | 272.65 | 272.63 | 272.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 273.80 | 272.93 | 272.50 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 272.00 | 272.47 | 272.53 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 14:15:00 | 274.25 | 272.76 | 272.65 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 13:15:00 | 270.65 | 272.66 | 272.74 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-11-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 14:15:00 | 274.90 | 273.11 | 272.94 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 11:15:00 | 272.05 | 272.74 | 272.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 12:15:00 | 270.20 | 272.23 | 272.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 11:15:00 | 272.35 | 271.03 | 271.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 11:15:00 | 272.35 | 271.03 | 271.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 272.35 | 271.03 | 271.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:00:00 | 272.35 | 271.03 | 271.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 274.00 | 271.63 | 271.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:00:00 | 274.00 | 271.63 | 271.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 13:15:00 | 274.35 | 272.17 | 272.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 14:15:00 | 276.45 | 273.03 | 272.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 269.35 | 274.38 | 273.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 269.35 | 274.38 | 273.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 269.35 | 274.38 | 273.89 | EMA400 retest candle locked (from upside) |

### Cycle 118 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 269.85 | 273.48 | 273.52 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 12:15:00 | 273.75 | 273.14 | 273.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 14:15:00 | 277.65 | 274.13 | 273.55 | Break + close above crossover candle high |

### Cycle 120 — SELL (started 2025-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 09:15:00 | 264.10 | 272.66 | 273.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 10:15:00 | 262.55 | 270.64 | 272.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 13:15:00 | 272.00 | 269.66 | 271.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 13:15:00 | 272.00 | 269.66 | 271.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 272.00 | 269.66 | 271.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 13:45:00 | 272.00 | 269.66 | 271.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 272.55 | 270.24 | 271.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 272.55 | 270.24 | 271.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 271.65 | 270.52 | 271.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 271.45 | 270.52 | 271.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 270.20 | 270.57 | 271.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 13:15:00 | 269.75 | 270.47 | 271.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 11:15:00 | 256.26 | 259.04 | 261.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 13:15:00 | 248.10 | 247.50 | 249.85 | SL hit (close>ema200) qty=0.50 sl=247.50 alert=retest2 |

### Cycle 121 — BUY (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 14:15:00 | 252.95 | 250.45 | 250.30 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 246.85 | 249.82 | 250.19 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 248.25 | 247.94 | 247.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 249.70 | 248.35 | 248.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 247.10 | 248.26 | 248.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 247.10 | 248.26 | 248.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 247.10 | 248.26 | 248.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:00:00 | 247.10 | 248.26 | 248.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 247.35 | 248.08 | 248.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:30:00 | 247.30 | 248.08 | 248.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 11:15:00 | 246.60 | 247.78 | 247.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 14:15:00 | 246.05 | 247.11 | 247.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 239.60 | 239.46 | 241.03 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-19 10:45:00 | 238.70 | 239.33 | 240.83 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 239.60 | 239.25 | 240.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:00:00 | 239.60 | 239.25 | 240.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 244.90 | 240.53 | 240.89 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 244.90 | 240.53 | 240.89 | SL hit (close>ema400) qty=1.00 sl=240.89 alert=retest1 |

### Cycle 125 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 246.30 | 241.69 | 241.39 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 10:15:00 | 240.00 | 241.26 | 241.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 14:15:00 | 239.50 | 240.38 | 240.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 15:15:00 | 237.10 | 237.08 | 238.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 09:15:00 | 236.30 | 237.08 | 238.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 237.70 | 236.47 | 237.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 237.70 | 236.47 | 237.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 237.90 | 236.76 | 237.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 236.15 | 236.76 | 237.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 236.60 | 236.77 | 237.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:30:00 | 237.70 | 236.77 | 237.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 237.45 | 236.91 | 237.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:00:00 | 237.45 | 236.91 | 237.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 237.90 | 237.11 | 237.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:30:00 | 237.70 | 237.11 | 237.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 237.60 | 237.20 | 237.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 13:30:00 | 237.90 | 237.20 | 237.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 14:15:00 | 240.00 | 237.76 | 237.56 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 14:15:00 | 237.20 | 238.26 | 238.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 15:15:00 | 234.50 | 237.51 | 238.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 232.30 | 232.27 | 233.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 09:45:00 | 233.23 | 232.27 | 233.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 220.90 | 220.65 | 223.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 219.93 | 220.63 | 222.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:45:00 | 220.00 | 220.51 | 222.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 220.25 | 220.46 | 222.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 15:00:00 | 219.69 | 220.30 | 222.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 208.93 | 213.35 | 216.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 209.00 | 213.35 | 216.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 209.24 | 213.35 | 216.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 208.71 | 213.35 | 216.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 13:15:00 | 212.90 | 212.11 | 214.66 | SL hit (close>ema200) qty=0.50 sl=212.11 alert=retest2 |

### Cycle 129 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 209.46 | 208.53 | 208.47 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 11:15:00 | 207.56 | 208.35 | 208.43 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2026-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 14:15:00 | 209.17 | 208.58 | 208.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 213.85 | 209.72 | 209.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 212.58 | 213.07 | 211.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 11:30:00 | 213.08 | 213.07 | 211.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 210.99 | 212.65 | 211.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 211.51 | 212.65 | 211.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 210.91 | 212.30 | 211.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 210.49 | 212.30 | 211.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 208.63 | 211.57 | 211.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 208.63 | 211.57 | 211.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 209.00 | 211.06 | 211.12 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 13:15:00 | 213.07 | 211.31 | 211.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 12:15:00 | 215.16 | 213.49 | 212.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 09:15:00 | 213.68 | 214.08 | 213.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 213.68 | 214.08 | 213.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 213.68 | 214.08 | 213.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 11:15:00 | 215.43 | 213.89 | 213.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 210.10 | 214.39 | 214.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 210.10 | 214.39 | 214.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 207.66 | 209.33 | 210.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 13:15:00 | 203.88 | 203.07 | 204.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 14:00:00 | 203.88 | 203.07 | 204.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 204.93 | 203.61 | 204.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 204.99 | 203.61 | 204.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 204.83 | 203.86 | 204.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 11:15:00 | 204.35 | 203.86 | 204.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 12:45:00 | 204.07 | 203.94 | 204.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:15:00 | 194.13 | 196.04 | 197.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:15:00 | 193.87 | 196.04 | 197.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 193.98 | 193.09 | 195.10 | SL hit (close>ema200) qty=0.50 sl=193.09 alert=retest2 |

### Cycle 135 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 180.55 | 178.11 | 177.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 11:15:00 | 196.28 | 181.81 | 179.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 179.30 | 182.26 | 180.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 179.30 | 182.26 | 180.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 179.30 | 182.26 | 180.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 179.30 | 182.26 | 180.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 178.60 | 181.52 | 180.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 177.60 | 181.52 | 180.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 176.60 | 180.54 | 179.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:15:00 | 176.84 | 180.54 | 179.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 178.20 | 180.07 | 179.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:15:00 | 179.70 | 180.07 | 179.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 12:15:00 | 178.09 | 179.45 | 179.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 12:15:00 | 178.09 | 179.45 | 179.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 14:15:00 | 176.00 | 178.53 | 179.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 176.45 | 173.95 | 175.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 176.45 | 173.95 | 175.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 176.45 | 173.95 | 175.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:15:00 | 176.00 | 173.95 | 175.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 176.40 | 174.44 | 175.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:45:00 | 175.75 | 174.44 | 175.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 176.19 | 174.79 | 175.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:00:00 | 176.19 | 174.79 | 175.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 175.57 | 175.10 | 175.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 175.57 | 175.10 | 175.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 175.00 | 175.08 | 175.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:45:00 | 175.82 | 175.08 | 175.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 175.00 | 175.07 | 175.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:15:00 | 179.40 | 175.07 | 175.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 177.55 | 175.56 | 175.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 09:15:00 | 184.49 | 178.20 | 177.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 178.89 | 184.85 | 181.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 09:15:00 | 178.89 | 184.85 | 181.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 178.89 | 184.85 | 181.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 178.89 | 184.85 | 181.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 177.29 | 183.34 | 181.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 11:00:00 | 177.29 | 183.34 | 181.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2026-03-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 13:15:00 | 176.90 | 179.76 | 180.15 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 186.27 | 179.94 | 179.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 188.26 | 181.60 | 180.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 182.88 | 184.51 | 182.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 15:00:00 | 182.88 | 184.51 | 182.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 182.90 | 184.19 | 182.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 183.14 | 184.19 | 182.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 183.48 | 184.05 | 182.44 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 178.65 | 182.23 | 182.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 12:15:00 | 178.23 | 181.04 | 181.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 182.61 | 180.13 | 180.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 182.61 | 180.13 | 180.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 182.61 | 180.13 | 180.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 183.52 | 180.13 | 180.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 186.35 | 181.63 | 181.53 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 178.55 | 181.52 | 181.61 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 184.00 | 181.35 | 181.35 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 12:15:00 | 180.40 | 181.28 | 181.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 09:15:00 | 179.50 | 180.41 | 180.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 181.52 | 179.92 | 180.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 181.52 | 179.92 | 180.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 181.52 | 179.92 | 180.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 11:00:00 | 180.16 | 179.97 | 180.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 11:15:00 | 183.45 | 180.17 | 179.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2026-04-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 11:15:00 | 183.45 | 180.17 | 179.74 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 15:15:00 | 180.00 | 180.29 | 180.30 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 184.00 | 181.03 | 180.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 185.37 | 182.71 | 182.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 184.17 | 184.19 | 183.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 09:15:00 | 184.17 | 184.19 | 183.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 184.17 | 184.19 | 183.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:30:00 | 182.80 | 184.19 | 183.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 184.08 | 184.94 | 184.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 185.65 | 184.94 | 184.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-27 09:15:00 | 204.22 | 198.63 | 196.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 13:15:00 | 200.05 | 201.58 | 201.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 15:15:00 | 199.20 | 201.00 | 201.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 198.61 | 197.87 | 199.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 198.61 | 197.87 | 199.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 198.61 | 197.87 | 199.22 | EMA400 retest candle locked (from downside) |

### Cycle 149 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 203.40 | 200.35 | 200.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 13:15:00 | 204.67 | 201.22 | 200.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 10:15:00 | 205.22 | 205.24 | 203.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-06 11:00:00 | 205.22 | 205.24 | 203.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 207.76 | 210.40 | 208.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 208.00 | 210.40 | 208.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 206.86 | 209.69 | 208.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:00:00 | 206.86 | 209.69 | 208.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 207.61 | 208.51 | 208.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:45:00 | 207.90 | 208.51 | 208.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — SELL (started 2026-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 15:15:00 | 206.00 | 207.62 | 207.72 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 10:30:00 | 327.85 | 2024-05-14 11:15:00 | 339.25 | STOP_HIT | 1.00 | -3.48% |
| BUY | retest2 | 2024-05-23 09:15:00 | 340.85 | 2024-05-27 11:15:00 | 342.00 | STOP_HIT | 1.00 | 0.34% |
| BUY | retest2 | 2024-05-23 11:00:00 | 340.75 | 2024-05-27 11:15:00 | 342.00 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2024-05-31 09:15:00 | 348.00 | 2024-06-03 09:15:00 | 382.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-04 09:30:00 | 347.00 | 2024-06-04 10:15:00 | 337.65 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2024-06-06 11:30:00 | 343.85 | 2024-06-07 10:15:00 | 344.00 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2024-06-12 09:15:00 | 345.90 | 2024-06-12 12:15:00 | 344.65 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2024-06-20 12:15:00 | 340.10 | 2024-07-02 09:15:00 | 340.05 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2024-07-23 12:15:00 | 318.70 | 2024-07-24 09:15:00 | 325.60 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2024-07-26 10:30:00 | 327.90 | 2024-08-02 09:15:00 | 360.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-29 09:15:00 | 328.90 | 2024-08-02 09:15:00 | 361.79 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-19 13:00:00 | 363.10 | 2024-08-21 09:15:00 | 380.75 | STOP_HIT | 1.00 | -4.86% |
| BUY | retest2 | 2024-08-26 09:30:00 | 384.10 | 2024-08-27 09:15:00 | 377.40 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2024-08-26 14:45:00 | 378.70 | 2024-08-27 09:15:00 | 377.40 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2024-08-27 09:15:00 | 378.70 | 2024-08-27 09:15:00 | 377.40 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2024-09-10 11:15:00 | 362.50 | 2024-09-13 09:15:00 | 366.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-09-10 11:45:00 | 362.45 | 2024-09-13 09:15:00 | 366.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-09-10 14:30:00 | 362.50 | 2024-09-13 09:15:00 | 366.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-09-11 09:45:00 | 362.00 | 2024-09-13 09:15:00 | 366.00 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-09-11 13:00:00 | 360.90 | 2024-09-13 09:15:00 | 366.00 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-09-23 11:15:00 | 350.50 | 2024-09-24 09:15:00 | 352.70 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-09-23 12:45:00 | 350.65 | 2024-09-24 09:15:00 | 352.70 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-09-27 11:15:00 | 347.00 | 2024-09-30 15:15:00 | 347.60 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2024-09-27 12:45:00 | 346.40 | 2024-09-30 15:15:00 | 347.60 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2024-10-08 12:15:00 | 334.80 | 2024-10-08 15:15:00 | 340.00 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-10-11 11:30:00 | 335.95 | 2024-10-22 10:15:00 | 319.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-14 11:45:00 | 336.00 | 2024-10-22 10:15:00 | 319.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 11:30:00 | 335.70 | 2024-10-22 10:15:00 | 318.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 13:15:00 | 335.70 | 2024-10-22 10:15:00 | 318.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-11 11:30:00 | 335.95 | 2024-10-23 10:15:00 | 318.80 | STOP_HIT | 0.50 | 5.10% |
| SELL | retest2 | 2024-10-14 11:45:00 | 336.00 | 2024-10-23 10:15:00 | 318.80 | STOP_HIT | 0.50 | 5.12% |
| SELL | retest2 | 2024-10-15 11:30:00 | 335.70 | 2024-10-23 10:15:00 | 318.80 | STOP_HIT | 0.50 | 5.03% |
| SELL | retest2 | 2024-10-15 13:15:00 | 335.70 | 2024-10-23 10:15:00 | 318.80 | STOP_HIT | 0.50 | 5.03% |
| SELL | retest2 | 2024-10-21 12:45:00 | 325.85 | 2024-10-24 14:15:00 | 339.40 | STOP_HIT | 1.00 | -4.16% |
| SELL | retest2 | 2024-10-21 13:30:00 | 325.65 | 2024-10-24 14:15:00 | 339.40 | STOP_HIT | 1.00 | -4.22% |
| SELL | retest2 | 2024-10-21 14:00:00 | 325.70 | 2024-10-24 14:15:00 | 339.40 | STOP_HIT | 1.00 | -4.21% |
| SELL | retest2 | 2024-11-12 11:15:00 | 334.55 | 2024-11-13 15:15:00 | 317.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 11:15:00 | 334.55 | 2024-11-14 09:15:00 | 330.40 | STOP_HIT | 0.50 | 1.24% |
| SELL | retest2 | 2024-11-13 09:30:00 | 332.00 | 2024-11-21 09:15:00 | 298.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2024-11-29 09:15:00 | 317.95 | 2024-12-02 12:15:00 | 309.95 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest1 | 2024-11-29 12:30:00 | 314.90 | 2024-12-02 12:15:00 | 309.95 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest1 | 2024-11-29 13:15:00 | 314.70 | 2024-12-02 12:15:00 | 309.95 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest1 | 2024-11-29 13:45:00 | 314.75 | 2024-12-02 12:15:00 | 309.95 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-12-20 13:00:00 | 294.55 | 2024-12-23 10:15:00 | 304.05 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2024-12-23 09:45:00 | 294.65 | 2024-12-23 10:15:00 | 304.05 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2025-01-16 12:00:00 | 271.55 | 2025-01-24 12:15:00 | 257.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-16 14:00:00 | 271.95 | 2025-01-24 12:15:00 | 258.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-16 12:00:00 | 271.55 | 2025-01-27 09:15:00 | 244.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-16 14:00:00 | 271.95 | 2025-01-27 09:15:00 | 244.75 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-07 11:15:00 | 269.95 | 2025-02-07 11:15:00 | 267.95 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-02-13 14:15:00 | 247.55 | 2025-02-14 12:15:00 | 235.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 14:15:00 | 247.55 | 2025-02-17 14:15:00 | 235.90 | STOP_HIT | 0.50 | 4.71% |
| BUY | retest2 | 2025-02-24 10:30:00 | 260.00 | 2025-02-27 09:15:00 | 251.70 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2025-02-24 11:15:00 | 260.00 | 2025-02-27 09:15:00 | 251.70 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2025-04-02 11:15:00 | 264.70 | 2025-04-07 09:15:00 | 253.25 | STOP_HIT | 1.00 | -4.33% |
| BUY | retest2 | 2025-04-04 12:30:00 | 264.80 | 2025-04-07 09:15:00 | 253.25 | STOP_HIT | 1.00 | -4.36% |
| BUY | retest2 | 2025-04-04 14:45:00 | 264.95 | 2025-04-07 09:15:00 | 253.25 | STOP_HIT | 1.00 | -4.42% |
| BUY | retest2 | 2025-04-11 09:15:00 | 275.65 | 2025-04-23 09:15:00 | 279.70 | STOP_HIT | 1.00 | 1.47% |
| SELL | retest2 | 2025-04-29 09:15:00 | 275.65 | 2025-04-29 12:15:00 | 261.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-29 09:15:00 | 275.65 | 2025-05-02 09:15:00 | 271.00 | STOP_HIT | 0.50 | 1.69% |
| SELL | retest2 | 2025-05-08 13:15:00 | 260.85 | 2025-05-12 09:15:00 | 266.95 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-05-12 12:00:00 | 263.00 | 2025-05-12 12:15:00 | 263.20 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-05-26 12:45:00 | 258.65 | 2025-05-27 11:15:00 | 272.90 | STOP_HIT | 1.00 | -5.51% |
| SELL | retest2 | 2025-05-26 14:15:00 | 258.50 | 2025-05-27 11:15:00 | 272.90 | STOP_HIT | 1.00 | -5.57% |
| SELL | retest2 | 2025-05-26 15:00:00 | 258.60 | 2025-05-27 11:15:00 | 272.90 | STOP_HIT | 1.00 | -5.53% |
| SELL | retest2 | 2025-05-27 09:30:00 | 258.80 | 2025-05-27 11:15:00 | 272.90 | STOP_HIT | 1.00 | -5.45% |
| SELL | retest2 | 2025-06-19 09:15:00 | 261.95 | 2025-06-25 12:15:00 | 258.65 | STOP_HIT | 1.00 | 1.26% |
| SELL | retest2 | 2025-07-02 13:15:00 | 257.15 | 2025-07-08 09:15:00 | 258.20 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-07-03 09:30:00 | 257.20 | 2025-07-08 09:15:00 | 258.20 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-07-03 12:15:00 | 257.15 | 2025-07-08 09:15:00 | 258.20 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-07-03 13:45:00 | 257.20 | 2025-07-08 09:15:00 | 258.20 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-07-14 15:15:00 | 269.00 | 2025-07-15 14:15:00 | 263.50 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-07-15 12:00:00 | 269.00 | 2025-07-15 14:15:00 | 263.50 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-07-15 12:45:00 | 268.85 | 2025-07-15 14:15:00 | 263.50 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-07-31 09:15:00 | 264.70 | 2025-08-07 09:15:00 | 251.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 13:45:00 | 264.80 | 2025-08-07 09:15:00 | 251.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 09:15:00 | 264.70 | 2025-08-07 15:15:00 | 251.50 | STOP_HIT | 0.50 | 4.99% |
| SELL | retest2 | 2025-07-31 13:45:00 | 264.80 | 2025-08-07 15:15:00 | 251.50 | STOP_HIT | 0.50 | 5.02% |
| SELL | retest2 | 2025-08-18 12:00:00 | 250.40 | 2025-08-19 10:15:00 | 256.05 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-08-26 09:15:00 | 255.75 | 2025-09-02 11:15:00 | 253.05 | STOP_HIT | 1.00 | 1.06% |
| BUY | retest2 | 2025-09-05 11:15:00 | 263.50 | 2025-09-05 11:15:00 | 261.65 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-09-05 14:00:00 | 263.65 | 2025-09-08 11:15:00 | 261.40 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-09-11 15:00:00 | 258.45 | 2025-09-12 09:15:00 | 261.50 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-09-16 10:45:00 | 258.45 | 2025-09-16 14:15:00 | 260.70 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-10-07 15:15:00 | 265.75 | 2025-10-08 09:15:00 | 263.20 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-10-08 12:30:00 | 266.50 | 2025-10-09 09:15:00 | 263.05 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-10-08 14:45:00 | 266.00 | 2025-10-09 09:15:00 | 263.05 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-10-13 15:00:00 | 269.15 | 2025-10-16 12:15:00 | 265.15 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-10-14 13:30:00 | 269.10 | 2025-10-16 12:15:00 | 265.15 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-10-14 14:30:00 | 268.80 | 2025-10-16 12:15:00 | 265.15 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-10-15 09:15:00 | 269.25 | 2025-10-16 12:15:00 | 265.15 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-11-03 13:45:00 | 273.00 | 2025-11-06 14:15:00 | 271.10 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-11-04 10:15:00 | 273.30 | 2025-11-06 14:15:00 | 271.10 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-11-04 11:00:00 | 273.30 | 2025-11-06 14:15:00 | 271.10 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-11-04 12:00:00 | 274.70 | 2025-11-06 14:15:00 | 271.10 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-11-10 14:45:00 | 270.10 | 2025-11-11 11:15:00 | 272.20 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-11-10 15:15:00 | 270.00 | 2025-11-11 11:15:00 | 272.20 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-11-11 10:30:00 | 270.75 | 2025-11-11 11:15:00 | 272.20 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-11-26 13:15:00 | 269.75 | 2025-12-01 11:15:00 | 256.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 13:15:00 | 269.75 | 2025-12-04 13:15:00 | 248.10 | STOP_HIT | 0.50 | 8.03% |
| SELL | retest1 | 2025-12-19 10:45:00 | 238.70 | 2025-12-19 14:15:00 | 244.90 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2026-01-13 12:00:00 | 219.93 | 2026-01-19 09:15:00 | 208.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:45:00 | 220.00 | 2026-01-19 09:15:00 | 209.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 14:00:00 | 220.25 | 2026-01-19 09:15:00 | 209.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 15:00:00 | 219.69 | 2026-01-19 09:15:00 | 208.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:00:00 | 219.93 | 2026-01-19 13:15:00 | 212.90 | STOP_HIT | 0.50 | 3.20% |
| SELL | retest2 | 2026-01-13 12:45:00 | 220.00 | 2026-01-19 13:15:00 | 212.90 | STOP_HIT | 0.50 | 3.23% |
| SELL | retest2 | 2026-01-13 14:00:00 | 220.25 | 2026-01-19 13:15:00 | 212.90 | STOP_HIT | 0.50 | 3.34% |
| SELL | retest2 | 2026-01-13 15:00:00 | 219.69 | 2026-01-19 13:15:00 | 212.90 | STOP_HIT | 0.50 | 3.09% |
| SELL | retest2 | 2026-01-22 10:45:00 | 210.00 | 2026-01-28 14:15:00 | 209.46 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2026-01-22 15:15:00 | 210.05 | 2026-01-28 14:15:00 | 209.46 | STOP_HIT | 1.00 | 0.28% |
| SELL | retest2 | 2026-01-23 11:45:00 | 209.69 | 2026-01-28 14:15:00 | 209.46 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2026-01-28 14:00:00 | 210.11 | 2026-01-28 14:15:00 | 209.46 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2026-02-04 11:15:00 | 215.43 | 2026-02-06 09:15:00 | 210.10 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2026-02-17 11:15:00 | 204.35 | 2026-02-24 09:15:00 | 194.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 12:45:00 | 204.07 | 2026-02-24 09:15:00 | 193.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 11:15:00 | 204.35 | 2026-02-25 09:15:00 | 193.98 | STOP_HIT | 0.50 | 5.07% |
| SELL | retest2 | 2026-02-17 12:45:00 | 204.07 | 2026-02-25 09:15:00 | 193.98 | STOP_HIT | 0.50 | 4.94% |
| BUY | retest2 | 2026-03-12 11:15:00 | 179.70 | 2026-03-12 12:15:00 | 178.09 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2026-04-08 11:00:00 | 180.16 | 2026-04-10 11:15:00 | 183.45 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2026-04-21 09:15:00 | 185.65 | 2026-04-27 09:15:00 | 204.22 | TARGET_HIT | 1.00 | 10.00% |
