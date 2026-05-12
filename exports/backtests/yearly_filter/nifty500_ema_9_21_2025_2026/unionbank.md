# Union Bank of India (UNIONBANK)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 166.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 75 |
| ALERT1 | 46 |
| ALERT2 | 46 |
| ALERT2_SKIP | 19 |
| ALERT3 | 114 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 40 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 40 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 43 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 32
- **Target hits / Stop hits / Partials:** 1 / 40 / 2
- **Avg / median % per leg:** -0.28% / -0.88%
- **Sum % (uncompounded):** -12.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 6 | 27.3% | 1 | 21 | 0 | -0.20% | -4.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 22 | 6 | 27.3% | 1 | 21 | 0 | -0.20% | -4.3% |
| SELL (all) | 21 | 5 | 23.8% | 0 | 19 | 2 | -0.37% | -7.8% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.99% | 10.0% |
| SELL @ 3rd Alert (retest2) | 19 | 3 | 15.8% | 0 | 18 | 1 | -0.94% | -17.8% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.99% | 10.0% |
| retest2 (combined) | 41 | 9 | 22.0% | 1 | 39 | 1 | -0.54% | -22.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 152.34 | 153.85 | 153.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 150.67 | 152.37 | 153.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 147.03 | 146.72 | 148.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 13:45:00 | 146.63 | 146.72 | 148.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 146.82 | 146.85 | 148.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 147.54 | 146.85 | 148.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 144.82 | 141.76 | 142.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 144.82 | 141.76 | 142.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 142.89 | 141.99 | 142.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 142.46 | 141.99 | 142.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 13:45:00 | 142.75 | 142.27 | 142.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:45:00 | 142.62 | 142.57 | 142.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 10:30:00 | 142.79 | 142.70 | 142.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 143.14 | 142.78 | 142.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:30:00 | 143.75 | 142.78 | 142.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 143.38 | 142.90 | 143.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 13:00:00 | 143.38 | 142.90 | 143.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-23 13:15:00 | 143.74 | 143.07 | 143.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 143.74 | 143.07 | 143.07 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 14:15:00 | 142.40 | 142.94 | 143.01 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 146.44 | 143.69 | 143.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 149.73 | 147.22 | 146.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 148.91 | 149.26 | 147.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 15:00:00 | 148.91 | 149.26 | 147.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 151.53 | 152.22 | 150.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:45:00 | 150.83 | 152.22 | 150.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 152.80 | 153.76 | 152.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 152.29 | 153.76 | 152.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 153.43 | 153.70 | 152.85 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 12:15:00 | 151.05 | 152.46 | 152.52 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 13:15:00 | 152.80 | 152.45 | 152.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 14:15:00 | 153.25 | 152.61 | 152.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 11:15:00 | 151.65 | 152.67 | 152.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 11:15:00 | 151.65 | 152.67 | 152.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 151.65 | 152.67 | 152.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:00:00 | 151.65 | 152.67 | 152.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 152.05 | 152.54 | 152.53 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 13:15:00 | 152.11 | 152.46 | 152.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 150.65 | 151.92 | 152.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 12:15:00 | 144.38 | 144.07 | 145.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 12:15:00 | 144.38 | 144.07 | 145.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 144.38 | 144.07 | 145.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:30:00 | 144.40 | 144.07 | 145.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 147.20 | 144.59 | 145.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:00:00 | 147.20 | 144.59 | 145.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 145.92 | 144.86 | 145.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:15:00 | 145.07 | 144.86 | 145.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 13:15:00 | 146.80 | 145.72 | 145.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 13:15:00 | 146.80 | 145.72 | 145.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 147.43 | 146.43 | 145.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 12:15:00 | 146.51 | 146.65 | 146.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 12:45:00 | 146.63 | 146.65 | 146.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 145.92 | 146.50 | 146.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:45:00 | 145.85 | 146.50 | 146.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 146.32 | 146.47 | 146.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:30:00 | 145.92 | 146.47 | 146.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 146.50 | 146.47 | 146.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 146.78 | 146.47 | 146.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 10:00:00 | 147.36 | 146.65 | 146.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 11:45:00 | 146.67 | 146.88 | 146.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 09:15:00 | 145.42 | 147.30 | 146.95 | SL hit (close<static) qty=1.00 sl=146.17 alert=retest2 |

### Cycle 9 — SELL (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 11:15:00 | 145.58 | 146.66 | 146.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 09:15:00 | 142.65 | 145.45 | 145.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 09:15:00 | 144.20 | 143.66 | 144.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 144.20 | 143.66 | 144.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 144.20 | 143.66 | 144.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 11:15:00 | 143.27 | 143.64 | 144.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:30:00 | 143.37 | 143.51 | 144.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 14:15:00 | 143.02 | 143.50 | 144.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 09:30:00 | 142.12 | 143.10 | 143.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 143.31 | 142.97 | 143.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:45:00 | 143.25 | 142.97 | 143.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 143.71 | 143.12 | 143.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 143.29 | 143.12 | 143.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 143.00 | 143.09 | 143.44 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-24 14:15:00 | 145.26 | 143.79 | 143.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 14:15:00 | 145.26 | 143.79 | 143.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 15:15:00 | 145.45 | 144.12 | 143.82 | Break + close above crossover candle high |

### Cycle 11 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 140.00 | 143.30 | 143.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 137.80 | 142.20 | 142.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 11:15:00 | 131.34 | 131.05 | 132.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-31 11:45:00 | 130.98 | 131.05 | 132.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 132.23 | 131.28 | 132.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 13:00:00 | 132.23 | 131.28 | 132.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 131.81 | 131.39 | 132.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 13:30:00 | 132.22 | 131.39 | 132.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 129.04 | 128.98 | 129.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 13:00:00 | 128.68 | 128.97 | 129.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:30:00 | 128.50 | 128.34 | 128.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:00:00 | 128.14 | 128.34 | 128.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:30:00 | 128.39 | 128.35 | 128.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 129.33 | 128.54 | 128.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:00:00 | 129.33 | 128.54 | 128.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 130.89 | 129.01 | 129.15 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-06 12:15:00 | 130.89 | 129.01 | 129.15 | SL hit (close>static) qty=1.00 sl=129.87 alert=retest2 |

### Cycle 12 — BUY (started 2025-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 13:15:00 | 131.40 | 129.49 | 129.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 14:15:00 | 132.23 | 131.02 | 130.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 09:15:00 | 131.03 | 131.29 | 130.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-08 10:00:00 | 131.03 | 131.29 | 130.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 13:15:00 | 131.32 | 131.31 | 130.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 13:30:00 | 131.53 | 131.31 | 130.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 130.91 | 131.19 | 130.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 09:15:00 | 132.84 | 131.19 | 130.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 12:15:00 | 136.05 | 136.77 | 136.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2025-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 12:15:00 | 136.05 | 136.77 | 136.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 134.89 | 136.29 | 136.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 126.63 | 126.21 | 127.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 09:45:00 | 127.37 | 126.21 | 127.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 128.06 | 126.79 | 127.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:30:00 | 127.96 | 126.79 | 127.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 127.93 | 127.02 | 127.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 09:15:00 | 127.38 | 127.53 | 127.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 128.64 | 127.75 | 127.98 | SL hit (close>static) qty=1.00 sl=128.20 alert=retest2 |

### Cycle 14 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 128.98 | 128.18 | 128.15 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 13:15:00 | 127.74 | 128.07 | 128.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 15:15:00 | 127.52 | 127.90 | 128.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 09:15:00 | 128.77 | 128.07 | 128.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 09:15:00 | 128.77 | 128.07 | 128.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 128.77 | 128.07 | 128.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:00:00 | 128.77 | 128.07 | 128.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 128.55 | 128.17 | 128.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 09:15:00 | 129.34 | 128.51 | 128.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 11:15:00 | 128.40 | 128.50 | 128.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 11:15:00 | 128.40 | 128.50 | 128.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 128.40 | 128.50 | 128.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:15:00 | 128.14 | 128.50 | 128.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 127.99 | 128.40 | 128.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:45:00 | 127.87 | 128.40 | 128.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 127.57 | 128.23 | 128.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 126.71 | 127.93 | 128.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 14:15:00 | 127.23 | 126.91 | 127.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 14:15:00 | 127.23 | 126.91 | 127.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 127.23 | 126.91 | 127.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:45:00 | 127.43 | 126.91 | 127.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 127.41 | 127.01 | 127.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 127.59 | 127.01 | 127.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 128.25 | 127.26 | 127.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 128.25 | 127.26 | 127.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 128.40 | 127.48 | 127.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:45:00 | 128.10 | 127.48 | 127.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 11:15:00 | 128.55 | 127.70 | 127.64 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 11:15:00 | 126.82 | 127.70 | 127.74 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 130.70 | 128.14 | 127.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 132.09 | 128.93 | 128.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 10:15:00 | 132.96 | 133.27 | 132.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 11:00:00 | 132.96 | 133.27 | 132.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 133.08 | 133.46 | 132.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 14:15:00 | 134.83 | 133.33 | 132.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 11:15:00 | 138.70 | 139.36 | 139.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2025-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 11:15:00 | 138.70 | 139.36 | 139.38 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2025-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 13:15:00 | 139.78 | 139.45 | 139.41 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2025-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 14:15:00 | 138.47 | 139.25 | 139.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 15:15:00 | 138.35 | 139.07 | 139.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 136.12 | 135.87 | 137.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 09:45:00 | 136.24 | 135.87 | 137.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 135.48 | 135.78 | 136.61 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2025-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 11:15:00 | 138.98 | 137.03 | 136.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 09:15:00 | 139.78 | 138.22 | 137.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 10:15:00 | 137.30 | 138.04 | 137.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 10:15:00 | 137.30 | 138.04 | 137.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 137.30 | 138.04 | 137.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:00:00 | 137.30 | 138.04 | 137.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 136.74 | 137.78 | 137.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:45:00 | 136.85 | 137.78 | 137.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 138.14 | 137.85 | 137.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 13:30:00 | 139.05 | 138.06 | 137.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 15:00:00 | 138.73 | 138.19 | 137.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 11:30:00 | 138.61 | 138.82 | 138.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 13:15:00 | 136.17 | 138.08 | 137.99 | SL hit (close<static) qty=1.00 sl=136.40 alert=retest2 |

### Cycle 25 — SELL (started 2025-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 15:15:00 | 137.70 | 137.94 | 137.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 09:15:00 | 136.71 | 137.69 | 137.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 09:15:00 | 137.36 | 136.83 | 137.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 09:15:00 | 137.36 | 136.83 | 137.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 137.36 | 136.83 | 137.18 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2025-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 12:15:00 | 138.61 | 137.49 | 137.42 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 136.16 | 137.31 | 137.44 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 13:15:00 | 138.46 | 137.42 | 137.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 139.80 | 138.03 | 137.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 10:15:00 | 138.73 | 138.91 | 138.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 10:45:00 | 138.59 | 138.91 | 138.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 138.70 | 138.87 | 138.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:45:00 | 138.96 | 138.91 | 138.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 10:15:00 | 137.74 | 138.91 | 138.71 | SL hit (close<static) qty=1.00 sl=138.34 alert=retest2 |

### Cycle 29 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 137.14 | 138.56 | 138.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 14:15:00 | 136.84 | 137.82 | 138.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 138.73 | 137.87 | 138.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 138.73 | 137.87 | 138.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 138.73 | 137.87 | 138.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 138.73 | 137.87 | 138.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 139.33 | 138.16 | 138.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 139.33 | 138.16 | 138.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 139.70 | 138.47 | 138.38 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 09:15:00 | 137.73 | 138.62 | 138.64 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 10:15:00 | 138.95 | 138.68 | 138.67 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 11:15:00 | 138.33 | 138.61 | 138.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 136.41 | 138.17 | 138.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 09:15:00 | 139.55 | 137.84 | 138.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 09:15:00 | 139.55 | 137.84 | 138.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 139.55 | 137.84 | 138.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:30:00 | 139.80 | 137.84 | 138.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 10:15:00 | 141.44 | 138.56 | 138.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 12:15:00 | 142.42 | 139.88 | 139.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 142.72 | 142.84 | 141.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 15:00:00 | 142.72 | 142.84 | 141.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 141.53 | 142.57 | 142.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:00:00 | 141.53 | 142.57 | 142.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 141.85 | 142.43 | 142.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:45:00 | 141.45 | 142.43 | 142.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 143.48 | 142.53 | 142.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 15:15:00 | 145.50 | 143.18 | 142.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 13:15:00 | 141.45 | 144.58 | 144.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 13:15:00 | 141.45 | 144.58 | 144.86 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 10:15:00 | 147.94 | 145.37 | 145.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 11:15:00 | 150.40 | 146.37 | 145.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 150.79 | 151.14 | 150.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 10:00:00 | 150.79 | 151.14 | 150.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 150.15 | 150.97 | 150.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:30:00 | 149.88 | 150.97 | 150.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 151.27 | 151.03 | 150.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 12:45:00 | 152.07 | 150.90 | 150.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 10:45:00 | 151.98 | 152.73 | 152.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 13:15:00 | 152.46 | 152.37 | 152.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 15:15:00 | 151.95 | 153.90 | 153.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 15:15:00 | 151.95 | 153.51 | 153.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 15:15:00 | 151.95 | 153.51 | 153.55 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 155.60 | 153.67 | 153.42 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2025-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 14:15:00 | 153.08 | 153.59 | 153.62 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 12:15:00 | 154.60 | 153.65 | 153.61 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 150.95 | 153.36 | 153.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 149.93 | 152.67 | 153.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 13:15:00 | 152.69 | 152.32 | 152.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-21 14:00:00 | 152.69 | 152.32 | 152.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 152.00 | 151.86 | 152.56 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2025-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 15:15:00 | 152.63 | 152.18 | 152.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 156.42 | 153.03 | 152.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 154.00 | 154.85 | 154.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 154.00 | 154.85 | 154.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 154.00 | 154.85 | 154.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:45:00 | 153.85 | 154.85 | 154.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 153.95 | 154.67 | 153.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:15:00 | 153.62 | 154.67 | 153.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 153.24 | 154.38 | 153.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:00:00 | 153.24 | 154.38 | 153.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 153.73 | 154.25 | 153.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:15:00 | 153.16 | 154.25 | 153.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 153.30 | 154.06 | 153.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:30:00 | 153.35 | 154.06 | 153.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 154.07 | 154.10 | 153.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:15:00 | 153.15 | 154.10 | 153.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 152.80 | 153.84 | 153.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:30:00 | 152.86 | 153.84 | 153.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 10:15:00 | 153.49 | 153.77 | 153.78 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 155.10 | 153.70 | 153.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 09:15:00 | 157.69 | 154.76 | 154.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 14:15:00 | 156.24 | 156.94 | 155.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 15:00:00 | 156.24 | 156.94 | 155.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 155.59 | 156.67 | 155.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 154.48 | 156.67 | 155.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 154.04 | 156.14 | 155.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 154.04 | 156.14 | 155.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 152.15 | 155.35 | 155.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:00:00 | 152.15 | 155.35 | 155.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 11:15:00 | 152.94 | 154.86 | 155.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 13:15:00 | 151.19 | 153.67 | 154.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 10:15:00 | 153.16 | 152.81 | 153.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 11:00:00 | 153.16 | 152.81 | 153.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 153.26 | 152.64 | 153.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:00:00 | 153.26 | 152.64 | 153.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 151.89 | 152.49 | 153.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:45:00 | 150.78 | 152.41 | 152.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 150.64 | 149.82 | 149.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2025-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 13:15:00 | 150.64 | 149.82 | 149.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 12:15:00 | 151.54 | 150.50 | 150.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 12:15:00 | 152.75 | 152.80 | 152.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 12:45:00 | 152.80 | 152.80 | 152.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 152.75 | 153.10 | 152.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:45:00 | 152.30 | 153.07 | 152.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 153.50 | 153.83 | 153.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:00:00 | 153.50 | 153.83 | 153.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 153.55 | 153.77 | 153.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:00:00 | 153.55 | 153.77 | 153.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 153.78 | 153.77 | 153.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:45:00 | 153.32 | 153.77 | 153.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 153.20 | 153.66 | 153.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:00:00 | 153.20 | 153.66 | 153.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 153.38 | 153.60 | 153.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 15:00:00 | 154.00 | 153.68 | 153.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 154.95 | 153.72 | 153.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 152.94 | 153.63 | 153.54 | SL hit (close<static) qty=1.00 sl=153.18 alert=retest2 |

### Cycle 47 — SELL (started 2025-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 13:15:00 | 152.65 | 153.35 | 153.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 10:15:00 | 152.53 | 153.07 | 153.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 10:15:00 | 150.15 | 149.59 | 150.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 10:15:00 | 150.15 | 149.59 | 150.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 150.15 | 149.59 | 150.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:30:00 | 150.12 | 149.59 | 150.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 150.93 | 149.86 | 150.28 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 14:15:00 | 151.80 | 150.62 | 150.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 09:15:00 | 153.90 | 151.34 | 150.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 11:15:00 | 153.69 | 153.70 | 152.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 12:00:00 | 153.69 | 153.70 | 152.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 153.00 | 153.56 | 152.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:45:00 | 153.29 | 153.56 | 152.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 164.04 | 165.26 | 163.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:30:00 | 164.98 | 165.26 | 163.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 161.86 | 164.58 | 163.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 161.86 | 164.58 | 163.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 161.90 | 164.04 | 163.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 161.90 | 164.04 | 163.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2026-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 14:15:00 | 161.49 | 162.87 | 163.02 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2026-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 10:15:00 | 164.22 | 163.27 | 163.16 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 161.78 | 163.00 | 163.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 161.68 | 162.60 | 162.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 11:15:00 | 162.39 | 162.37 | 162.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:00:00 | 162.39 | 162.37 | 162.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 163.40 | 162.57 | 162.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 13:00:00 | 163.40 | 162.57 | 162.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 162.91 | 162.64 | 162.77 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2026-01-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 14:15:00 | 163.93 | 162.90 | 162.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 09:15:00 | 165.86 | 163.71 | 163.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 10:15:00 | 175.01 | 175.68 | 173.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 11:00:00 | 175.01 | 175.68 | 173.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 13:15:00 | 175.18 | 176.58 | 175.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 14:00:00 | 175.18 | 176.58 | 175.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 175.79 | 176.42 | 175.44 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2026-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 11:15:00 | 173.20 | 174.84 | 174.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 13:15:00 | 171.40 | 174.00 | 174.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 175.98 | 174.01 | 174.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 175.98 | 174.01 | 174.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 175.98 | 174.01 | 174.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 177.18 | 174.01 | 174.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 175.48 | 174.30 | 174.47 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2026-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 12:15:00 | 175.02 | 174.66 | 174.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 10:15:00 | 178.05 | 175.58 | 175.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 12:15:00 | 175.37 | 175.83 | 175.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 12:15:00 | 175.37 | 175.83 | 175.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 175.37 | 175.83 | 175.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:00:00 | 175.37 | 175.83 | 175.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 172.78 | 175.22 | 175.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:00:00 | 172.78 | 175.22 | 175.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — SELL (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 14:15:00 | 173.01 | 174.78 | 174.88 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 175.85 | 174.57 | 174.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 12:15:00 | 178.91 | 176.22 | 175.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 11:15:00 | 178.54 | 179.15 | 177.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 12:00:00 | 178.54 | 179.15 | 177.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 181.18 | 179.56 | 178.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 12:00:00 | 182.55 | 180.32 | 178.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 11:15:00 | 171.90 | 178.63 | 178.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 171.90 | 178.63 | 178.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 169.29 | 175.15 | 177.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 171.25 | 170.15 | 172.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 171.25 | 170.15 | 172.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 172.39 | 170.77 | 172.77 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 174.99 | 173.40 | 173.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 11:15:00 | 176.19 | 174.17 | 173.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 15:15:00 | 176.75 | 176.87 | 175.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 09:15:00 | 176.97 | 176.87 | 175.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 175.46 | 176.63 | 175.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 174.88 | 176.63 | 175.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 176.20 | 176.54 | 175.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 14:15:00 | 176.76 | 176.39 | 175.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 10:15:00 | 177.77 | 178.69 | 178.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 15:15:00 | 178.76 | 179.45 | 179.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 15:15:00 | 178.76 | 179.45 | 179.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 177.43 | 179.04 | 179.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 12:15:00 | 179.00 | 178.63 | 179.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 12:15:00 | 179.00 | 178.63 | 179.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 179.00 | 178.63 | 179.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 12:30:00 | 178.49 | 178.63 | 179.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 180.63 | 179.03 | 179.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:00:00 | 180.63 | 179.03 | 179.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2026-02-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 14:15:00 | 182.76 | 179.78 | 179.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 15:15:00 | 183.98 | 180.62 | 179.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 13:15:00 | 191.08 | 192.24 | 189.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 14:00:00 | 191.08 | 192.24 | 189.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 189.57 | 191.71 | 189.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 189.57 | 191.71 | 189.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 189.00 | 191.17 | 189.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 191.96 | 191.17 | 189.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 198.60 | 199.74 | 197.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:45:00 | 198.87 | 199.74 | 197.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 196.94 | 198.98 | 197.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 12:45:00 | 197.04 | 198.98 | 197.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 197.15 | 198.61 | 197.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 14:45:00 | 198.40 | 198.78 | 197.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 12:15:00 | 196.90 | 199.83 | 200.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 196.90 | 199.83 | 200.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 191.90 | 197.70 | 198.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 194.20 | 193.23 | 195.50 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:45:00 | 192.13 | 193.05 | 195.21 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 191.60 | 191.23 | 192.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 12:45:00 | 192.39 | 191.23 | 192.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 182.52 | 187.86 | 190.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 182.55 | 181.25 | 184.66 | SL hit (close>ema200) qty=0.50 sl=181.25 alert=retest1 |

### Cycle 62 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 178.07 | 176.26 | 176.23 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 173.83 | 176.05 | 176.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 172.86 | 175.08 | 175.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 177.15 | 174.42 | 175.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 177.15 | 174.42 | 175.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 177.15 | 174.42 | 175.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 179.13 | 174.42 | 175.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 177.94 | 175.13 | 175.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 177.95 | 175.13 | 175.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 12:15:00 | 178.28 | 176.09 | 175.81 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 169.40 | 175.30 | 175.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 14:15:00 | 168.50 | 171.32 | 173.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 172.90 | 171.14 | 172.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 172.90 | 171.14 | 172.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 172.90 | 171.14 | 172.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:30:00 | 174.12 | 171.14 | 172.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 172.03 | 171.32 | 172.74 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 179.28 | 174.23 | 173.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 181.92 | 176.68 | 174.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 173.18 | 178.08 | 176.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 173.18 | 178.08 | 176.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 173.18 | 178.08 | 176.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 173.18 | 178.08 | 176.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 173.70 | 177.20 | 176.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 173.70 | 177.20 | 176.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 174.54 | 175.86 | 175.86 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 15:15:00 | 177.17 | 176.03 | 175.93 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 170.99 | 175.02 | 175.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 168.49 | 173.72 | 174.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 10:15:00 | 168.86 | 168.39 | 170.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-01 11:00:00 | 168.86 | 168.39 | 170.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 171.57 | 169.03 | 171.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 12:00:00 | 171.57 | 169.03 | 171.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 172.68 | 169.76 | 171.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 12:30:00 | 172.53 | 169.76 | 171.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 13:15:00 | 171.82 | 170.17 | 171.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 165.13 | 170.77 | 171.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 09:15:00 | 174.24 | 170.85 | 170.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 09:15:00 | 174.24 | 170.85 | 170.59 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2026-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 11:15:00 | 169.26 | 171.24 | 171.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 12:15:00 | 168.36 | 170.67 | 171.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 14:15:00 | 172.10 | 170.82 | 171.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 14:15:00 | 172.10 | 170.82 | 171.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 172.10 | 170.82 | 171.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 15:00:00 | 172.10 | 170.82 | 171.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 172.30 | 171.12 | 171.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:15:00 | 181.60 | 171.12 | 171.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 182.27 | 173.35 | 172.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 184.42 | 175.56 | 173.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 184.54 | 184.63 | 180.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:45:00 | 184.65 | 184.63 | 180.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 182.25 | 186.82 | 184.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:00:00 | 183.30 | 185.26 | 184.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 12:15:00 | 183.57 | 190.62 | 191.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 12:15:00 | 183.57 | 190.62 | 191.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 13:15:00 | 178.20 | 188.14 | 190.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 167.24 | 166.29 | 168.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 09:30:00 | 167.90 | 166.29 | 168.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 166.55 | 164.35 | 165.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:15:00 | 165.58 | 165.04 | 165.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 14:15:00 | 168.56 | 166.02 | 165.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 168.56 | 166.02 | 165.84 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 11:15:00 | 164.81 | 166.13 | 166.27 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-19 10:15:00 | 135.13 | 2025-06-02 09:15:00 | 148.64 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-20 12:15:00 | 142.46 | 2025-06-23 13:15:00 | 143.74 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-06-20 13:45:00 | 142.75 | 2025-06-23 13:15:00 | 143.74 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-06-23 09:45:00 | 142.62 | 2025-06-23 13:15:00 | 143.74 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-06-23 10:30:00 | 142.79 | 2025-06-23 13:15:00 | 143.74 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-07-14 11:15:00 | 145.07 | 2025-07-14 13:15:00 | 146.80 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-07-16 09:15:00 | 146.78 | 2025-07-17 09:15:00 | 145.42 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-07-16 10:00:00 | 147.36 | 2025-07-17 09:15:00 | 145.42 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-07-16 11:45:00 | 146.67 | 2025-07-17 09:15:00 | 145.42 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-07-22 11:15:00 | 143.27 | 2025-07-24 14:15:00 | 145.26 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-07-22 12:30:00 | 143.37 | 2025-07-24 14:15:00 | 145.26 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-07-22 14:15:00 | 143.02 | 2025-07-24 14:15:00 | 145.26 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-07-23 09:30:00 | 142.12 | 2025-07-24 14:15:00 | 145.26 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-08-05 13:00:00 | 128.68 | 2025-08-06 12:15:00 | 130.89 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-08-06 09:30:00 | 128.50 | 2025-08-06 12:15:00 | 130.89 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-08-06 10:00:00 | 128.14 | 2025-08-06 12:15:00 | 130.89 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-08-06 10:30:00 | 128.39 | 2025-08-06 12:15:00 | 130.89 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-08-11 09:15:00 | 132.84 | 2025-08-22 12:15:00 | 136.05 | STOP_HIT | 1.00 | 2.42% |
| SELL | retest2 | 2025-09-02 09:15:00 | 127.38 | 2025-09-02 09:15:00 | 128.64 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-09-15 14:15:00 | 134.83 | 2025-09-25 11:15:00 | 138.70 | STOP_HIT | 1.00 | 2.87% |
| BUY | retest2 | 2025-10-01 13:30:00 | 139.05 | 2025-10-03 13:15:00 | 136.17 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-10-01 15:00:00 | 138.73 | 2025-10-03 13:15:00 | 136.17 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-10-03 11:30:00 | 138.61 | 2025-10-03 13:15:00 | 136.17 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-10-13 12:45:00 | 138.96 | 2025-10-14 10:15:00 | 137.74 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-10-27 15:15:00 | 145.50 | 2025-10-30 13:15:00 | 141.45 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2025-11-07 12:45:00 | 152.07 | 2025-11-13 15:15:00 | 151.95 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-11-11 10:45:00 | 151.98 | 2025-11-13 15:15:00 | 151.95 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2025-11-11 13:15:00 | 152.46 | 2025-11-13 15:15:00 | 151.95 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-11-13 15:15:00 | 151.95 | 2025-11-13 15:15:00 | 151.95 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-12-08 09:45:00 | 150.78 | 2025-12-11 13:15:00 | 150.64 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-12-19 15:00:00 | 154.00 | 2025-12-22 11:15:00 | 152.94 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-12-22 09:15:00 | 154.95 | 2025-12-22 11:15:00 | 152.94 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2026-01-30 12:00:00 | 182.55 | 2026-02-01 11:15:00 | 171.90 | STOP_HIT | 1.00 | -5.83% |
| BUY | retest2 | 2026-02-06 14:15:00 | 176.76 | 2026-02-13 15:15:00 | 178.76 | STOP_HIT | 1.00 | 1.13% |
| BUY | retest2 | 2026-02-11 10:15:00 | 177.77 | 2026-02-13 15:15:00 | 178.76 | STOP_HIT | 1.00 | 0.56% |
| BUY | retest2 | 2026-02-25 14:45:00 | 198.40 | 2026-03-02 12:15:00 | 196.90 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest1 | 2026-03-05 10:45:00 | 192.13 | 2026-03-09 09:15:00 | 182.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-03-05 10:45:00 | 192.13 | 2026-03-10 10:15:00 | 182.55 | STOP_HIT | 0.50 | 4.99% |
| SELL | retest2 | 2026-03-11 11:15:00 | 182.55 | 2026-03-13 14:15:00 | 173.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 11:15:00 | 182.55 | 2026-03-16 14:15:00 | 175.66 | STOP_HIT | 0.50 | 3.77% |
| SELL | retest2 | 2026-04-02 09:15:00 | 165.13 | 2026-04-06 09:15:00 | 174.24 | STOP_HIT | 1.00 | -5.52% |
| BUY | retest2 | 2026-04-13 12:00:00 | 183.30 | 2026-04-23 12:15:00 | 183.57 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2026-05-06 12:15:00 | 165.58 | 2026-05-06 14:15:00 | 168.56 | STOP_HIT | 1.00 | -1.80% |
