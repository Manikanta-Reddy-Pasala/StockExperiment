# FSN E-Commerce Ventures Ltd. (NYKAA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 273.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 265 |
| ALERT1 | 160 |
| ALERT2 | 160 |
| ALERT2_SKIP | 81 |
| ALERT3 | 413 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 191 |
| PARTIAL | 20 |
| TARGET_HIT | 3 |
| STOP_HIT | 191 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 214 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 72 / 142
- **Target hits / Stop hits / Partials:** 3 / 191 / 20
- **Avg / median % per leg:** 0.26% / -0.85%
- **Sum % (uncompounded):** 56.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 114 | 32 | 28.1% | 3 | 111 | 0 | -0.04% | -4.4% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.52% | -3.0% |
| BUY @ 3rd Alert (retest2) | 112 | 32 | 28.6% | 3 | 109 | 0 | -0.01% | -1.3% |
| SELL (all) | 100 | 40 | 40.0% | 0 | 80 | 20 | 0.61% | 60.6% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.10% | -6.3% |
| SELL @ 3rd Alert (retest2) | 97 | 40 | 41.2% | 0 | 77 | 20 | 0.69% | 66.9% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.87% | -9.3% |
| retest2 (combined) | 209 | 72 | 34.4% | 3 | 186 | 20 | 0.31% | 65.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 12:15:00 | 127.90 | 126.53 | 126.35 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 11:15:00 | 125.05 | 126.41 | 126.43 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-17 15:15:00 | 126.80 | 126.40 | 126.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-18 09:15:00 | 128.00 | 126.72 | 126.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-18 13:15:00 | 126.65 | 127.05 | 126.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-18 13:15:00 | 126.65 | 127.05 | 126.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 13:15:00 | 126.65 | 127.05 | 126.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-18 14:00:00 | 126.65 | 127.05 | 126.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 14:15:00 | 126.00 | 126.84 | 126.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-18 15:00:00 | 126.00 | 126.84 | 126.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2023-05-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 15:15:00 | 125.65 | 126.60 | 126.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-19 09:15:00 | 124.70 | 126.22 | 126.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 10:15:00 | 127.15 | 126.41 | 126.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-19 10:15:00 | 127.15 | 126.41 | 126.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 10:15:00 | 127.15 | 126.41 | 126.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 10:30:00 | 127.20 | 126.41 | 126.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 11:15:00 | 127.05 | 126.54 | 126.56 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2023-05-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 12:15:00 | 127.30 | 126.69 | 126.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-19 13:15:00 | 127.80 | 126.91 | 126.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-22 09:15:00 | 126.20 | 127.15 | 126.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 09:15:00 | 126.20 | 127.15 | 126.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 126.20 | 127.15 | 126.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-22 10:00:00 | 126.20 | 127.15 | 126.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 10:15:00 | 127.20 | 127.16 | 126.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-22 11:15:00 | 127.50 | 127.16 | 126.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-22 14:15:00 | 127.40 | 127.25 | 127.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-23 09:15:00 | 126.25 | 126.92 | 126.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2023-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 09:15:00 | 126.25 | 126.92 | 126.94 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 13:15:00 | 128.15 | 127.02 | 126.97 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 10:15:00 | 126.00 | 127.02 | 127.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-24 14:15:00 | 125.25 | 126.26 | 126.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-25 10:15:00 | 126.70 | 126.15 | 126.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 10:15:00 | 126.70 | 126.15 | 126.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 10:15:00 | 126.70 | 126.15 | 126.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-25 10:45:00 | 127.00 | 126.15 | 126.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 11:15:00 | 127.55 | 126.43 | 126.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-25 12:00:00 | 127.55 | 126.43 | 126.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 12:15:00 | 127.00 | 126.54 | 126.60 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2023-05-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 13:15:00 | 127.15 | 126.66 | 126.65 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2023-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-26 09:15:00 | 125.60 | 126.55 | 126.62 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-06-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 12:15:00 | 126.00 | 125.72 | 125.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-02 09:15:00 | 127.45 | 126.16 | 125.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-06 12:15:00 | 134.75 | 135.22 | 133.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-06 12:45:00 | 134.95 | 135.22 | 133.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 15:15:00 | 133.90 | 134.76 | 133.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-07 09:15:00 | 133.05 | 134.76 | 133.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 09:15:00 | 133.40 | 134.49 | 133.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-07 11:00:00 | 136.25 | 134.84 | 133.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-07 12:15:00 | 135.85 | 134.93 | 133.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-07 13:30:00 | 135.45 | 135.06 | 134.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-07 14:15:00 | 135.45 | 135.06 | 134.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 15:15:00 | 134.10 | 134.80 | 134.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-08 10:15:00 | 136.20 | 134.77 | 134.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-15 14:15:00 | 138.05 | 139.89 | 140.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2023-06-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 14:15:00 | 138.05 | 139.89 | 140.14 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 11:15:00 | 143.20 | 140.43 | 140.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-16 12:15:00 | 144.10 | 141.17 | 140.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-20 10:15:00 | 147.75 | 148.08 | 145.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-20 10:45:00 | 148.20 | 148.08 | 145.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 15:15:00 | 146.70 | 147.61 | 146.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-21 09:15:00 | 150.25 | 147.61 | 146.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-27 09:15:00 | 147.95 | 148.94 | 148.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2023-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-27 09:15:00 | 147.95 | 148.94 | 148.97 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 11:15:00 | 149.90 | 148.97 | 148.97 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-27 12:15:00 | 148.85 | 148.94 | 148.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-27 14:15:00 | 148.10 | 148.74 | 148.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-28 09:15:00 | 148.60 | 148.59 | 148.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-28 09:15:00 | 148.60 | 148.59 | 148.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 09:15:00 | 148.60 | 148.59 | 148.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-28 12:45:00 | 148.20 | 148.45 | 148.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-28 14:45:00 | 147.80 | 148.32 | 148.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-30 15:15:00 | 148.20 | 147.69 | 147.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-03 10:15:00 | 149.05 | 148.22 | 148.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2023-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-03 10:15:00 | 149.05 | 148.22 | 148.17 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 09:15:00 | 147.60 | 148.24 | 148.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-04 10:15:00 | 147.30 | 148.05 | 148.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-06 09:15:00 | 144.30 | 144.26 | 145.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-06 09:30:00 | 144.40 | 144.26 | 145.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 09:15:00 | 145.85 | 144.22 | 144.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-07 10:00:00 | 145.85 | 144.22 | 144.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 10:15:00 | 143.35 | 144.05 | 144.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-07 15:00:00 | 141.70 | 143.23 | 144.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-11 10:45:00 | 142.30 | 141.35 | 142.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-11 13:15:00 | 145.65 | 143.09 | 142.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2023-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 13:15:00 | 145.65 | 143.09 | 142.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 15:15:00 | 147.60 | 144.49 | 143.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-12 12:15:00 | 145.00 | 145.43 | 144.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-12 12:30:00 | 144.80 | 145.43 | 144.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 14:15:00 | 144.15 | 145.09 | 144.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-12 14:45:00 | 143.60 | 145.09 | 144.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 15:15:00 | 144.00 | 144.87 | 144.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-13 09:15:00 | 144.85 | 144.87 | 144.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-13 15:15:00 | 143.10 | 144.43 | 144.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2023-07-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 15:15:00 | 143.10 | 144.43 | 144.47 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 12:15:00 | 145.50 | 144.41 | 144.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-14 13:15:00 | 146.15 | 144.76 | 144.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-17 12:15:00 | 145.25 | 145.49 | 145.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-17 12:45:00 | 145.15 | 145.49 | 145.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 13:15:00 | 145.45 | 145.48 | 145.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-17 14:45:00 | 145.75 | 145.54 | 145.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-18 10:15:00 | 144.45 | 145.27 | 145.14 | SL hit (close<static) qty=1.00 sl=144.65 alert=retest2 |

### Cycle 22 — SELL (started 2023-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 12:15:00 | 144.50 | 145.00 | 145.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-19 10:15:00 | 143.40 | 144.53 | 144.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-19 12:15:00 | 145.20 | 144.54 | 144.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-19 12:15:00 | 145.20 | 144.54 | 144.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 12:15:00 | 145.20 | 144.54 | 144.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-19 13:00:00 | 145.20 | 144.54 | 144.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 13:15:00 | 145.10 | 144.65 | 144.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-19 13:45:00 | 145.15 | 144.65 | 144.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2023-07-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 14:15:00 | 145.75 | 144.87 | 144.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-19 15:15:00 | 146.60 | 145.22 | 145.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-20 10:15:00 | 144.85 | 145.21 | 145.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-20 10:15:00 | 144.85 | 145.21 | 145.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 10:15:00 | 144.85 | 145.21 | 145.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 10:30:00 | 144.60 | 145.21 | 145.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 11:15:00 | 144.85 | 145.13 | 145.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-21 10:45:00 | 145.40 | 145.12 | 145.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-21 11:15:00 | 145.55 | 145.12 | 145.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-24 14:15:00 | 145.00 | 145.08 | 145.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2023-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 14:15:00 | 145.00 | 145.08 | 145.09 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-24 15:15:00 | 145.35 | 145.13 | 145.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-25 09:15:00 | 145.50 | 145.21 | 145.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-25 10:15:00 | 145.00 | 145.17 | 145.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-25 10:15:00 | 145.00 | 145.17 | 145.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 10:15:00 | 145.00 | 145.17 | 145.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 11:00:00 | 145.00 | 145.17 | 145.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 11:15:00 | 145.10 | 145.15 | 145.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 11:30:00 | 145.00 | 145.15 | 145.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2023-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-25 12:15:00 | 144.95 | 145.11 | 145.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-25 13:15:00 | 144.80 | 145.05 | 145.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-26 09:15:00 | 145.00 | 144.91 | 145.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-26 09:15:00 | 145.00 | 144.91 | 145.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 09:15:00 | 145.00 | 144.91 | 145.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-26 09:30:00 | 145.15 | 144.91 | 145.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 10:15:00 | 145.10 | 144.95 | 145.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-26 11:00:00 | 145.10 | 144.95 | 145.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2023-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 11:15:00 | 148.05 | 145.57 | 145.29 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 09:15:00 | 145.25 | 145.54 | 145.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-28 12:15:00 | 144.45 | 145.11 | 145.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-31 09:15:00 | 145.70 | 144.91 | 145.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-31 09:15:00 | 145.70 | 144.91 | 145.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 09:15:00 | 145.70 | 144.91 | 145.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 10:00:00 | 145.70 | 144.91 | 145.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 10:15:00 | 145.75 | 145.08 | 145.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 10:45:00 | 145.50 | 145.08 | 145.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 09:15:00 | 145.50 | 144.74 | 144.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-01 10:00:00 | 145.50 | 144.74 | 144.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2023-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-01 10:15:00 | 146.55 | 145.10 | 145.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 14:15:00 | 147.95 | 146.24 | 145.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 10:15:00 | 146.60 | 146.76 | 146.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 10:15:00 | 146.60 | 146.76 | 146.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 10:15:00 | 146.60 | 146.76 | 146.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:00:00 | 146.60 | 146.76 | 146.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 13:15:00 | 145.90 | 146.63 | 146.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 13:30:00 | 145.95 | 146.63 | 146.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 14:15:00 | 147.15 | 146.73 | 146.30 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2023-08-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 10:15:00 | 144.30 | 145.80 | 145.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-03 14:15:00 | 143.25 | 144.59 | 145.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-04 09:15:00 | 146.10 | 144.79 | 145.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 09:15:00 | 146.10 | 144.79 | 145.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 146.10 | 144.79 | 145.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 09:30:00 | 146.95 | 144.79 | 145.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 10:15:00 | 145.95 | 145.02 | 145.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 10:30:00 | 146.20 | 145.02 | 145.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2023-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 13:15:00 | 149.00 | 146.03 | 145.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 09:15:00 | 150.40 | 147.25 | 146.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-07 14:15:00 | 147.65 | 148.06 | 147.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-07 14:45:00 | 147.90 | 148.06 | 147.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 15:15:00 | 146.95 | 147.84 | 147.17 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2023-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 13:15:00 | 146.40 | 146.93 | 146.93 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2023-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 14:15:00 | 147.00 | 146.95 | 146.94 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 09:15:00 | 146.35 | 146.84 | 146.89 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-08-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-10 09:15:00 | 147.00 | 146.83 | 146.82 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-08-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 14:15:00 | 146.00 | 146.69 | 146.77 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-11 12:15:00 | 147.35 | 146.82 | 146.78 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2023-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 14:15:00 | 146.40 | 146.74 | 146.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-14 09:15:00 | 134.45 | 144.25 | 145.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-17 14:15:00 | 133.60 | 133.55 | 135.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-17 15:00:00 | 133.60 | 133.55 | 135.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 133.25 | 132.40 | 133.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-21 11:00:00 | 132.00 | 132.32 | 133.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-21 13:30:00 | 132.60 | 132.43 | 133.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-22 12:15:00 | 135.15 | 133.31 | 133.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2023-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 12:15:00 | 135.15 | 133.31 | 133.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-23 11:15:00 | 135.95 | 134.86 | 134.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-25 09:15:00 | 136.00 | 136.34 | 135.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 09:15:00 | 136.00 | 136.34 | 135.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 136.00 | 136.34 | 135.66 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2023-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-28 12:15:00 | 135.05 | 135.57 | 135.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-28 13:15:00 | 134.35 | 135.32 | 135.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-30 09:15:00 | 135.30 | 134.18 | 134.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 09:15:00 | 135.30 | 134.18 | 134.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 09:15:00 | 135.30 | 134.18 | 134.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 09:30:00 | 135.20 | 134.18 | 134.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 10:15:00 | 135.15 | 134.37 | 134.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 11:15:00 | 136.60 | 134.37 | 134.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2023-08-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 11:15:00 | 136.20 | 134.74 | 134.73 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2023-08-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 11:15:00 | 134.10 | 134.94 | 134.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 13:15:00 | 133.85 | 134.59 | 134.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 10:15:00 | 134.45 | 134.28 | 134.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-01 10:30:00 | 134.30 | 134.28 | 134.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 11:15:00 | 134.50 | 134.33 | 134.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 11:30:00 | 134.55 | 134.33 | 134.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 12:15:00 | 134.40 | 134.34 | 134.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 12:30:00 | 134.45 | 134.34 | 134.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 14:15:00 | 135.55 | 134.61 | 134.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 15:00:00 | 135.55 | 134.61 | 134.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2023-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 15:15:00 | 135.30 | 134.75 | 134.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-05 12:15:00 | 137.70 | 135.40 | 135.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-11 09:15:00 | 144.45 | 145.34 | 143.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-11 09:15:00 | 144.45 | 145.34 | 143.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 09:15:00 | 144.45 | 145.34 | 143.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-11 09:30:00 | 144.25 | 145.34 | 143.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 145.80 | 145.68 | 144.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:45:00 | 145.60 | 145.68 | 144.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 14:15:00 | 144.75 | 145.36 | 144.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 15:00:00 | 144.75 | 145.36 | 144.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 15:15:00 | 145.05 | 145.30 | 144.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-13 09:15:00 | 147.75 | 145.30 | 144.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-20 09:15:00 | 146.30 | 149.81 | 150.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2023-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 09:15:00 | 146.30 | 149.81 | 150.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 11:15:00 | 145.30 | 148.24 | 149.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 09:15:00 | 145.00 | 143.54 | 145.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 09:15:00 | 145.00 | 143.54 | 145.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 145.00 | 143.54 | 145.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 09:45:00 | 144.40 | 143.54 | 145.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 10:15:00 | 143.65 | 143.57 | 145.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 10:30:00 | 145.25 | 143.57 | 145.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 15:15:00 | 143.40 | 142.45 | 143.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 09:15:00 | 143.30 | 142.45 | 143.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 144.50 | 142.86 | 143.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 09:30:00 | 144.05 | 142.86 | 143.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 10:15:00 | 145.65 | 143.42 | 143.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 11:00:00 | 145.65 | 143.42 | 143.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2023-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 11:15:00 | 145.25 | 143.78 | 143.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-26 15:15:00 | 146.15 | 144.98 | 144.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 09:15:00 | 150.50 | 151.89 | 150.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 09:15:00 | 150.50 | 151.89 | 150.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 150.50 | 151.89 | 150.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-03 10:30:00 | 152.00 | 151.82 | 150.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-03 11:15:00 | 151.75 | 151.82 | 150.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-03 11:45:00 | 151.75 | 151.87 | 150.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-03 14:45:00 | 151.90 | 151.71 | 150.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 09:15:00 | 149.80 | 151.27 | 150.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 09:45:00 | 149.95 | 151.27 | 150.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-10-04 12:15:00 | 147.55 | 150.17 | 150.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2023-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 12:15:00 | 147.55 | 150.17 | 150.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-05 11:15:00 | 146.30 | 147.71 | 148.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 14:15:00 | 147.05 | 147.05 | 148.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-05 14:30:00 | 147.10 | 147.05 | 148.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 09:15:00 | 148.80 | 147.34 | 148.14 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2023-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 14:15:00 | 149.70 | 148.64 | 148.56 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2023-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 09:15:00 | 146.35 | 148.32 | 148.44 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 09:15:00 | 149.40 | 148.11 | 148.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 09:15:00 | 150.40 | 148.69 | 148.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-12 15:15:00 | 149.45 | 149.75 | 149.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-12 15:15:00 | 149.45 | 149.75 | 149.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 15:15:00 | 149.45 | 149.75 | 149.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 09:15:00 | 148.20 | 149.75 | 149.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 148.05 | 149.41 | 149.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 09:45:00 | 148.05 | 149.41 | 149.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 10:15:00 | 147.65 | 149.06 | 148.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 11:00:00 | 147.65 | 149.06 | 148.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2023-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 11:15:00 | 147.00 | 148.64 | 148.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-16 09:15:00 | 145.15 | 147.29 | 148.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-18 14:15:00 | 146.20 | 145.27 | 145.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 14:15:00 | 146.20 | 145.27 | 145.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 14:15:00 | 146.20 | 145.27 | 145.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-18 15:00:00 | 146.20 | 145.27 | 145.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 15:15:00 | 146.10 | 145.43 | 145.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-19 09:15:00 | 144.55 | 145.43 | 145.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 09:15:00 | 137.32 | 140.01 | 141.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-25 13:15:00 | 139.95 | 139.61 | 141.04 | SL hit (close>ema200) qty=0.50 sl=139.61 alert=retest2 |

### Cycle 51 — BUY (started 2023-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 10:15:00 | 140.65 | 140.03 | 139.98 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2023-11-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 11:15:00 | 139.15 | 139.97 | 140.04 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2023-11-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 10:15:00 | 140.45 | 140.01 | 140.00 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2023-11-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-02 13:15:00 | 139.85 | 139.98 | 139.99 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2023-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 09:15:00 | 141.40 | 140.24 | 140.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 09:15:00 | 146.90 | 141.73 | 140.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-08 14:15:00 | 149.55 | 149.73 | 148.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-08 14:30:00 | 149.15 | 149.73 | 148.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 14:15:00 | 147.75 | 149.19 | 148.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 15:00:00 | 147.75 | 149.19 | 148.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 15:15:00 | 147.50 | 148.85 | 148.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-10 09:15:00 | 146.10 | 148.85 | 148.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2023-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 09:15:00 | 146.10 | 148.30 | 148.33 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2023-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-10 15:15:00 | 150.00 | 148.39 | 148.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-13 09:15:00 | 151.50 | 148.99 | 148.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-16 14:15:00 | 153.75 | 154.18 | 153.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-16 15:00:00 | 153.75 | 154.18 | 153.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 15:15:00 | 153.45 | 154.03 | 153.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-17 09:15:00 | 154.85 | 154.03 | 153.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-11-20 09:15:00 | 170.34 | 165.35 | 160.24 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2023-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 09:15:00 | 169.00 | 170.86 | 170.86 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2023-11-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 14:15:00 | 172.35 | 170.36 | 170.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 11:15:00 | 175.00 | 172.12 | 171.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 09:15:00 | 174.00 | 174.27 | 172.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-01 09:30:00 | 173.90 | 174.27 | 172.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 11:15:00 | 174.35 | 174.27 | 173.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-01 11:30:00 | 174.20 | 174.27 | 173.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 14:15:00 | 173.90 | 175.51 | 174.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-04 15:00:00 | 173.90 | 175.51 | 174.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 15:15:00 | 173.40 | 175.08 | 174.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 09:15:00 | 171.75 | 175.08 | 174.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2023-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 09:15:00 | 170.50 | 174.17 | 174.25 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2023-12-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 10:15:00 | 174.30 | 173.26 | 173.20 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2023-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 11:15:00 | 171.90 | 173.06 | 173.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 12:15:00 | 171.20 | 172.69 | 172.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-11 10:15:00 | 171.75 | 171.53 | 172.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-11 10:30:00 | 171.90 | 171.53 | 172.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 09:15:00 | 168.10 | 168.11 | 169.44 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2023-12-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 09:15:00 | 174.55 | 170.13 | 169.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 10:15:00 | 177.20 | 171.54 | 170.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 13:15:00 | 177.60 | 178.18 | 176.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-18 14:00:00 | 177.60 | 178.18 | 176.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 14:15:00 | 174.85 | 177.51 | 176.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 15:00:00 | 174.85 | 177.51 | 176.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 15:15:00 | 172.75 | 176.56 | 176.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 09:45:00 | 172.95 | 175.78 | 175.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2023-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-19 10:15:00 | 173.00 | 175.22 | 175.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-19 14:15:00 | 171.70 | 173.65 | 174.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-20 09:15:00 | 175.55 | 173.73 | 174.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 09:15:00 | 175.55 | 173.73 | 174.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 09:15:00 | 175.55 | 173.73 | 174.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-20 10:00:00 | 175.55 | 173.73 | 174.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 10:15:00 | 174.75 | 173.94 | 174.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-20 11:30:00 | 173.85 | 173.96 | 174.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 14:15:00 | 165.16 | 171.43 | 173.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-12-22 09:15:00 | 170.30 | 167.66 | 169.36 | SL hit (close>ema200) qty=0.50 sl=167.66 alert=retest2 |

### Cycle 65 — BUY (started 2023-12-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 15:15:00 | 172.05 | 170.31 | 170.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 09:15:00 | 174.10 | 171.32 | 170.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 11:15:00 | 171.25 | 171.76 | 171.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-27 11:15:00 | 171.25 | 171.76 | 171.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 11:15:00 | 171.25 | 171.76 | 171.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 11:45:00 | 171.55 | 171.76 | 171.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 12:15:00 | 172.80 | 171.97 | 171.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 12:30:00 | 170.80 | 171.97 | 171.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 10:15:00 | 172.15 | 172.46 | 171.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 10:45:00 | 172.00 | 172.46 | 171.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 11:15:00 | 173.00 | 172.57 | 171.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-28 13:30:00 | 174.00 | 173.14 | 172.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 11:00:00 | 173.65 | 172.82 | 172.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 14:30:00 | 173.70 | 173.26 | 172.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-01 10:15:00 | 171.70 | 172.70 | 172.59 | SL hit (close<static) qty=1.00 sl=171.85 alert=retest2 |

### Cycle 66 — SELL (started 2024-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-01 14:15:00 | 171.70 | 172.39 | 172.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-01 15:15:00 | 171.10 | 172.13 | 172.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-03 09:15:00 | 171.00 | 170.57 | 171.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-03 09:15:00 | 171.00 | 170.57 | 171.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 09:15:00 | 171.00 | 170.57 | 171.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-03 09:30:00 | 171.05 | 170.57 | 171.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 10:15:00 | 170.90 | 170.63 | 171.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-03 10:30:00 | 171.00 | 170.63 | 171.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 09:15:00 | 168.90 | 169.39 | 170.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-04 09:45:00 | 169.55 | 169.39 | 170.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 11:15:00 | 169.85 | 169.46 | 170.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-04 11:30:00 | 170.20 | 169.46 | 170.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 14:15:00 | 169.90 | 169.43 | 169.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-04 15:15:00 | 170.10 | 169.43 | 169.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 15:15:00 | 170.10 | 169.57 | 169.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-05 09:15:00 | 170.75 | 169.57 | 169.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 09:15:00 | 172.60 | 170.17 | 170.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-05 10:00:00 | 172.60 | 170.17 | 170.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2024-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 10:15:00 | 174.85 | 171.11 | 170.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-08 10:15:00 | 177.45 | 174.45 | 172.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-12 09:15:00 | 191.50 | 192.15 | 189.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-12 10:15:00 | 190.00 | 191.72 | 189.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 10:15:00 | 190.00 | 191.72 | 189.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-12 11:00:00 | 190.00 | 191.72 | 189.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 11:15:00 | 188.30 | 191.03 | 189.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-12 12:15:00 | 190.35 | 191.03 | 189.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-15 09:15:00 | 184.15 | 188.31 | 188.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2024-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 09:15:00 | 184.15 | 188.31 | 188.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-15 12:15:00 | 183.05 | 186.03 | 187.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 169.25 | 166.98 | 170.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 09:15:00 | 169.25 | 166.98 | 170.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 169.25 | 166.98 | 170.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 09:30:00 | 169.10 | 166.98 | 170.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 10:15:00 | 172.00 | 167.98 | 170.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 11:00:00 | 172.00 | 167.98 | 170.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 11:15:00 | 170.10 | 168.41 | 170.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 11:30:00 | 171.75 | 168.41 | 170.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 12:15:00 | 170.05 | 168.73 | 170.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 12:30:00 | 170.85 | 168.73 | 170.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 13:15:00 | 170.00 | 168.99 | 170.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 13:30:00 | 170.25 | 168.99 | 170.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 14:15:00 | 171.85 | 169.56 | 170.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 15:00:00 | 171.85 | 169.56 | 170.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 15:15:00 | 171.70 | 169.99 | 170.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-20 09:15:00 | 172.90 | 169.99 | 170.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2024-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 11:15:00 | 171.70 | 170.92 | 170.84 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 09:15:00 | 167.00 | 170.27 | 170.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 11:15:00 | 165.30 | 168.78 | 169.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 15:15:00 | 163.00 | 162.98 | 165.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-25 09:15:00 | 163.00 | 162.98 | 165.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 165.10 | 163.41 | 165.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 10:00:00 | 165.10 | 163.41 | 165.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 10:15:00 | 162.25 | 163.18 | 164.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 11:15:00 | 162.05 | 163.18 | 164.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-30 10:15:00 | 165.40 | 162.64 | 162.83 | SL hit (close>static) qty=1.00 sl=165.25 alert=retest2 |

### Cycle 71 — BUY (started 2024-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 11:15:00 | 166.30 | 163.37 | 163.15 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 09:15:00 | 161.45 | 163.39 | 163.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-01 11:15:00 | 160.80 | 162.55 | 163.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-01 14:15:00 | 162.25 | 161.92 | 162.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 14:15:00 | 162.25 | 161.92 | 162.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 14:15:00 | 162.25 | 161.92 | 162.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-01 15:15:00 | 163.95 | 161.92 | 162.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 15:15:00 | 163.95 | 162.33 | 162.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-02 09:15:00 | 163.90 | 162.33 | 162.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 09:15:00 | 165.55 | 162.97 | 163.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-02 10:00:00 | 165.55 | 162.97 | 163.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2024-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 11:15:00 | 164.35 | 163.34 | 163.22 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 09:15:00 | 161.20 | 163.08 | 163.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 10:15:00 | 160.25 | 162.51 | 162.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-05 15:15:00 | 161.60 | 161.37 | 162.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-05 15:15:00 | 161.60 | 161.37 | 162.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 15:15:00 | 161.60 | 161.37 | 162.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 09:15:00 | 163.00 | 161.37 | 162.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 160.95 | 161.29 | 161.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 09:30:00 | 162.40 | 161.29 | 161.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 10:15:00 | 161.90 | 161.41 | 161.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 10:30:00 | 162.95 | 161.41 | 161.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 11:15:00 | 161.15 | 161.36 | 161.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-06 14:00:00 | 160.05 | 160.96 | 161.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-07 09:15:00 | 162.10 | 161.06 | 161.49 | SL hit (close>static) qty=1.00 sl=161.95 alert=retest2 |

### Cycle 75 — BUY (started 2024-02-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 11:15:00 | 151.75 | 148.99 | 148.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 14:15:00 | 154.00 | 150.90 | 149.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-15 10:15:00 | 149.45 | 151.14 | 150.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-15 10:15:00 | 149.45 | 151.14 | 150.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 10:15:00 | 149.45 | 151.14 | 150.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-15 11:00:00 | 149.45 | 151.14 | 150.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 11:15:00 | 150.85 | 151.08 | 150.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-15 11:30:00 | 149.35 | 151.08 | 150.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 12:15:00 | 152.00 | 151.27 | 150.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-15 14:30:00 | 153.10 | 151.91 | 150.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-19 09:15:00 | 152.95 | 151.80 | 151.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-19 10:15:00 | 153.00 | 151.85 | 151.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-19 11:15:00 | 153.00 | 151.97 | 151.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 153.45 | 153.01 | 152.35 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-02-21 10:15:00 | 151.10 | 152.33 | 152.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2024-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 10:15:00 | 151.10 | 152.33 | 152.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 12:15:00 | 148.60 | 151.33 | 151.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-21 14:15:00 | 151.50 | 151.26 | 151.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-21 15:00:00 | 151.50 | 151.26 | 151.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 15:15:00 | 151.00 | 151.21 | 151.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 09:15:00 | 150.70 | 151.21 | 151.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 150.75 | 151.12 | 151.60 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2024-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 10:15:00 | 151.85 | 151.63 | 151.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-26 13:15:00 | 152.70 | 152.09 | 151.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-28 14:15:00 | 153.50 | 153.84 | 153.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-28 15:00:00 | 153.50 | 153.84 | 153.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 15:15:00 | 153.00 | 153.67 | 153.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-29 09:15:00 | 155.60 | 153.67 | 153.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 09:15:00 | 155.80 | 154.10 | 153.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-01 09:15:00 | 158.20 | 156.09 | 154.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-01 14:30:00 | 157.95 | 156.97 | 155.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-02 09:15:00 | 159.45 | 156.97 | 156.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-06 10:15:00 | 156.10 | 158.20 | 158.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2024-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 10:15:00 | 156.10 | 158.20 | 158.33 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2024-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-11 10:15:00 | 160.20 | 158.35 | 158.09 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2024-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 11:15:00 | 155.80 | 157.88 | 158.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 10:15:00 | 153.35 | 156.08 | 157.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 10:15:00 | 154.20 | 152.16 | 154.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 10:15:00 | 154.20 | 152.16 | 154.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 10:15:00 | 154.20 | 152.16 | 154.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 11:00:00 | 154.20 | 152.16 | 154.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 11:15:00 | 152.95 | 152.32 | 154.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-14 12:45:00 | 151.95 | 152.34 | 153.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-14 14:00:00 | 151.25 | 152.12 | 153.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-14 15:00:00 | 150.95 | 151.89 | 153.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-19 10:15:00 | 152.75 | 151.63 | 151.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2024-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-19 10:15:00 | 152.75 | 151.63 | 151.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-19 12:15:00 | 153.25 | 152.07 | 151.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-20 09:15:00 | 151.95 | 152.64 | 152.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 09:15:00 | 151.95 | 152.64 | 152.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 151.95 | 152.64 | 152.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 10:00:00 | 151.95 | 152.64 | 152.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 10:15:00 | 151.65 | 152.44 | 152.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 10:45:00 | 151.55 | 152.44 | 152.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 11:15:00 | 151.45 | 152.24 | 152.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 12:00:00 | 151.45 | 152.24 | 152.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 12:15:00 | 152.00 | 152.20 | 152.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 13:30:00 | 152.50 | 152.23 | 152.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 14:45:00 | 152.50 | 152.38 | 152.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-03 09:15:00 | 162.00 | 163.38 | 163.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2024-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-03 09:15:00 | 162.00 | 163.38 | 163.39 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-03 11:15:00 | 163.95 | 163.42 | 163.40 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-04-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-03 12:15:00 | 163.20 | 163.38 | 163.38 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2024-04-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-03 13:15:00 | 164.55 | 163.61 | 163.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-03 14:15:00 | 165.15 | 163.92 | 163.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-04 10:15:00 | 163.55 | 163.89 | 163.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-04 10:15:00 | 163.55 | 163.89 | 163.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 10:15:00 | 163.55 | 163.89 | 163.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 10:45:00 | 163.40 | 163.89 | 163.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 11:15:00 | 163.10 | 163.74 | 163.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 11:30:00 | 163.15 | 163.74 | 163.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 13:15:00 | 163.45 | 163.66 | 163.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 13:45:00 | 163.25 | 163.66 | 163.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 14:15:00 | 164.05 | 163.74 | 163.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-04 15:15:00 | 165.00 | 163.74 | 163.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-09 09:15:00 | 181.50 | 176.94 | 172.53 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2024-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-16 12:15:00 | 175.70 | 177.58 | 177.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-16 14:15:00 | 173.75 | 176.64 | 177.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-19 12:15:00 | 170.00 | 168.39 | 171.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-19 13:00:00 | 170.00 | 168.39 | 171.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 166.25 | 168.23 | 170.19 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2024-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 11:15:00 | 173.15 | 169.74 | 169.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 09:15:00 | 176.60 | 172.42 | 171.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 11:15:00 | 175.75 | 175.76 | 174.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-25 12:00:00 | 175.75 | 175.76 | 174.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 10:15:00 | 176.85 | 177.28 | 176.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 11:30:00 | 179.10 | 177.32 | 176.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-02 11:15:00 | 174.75 | 176.61 | 176.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2024-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 11:15:00 | 174.75 | 176.61 | 176.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-02 12:15:00 | 173.10 | 175.91 | 176.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-02 14:15:00 | 178.05 | 175.87 | 176.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-02 14:15:00 | 178.05 | 175.87 | 176.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 14:15:00 | 178.05 | 175.87 | 176.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 15:00:00 | 178.05 | 175.87 | 176.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 15:15:00 | 177.05 | 176.11 | 176.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 10:15:00 | 176.45 | 176.25 | 176.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 15:15:00 | 167.63 | 170.89 | 172.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-07 09:15:00 | 176.95 | 172.10 | 173.11 | SL hit (close>ema200) qty=0.50 sl=172.10 alert=retest2 |

### Cycle 89 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 172.95 | 169.32 | 168.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 09:15:00 | 175.00 | 171.99 | 170.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 13:15:00 | 175.75 | 176.39 | 175.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 13:45:00 | 175.70 | 176.39 | 175.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 13:15:00 | 177.95 | 179.14 | 178.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 13:30:00 | 177.95 | 179.14 | 178.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 14:15:00 | 173.35 | 177.98 | 177.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 15:00:00 | 173.35 | 177.98 | 177.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2024-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 15:15:00 | 173.50 | 177.08 | 177.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 09:15:00 | 170.00 | 175.67 | 176.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 13:15:00 | 161.55 | 161.51 | 163.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-30 13:45:00 | 161.70 | 161.51 | 163.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 164.00 | 161.97 | 163.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 10:00:00 | 164.00 | 161.97 | 163.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 164.30 | 162.43 | 163.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 11:00:00 | 164.30 | 162.43 | 163.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 11:15:00 | 164.00 | 162.75 | 163.50 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 165.60 | 163.94 | 163.85 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-03 12:15:00 | 162.95 | 163.84 | 163.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-03 13:15:00 | 160.50 | 163.17 | 163.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 14:15:00 | 163.90 | 163.32 | 163.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 14:15:00 | 163.90 | 163.32 | 163.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 14:15:00 | 163.90 | 163.32 | 163.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 15:00:00 | 163.90 | 163.32 | 163.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 15:15:00 | 162.50 | 163.15 | 163.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 158.35 | 163.15 | 163.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 10:15:00 | 161.25 | 162.86 | 163.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 11:15:00 | 150.43 | 159.47 | 161.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 11:15:00 | 153.19 | 159.47 | 161.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-05 10:15:00 | 159.40 | 157.95 | 159.69 | SL hit (close>ema200) qty=0.50 sl=157.95 alert=retest2 |

### Cycle 93 — BUY (started 2024-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 14:15:00 | 162.50 | 160.53 | 160.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 166.10 | 161.97 | 161.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 12:15:00 | 169.62 | 170.17 | 168.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 13:00:00 | 169.62 | 170.17 | 168.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 13:15:00 | 168.61 | 169.86 | 168.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:00:00 | 168.61 | 169.86 | 168.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 169.02 | 169.69 | 168.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:30:00 | 168.80 | 169.69 | 168.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 169.05 | 169.56 | 168.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:15:00 | 173.00 | 169.56 | 168.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 14:30:00 | 169.49 | 171.05 | 170.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 15:00:00 | 170.25 | 171.05 | 170.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 09:15:00 | 169.90 | 170.64 | 170.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-13 09:15:00 | 167.69 | 170.05 | 170.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2024-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 09:15:00 | 167.69 | 170.05 | 170.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-13 10:15:00 | 167.12 | 169.46 | 169.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-14 13:15:00 | 168.19 | 166.69 | 167.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-14 13:15:00 | 168.19 | 166.69 | 167.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 13:15:00 | 168.19 | 166.69 | 167.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-14 14:00:00 | 168.19 | 166.69 | 167.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 14:15:00 | 170.88 | 167.53 | 168.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-14 14:30:00 | 172.10 | 167.53 | 168.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 15:15:00 | 170.28 | 168.08 | 168.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 09:15:00 | 172.96 | 168.08 | 168.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2024-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 09:15:00 | 175.99 | 169.66 | 168.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 10:15:00 | 177.00 | 171.13 | 169.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 09:15:00 | 173.36 | 173.73 | 171.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-19 09:30:00 | 174.18 | 173.73 | 171.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 11:15:00 | 171.95 | 173.24 | 171.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 11:45:00 | 171.99 | 173.24 | 171.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 12:15:00 | 172.00 | 172.99 | 171.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 13:15:00 | 173.57 | 172.99 | 171.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 14:45:00 | 173.25 | 172.76 | 172.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 09:30:00 | 174.40 | 173.01 | 172.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 11:30:00 | 173.54 | 172.95 | 172.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 12:15:00 | 172.58 | 172.87 | 172.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 13:00:00 | 172.58 | 172.87 | 172.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 173.74 | 173.05 | 172.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 14:30:00 | 174.20 | 173.39 | 172.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 09:45:00 | 174.85 | 173.78 | 173.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 11:15:00 | 173.87 | 175.02 | 175.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2024-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 11:15:00 | 173.87 | 175.02 | 175.12 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2024-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 14:15:00 | 176.71 | 175.16 | 175.05 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 09:15:00 | 174.65 | 175.80 | 175.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-03 10:15:00 | 172.92 | 174.64 | 175.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 13:15:00 | 174.58 | 174.40 | 174.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 13:15:00 | 174.58 | 174.40 | 174.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 13:15:00 | 174.58 | 174.40 | 174.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 13:45:00 | 174.60 | 174.40 | 174.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 175.22 | 174.56 | 174.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 15:00:00 | 175.22 | 174.56 | 174.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 15:15:00 | 174.40 | 174.53 | 174.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:15:00 | 174.30 | 174.53 | 174.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 173.10 | 174.24 | 174.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 14:30:00 | 172.53 | 173.33 | 174.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 09:15:00 | 172.32 | 173.26 | 173.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 11:15:00 | 177.20 | 174.28 | 174.29 | SL hit (close>static) qty=1.00 sl=176.84 alert=retest2 |

### Cycle 99 — BUY (started 2024-07-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 12:15:00 | 175.96 | 174.62 | 174.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 09:15:00 | 178.27 | 176.24 | 175.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 14:15:00 | 177.44 | 177.66 | 176.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-08 14:30:00 | 177.49 | 177.66 | 176.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 15:15:00 | 176.10 | 177.35 | 176.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:45:00 | 174.85 | 176.88 | 176.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 174.70 | 176.44 | 176.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:30:00 | 174.79 | 176.44 | 176.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2024-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 11:15:00 | 174.08 | 175.97 | 175.99 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 09:15:00 | 176.90 | 176.00 | 175.93 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 10:15:00 | 174.62 | 175.72 | 175.81 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2024-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 11:15:00 | 176.52 | 175.88 | 175.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 09:15:00 | 178.20 | 176.38 | 176.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 11:15:00 | 176.30 | 176.43 | 176.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 11:15:00 | 176.30 | 176.43 | 176.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 176.30 | 176.43 | 176.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:00:00 | 176.30 | 176.43 | 176.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 12:15:00 | 176.49 | 176.44 | 176.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 13:15:00 | 176.40 | 176.44 | 176.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 13:15:00 | 178.10 | 176.77 | 176.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 09:15:00 | 179.02 | 177.25 | 176.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 09:15:00 | 177.41 | 179.88 | 180.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 177.41 | 179.88 | 180.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 10:15:00 | 177.14 | 179.33 | 179.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 12:15:00 | 177.35 | 177.23 | 178.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 12:45:00 | 177.45 | 177.23 | 178.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 174.95 | 176.43 | 177.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 10:15:00 | 174.80 | 176.43 | 177.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 172.56 | 176.08 | 177.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 11:15:00 | 178.50 | 177.16 | 177.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2024-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 11:15:00 | 178.50 | 177.16 | 177.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 12:15:00 | 180.32 | 177.80 | 177.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 15:15:00 | 181.50 | 181.67 | 180.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 09:15:00 | 180.48 | 181.43 | 180.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 180.48 | 181.43 | 180.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:00:00 | 180.48 | 181.43 | 180.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 181.34 | 181.41 | 180.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 11:30:00 | 181.70 | 181.49 | 180.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-30 14:15:00 | 199.87 | 192.75 | 188.13 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 185.17 | 194.38 | 194.50 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2024-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 13:15:00 | 194.81 | 191.76 | 191.70 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 10:15:00 | 190.80 | 191.66 | 191.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 11:15:00 | 188.86 | 191.10 | 191.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 09:15:00 | 190.68 | 190.35 | 190.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 09:15:00 | 190.68 | 190.35 | 190.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 190.68 | 190.35 | 190.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 09:45:00 | 190.66 | 190.35 | 190.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 10:15:00 | 191.70 | 190.62 | 190.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 11:00:00 | 191.70 | 190.62 | 190.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 11:15:00 | 192.19 | 190.93 | 191.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 11:30:00 | 192.08 | 190.93 | 191.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2024-08-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 12:15:00 | 192.40 | 191.22 | 191.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 10:15:00 | 194.60 | 192.57 | 191.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 15:15:00 | 193.45 | 193.75 | 192.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 15:15:00 | 193.45 | 193.75 | 192.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 15:15:00 | 193.45 | 193.75 | 192.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 09:15:00 | 191.05 | 193.75 | 192.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 189.68 | 192.94 | 192.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 09:30:00 | 190.27 | 192.94 | 192.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 191.10 | 192.57 | 192.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 11:15:00 | 191.33 | 192.57 | 192.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 11:15:00 | 189.92 | 192.04 | 192.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2024-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 11:15:00 | 189.92 | 192.04 | 192.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 12:15:00 | 189.59 | 191.55 | 191.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 10:15:00 | 190.20 | 189.23 | 190.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 10:15:00 | 190.20 | 189.23 | 190.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 190.20 | 189.23 | 190.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 10:30:00 | 189.81 | 189.23 | 190.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 11:15:00 | 191.35 | 189.65 | 190.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 15:00:00 | 188.90 | 190.31 | 190.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 09:15:00 | 187.25 | 190.15 | 190.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 14:15:00 | 193.60 | 191.00 | 190.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2024-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 14:15:00 | 193.60 | 191.00 | 190.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 09:15:00 | 197.00 | 193.63 | 192.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 14:15:00 | 210.75 | 211.33 | 206.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 14:45:00 | 208.35 | 211.33 | 206.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 14:15:00 | 219.15 | 221.41 | 218.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 14:45:00 | 218.57 | 221.41 | 218.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 220.86 | 222.30 | 220.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 10:45:00 | 221.15 | 222.30 | 220.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 11:15:00 | 220.67 | 221.98 | 220.63 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2024-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 14:15:00 | 216.54 | 219.83 | 219.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 09:15:00 | 214.18 | 218.42 | 219.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-02 09:15:00 | 209.88 | 209.61 | 212.00 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 09:30:00 | 206.40 | 208.81 | 210.40 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 11:15:00 | 209.56 | 208.84 | 210.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 12:00:00 | 209.56 | 208.84 | 210.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 12:15:00 | 207.92 | 208.66 | 209.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 12:30:00 | 209.85 | 208.66 | 209.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 209.04 | 208.84 | 209.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 14:30:00 | 209.26 | 208.84 | 209.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 209.30 | 208.94 | 209.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 09:15:00 | 206.75 | 208.94 | 209.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 14:15:00 | 209.70 | 207.72 | 208.63 | SL hit (close>ema400) qty=1.00 sl=208.63 alert=retest1 |

### Cycle 113 — BUY (started 2024-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 09:15:00 | 213.31 | 209.23 | 209.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 10:15:00 | 214.47 | 210.28 | 209.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 10:15:00 | 210.85 | 213.58 | 212.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 10:15:00 | 210.85 | 213.58 | 212.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 210.85 | 213.58 | 212.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 11:00:00 | 210.85 | 213.58 | 212.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 11:15:00 | 214.99 | 213.86 | 212.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 13:30:00 | 215.36 | 214.18 | 212.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 15:00:00 | 216.95 | 214.73 | 213.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 09:30:00 | 218.05 | 215.76 | 213.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 10:15:00 | 215.31 | 216.32 | 215.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 10:15:00 | 215.11 | 216.08 | 215.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 10:30:00 | 214.50 | 216.08 | 215.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 11:15:00 | 215.50 | 215.96 | 215.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 12:15:00 | 214.98 | 215.96 | 215.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 12:15:00 | 215.20 | 215.81 | 215.80 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-11 13:15:00 | 213.20 | 215.29 | 215.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2024-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 13:15:00 | 213.20 | 215.29 | 215.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 14:15:00 | 212.57 | 214.74 | 215.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-13 10:15:00 | 209.14 | 208.96 | 211.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-13 11:00:00 | 209.14 | 208.96 | 211.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 204.27 | 206.66 | 208.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 10:45:00 | 204.02 | 205.91 | 208.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 13:15:00 | 203.98 | 205.13 | 207.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 13:45:00 | 204.00 | 204.92 | 207.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 14:15:00 | 204.03 | 204.92 | 207.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 202.78 | 201.27 | 203.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:30:00 | 204.00 | 201.27 | 203.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 199.86 | 200.30 | 201.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 11:15:00 | 197.50 | 200.14 | 201.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 11:15:00 | 206.00 | 202.03 | 201.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2024-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 11:15:00 | 206.00 | 202.03 | 201.69 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2024-09-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-20 15:15:00 | 200.75 | 201.63 | 201.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-23 12:15:00 | 199.48 | 200.60 | 201.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 14:15:00 | 202.35 | 200.88 | 201.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-23 14:15:00 | 202.35 | 200.88 | 201.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 14:15:00 | 202.35 | 200.88 | 201.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 15:00:00 | 202.35 | 200.88 | 201.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 15:15:00 | 201.60 | 201.02 | 201.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 09:15:00 | 200.46 | 201.02 | 201.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 13:15:00 | 201.84 | 199.14 | 198.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2024-09-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 13:15:00 | 201.84 | 199.14 | 198.88 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2024-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 09:15:00 | 195.97 | 199.60 | 199.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 13:15:00 | 195.47 | 197.36 | 198.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 197.00 | 196.96 | 197.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-01 09:30:00 | 197.16 | 196.96 | 197.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 197.34 | 197.04 | 197.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 14:45:00 | 197.59 | 197.04 | 197.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 197.25 | 197.08 | 197.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:15:00 | 195.20 | 197.08 | 197.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-08 15:15:00 | 194.16 | 193.32 | 193.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2024-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 15:15:00 | 194.16 | 193.32 | 193.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 10:15:00 | 196.78 | 194.04 | 193.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 10:15:00 | 196.05 | 197.08 | 195.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 11:00:00 | 196.05 | 197.08 | 195.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 11:15:00 | 193.30 | 196.32 | 195.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 12:00:00 | 193.30 | 196.32 | 195.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 12:15:00 | 192.93 | 195.64 | 195.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 12:30:00 | 192.35 | 195.64 | 195.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2024-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 13:15:00 | 192.55 | 195.02 | 195.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 13:15:00 | 190.65 | 192.79 | 193.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 09:15:00 | 192.97 | 192.61 | 193.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 09:15:00 | 192.97 | 192.61 | 193.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 192.97 | 192.61 | 193.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:45:00 | 192.83 | 192.61 | 193.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 11:15:00 | 193.30 | 192.76 | 193.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 12:00:00 | 193.30 | 192.76 | 193.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 12:15:00 | 192.52 | 192.72 | 193.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 13:15:00 | 192.29 | 192.72 | 193.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 14:45:00 | 192.30 | 192.57 | 193.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 09:30:00 | 191.90 | 192.18 | 192.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 09:15:00 | 182.68 | 186.34 | 188.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 09:15:00 | 182.69 | 186.34 | 188.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 09:15:00 | 182.31 | 186.34 | 188.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-17 14:15:00 | 185.20 | 184.57 | 186.65 | SL hit (close>ema200) qty=0.50 sl=184.57 alert=retest2 |

### Cycle 121 — BUY (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 14:15:00 | 178.91 | 176.96 | 176.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 180.00 | 177.89 | 177.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 09:15:00 | 180.00 | 180.61 | 179.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 09:15:00 | 180.00 | 180.61 | 179.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 180.00 | 180.61 | 179.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 15:00:00 | 182.00 | 180.64 | 179.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 10:15:00 | 181.33 | 181.72 | 180.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 11:15:00 | 181.71 | 181.54 | 180.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 14:30:00 | 181.11 | 181.36 | 180.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 15:15:00 | 180.00 | 181.09 | 180.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:15:00 | 183.23 | 181.09 | 180.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:45:00 | 182.85 | 181.60 | 181.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 10:45:00 | 183.00 | 182.03 | 181.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 15:00:00 | 183.43 | 182.73 | 181.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 182.00 | 182.52 | 181.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:00:00 | 182.00 | 182.52 | 181.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 182.00 | 182.42 | 181.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:30:00 | 181.97 | 182.42 | 181.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 11:15:00 | 182.13 | 182.36 | 181.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 11:45:00 | 182.00 | 182.36 | 181.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 12:15:00 | 182.00 | 182.29 | 181.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 12:45:00 | 181.95 | 182.29 | 181.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 13:15:00 | 182.00 | 182.23 | 181.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 13:30:00 | 181.95 | 182.23 | 181.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 14:15:00 | 181.25 | 182.03 | 181.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 15:00:00 | 181.25 | 182.03 | 181.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 15:15:00 | 181.50 | 181.93 | 181.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 09:15:00 | 182.55 | 181.93 | 181.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 12:15:00 | 183.09 | 184.65 | 184.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2024-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 12:15:00 | 183.09 | 184.65 | 184.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 13:15:00 | 182.48 | 184.22 | 184.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 184.92 | 184.08 | 184.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 184.92 | 184.08 | 184.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 184.92 | 184.08 | 184.41 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2024-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 10:15:00 | 186.95 | 184.65 | 184.64 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2024-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 11:15:00 | 182.34 | 184.19 | 184.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 12:15:00 | 179.99 | 183.35 | 184.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 14:15:00 | 169.35 | 169.00 | 171.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-18 14:45:00 | 169.69 | 169.00 | 171.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 172.70 | 169.91 | 171.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:45:00 | 172.20 | 169.91 | 171.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 172.57 | 170.44 | 171.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:45:00 | 173.05 | 170.44 | 171.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 168.52 | 169.83 | 170.83 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 170.95 | 169.67 | 169.55 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2024-11-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-25 12:15:00 | 168.07 | 169.40 | 169.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-25 15:15:00 | 167.40 | 168.66 | 169.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 13:15:00 | 167.32 | 167.26 | 168.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-26 14:00:00 | 167.32 | 167.26 | 168.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 169.50 | 167.69 | 168.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:15:00 | 170.60 | 167.69 | 168.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 170.65 | 168.29 | 168.39 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2024-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 10:15:00 | 169.68 | 168.56 | 168.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 172.64 | 170.09 | 169.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 09:15:00 | 172.20 | 172.57 | 171.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-29 10:00:00 | 172.20 | 172.57 | 171.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 172.30 | 172.51 | 171.30 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2024-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 13:15:00 | 170.06 | 171.04 | 171.06 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2024-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 14:15:00 | 171.54 | 171.14 | 171.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 09:15:00 | 173.21 | 171.61 | 171.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 11:15:00 | 171.57 | 171.69 | 171.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 11:15:00 | 171.57 | 171.69 | 171.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 11:15:00 | 171.57 | 171.69 | 171.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 12:30:00 | 172.48 | 171.65 | 171.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 15:15:00 | 172.50 | 171.67 | 171.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-04 11:15:00 | 170.45 | 171.75 | 171.62 | SL hit (close<static) qty=1.00 sl=170.90 alert=retest2 |

### Cycle 130 — SELL (started 2024-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 12:15:00 | 170.05 | 171.41 | 171.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 14:15:00 | 169.26 | 170.79 | 171.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 14:15:00 | 166.43 | 165.91 | 167.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-06 14:15:00 | 166.43 | 165.91 | 167.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 14:15:00 | 166.43 | 165.91 | 167.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 14:45:00 | 167.21 | 165.91 | 167.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 165.75 | 166.01 | 167.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 10:45:00 | 164.98 | 165.84 | 167.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 09:15:00 | 171.44 | 167.09 | 167.12 | SL hit (close>static) qty=1.00 sl=168.70 alert=retest2 |

### Cycle 131 — BUY (started 2024-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 10:15:00 | 172.70 | 168.22 | 167.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 13:15:00 | 173.56 | 170.34 | 168.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-11 10:15:00 | 170.50 | 171.25 | 169.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-11 11:00:00 | 170.50 | 171.25 | 169.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 11:15:00 | 170.24 | 171.05 | 169.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 09:15:00 | 171.00 | 170.32 | 169.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 09:45:00 | 170.76 | 170.36 | 169.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 10:15:00 | 171.05 | 170.36 | 169.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 11:15:00 | 171.04 | 170.36 | 169.96 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 170.51 | 170.39 | 170.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 11:45:00 | 170.59 | 170.39 | 170.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 13:15:00 | 169.86 | 170.32 | 170.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 14:00:00 | 169.86 | 170.32 | 170.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 14:15:00 | 169.61 | 170.18 | 170.01 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-12 14:15:00 | 169.61 | 170.18 | 170.01 | SL hit (close<static) qty=1.00 sl=169.75 alert=retest2 |

### Cycle 132 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 166.90 | 169.46 | 169.70 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2024-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 12:15:00 | 170.92 | 169.27 | 169.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 13:15:00 | 172.06 | 169.83 | 169.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 09:15:00 | 172.35 | 174.97 | 173.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 09:15:00 | 172.35 | 174.97 | 173.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 172.35 | 174.97 | 173.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 10:00:00 | 172.35 | 174.97 | 173.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 172.08 | 174.40 | 173.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 11:00:00 | 172.08 | 174.40 | 173.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 11:15:00 | 171.09 | 173.73 | 173.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 12:00:00 | 171.09 | 173.73 | 173.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 15:15:00 | 173.00 | 173.56 | 173.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-19 09:15:00 | 169.16 | 173.56 | 173.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2024-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 09:15:00 | 170.00 | 172.85 | 172.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 10:15:00 | 167.62 | 169.88 | 171.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 13:15:00 | 161.63 | 161.43 | 163.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 14:00:00 | 161.63 | 161.43 | 163.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 161.40 | 160.33 | 160.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 11:00:00 | 161.40 | 160.33 | 160.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 11:15:00 | 162.02 | 160.66 | 161.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 14:15:00 | 160.51 | 160.90 | 161.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 10:00:00 | 160.28 | 161.03 | 161.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-31 10:15:00 | 162.05 | 161.24 | 161.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2024-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 10:15:00 | 162.05 | 161.24 | 161.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 11:15:00 | 163.39 | 161.67 | 161.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 10:15:00 | 163.83 | 164.54 | 163.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-02 11:00:00 | 163.83 | 164.54 | 163.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 163.51 | 164.34 | 163.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 11:45:00 | 163.70 | 164.34 | 163.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 12:15:00 | 163.52 | 164.17 | 163.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 13:30:00 | 164.28 | 164.33 | 163.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-08 14:15:00 | 169.73 | 170.66 | 170.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 14:15:00 | 169.73 | 170.66 | 170.70 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 09:15:00 | 173.14 | 170.99 | 170.83 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 13:15:00 | 170.00 | 170.71 | 170.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 12:15:00 | 169.61 | 170.39 | 170.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 15:15:00 | 170.10 | 169.99 | 170.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 15:15:00 | 170.10 | 169.99 | 170.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 15:15:00 | 170.10 | 169.99 | 170.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 09:15:00 | 166.45 | 169.99 | 170.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-15 11:15:00 | 171.83 | 167.24 | 166.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2025-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 11:15:00 | 171.83 | 167.24 | 166.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 13:15:00 | 172.27 | 169.02 | 167.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 173.62 | 174.20 | 172.11 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 10:15:00 | 174.98 | 174.20 | 172.11 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 12:00:00 | 174.56 | 174.27 | 172.50 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 172.73 | 173.90 | 172.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:00:00 | 172.73 | 173.90 | 172.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 172.94 | 173.70 | 172.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 15:15:00 | 172.12 | 173.70 | 172.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 172.12 | 173.39 | 172.61 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-17 15:15:00 | 172.12 | 173.39 | 172.61 | SL hit (close<ema400) qty=1.00 sl=172.61 alert=retest1 |

### Cycle 140 — SELL (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 10:15:00 | 168.54 | 171.51 | 171.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 13:15:00 | 167.09 | 169.07 | 170.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 15:15:00 | 166.00 | 165.66 | 167.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 09:15:00 | 164.75 | 165.66 | 167.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 168.84 | 166.29 | 167.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 168.84 | 166.29 | 167.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 169.88 | 167.01 | 167.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:45:00 | 170.19 | 167.01 | 167.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2025-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 14:15:00 | 170.13 | 168.33 | 168.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 15:15:00 | 170.73 | 168.81 | 168.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 11:15:00 | 168.21 | 168.76 | 168.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 11:15:00 | 168.21 | 168.76 | 168.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 11:15:00 | 168.21 | 168.76 | 168.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 12:15:00 | 168.43 | 168.76 | 168.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 12:15:00 | 169.69 | 168.94 | 168.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 12:30:00 | 169.00 | 168.94 | 168.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 13:15:00 | 168.14 | 168.78 | 168.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:00:00 | 168.14 | 168.78 | 168.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — SELL (started 2025-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 14:15:00 | 166.69 | 168.36 | 168.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 162.67 | 167.17 | 167.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 14:15:00 | 166.57 | 166.02 | 166.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-27 14:15:00 | 166.57 | 166.02 | 166.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 14:15:00 | 166.57 | 166.02 | 166.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 15:00:00 | 166.57 | 166.02 | 166.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 15:15:00 | 167.44 | 166.30 | 166.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 09:15:00 | 169.11 | 166.30 | 166.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 167.57 | 166.56 | 166.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 09:30:00 | 168.19 | 166.56 | 166.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 10:15:00 | 168.78 | 167.00 | 167.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 11:00:00 | 168.78 | 167.00 | 167.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2025-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 11:15:00 | 171.57 | 167.91 | 167.56 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 14:15:00 | 168.63 | 168.95 | 168.95 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-01-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 12:15:00 | 169.96 | 169.08 | 168.98 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 10:15:00 | 165.68 | 168.32 | 168.66 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 12:15:00 | 176.38 | 169.64 | 169.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 13:15:00 | 177.18 | 171.15 | 169.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-04 10:15:00 | 178.01 | 178.28 | 175.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-04 10:30:00 | 177.08 | 178.28 | 175.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 175.77 | 177.91 | 176.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 11:00:00 | 175.77 | 177.91 | 176.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 11:15:00 | 176.04 | 177.54 | 176.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 11:30:00 | 176.39 | 177.54 | 176.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 12:15:00 | 174.97 | 177.02 | 176.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 13:00:00 | 174.97 | 177.02 | 176.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 13:15:00 | 174.75 | 176.57 | 176.50 | EMA400 retest candle locked (from upside) |

### Cycle 148 — SELL (started 2025-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 14:15:00 | 174.43 | 176.14 | 176.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 11:15:00 | 173.90 | 175.08 | 175.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 14:15:00 | 173.37 | 172.73 | 173.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 14:15:00 | 173.37 | 172.73 | 173.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 173.37 | 172.73 | 173.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 15:00:00 | 173.37 | 172.73 | 173.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 15:15:00 | 173.53 | 172.89 | 173.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 09:15:00 | 173.85 | 172.89 | 173.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 172.60 | 172.83 | 173.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 09:45:00 | 174.43 | 172.83 | 173.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 172.20 | 172.71 | 173.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:30:00 | 173.37 | 172.71 | 173.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 172.97 | 171.08 | 172.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 12:00:00 | 167.80 | 170.71 | 171.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 13:45:00 | 167.99 | 169.53 | 171.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-12 14:15:00 | 172.04 | 170.70 | 170.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2025-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 14:15:00 | 172.04 | 170.70 | 170.68 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2025-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 09:15:00 | 170.05 | 170.79 | 170.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 11:15:00 | 168.50 | 170.03 | 170.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 09:15:00 | 170.28 | 169.28 | 169.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 09:15:00 | 170.28 | 169.28 | 169.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 09:15:00 | 170.28 | 169.28 | 169.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:00:00 | 170.28 | 169.28 | 169.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 172.28 | 169.88 | 170.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:30:00 | 173.35 | 169.88 | 170.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2025-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 12:15:00 | 171.53 | 170.36 | 170.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-17 14:15:00 | 173.73 | 171.24 | 170.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-18 09:15:00 | 170.50 | 171.44 | 170.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-18 09:15:00 | 170.50 | 171.44 | 170.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 170.50 | 171.44 | 170.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:30:00 | 169.85 | 171.44 | 170.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 10:15:00 | 169.55 | 171.06 | 170.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-18 10:30:00 | 169.71 | 171.06 | 170.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 11:15:00 | 169.59 | 170.77 | 170.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-18 13:00:00 | 170.11 | 170.64 | 170.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-18 15:15:00 | 170.10 | 170.59 | 170.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2025-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 15:15:00 | 170.10 | 170.59 | 170.61 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 170.76 | 170.63 | 170.62 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-19 10:15:00 | 170.39 | 170.58 | 170.60 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 170.94 | 170.65 | 170.63 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-19 12:15:00 | 170.39 | 170.60 | 170.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-19 13:15:00 | 169.94 | 170.47 | 170.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-19 14:15:00 | 170.93 | 170.56 | 170.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 14:15:00 | 170.93 | 170.56 | 170.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 14:15:00 | 170.93 | 170.56 | 170.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 15:00:00 | 170.93 | 170.56 | 170.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2025-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 15:15:00 | 170.80 | 170.61 | 170.60 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 09:15:00 | 168.52 | 170.19 | 170.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-20 12:15:00 | 165.81 | 169.07 | 169.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 09:15:00 | 164.95 | 164.92 | 166.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-24 09:30:00 | 163.84 | 164.92 | 166.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 165.92 | 164.49 | 165.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:45:00 | 166.06 | 164.49 | 165.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 165.14 | 164.62 | 165.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 11:15:00 | 164.72 | 164.62 | 165.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 11:45:00 | 164.55 | 164.62 | 165.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 13:30:00 | 164.72 | 164.62 | 165.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 14:15:00 | 164.14 | 164.62 | 165.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 162.86 | 163.77 | 164.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 13:45:00 | 161.43 | 162.71 | 163.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 156.48 | 161.03 | 162.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 156.32 | 161.03 | 162.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 156.48 | 161.03 | 162.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-28 15:15:00 | 160.50 | 159.00 | 160.66 | SL hit (close>ema200) qty=0.50 sl=159.00 alert=retest2 |

### Cycle 159 — BUY (started 2025-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 13:15:00 | 161.67 | 160.37 | 160.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 14:15:00 | 162.06 | 160.71 | 160.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 09:15:00 | 165.93 | 166.93 | 165.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 09:15:00 | 165.93 | 166.93 | 165.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 165.93 | 166.93 | 165.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:45:00 | 165.26 | 166.93 | 165.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 165.46 | 166.40 | 165.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:00:00 | 165.46 | 166.40 | 165.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 164.82 | 166.09 | 165.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:00:00 | 164.82 | 166.09 | 165.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 165.06 | 165.88 | 165.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:45:00 | 164.76 | 165.88 | 165.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 164.92 | 165.69 | 165.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 15:00:00 | 164.92 | 165.69 | 165.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 165.00 | 165.55 | 165.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 162.23 | 165.55 | 165.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — SELL (started 2025-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 09:15:00 | 161.36 | 164.71 | 164.78 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2025-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 14:15:00 | 165.61 | 164.13 | 163.97 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 10:15:00 | 162.00 | 163.78 | 163.87 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 09:15:00 | 165.24 | 163.95 | 163.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 09:15:00 | 166.95 | 164.81 | 164.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-17 12:15:00 | 164.93 | 165.14 | 164.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 12:15:00 | 164.93 | 165.14 | 164.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 12:15:00 | 164.93 | 165.14 | 164.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 12:45:00 | 164.59 | 165.14 | 164.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 164.58 | 164.99 | 164.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 09:15:00 | 165.55 | 164.99 | 164.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 10:15:00 | 165.10 | 165.01 | 164.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 11:00:00 | 165.49 | 165.11 | 164.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-19 11:15:00 | 164.19 | 165.42 | 165.25 | SL hit (close<static) qty=1.00 sl=164.54 alert=retest2 |

### Cycle 164 — SELL (started 2025-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-19 13:15:00 | 164.53 | 165.15 | 165.15 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2025-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 09:15:00 | 168.64 | 165.74 | 165.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-21 09:15:00 | 168.95 | 167.80 | 166.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 10:15:00 | 172.75 | 173.03 | 171.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 11:00:00 | 172.75 | 173.03 | 171.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 170.31 | 172.49 | 171.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:00:00 | 170.31 | 172.49 | 171.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 170.00 | 171.99 | 171.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:30:00 | 170.17 | 171.99 | 171.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — SELL (started 2025-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 15:15:00 | 169.57 | 170.85 | 170.94 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 09:15:00 | 172.36 | 171.15 | 171.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-26 11:15:00 | 172.73 | 171.60 | 171.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-26 15:15:00 | 171.76 | 171.92 | 171.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-26 15:15:00 | 171.76 | 171.92 | 171.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 15:15:00 | 171.76 | 171.92 | 171.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:15:00 | 170.84 | 171.92 | 171.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 171.24 | 171.79 | 171.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:30:00 | 170.16 | 171.79 | 171.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 170.90 | 171.61 | 171.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:00:00 | 170.90 | 171.61 | 171.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 170.89 | 171.47 | 171.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 12:00:00 | 170.89 | 171.47 | 171.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — SELL (started 2025-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 12:15:00 | 170.52 | 171.28 | 171.35 | EMA200 below EMA400 |

### Cycle 169 — BUY (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 14:15:00 | 174.29 | 171.78 | 171.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 15:15:00 | 175.90 | 172.61 | 171.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 15:15:00 | 179.69 | 179.77 | 178.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-03 09:15:00 | 180.18 | 179.77 | 178.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 180.10 | 179.84 | 178.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:45:00 | 178.83 | 179.84 | 178.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 15:15:00 | 179.01 | 179.68 | 179.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 09:15:00 | 176.37 | 179.68 | 179.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 176.59 | 179.06 | 178.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 09:30:00 | 176.89 | 179.06 | 178.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 177.29 | 178.71 | 178.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 11:15:00 | 175.87 | 178.14 | 178.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-04 15:15:00 | 177.51 | 177.32 | 177.94 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-07 09:15:00 | 172.67 | 177.32 | 177.94 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 176.05 | 173.38 | 174.95 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-08 09:15:00 | 176.05 | 173.38 | 174.95 | SL hit (close>ema400) qty=1.00 sl=174.95 alert=retest1 |

### Cycle 171 — BUY (started 2025-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 13:15:00 | 177.35 | 175.73 | 175.71 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 09:15:00 | 173.70 | 175.60 | 175.69 | EMA200 below EMA400 |

### Cycle 173 — BUY (started 2025-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 14:15:00 | 176.97 | 175.86 | 175.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 179.00 | 176.67 | 176.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 13:15:00 | 177.99 | 178.22 | 177.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-11 14:00:00 | 177.99 | 178.22 | 177.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 10:15:00 | 189.13 | 191.44 | 189.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 11:00:00 | 189.13 | 191.44 | 189.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 11:15:00 | 187.31 | 190.61 | 189.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 11:30:00 | 187.79 | 190.61 | 189.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — SELL (started 2025-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-22 14:15:00 | 187.96 | 189.25 | 189.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-22 15:15:00 | 187.64 | 188.92 | 189.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-23 10:15:00 | 190.00 | 189.12 | 189.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-23 10:15:00 | 190.00 | 189.12 | 189.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 190.00 | 189.12 | 189.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-23 11:00:00 | 190.00 | 189.12 | 189.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — BUY (started 2025-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 11:15:00 | 191.92 | 189.68 | 189.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 12:15:00 | 192.80 | 190.31 | 189.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 15:15:00 | 193.31 | 193.64 | 192.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-25 09:15:00 | 192.87 | 193.64 | 192.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 190.95 | 193.10 | 192.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 190.95 | 193.10 | 192.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 189.58 | 192.40 | 191.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 189.58 | 192.40 | 191.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 12:15:00 | 190.97 | 192.03 | 191.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 13:00:00 | 190.97 | 192.03 | 191.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 13:15:00 | 192.03 | 192.03 | 191.89 | EMA400 retest candle locked (from upside) |

### Cycle 176 — SELL (started 2025-04-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 15:15:00 | 191.18 | 191.78 | 191.79 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 09:15:00 | 194.49 | 192.32 | 192.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 197.50 | 194.16 | 193.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 15:15:00 | 194.46 | 194.76 | 194.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-30 09:15:00 | 193.63 | 194.76 | 194.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 193.15 | 194.44 | 193.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 09:30:00 | 195.31 | 194.43 | 194.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 13:15:00 | 194.81 | 194.03 | 193.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 14:00:00 | 194.80 | 194.18 | 194.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 14:30:00 | 194.80 | 194.04 | 193.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-02 15:15:00 | 193.00 | 193.83 | 193.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — SELL (started 2025-05-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 15:15:00 | 193.00 | 193.83 | 193.89 | EMA200 below EMA400 |

### Cycle 179 — BUY (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 10:15:00 | 195.50 | 194.14 | 194.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 11:15:00 | 196.90 | 194.69 | 194.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-05 14:15:00 | 195.21 | 195.73 | 194.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-05 14:45:00 | 195.20 | 195.73 | 194.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 13:15:00 | 196.00 | 196.38 | 195.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:00:00 | 196.00 | 196.38 | 195.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 195.01 | 196.10 | 195.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:45:00 | 194.50 | 196.10 | 195.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 193.00 | 195.48 | 195.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:15:00 | 194.18 | 195.48 | 195.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — SELL (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 09:15:00 | 192.13 | 194.81 | 195.10 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2025-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 12:15:00 | 196.36 | 195.44 | 195.34 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 13:15:00 | 193.91 | 195.17 | 195.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 14:15:00 | 193.41 | 194.82 | 195.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 15:15:00 | 193.50 | 192.92 | 193.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 15:15:00 | 193.50 | 192.92 | 193.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 193.50 | 192.92 | 193.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 196.42 | 192.92 | 193.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 197.85 | 193.91 | 194.04 | EMA400 retest candle locked (from downside) |

### Cycle 183 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 198.08 | 194.74 | 194.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 198.40 | 195.47 | 194.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 196.16 | 197.70 | 196.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 12:15:00 | 196.16 | 197.70 | 196.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 196.16 | 197.70 | 196.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:00:00 | 196.16 | 197.70 | 196.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 197.94 | 197.74 | 196.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 14:15:00 | 198.58 | 197.74 | 196.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 200.79 | 197.62 | 196.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 11:00:00 | 198.00 | 197.89 | 197.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 13:00:00 | 198.01 | 197.91 | 197.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 197.31 | 197.84 | 197.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 14:45:00 | 197.50 | 197.84 | 197.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 15:15:00 | 197.15 | 197.70 | 197.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 10:15:00 | 198.81 | 197.72 | 197.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 199.84 | 197.46 | 197.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 15:15:00 | 197.06 | 199.37 | 199.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — SELL (started 2025-05-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 15:15:00 | 197.06 | 199.37 | 199.48 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2025-05-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 09:15:00 | 200.87 | 199.67 | 199.61 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 10:15:00 | 199.11 | 199.56 | 199.56 | EMA200 below EMA400 |

### Cycle 187 — BUY (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 12:15:00 | 200.55 | 199.75 | 199.65 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 198.58 | 199.46 | 199.54 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 201.00 | 199.77 | 199.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 10:15:00 | 202.99 | 200.42 | 199.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-21 13:15:00 | 200.39 | 200.61 | 200.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 13:15:00 | 200.39 | 200.61 | 200.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 13:15:00 | 200.39 | 200.61 | 200.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 13:45:00 | 199.96 | 200.61 | 200.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 200.46 | 200.58 | 200.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 15:15:00 | 201.30 | 200.58 | 200.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 15:15:00 | 201.30 | 200.72 | 200.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:15:00 | 201.29 | 200.72 | 200.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 202.88 | 201.16 | 200.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 09:45:00 | 204.00 | 202.14 | 201.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 11:30:00 | 203.96 | 202.43 | 201.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 09:15:00 | 204.36 | 202.79 | 202.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 14:15:00 | 200.17 | 201.97 | 202.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — SELL (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 14:15:00 | 200.17 | 201.97 | 202.02 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 204.10 | 201.64 | 201.60 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 14:15:00 | 200.93 | 201.62 | 201.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 09:15:00 | 200.00 | 201.20 | 201.44 | Break + close below crossover candle low |

### Cycle 193 — BUY (started 2025-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 14:15:00 | 205.10 | 201.46 | 201.39 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 09:15:00 | 200.08 | 202.03 | 202.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 11:15:00 | 195.00 | 199.75 | 200.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 13:15:00 | 195.77 | 195.44 | 197.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-03 14:00:00 | 195.77 | 195.44 | 197.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 196.50 | 195.40 | 196.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 10:30:00 | 195.20 | 195.54 | 196.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 12:15:00 | 199.17 | 196.50 | 196.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — BUY (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 12:15:00 | 199.17 | 196.50 | 196.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 10:15:00 | 201.08 | 198.00 | 197.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 09:15:00 | 200.00 | 200.67 | 199.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 10:00:00 | 200.00 | 200.67 | 199.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 199.21 | 200.27 | 199.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:00:00 | 199.21 | 200.27 | 199.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 198.90 | 199.99 | 199.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:30:00 | 198.82 | 199.99 | 199.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 199.15 | 199.82 | 199.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 199.98 | 199.65 | 199.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 11:00:00 | 199.96 | 199.74 | 199.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 13:15:00 | 198.27 | 199.29 | 199.27 | SL hit (close<static) qty=1.00 sl=198.80 alert=retest2 |

### Cycle 196 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 198.45 | 199.13 | 199.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 198.28 | 198.96 | 199.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 194.45 | 194.39 | 195.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 13:45:00 | 194.12 | 194.39 | 195.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 196.18 | 194.71 | 195.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 196.18 | 194.71 | 195.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 195.34 | 194.84 | 195.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 15:15:00 | 194.57 | 194.84 | 195.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 10:15:00 | 196.89 | 195.06 | 194.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — BUY (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 10:15:00 | 196.89 | 195.06 | 194.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 11:15:00 | 197.80 | 195.61 | 195.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 13:15:00 | 195.21 | 195.62 | 195.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 13:15:00 | 195.21 | 195.62 | 195.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 195.21 | 195.62 | 195.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 14:00:00 | 195.21 | 195.62 | 195.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 196.31 | 195.75 | 195.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 09:15:00 | 196.83 | 195.81 | 195.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 11:15:00 | 194.73 | 195.74 | 195.52 | SL hit (close<static) qty=1.00 sl=194.78 alert=retest2 |

### Cycle 198 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 193.46 | 195.28 | 195.34 | EMA200 below EMA400 |

### Cycle 199 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 197.11 | 195.26 | 195.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 10:15:00 | 200.17 | 196.77 | 195.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 09:15:00 | 198.29 | 200.67 | 198.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 198.29 | 200.67 | 198.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 198.29 | 200.67 | 198.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:45:00 | 196.64 | 200.67 | 198.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 198.82 | 200.30 | 198.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 11:15:00 | 199.47 | 200.30 | 198.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 12:30:00 | 199.34 | 199.85 | 198.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 201.03 | 199.01 | 198.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 09:15:00 | 204.95 | 207.07 | 207.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 200 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 204.95 | 207.07 | 207.08 | EMA200 below EMA400 |

### Cycle 201 — BUY (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 10:15:00 | 207.90 | 207.24 | 207.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 11:15:00 | 209.07 | 207.60 | 207.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 09:15:00 | 204.08 | 208.46 | 208.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 204.08 | 208.46 | 208.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 204.08 | 208.46 | 208.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:45:00 | 202.43 | 208.46 | 208.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — SELL (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 10:15:00 | 203.13 | 207.40 | 207.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 14:15:00 | 202.51 | 204.84 | 206.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 09:15:00 | 202.46 | 200.33 | 202.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 09:15:00 | 202.46 | 200.33 | 202.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 202.46 | 200.33 | 202.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 202.46 | 200.33 | 202.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 202.33 | 200.73 | 202.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:30:00 | 201.61 | 200.59 | 202.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 11:00:00 | 201.81 | 201.28 | 201.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 205.85 | 202.83 | 202.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 205.85 | 202.83 | 202.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 11:15:00 | 209.70 | 204.57 | 203.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 09:15:00 | 218.40 | 218.84 | 216.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 09:45:00 | 218.40 | 218.84 | 216.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 217.02 | 218.31 | 216.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:30:00 | 216.21 | 218.31 | 216.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 217.66 | 218.18 | 217.04 | EMA400 retest candle locked (from upside) |

### Cycle 204 — SELL (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 10:15:00 | 214.67 | 216.43 | 216.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 11:15:00 | 214.25 | 216.00 | 216.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 14:15:00 | 216.00 | 215.74 | 216.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 14:15:00 | 216.00 | 215.74 | 216.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 216.00 | 215.74 | 216.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 15:00:00 | 216.00 | 215.74 | 216.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 216.00 | 215.79 | 216.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:15:00 | 217.10 | 215.79 | 216.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 216.58 | 215.95 | 216.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:30:00 | 218.28 | 215.95 | 216.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 205 — BUY (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 10:15:00 | 218.71 | 216.50 | 216.36 | EMA200 above EMA400 |

### Cycle 206 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 214.88 | 216.30 | 216.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 12:15:00 | 213.32 | 215.71 | 216.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 11:15:00 | 214.38 | 213.93 | 214.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 12:00:00 | 214.38 | 213.93 | 214.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 214.58 | 214.15 | 214.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:30:00 | 214.66 | 214.15 | 214.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 215.04 | 214.33 | 214.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 15:00:00 | 215.04 | 214.33 | 214.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 216.00 | 214.66 | 214.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 216.90 | 214.66 | 214.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — BUY (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 09:15:00 | 219.17 | 215.56 | 215.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 12:15:00 | 219.81 | 217.26 | 216.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 09:15:00 | 217.98 | 218.58 | 217.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-23 10:00:00 | 217.98 | 218.58 | 217.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 216.83 | 218.23 | 217.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:00:00 | 216.83 | 218.23 | 217.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 217.40 | 218.07 | 217.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:45:00 | 216.60 | 218.07 | 217.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 216.08 | 217.67 | 217.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:00:00 | 216.08 | 217.67 | 217.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 216.10 | 217.36 | 217.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:00:00 | 216.10 | 217.36 | 217.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 216.75 | 217.71 | 217.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:00:00 | 216.75 | 217.71 | 217.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 216.38 | 217.44 | 217.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 11:00:00 | 216.38 | 217.44 | 217.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 216.21 | 217.19 | 217.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 11:30:00 | 216.48 | 217.19 | 217.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 208 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 215.93 | 216.94 | 217.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 13:15:00 | 215.60 | 216.67 | 216.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 214.20 | 213.51 | 214.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 09:30:00 | 213.71 | 213.51 | 214.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 212.49 | 213.30 | 214.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 11:30:00 | 211.85 | 213.11 | 214.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:15:00 | 212.20 | 213.11 | 214.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 12:00:00 | 212.07 | 210.76 | 211.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 12:45:00 | 212.07 | 211.02 | 211.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 14:15:00 | 214.00 | 211.99 | 211.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — BUY (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 14:15:00 | 214.00 | 211.99 | 211.86 | EMA200 above EMA400 |

### Cycle 210 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 210.30 | 211.76 | 211.80 | EMA200 below EMA400 |

### Cycle 211 — BUY (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 12:15:00 | 212.11 | 211.82 | 211.82 | EMA200 above EMA400 |

### Cycle 212 — SELL (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 13:15:00 | 210.78 | 211.61 | 211.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 209.42 | 211.17 | 211.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 14:15:00 | 210.46 | 210.28 | 210.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-01 15:00:00 | 210.46 | 210.28 | 210.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 210.00 | 210.02 | 210.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:30:00 | 210.52 | 210.02 | 210.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 209.46 | 209.91 | 210.46 | EMA400 retest candle locked (from downside) |

### Cycle 213 — BUY (started 2025-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 12:15:00 | 213.79 | 210.82 | 210.79 | EMA200 above EMA400 |

### Cycle 214 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 208.95 | 210.97 | 211.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 10:15:00 | 207.60 | 210.29 | 210.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 209.50 | 207.74 | 208.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 209.50 | 207.74 | 208.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 209.50 | 207.74 | 208.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 209.50 | 207.74 | 208.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 209.52 | 208.10 | 208.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 206.89 | 208.10 | 208.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 09:15:00 | 211.10 | 205.73 | 205.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 215 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 211.10 | 205.73 | 205.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 11:15:00 | 219.69 | 209.77 | 207.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 10:15:00 | 211.75 | 212.94 | 210.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 11:00:00 | 211.75 | 212.94 | 210.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 219.55 | 218.35 | 215.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 10:45:00 | 221.36 | 219.10 | 216.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 12:15:00 | 228.44 | 230.09 | 230.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 216 — SELL (started 2025-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 12:15:00 | 228.44 | 230.09 | 230.13 | EMA200 below EMA400 |

### Cycle 217 — BUY (started 2025-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 15:15:00 | 230.80 | 230.22 | 230.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 09:15:00 | 231.80 | 230.54 | 230.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 232.50 | 233.07 | 232.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 232.50 | 233.07 | 232.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 232.50 | 233.07 | 232.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 13:45:00 | 232.24 | 233.07 | 232.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 232.55 | 232.97 | 232.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:30:00 | 232.90 | 232.97 | 232.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 231.34 | 232.64 | 232.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 233.78 | 232.64 | 232.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 15:15:00 | 241.00 | 241.87 | 241.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 218 — SELL (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 15:15:00 | 241.00 | 241.87 | 241.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 09:15:00 | 240.67 | 241.63 | 241.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 15:15:00 | 238.22 | 238.08 | 239.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-12 09:15:00 | 237.80 | 238.08 | 239.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 237.79 | 238.02 | 238.94 | EMA400 retest candle locked (from downside) |

### Cycle 219 — BUY (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 11:15:00 | 241.80 | 239.46 | 239.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 13:15:00 | 243.00 | 240.54 | 239.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 09:15:00 | 242.66 | 242.68 | 241.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 09:45:00 | 242.50 | 242.68 | 241.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 241.45 | 242.39 | 241.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:00:00 | 241.45 | 242.39 | 241.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 241.21 | 242.16 | 241.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:45:00 | 241.06 | 242.16 | 241.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 241.16 | 241.96 | 241.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 13:45:00 | 241.63 | 241.96 | 241.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 242.64 | 242.09 | 241.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 15:15:00 | 243.25 | 242.09 | 241.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 13:15:00 | 242.91 | 242.88 | 242.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 10:15:00 | 243.45 | 243.84 | 243.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 15:15:00 | 242.59 | 243.49 | 243.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 220 — SELL (started 2025-09-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 15:15:00 | 242.59 | 243.49 | 243.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 238.30 | 242.45 | 243.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 15:15:00 | 238.55 | 238.37 | 239.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 09:15:00 | 237.72 | 238.37 | 239.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 232.32 | 232.04 | 233.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 11:00:00 | 232.32 | 232.04 | 233.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 232.70 | 231.88 | 232.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 14:45:00 | 232.31 | 231.88 | 232.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 234.30 | 232.36 | 233.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:15:00 | 235.59 | 232.36 | 233.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 237.00 | 233.29 | 233.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:45:00 | 237.11 | 233.29 | 233.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 221 — BUY (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 10:15:00 | 237.89 | 234.21 | 233.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 11:15:00 | 238.31 | 235.03 | 234.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 13:15:00 | 238.40 | 238.64 | 237.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 14:00:00 | 238.40 | 238.64 | 237.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 263.00 | 264.46 | 262.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:45:00 | 262.45 | 264.46 | 262.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 261.36 | 263.84 | 262.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 12:00:00 | 261.36 | 263.84 | 262.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 262.22 | 263.52 | 262.38 | EMA400 retest candle locked (from upside) |

### Cycle 222 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 259.18 | 261.66 | 261.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 257.54 | 260.84 | 261.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 259.71 | 258.31 | 259.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 259.71 | 258.31 | 259.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 259.71 | 258.31 | 259.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 259.71 | 258.31 | 259.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 257.93 | 258.23 | 259.54 | EMA400 retest candle locked (from downside) |

### Cycle 223 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 264.62 | 260.90 | 260.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 11:15:00 | 265.15 | 262.10 | 261.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 15:15:00 | 262.68 | 263.04 | 261.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 09:15:00 | 263.56 | 263.04 | 261.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 263.96 | 263.23 | 262.13 | EMA400 retest candle locked (from upside) |

### Cycle 224 — SELL (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 15:15:00 | 258.93 | 261.65 | 261.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 13:15:00 | 257.91 | 259.96 | 260.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 259.19 | 259.12 | 260.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 259.19 | 259.12 | 260.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 259.19 | 259.12 | 260.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 259.11 | 259.12 | 260.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 257.39 | 258.43 | 259.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:30:00 | 258.90 | 258.43 | 259.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 253.99 | 252.30 | 254.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 253.99 | 252.30 | 254.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 253.38 | 252.65 | 254.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:45:00 | 253.51 | 252.65 | 254.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 254.87 | 253.24 | 254.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:00:00 | 254.87 | 253.24 | 254.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 255.16 | 253.63 | 254.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:45:00 | 254.87 | 253.63 | 254.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 254.92 | 253.89 | 254.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 255.66 | 253.89 | 254.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 225 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 256.35 | 254.63 | 254.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 15:15:00 | 258.77 | 256.30 | 255.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 14:15:00 | 257.61 | 257.79 | 256.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-29 15:00:00 | 257.61 | 257.79 | 256.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 255.71 | 257.57 | 256.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 255.71 | 257.57 | 256.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 256.80 | 257.41 | 256.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:30:00 | 254.24 | 257.41 | 256.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 256.00 | 257.13 | 256.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 12:00:00 | 256.00 | 257.13 | 256.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 257.60 | 257.22 | 256.86 | EMA400 retest candle locked (from upside) |

### Cycle 226 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 253.93 | 256.34 | 256.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 12:15:00 | 253.00 | 255.21 | 255.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 12:15:00 | 250.51 | 250.21 | 252.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 13:00:00 | 250.51 | 250.21 | 252.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 251.31 | 250.16 | 251.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 13:15:00 | 251.91 | 250.16 | 251.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 250.87 | 250.30 | 251.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 14:15:00 | 250.68 | 250.30 | 251.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 15:00:00 | 250.28 | 250.30 | 251.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 09:15:00 | 249.66 | 250.42 | 251.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 09:15:00 | 256.68 | 248.33 | 248.52 | SL hit (close>static) qty=1.00 sl=252.68 alert=retest2 |

### Cycle 227 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 260.44 | 250.76 | 249.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 09:15:00 | 263.05 | 260.04 | 259.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 15:15:00 | 268.50 | 268.64 | 266.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-19 09:15:00 | 268.86 | 268.64 | 266.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 267.07 | 268.08 | 266.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:45:00 | 267.76 | 268.08 | 266.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 268.45 | 268.54 | 267.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 12:00:00 | 269.25 | 268.68 | 267.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 12:45:00 | 269.00 | 268.88 | 267.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 270.04 | 269.03 | 268.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 12:15:00 | 269.47 | 269.29 | 268.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 12:15:00 | 269.33 | 269.30 | 268.62 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-24 11:15:00 | 266.94 | 268.52 | 268.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 228 — SELL (started 2025-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 11:15:00 | 266.94 | 268.52 | 268.53 | EMA200 below EMA400 |

### Cycle 229 — BUY (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 12:15:00 | 269.07 | 268.63 | 268.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 13:15:00 | 270.83 | 269.07 | 268.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 11:15:00 | 270.00 | 270.22 | 269.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-25 11:45:00 | 270.10 | 270.22 | 269.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 269.79 | 270.14 | 269.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 10:15:00 | 271.30 | 270.30 | 269.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 12:15:00 | 265.06 | 269.16 | 269.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 230 — SELL (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 12:15:00 | 265.06 | 269.16 | 269.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-26 14:15:00 | 264.51 | 267.75 | 268.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 09:15:00 | 268.17 | 266.01 | 266.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 268.17 | 266.01 | 266.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 268.17 | 266.01 | 266.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:30:00 | 268.10 | 266.01 | 266.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 267.99 | 266.41 | 266.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:30:00 | 268.19 | 266.41 | 266.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 267.42 | 267.29 | 267.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 13:45:00 | 267.50 | 267.29 | 267.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 231 — BUY (started 2025-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 14:15:00 | 267.46 | 267.32 | 267.31 | EMA200 above EMA400 |

### Cycle 232 — SELL (started 2025-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 15:15:00 | 267.02 | 267.26 | 267.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 09:15:00 | 264.20 | 266.65 | 267.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 14:15:00 | 265.55 | 265.13 | 265.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 15:00:00 | 265.55 | 265.13 | 265.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 264.60 | 265.03 | 265.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 263.90 | 265.03 | 265.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 12:15:00 | 250.70 | 253.64 | 255.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 15:15:00 | 253.65 | 253.05 | 254.65 | SL hit (close>ema200) qty=0.50 sl=253.05 alert=retest2 |

### Cycle 233 — BUY (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 11:15:00 | 252.80 | 250.74 | 250.64 | EMA200 above EMA400 |

### Cycle 234 — SELL (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 10:15:00 | 247.35 | 250.42 | 250.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 246.05 | 248.68 | 249.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 14:15:00 | 248.00 | 247.98 | 249.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-16 15:00:00 | 248.00 | 247.98 | 249.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 242.25 | 244.89 | 246.56 | EMA400 retest candle locked (from downside) |

### Cycle 235 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 249.20 | 246.12 | 245.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 253.65 | 248.19 | 246.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 13:15:00 | 258.05 | 258.11 | 255.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 13:45:00 | 258.00 | 258.11 | 255.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 258.65 | 259.08 | 257.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 09:30:00 | 260.40 | 259.12 | 258.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:00:00 | 260.00 | 259.12 | 258.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 11:15:00 | 260.15 | 259.22 | 258.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 14:15:00 | 266.00 | 267.36 | 267.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 236 — SELL (started 2026-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 14:15:00 | 266.00 | 267.36 | 267.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 264.05 | 266.53 | 267.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 252.70 | 252.64 | 255.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:45:00 | 253.15 | 252.64 | 255.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 254.10 | 252.04 | 253.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:45:00 | 253.95 | 252.04 | 253.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 255.95 | 252.82 | 254.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 255.60 | 252.82 | 254.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 252.25 | 252.71 | 253.85 | EMA400 retest candle locked (from downside) |

### Cycle 237 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 254.80 | 253.88 | 253.88 | EMA200 above EMA400 |

### Cycle 238 — SELL (started 2026-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 13:15:00 | 252.80 | 253.80 | 253.86 | EMA200 below EMA400 |

### Cycle 239 — BUY (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 09:15:00 | 255.90 | 253.95 | 253.89 | EMA200 above EMA400 |

### Cycle 240 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 252.00 | 253.84 | 253.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 13:15:00 | 251.35 | 253.34 | 253.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 244.75 | 242.71 | 245.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 244.75 | 242.71 | 245.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 244.75 | 242.71 | 245.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 246.65 | 242.71 | 245.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 241.80 | 242.53 | 244.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 240.90 | 242.53 | 244.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 11:15:00 | 237.05 | 236.73 | 236.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 241 — BUY (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 11:15:00 | 237.05 | 236.73 | 236.70 | EMA200 above EMA400 |

### Cycle 242 — SELL (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 12:15:00 | 236.00 | 236.59 | 236.63 | EMA200 below EMA400 |

### Cycle 243 — BUY (started 2026-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 14:15:00 | 237.25 | 236.75 | 236.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 239.71 | 237.34 | 236.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 237.57 | 239.27 | 238.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 15:15:00 | 237.57 | 239.27 | 238.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 237.57 | 239.27 | 238.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 234.78 | 239.27 | 238.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 236.86 | 238.79 | 238.25 | EMA400 retest candle locked (from upside) |

### Cycle 244 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 236.26 | 237.91 | 237.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 13:15:00 | 235.35 | 237.09 | 237.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 238.05 | 237.29 | 237.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 238.05 | 237.29 | 237.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 238.05 | 237.29 | 237.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 238.05 | 237.29 | 237.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 235.79 | 236.99 | 237.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 244.50 | 236.99 | 237.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 245 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 244.58 | 238.50 | 238.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 10:15:00 | 247.15 | 240.23 | 238.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 10:15:00 | 279.81 | 280.16 | 275.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 11:00:00 | 279.81 | 280.16 | 275.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 276.95 | 278.79 | 276.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:45:00 | 276.88 | 278.79 | 276.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 277.72 | 278.58 | 276.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 11:00:00 | 278.37 | 277.99 | 276.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 09:30:00 | 278.53 | 278.92 | 277.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 274.12 | 277.96 | 277.54 | SL hit (close<static) qty=1.00 sl=276.22 alert=retest2 |

### Cycle 246 — SELL (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 12:15:00 | 274.40 | 276.86 | 277.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 271.16 | 275.21 | 276.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 274.52 | 274.40 | 275.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 10:00:00 | 274.52 | 274.40 | 275.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 276.49 | 274.82 | 275.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 11:00:00 | 276.49 | 274.82 | 275.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 275.17 | 274.89 | 275.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 12:15:00 | 274.68 | 274.89 | 275.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 274.22 | 274.74 | 275.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 10:15:00 | 260.95 | 264.48 | 266.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 10:15:00 | 260.51 | 264.48 | 266.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-24 15:15:00 | 262.10 | 262.01 | 264.15 | SL hit (close>ema200) qty=0.50 sl=262.01 alert=retest2 |

### Cycle 247 — BUY (started 2026-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 13:15:00 | 266.03 | 265.18 | 265.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 268.33 | 266.25 | 265.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 265.79 | 266.65 | 265.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 11:15:00 | 265.79 | 266.65 | 265.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 265.79 | 266.65 | 265.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:00:00 | 265.79 | 266.65 | 265.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 264.99 | 266.32 | 265.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:45:00 | 264.76 | 266.32 | 265.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 266.03 | 266.26 | 265.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 14:15:00 | 266.34 | 266.26 | 265.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 263.90 | 266.23 | 266.02 | SL hit (close<static) qty=1.00 sl=264.06 alert=retest2 |

### Cycle 248 — SELL (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 14:15:00 | 265.10 | 265.93 | 265.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 261.75 | 264.93 | 265.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 15:15:00 | 261.00 | 260.41 | 262.48 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:15:00 | 253.50 | 260.41 | 262.48 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 15:15:00 | 256.65 | 255.60 | 258.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:15:00 | 258.70 | 255.60 | 258.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 256.10 | 255.70 | 258.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 10:15:00 | 255.90 | 255.70 | 258.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 11:00:00 | 255.60 | 255.68 | 257.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:45:00 | 256.00 | 255.69 | 257.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 260.45 | 256.64 | 257.62 | SL hit (close>ema400) qty=1.00 sl=257.62 alert=retest1 |

### Cycle 249 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 261.60 | 258.27 | 258.23 | EMA200 above EMA400 |

### Cycle 250 — SELL (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 12:15:00 | 256.45 | 257.93 | 258.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 14:15:00 | 254.85 | 257.05 | 257.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 253.05 | 250.56 | 252.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 253.05 | 250.56 | 252.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 253.05 | 250.56 | 252.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:00:00 | 253.05 | 250.56 | 252.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 255.75 | 251.59 | 253.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 11:00:00 | 255.75 | 251.59 | 253.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 255.30 | 252.34 | 253.26 | EMA400 retest candle locked (from downside) |

### Cycle 251 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 257.35 | 253.79 | 253.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 258.70 | 255.59 | 254.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 11:15:00 | 255.00 | 255.61 | 254.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 12:00:00 | 255.00 | 255.61 | 254.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 253.95 | 255.28 | 254.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 253.95 | 255.28 | 254.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 253.95 | 255.01 | 254.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:00:00 | 253.95 | 255.01 | 254.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 252 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 251.70 | 254.35 | 254.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 15:15:00 | 251.00 | 253.68 | 254.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 10:15:00 | 237.10 | 237.00 | 239.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:45:00 | 237.10 | 237.00 | 239.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 240.45 | 238.02 | 239.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 240.45 | 238.02 | 239.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 240.20 | 238.45 | 239.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:30:00 | 240.45 | 238.45 | 239.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 238.15 | 238.39 | 239.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:15:00 | 242.30 | 238.39 | 239.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 242.70 | 239.25 | 239.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 242.80 | 239.25 | 239.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 241.85 | 239.77 | 239.97 | EMA400 retest candle locked (from downside) |

### Cycle 253 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 242.30 | 240.28 | 240.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 242.80 | 240.78 | 240.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 238.30 | 241.12 | 240.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 238.30 | 241.12 | 240.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 238.30 | 241.12 | 240.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 10:45:00 | 240.30 | 240.88 | 240.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 11:15:00 | 239.50 | 240.61 | 240.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 254 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 239.50 | 240.61 | 240.61 | EMA200 below EMA400 |

### Cycle 255 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 242.65 | 240.68 | 240.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 11:15:00 | 243.20 | 241.19 | 240.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-20 14:15:00 | 240.80 | 241.48 | 241.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 14:15:00 | 240.80 | 241.48 | 241.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 240.80 | 241.48 | 241.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 240.80 | 241.48 | 241.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 239.80 | 241.15 | 240.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:15:00 | 234.20 | 241.15 | 240.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 256 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 234.70 | 239.86 | 240.38 | EMA200 below EMA400 |

### Cycle 257 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 240.00 | 238.05 | 238.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 247.40 | 240.23 | 239.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 243.50 | 243.68 | 241.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:15:00 | 243.00 | 243.68 | 241.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 240.95 | 243.13 | 241.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 240.95 | 243.13 | 241.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 239.50 | 242.41 | 241.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 239.50 | 242.41 | 241.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 238.55 | 241.64 | 241.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:15:00 | 238.90 | 241.64 | 241.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 258 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 239.75 | 240.88 | 240.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 233.70 | 238.92 | 239.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 15:15:00 | 236.75 | 236.09 | 237.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-01 09:15:00 | 242.62 | 236.09 | 237.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 240.19 | 236.91 | 237.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:45:00 | 237.88 | 236.93 | 237.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 14:15:00 | 240.08 | 238.60 | 238.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 259 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 240.08 | 238.60 | 238.44 | EMA200 above EMA400 |

### Cycle 260 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 235.36 | 238.06 | 238.23 | EMA200 below EMA400 |

### Cycle 261 — BUY (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 11:15:00 | 238.81 | 238.38 | 238.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 246.66 | 240.33 | 239.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 247.35 | 249.61 | 246.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 247.35 | 249.61 | 246.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 247.35 | 249.61 | 246.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:30:00 | 253.46 | 248.96 | 247.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 10:00:00 | 253.59 | 248.96 | 247.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 13:00:00 | 253.40 | 251.35 | 248.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 10:15:00 | 263.67 | 264.81 | 264.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 262 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 263.67 | 264.81 | 264.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 11:15:00 | 261.69 | 264.19 | 264.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 09:15:00 | 263.51 | 262.17 | 263.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 09:15:00 | 263.51 | 262.17 | 263.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 263.51 | 262.17 | 263.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:00:00 | 263.51 | 262.17 | 263.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 263.70 | 262.48 | 263.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:00:00 | 263.70 | 262.48 | 263.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 262.88 | 262.56 | 263.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 12:45:00 | 261.74 | 262.39 | 263.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 14:15:00 | 261.92 | 262.49 | 263.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 09:15:00 | 261.09 | 262.36 | 262.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 11:00:00 | 261.71 | 262.39 | 262.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 262.64 | 262.10 | 262.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:45:00 | 263.67 | 262.10 | 262.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 262.90 | 262.26 | 262.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 264.89 | 262.26 | 262.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 264.80 | 262.77 | 262.77 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 264.80 | 262.77 | 262.77 | SL hit (close>static) qty=1.00 sl=264.19 alert=retest2 |

### Cycle 263 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 267.12 | 263.64 | 263.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 11:15:00 | 269.70 | 264.85 | 263.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 268.50 | 268.73 | 267.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 13:30:00 | 268.55 | 268.73 | 267.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 267.50 | 268.85 | 267.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:45:00 | 267.43 | 268.85 | 267.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 268.33 | 268.74 | 267.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 12:00:00 | 270.00 | 268.99 | 267.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 13:15:00 | 267.25 | 268.73 | 267.93 | SL hit (close<static) qty=1.00 sl=267.40 alert=retest2 |

### Cycle 264 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 264.68 | 267.01 | 267.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 12:15:00 | 263.05 | 265.66 | 266.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 11:15:00 | 266.80 | 265.43 | 265.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 11:15:00 | 266.80 | 265.43 | 265.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 266.80 | 265.43 | 265.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:00:00 | 266.80 | 265.43 | 265.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 265 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 268.65 | 266.07 | 266.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 14:15:00 | 269.50 | 266.86 | 266.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 10:15:00 | 271.25 | 271.62 | 269.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 11:00:00 | 271.25 | 271.62 | 269.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 12:15:00 | 271.25 | 271.52 | 270.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 12:45:00 | 270.10 | 271.52 | 270.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 14:15:00 | 269.75 | 271.20 | 270.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 15:00:00 | 269.75 | 271.20 | 270.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 15:15:00 | 269.70 | 270.90 | 270.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 271.45 | 270.90 | 270.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:45:00 | 272.40 | 271.06 | 270.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-22 11:15:00 | 127.50 | 2023-05-23 09:15:00 | 126.25 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2023-05-22 14:15:00 | 127.40 | 2023-05-23 09:15:00 | 126.25 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2023-06-07 11:00:00 | 136.25 | 2023-06-15 14:15:00 | 138.05 | STOP_HIT | 1.00 | 1.32% |
| BUY | retest2 | 2023-06-07 12:15:00 | 135.85 | 2023-06-15 14:15:00 | 138.05 | STOP_HIT | 1.00 | 1.62% |
| BUY | retest2 | 2023-06-07 13:30:00 | 135.45 | 2023-06-15 14:15:00 | 138.05 | STOP_HIT | 1.00 | 1.92% |
| BUY | retest2 | 2023-06-07 14:15:00 | 135.45 | 2023-06-15 14:15:00 | 138.05 | STOP_HIT | 1.00 | 1.92% |
| BUY | retest2 | 2023-06-08 10:15:00 | 136.20 | 2023-06-15 14:15:00 | 138.05 | STOP_HIT | 1.00 | 1.36% |
| BUY | retest2 | 2023-06-21 09:15:00 | 150.25 | 2023-06-27 09:15:00 | 147.95 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2023-06-28 12:45:00 | 148.20 | 2023-07-03 10:15:00 | 149.05 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2023-06-28 14:45:00 | 147.80 | 2023-07-03 10:15:00 | 149.05 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2023-06-30 15:15:00 | 148.20 | 2023-07-03 10:15:00 | 149.05 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2023-07-07 15:00:00 | 141.70 | 2023-07-11 13:15:00 | 145.65 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2023-07-11 10:45:00 | 142.30 | 2023-07-11 13:15:00 | 145.65 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2023-07-13 09:15:00 | 144.85 | 2023-07-13 15:15:00 | 143.10 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2023-07-17 14:45:00 | 145.75 | 2023-07-18 10:15:00 | 144.45 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2023-07-21 10:45:00 | 145.40 | 2023-07-24 14:15:00 | 145.00 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2023-07-21 11:15:00 | 145.55 | 2023-07-24 14:15:00 | 145.00 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2023-08-21 11:00:00 | 132.00 | 2023-08-22 12:15:00 | 135.15 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2023-08-21 13:30:00 | 132.60 | 2023-08-22 12:15:00 | 135.15 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2023-09-13 09:15:00 | 147.75 | 2023-09-20 09:15:00 | 146.30 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2023-10-03 10:30:00 | 152.00 | 2023-10-04 12:15:00 | 147.55 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2023-10-03 11:15:00 | 151.75 | 2023-10-04 12:15:00 | 147.55 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2023-10-03 11:45:00 | 151.75 | 2023-10-04 12:15:00 | 147.55 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2023-10-03 14:45:00 | 151.90 | 2023-10-04 12:15:00 | 147.55 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2023-10-19 09:15:00 | 144.55 | 2023-10-25 09:15:00 | 137.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-19 09:15:00 | 144.55 | 2023-10-25 13:15:00 | 139.95 | STOP_HIT | 0.50 | 3.18% |
| BUY | retest2 | 2023-11-17 09:15:00 | 154.85 | 2023-11-20 09:15:00 | 170.34 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-12-20 11:30:00 | 173.85 | 2023-12-20 14:15:00 | 165.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-20 11:30:00 | 173.85 | 2023-12-22 09:15:00 | 170.30 | STOP_HIT | 0.50 | 2.04% |
| BUY | retest2 | 2023-12-28 13:30:00 | 174.00 | 2024-01-01 10:15:00 | 171.70 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2023-12-29 11:00:00 | 173.65 | 2024-01-01 10:15:00 | 171.70 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2023-12-29 14:30:00 | 173.70 | 2024-01-01 10:15:00 | 171.70 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-01-12 12:15:00 | 190.35 | 2024-01-15 09:15:00 | 184.15 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2024-01-25 11:15:00 | 162.05 | 2024-01-30 10:15:00 | 165.40 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2024-01-30 10:30:00 | 162.05 | 2024-01-30 11:15:00 | 166.30 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2024-02-06 14:00:00 | 160.05 | 2024-02-07 09:15:00 | 162.10 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-02-07 11:45:00 | 159.95 | 2024-02-08 11:15:00 | 151.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-07 12:45:00 | 159.80 | 2024-02-08 11:15:00 | 151.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-07 11:45:00 | 159.95 | 2024-02-09 15:15:00 | 151.40 | STOP_HIT | 0.50 | 5.35% |
| SELL | retest2 | 2024-02-07 12:45:00 | 159.80 | 2024-02-09 15:15:00 | 151.40 | STOP_HIT | 0.50 | 5.26% |
| BUY | retest2 | 2024-02-15 14:30:00 | 153.10 | 2024-02-21 10:15:00 | 151.10 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-02-19 09:15:00 | 152.95 | 2024-02-21 10:15:00 | 151.10 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-02-19 10:15:00 | 153.00 | 2024-02-21 10:15:00 | 151.10 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-02-19 11:15:00 | 153.00 | 2024-02-21 10:15:00 | 151.10 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-03-01 09:15:00 | 158.20 | 2024-03-06 10:15:00 | 156.10 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-03-01 14:30:00 | 157.95 | 2024-03-06 10:15:00 | 156.10 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2024-03-02 09:15:00 | 159.45 | 2024-03-06 10:15:00 | 156.10 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2024-03-14 12:45:00 | 151.95 | 2024-03-19 10:15:00 | 152.75 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2024-03-14 14:00:00 | 151.25 | 2024-03-19 10:15:00 | 152.75 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-03-14 15:00:00 | 150.95 | 2024-03-19 10:15:00 | 152.75 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-03-20 13:30:00 | 152.50 | 2024-04-03 09:15:00 | 162.00 | STOP_HIT | 1.00 | 6.23% |
| BUY | retest2 | 2024-03-20 14:45:00 | 152.50 | 2024-04-03 09:15:00 | 162.00 | STOP_HIT | 1.00 | 6.23% |
| BUY | retest2 | 2024-04-04 15:15:00 | 165.00 | 2024-04-09 09:15:00 | 181.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-30 11:30:00 | 179.10 | 2024-05-02 11:15:00 | 174.75 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2024-05-03 10:15:00 | 176.45 | 2024-05-06 15:15:00 | 167.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-03 10:15:00 | 176.45 | 2024-05-07 09:15:00 | 176.95 | STOP_HIT | 0.50 | -0.28% |
| SELL | retest2 | 2024-05-07 09:45:00 | 175.30 | 2024-05-09 10:15:00 | 167.63 | PARTIAL | 0.50 | 4.38% |
| SELL | retest2 | 2024-05-07 10:30:00 | 176.45 | 2024-05-09 14:15:00 | 166.53 | PARTIAL | 0.50 | 5.62% |
| SELL | retest2 | 2024-05-07 11:15:00 | 174.40 | 2024-05-10 09:15:00 | 165.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-07 09:45:00 | 175.30 | 2024-05-10 10:15:00 | 168.90 | STOP_HIT | 0.50 | 3.65% |
| SELL | retest2 | 2024-05-07 10:30:00 | 176.45 | 2024-05-10 10:15:00 | 168.90 | STOP_HIT | 0.50 | 4.28% |
| SELL | retest2 | 2024-05-07 11:15:00 | 174.40 | 2024-05-10 10:15:00 | 168.90 | STOP_HIT | 0.50 | 3.15% |
| SELL | retest2 | 2024-05-07 13:30:00 | 171.15 | 2024-05-15 09:15:00 | 172.95 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-05-08 09:15:00 | 170.25 | 2024-05-15 09:15:00 | 172.95 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-06-04 09:15:00 | 158.35 | 2024-06-04 11:15:00 | 150.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 10:15:00 | 161.25 | 2024-06-04 11:15:00 | 153.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 158.35 | 2024-06-05 10:15:00 | 159.40 | STOP_HIT | 0.50 | -0.66% |
| SELL | retest2 | 2024-06-04 10:15:00 | 161.25 | 2024-06-05 10:15:00 | 159.40 | STOP_HIT | 0.50 | 1.15% |
| BUY | retest2 | 2024-06-11 09:15:00 | 173.00 | 2024-06-13 09:15:00 | 167.69 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2024-06-12 14:30:00 | 169.49 | 2024-06-13 09:15:00 | 167.69 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-06-12 15:00:00 | 170.25 | 2024-06-13 09:15:00 | 167.69 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-06-13 09:15:00 | 169.90 | 2024-06-13 09:15:00 | 167.69 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-06-19 13:15:00 | 173.57 | 2024-06-26 11:15:00 | 173.87 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2024-06-20 14:45:00 | 173.25 | 2024-06-26 11:15:00 | 173.87 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2024-06-21 09:30:00 | 174.40 | 2024-06-26 11:15:00 | 173.87 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2024-06-21 11:30:00 | 173.54 | 2024-06-26 11:15:00 | 173.87 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2024-06-21 14:30:00 | 174.20 | 2024-06-26 11:15:00 | 173.87 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2024-06-24 09:45:00 | 174.85 | 2024-06-26 11:15:00 | 173.87 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2024-07-04 14:30:00 | 172.53 | 2024-07-05 11:15:00 | 177.20 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2024-07-05 09:15:00 | 172.32 | 2024-07-05 11:15:00 | 177.20 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2024-07-12 09:15:00 | 179.02 | 2024-07-19 09:15:00 | 177.41 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-07-23 10:15:00 | 174.80 | 2024-07-24 11:15:00 | 178.50 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2024-07-23 12:15:00 | 172.56 | 2024-07-24 11:15:00 | 178.50 | STOP_HIT | 1.00 | -3.44% |
| BUY | retest2 | 2024-07-26 11:30:00 | 181.70 | 2024-07-30 14:15:00 | 199.87 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-13 11:15:00 | 191.33 | 2024-08-13 11:15:00 | 189.92 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-08-14 15:00:00 | 188.90 | 2024-08-16 14:15:00 | 193.60 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2024-08-16 09:15:00 | 187.25 | 2024-08-16 14:15:00 | 193.60 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest1 | 2024-09-03 09:30:00 | 206.40 | 2024-09-04 14:15:00 | 209.70 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-09-04 09:15:00 | 206.75 | 2024-09-04 15:15:00 | 210.15 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2024-09-06 13:30:00 | 215.36 | 2024-09-11 13:15:00 | 213.20 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-09-06 15:00:00 | 216.95 | 2024-09-11 13:15:00 | 213.20 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-09-09 09:30:00 | 218.05 | 2024-09-11 13:15:00 | 213.20 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2024-09-11 10:15:00 | 215.31 | 2024-09-11 13:15:00 | 213.20 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-09-16 10:45:00 | 204.02 | 2024-09-20 11:15:00 | 206.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-09-16 13:15:00 | 203.98 | 2024-09-20 11:15:00 | 206.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-09-16 13:45:00 | 204.00 | 2024-09-20 11:15:00 | 206.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-09-16 14:15:00 | 204.03 | 2024-09-20 11:15:00 | 206.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-09-19 11:15:00 | 197.50 | 2024-09-20 11:15:00 | 206.00 | STOP_HIT | 1.00 | -4.30% |
| SELL | retest2 | 2024-09-24 09:15:00 | 200.46 | 2024-09-26 13:15:00 | 201.84 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-10-03 09:15:00 | 195.20 | 2024-10-08 15:15:00 | 194.16 | STOP_HIT | 1.00 | 0.53% |
| SELL | retest2 | 2024-10-14 13:15:00 | 192.29 | 2024-10-17 09:15:00 | 182.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-14 14:45:00 | 192.30 | 2024-10-17 09:15:00 | 182.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 09:30:00 | 191.90 | 2024-10-17 09:15:00 | 182.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-14 13:15:00 | 192.29 | 2024-10-17 14:15:00 | 185.20 | STOP_HIT | 0.50 | 3.69% |
| SELL | retest2 | 2024-10-14 14:45:00 | 192.30 | 2024-10-17 14:15:00 | 185.20 | STOP_HIT | 0.50 | 3.69% |
| SELL | retest2 | 2024-10-15 09:30:00 | 191.90 | 2024-10-17 14:15:00 | 185.20 | STOP_HIT | 0.50 | 3.49% |
| BUY | retest2 | 2024-10-31 15:00:00 | 182.00 | 2024-11-11 12:15:00 | 183.09 | STOP_HIT | 1.00 | 0.60% |
| BUY | retest2 | 2024-11-04 10:15:00 | 181.33 | 2024-11-11 12:15:00 | 183.09 | STOP_HIT | 1.00 | 0.97% |
| BUY | retest2 | 2024-11-04 11:15:00 | 181.71 | 2024-11-11 12:15:00 | 183.09 | STOP_HIT | 1.00 | 0.76% |
| BUY | retest2 | 2024-11-04 14:30:00 | 181.11 | 2024-11-11 12:15:00 | 183.09 | STOP_HIT | 1.00 | 1.09% |
| BUY | retest2 | 2024-11-05 09:15:00 | 183.23 | 2024-11-11 12:15:00 | 183.09 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2024-11-05 09:45:00 | 182.85 | 2024-11-11 12:15:00 | 183.09 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2024-11-05 10:45:00 | 183.00 | 2024-11-11 12:15:00 | 183.09 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2024-11-05 15:00:00 | 183.43 | 2024-11-11 12:15:00 | 183.09 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2024-11-07 09:15:00 | 182.55 | 2024-11-11 12:15:00 | 183.09 | STOP_HIT | 1.00 | 0.30% |
| BUY | retest2 | 2024-12-03 12:30:00 | 172.48 | 2024-12-04 11:15:00 | 170.45 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-12-03 15:15:00 | 172.50 | 2024-12-04 11:15:00 | 170.45 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-12-09 10:45:00 | 164.98 | 2024-12-10 09:15:00 | 171.44 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest2 | 2024-12-12 09:15:00 | 171.00 | 2024-12-12 14:15:00 | 169.61 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-12-12 09:45:00 | 170.76 | 2024-12-12 14:15:00 | 169.61 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-12-12 10:15:00 | 171.05 | 2024-12-12 14:15:00 | 169.61 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-12-12 11:15:00 | 171.04 | 2024-12-12 14:15:00 | 169.61 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-12-30 14:15:00 | 160.51 | 2024-12-31 10:15:00 | 162.05 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-12-31 10:00:00 | 160.28 | 2024-12-31 10:15:00 | 162.05 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-01-02 13:30:00 | 164.28 | 2025-01-08 14:15:00 | 169.73 | STOP_HIT | 1.00 | 3.32% |
| SELL | retest2 | 2025-01-13 09:15:00 | 166.45 | 2025-01-15 11:15:00 | 171.83 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest1 | 2025-01-17 10:15:00 | 174.98 | 2025-01-17 15:15:00 | 172.12 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest1 | 2025-01-17 12:00:00 | 174.56 | 2025-01-17 15:15:00 | 172.12 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-02-11 12:00:00 | 167.80 | 2025-02-12 14:15:00 | 172.04 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2025-02-11 13:45:00 | 167.99 | 2025-02-12 14:15:00 | 172.04 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-02-18 13:00:00 | 170.11 | 2025-02-18 15:15:00 | 170.10 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-02-25 11:15:00 | 164.72 | 2025-02-28 09:15:00 | 156.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 11:45:00 | 164.55 | 2025-02-28 09:15:00 | 156.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 13:30:00 | 164.72 | 2025-02-28 09:15:00 | 156.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 11:15:00 | 164.72 | 2025-02-28 15:15:00 | 160.50 | STOP_HIT | 0.50 | 2.56% |
| SELL | retest2 | 2025-02-25 11:45:00 | 164.55 | 2025-02-28 15:15:00 | 160.50 | STOP_HIT | 0.50 | 2.46% |
| SELL | retest2 | 2025-02-25 13:30:00 | 164.72 | 2025-02-28 15:15:00 | 160.50 | STOP_HIT | 0.50 | 2.56% |
| SELL | retest2 | 2025-02-25 14:15:00 | 164.14 | 2025-03-04 09:15:00 | 155.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 14:15:00 | 164.14 | 2025-03-04 09:15:00 | 159.31 | STOP_HIT | 0.50 | 2.94% |
| SELL | retest2 | 2025-02-27 13:45:00 | 161.43 | 2025-03-04 13:15:00 | 161.67 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2025-03-18 09:15:00 | 165.55 | 2025-03-19 11:15:00 | 164.19 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-03-18 10:15:00 | 165.10 | 2025-03-19 11:15:00 | 164.19 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-03-18 11:00:00 | 165.49 | 2025-03-19 11:15:00 | 164.19 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest1 | 2025-04-07 09:15:00 | 172.67 | 2025-04-08 09:15:00 | 176.05 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-05-02 09:30:00 | 195.31 | 2025-05-02 15:15:00 | 193.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-05-02 13:15:00 | 194.81 | 2025-05-02 15:15:00 | 193.00 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-05-02 14:00:00 | 194.80 | 2025-05-02 15:15:00 | 193.00 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-05-02 14:30:00 | 194.80 | 2025-05-02 15:15:00 | 193.00 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-05-13 14:15:00 | 198.58 | 2025-05-19 15:15:00 | 197.06 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-05-14 09:15:00 | 200.79 | 2025-05-19 15:15:00 | 197.06 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-05-14 11:00:00 | 198.00 | 2025-05-19 15:15:00 | 197.06 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-05-14 13:00:00 | 198.01 | 2025-05-19 15:15:00 | 197.06 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-05-15 10:15:00 | 198.81 | 2025-05-19 15:15:00 | 197.06 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-05-16 09:15:00 | 199.84 | 2025-05-19 15:15:00 | 197.06 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-05-23 09:45:00 | 204.00 | 2025-05-26 14:15:00 | 200.17 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-05-23 11:30:00 | 203.96 | 2025-05-26 14:15:00 | 200.17 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-05-26 09:15:00 | 204.36 | 2025-05-26 14:15:00 | 200.17 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-06-05 10:30:00 | 195.20 | 2025-06-05 12:15:00 | 199.17 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-06-11 09:15:00 | 199.98 | 2025-06-11 13:15:00 | 198.27 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-06-11 11:00:00 | 199.96 | 2025-06-11 13:15:00 | 198.27 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-06-16 15:15:00 | 194.57 | 2025-06-18 10:15:00 | 196.89 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-06-19 09:15:00 | 196.83 | 2025-06-19 11:15:00 | 194.73 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-06-24 11:15:00 | 199.47 | 2025-07-02 09:15:00 | 204.95 | STOP_HIT | 1.00 | 2.75% |
| BUY | retest2 | 2025-06-24 12:30:00 | 199.34 | 2025-07-02 09:15:00 | 204.95 | STOP_HIT | 1.00 | 2.81% |
| BUY | retest2 | 2025-06-25 09:15:00 | 201.03 | 2025-07-02 09:15:00 | 204.95 | STOP_HIT | 1.00 | 1.95% |
| SELL | retest2 | 2025-07-07 11:30:00 | 201.61 | 2025-07-09 09:15:00 | 205.85 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-07-08 11:00:00 | 201.81 | 2025-07-09 09:15:00 | 205.85 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-07-28 11:30:00 | 211.85 | 2025-07-30 14:15:00 | 214.00 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-07-28 12:15:00 | 212.20 | 2025-07-30 14:15:00 | 214.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-07-30 12:00:00 | 212.07 | 2025-07-30 14:15:00 | 214.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-07-30 12:45:00 | 212.07 | 2025-07-30 14:15:00 | 214.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-08-08 09:15:00 | 206.89 | 2025-08-13 09:15:00 | 211.10 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-08-19 10:45:00 | 221.36 | 2025-08-29 12:15:00 | 228.44 | STOP_HIT | 1.00 | 3.20% |
| BUY | retest2 | 2025-09-03 09:15:00 | 233.78 | 2025-09-09 15:15:00 | 241.00 | STOP_HIT | 1.00 | 3.09% |
| BUY | retest2 | 2025-09-17 15:15:00 | 243.25 | 2025-09-22 15:15:00 | 242.59 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-09-18 13:15:00 | 242.91 | 2025-09-22 15:15:00 | 242.59 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-09-22 10:15:00 | 243.45 | 2025-09-22 15:15:00 | 242.59 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-11-04 14:15:00 | 250.68 | 2025-11-10 09:15:00 | 256.68 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2025-11-04 15:00:00 | 250.28 | 2025-11-10 09:15:00 | 256.68 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-11-06 09:15:00 | 249.66 | 2025-11-10 09:15:00 | 256.68 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2025-11-20 12:00:00 | 269.25 | 2025-11-24 11:15:00 | 266.94 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-11-20 12:45:00 | 269.00 | 2025-11-24 11:15:00 | 266.94 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-11-21 09:15:00 | 270.04 | 2025-11-24 11:15:00 | 266.94 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-11-21 12:15:00 | 269.47 | 2025-11-24 11:15:00 | 266.94 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-11-26 10:15:00 | 271.30 | 2025-11-26 12:15:00 | 265.06 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-12-02 09:15:00 | 263.90 | 2025-12-08 12:15:00 | 250.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 09:15:00 | 263.90 | 2025-12-08 15:15:00 | 253.65 | STOP_HIT | 0.50 | 3.88% |
| BUY | retest2 | 2025-12-30 09:30:00 | 260.40 | 2026-01-07 14:15:00 | 266.00 | STOP_HIT | 1.00 | 2.15% |
| BUY | retest2 | 2025-12-30 10:00:00 | 260.00 | 2026-01-07 14:15:00 | 266.00 | STOP_HIT | 1.00 | 2.31% |
| BUY | retest2 | 2025-12-30 11:15:00 | 260.15 | 2026-01-07 14:15:00 | 266.00 | STOP_HIT | 1.00 | 2.25% |
| SELL | retest2 | 2026-01-22 11:15:00 | 240.90 | 2026-01-30 11:15:00 | 237.05 | STOP_HIT | 1.00 | 1.60% |
| BUY | retest2 | 2026-02-12 11:00:00 | 278.37 | 2026-02-13 10:15:00 | 274.12 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2026-02-13 09:30:00 | 278.53 | 2026-02-13 10:15:00 | 274.12 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2026-02-16 12:15:00 | 274.68 | 2026-02-24 10:15:00 | 260.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 09:15:00 | 274.22 | 2026-02-24 10:15:00 | 260.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-16 12:15:00 | 274.68 | 2026-02-24 15:15:00 | 262.10 | STOP_HIT | 0.50 | 4.58% |
| SELL | retest2 | 2026-02-17 09:15:00 | 274.22 | 2026-02-24 15:15:00 | 262.10 | STOP_HIT | 0.50 | 4.42% |
| BUY | retest2 | 2026-02-26 14:15:00 | 266.34 | 2026-02-27 09:15:00 | 263.90 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2026-02-27 12:30:00 | 266.60 | 2026-02-27 14:15:00 | 265.10 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2026-02-27 13:00:00 | 266.42 | 2026-02-27 14:15:00 | 265.10 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2026-03-04 09:15:00 | 253.50 | 2026-03-05 14:15:00 | 260.45 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2026-03-05 10:15:00 | 255.90 | 2026-03-05 14:15:00 | 260.45 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-03-05 11:00:00 | 255.60 | 2026-03-05 14:15:00 | 260.45 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2026-03-05 13:45:00 | 256.00 | 2026-03-05 14:15:00 | 260.45 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-03-19 10:45:00 | 240.30 | 2026-03-19 11:15:00 | 239.50 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2026-04-01 10:45:00 | 237.88 | 2026-04-01 14:15:00 | 240.08 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2026-04-08 09:30:00 | 253.46 | 2026-04-22 10:15:00 | 263.67 | STOP_HIT | 1.00 | 4.03% |
| BUY | retest2 | 2026-04-08 10:00:00 | 253.59 | 2026-04-22 10:15:00 | 263.67 | STOP_HIT | 1.00 | 3.97% |
| BUY | retest2 | 2026-04-08 13:00:00 | 253.40 | 2026-04-22 10:15:00 | 263.67 | STOP_HIT | 1.00 | 4.05% |
| SELL | retest2 | 2026-04-23 12:45:00 | 261.74 | 2026-04-27 09:15:00 | 264.80 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2026-04-23 14:15:00 | 261.92 | 2026-04-27 09:15:00 | 264.80 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-04-24 09:15:00 | 261.09 | 2026-04-27 09:15:00 | 264.80 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2026-04-24 11:00:00 | 261.71 | 2026-04-27 09:15:00 | 264.80 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2026-04-29 12:00:00 | 270.00 | 2026-04-29 13:15:00 | 267.25 | STOP_HIT | 1.00 | -1.02% |
