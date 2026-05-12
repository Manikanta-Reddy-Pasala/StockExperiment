# SJVN Ltd. (SJVN)

## Backtest Summary

- **Window:** 2024-03-13 10:15:00 → 2026-05-08 15:15:00 (3709 bars)
- **Last close:** 78.69
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 162 |
| ALERT1 | 91 |
| ALERT2 | 89 |
| ALERT2_SKIP | 40 |
| ALERT3 | 217 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 91 |
| PARTIAL | 14 |
| TARGET_HIT | 6 |
| STOP_HIT | 90 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 110 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 46 / 64
- **Target hits / Stop hits / Partials:** 6 / 90 / 14
- **Avg / median % per leg:** 0.89% / -0.20%
- **Sum % (uncompounded):** 97.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 32 | 6 | 18.8% | 4 | 27 | 1 | 0.35% | 11.1% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 4 | 1 | 2.26% | 13.5% |
| BUY @ 3rd Alert (retest2) | 26 | 3 | 11.5% | 3 | 23 | 0 | -0.09% | -2.5% |
| SELL (all) | 78 | 40 | 51.3% | 2 | 63 | 13 | 1.11% | 86.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 78 | 40 | 51.3% | 2 | 63 | 13 | 1.11% | 86.3% |
| retest1 (combined) | 6 | 3 | 50.0% | 1 | 4 | 1 | 2.26% | 13.5% |
| retest2 (combined) | 104 | 43 | 41.3% | 5 | 86 | 13 | 0.81% | 83.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 128.05 | 125.58 | 125.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 14:15:00 | 131.15 | 127.35 | 126.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 15:15:00 | 129.20 | 129.26 | 128.13 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 09:15:00 | 132.95 | 129.26 | 128.13 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-21 10:15:00 | 139.60 | 136.73 | 134.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-05-21 11:15:00 | 146.25 | 138.78 | 135.25 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 2 — SELL (started 2024-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 11:15:00 | 143.80 | 144.90 | 145.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 09:15:00 | 140.15 | 143.81 | 144.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 13:15:00 | 141.20 | 140.28 | 141.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 13:15:00 | 141.20 | 140.28 | 141.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 13:15:00 | 141.20 | 140.28 | 141.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 13:30:00 | 141.30 | 140.28 | 141.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 140.00 | 140.23 | 141.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 14:30:00 | 140.35 | 140.23 | 141.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 138.10 | 134.89 | 136.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 138.10 | 134.89 | 136.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 143.00 | 136.51 | 136.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 143.60 | 136.51 | 136.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 143.80 | 137.97 | 137.45 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 125.80 | 137.39 | 138.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 118.20 | 133.55 | 136.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 13:15:00 | 126.40 | 126.13 | 129.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 14:00:00 | 126.40 | 126.13 | 129.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 132.40 | 127.54 | 129.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 132.50 | 127.54 | 129.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 132.90 | 128.61 | 129.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 11:30:00 | 131.25 | 129.01 | 129.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-07 09:15:00 | 131.55 | 130.40 | 130.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 09:15:00 | 131.55 | 130.40 | 130.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 13:15:00 | 134.10 | 131.62 | 130.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 132.40 | 132.75 | 132.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-10 14:15:00 | 132.40 | 132.75 | 132.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 132.40 | 132.75 | 132.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:30:00 | 132.56 | 132.75 | 132.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 132.50 | 132.70 | 132.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:15:00 | 133.97 | 132.70 | 132.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 11:15:00 | 134.40 | 132.95 | 132.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 11:45:00 | 133.40 | 133.85 | 133.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-13 14:15:00 | 133.12 | 133.53 | 133.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 14:15:00 | 133.12 | 133.53 | 133.53 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 09:15:00 | 133.80 | 133.55 | 133.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-14 11:15:00 | 136.25 | 134.13 | 133.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-18 09:15:00 | 134.50 | 134.79 | 134.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-18 10:00:00 | 134.50 | 134.79 | 134.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 134.61 | 134.75 | 134.36 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2024-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 09:15:00 | 132.26 | 133.95 | 134.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 12:15:00 | 130.46 | 131.07 | 131.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 11:15:00 | 131.59 | 130.20 | 130.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 11:15:00 | 131.59 | 130.20 | 130.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 131.59 | 130.20 | 130.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 12:00:00 | 131.59 | 130.20 | 130.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 130.80 | 130.32 | 130.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 12:45:00 | 131.50 | 130.32 | 130.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 13:15:00 | 130.60 | 130.38 | 130.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 14:15:00 | 130.45 | 130.38 | 130.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 09:15:00 | 132.62 | 130.56 | 130.77 | SL hit (close>static) qty=1.00 sl=131.47 alert=retest2 |

### Cycle 9 — BUY (started 2024-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 11:15:00 | 131.75 | 130.92 | 130.91 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 129.82 | 130.72 | 130.82 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 14:15:00 | 131.89 | 130.96 | 130.92 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 131.20 | 131.56 | 131.56 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 10:15:00 | 136.23 | 132.25 | 131.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 140.80 | 135.80 | 133.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 140.94 | 144.22 | 143.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 09:15:00 | 140.94 | 144.22 | 143.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 140.94 | 144.22 | 143.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 140.94 | 144.22 | 143.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 140.65 | 143.50 | 142.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 139.35 | 143.50 | 142.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2024-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 13:15:00 | 140.60 | 142.35 | 142.41 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 14:15:00 | 142.85 | 142.45 | 142.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 09:15:00 | 145.60 | 143.10 | 142.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 14:15:00 | 150.60 | 151.02 | 148.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-12 15:00:00 | 150.60 | 151.02 | 148.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 150.56 | 150.93 | 148.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:30:00 | 148.82 | 150.93 | 148.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 151.64 | 152.91 | 151.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:00:00 | 151.64 | 152.91 | 151.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 151.16 | 152.56 | 151.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:45:00 | 151.55 | 152.56 | 151.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 151.75 | 152.40 | 151.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 148.25 | 152.40 | 151.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 148.75 | 151.67 | 151.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 147.07 | 151.67 | 151.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 146.20 | 150.58 | 151.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 142.50 | 147.05 | 148.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 144.13 | 143.18 | 145.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 09:45:00 | 143.90 | 143.18 | 145.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 143.10 | 143.54 | 144.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 133.70 | 143.87 | 144.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 14:15:00 | 147.75 | 143.32 | 144.04 | SL hit (close>static) qty=1.00 sl=145.35 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 10:15:00 | 145.78 | 144.66 | 144.55 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 09:15:00 | 141.50 | 144.00 | 144.31 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 09:15:00 | 152.35 | 144.24 | 144.01 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 10:15:00 | 148.37 | 149.37 | 149.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 12:15:00 | 146.99 | 148.59 | 149.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 141.69 | 140.37 | 142.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 141.69 | 140.37 | 142.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 141.69 | 140.37 | 142.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:00:00 | 137.89 | 140.28 | 141.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 15:00:00 | 139.85 | 138.94 | 139.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:45:00 | 139.70 | 139.37 | 139.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 10:30:00 | 140.04 | 139.46 | 139.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 11:15:00 | 140.33 | 139.63 | 140.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 11:45:00 | 140.07 | 139.63 | 140.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 12:15:00 | 139.55 | 139.61 | 139.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 12:30:00 | 140.38 | 139.61 | 139.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 13:15:00 | 139.10 | 139.51 | 139.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 13:30:00 | 139.75 | 139.51 | 139.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 139.29 | 139.47 | 139.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 14:45:00 | 141.60 | 139.47 | 139.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 140.76 | 139.70 | 139.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 10:15:00 | 140.01 | 139.70 | 139.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 11:45:00 | 140.25 | 139.98 | 139.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 12:15:00 | 140.00 | 139.98 | 139.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-08-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 12:15:00 | 140.00 | 139.98 | 139.98 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 13:15:00 | 139.91 | 139.97 | 139.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 14:15:00 | 139.60 | 139.89 | 139.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-12 10:15:00 | 140.20 | 139.35 | 139.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 10:15:00 | 140.20 | 139.35 | 139.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 140.20 | 139.35 | 139.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 11:00:00 | 140.20 | 139.35 | 139.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 11:15:00 | 140.44 | 139.56 | 139.70 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2024-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 13:15:00 | 141.50 | 140.11 | 139.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-13 09:15:00 | 141.83 | 140.52 | 140.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 12:15:00 | 139.77 | 140.53 | 140.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 12:15:00 | 139.77 | 140.53 | 140.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 139.77 | 140.53 | 140.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 13:00:00 | 139.77 | 140.53 | 140.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 139.17 | 140.26 | 140.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 139.17 | 140.26 | 140.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2024-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 14:15:00 | 136.55 | 139.52 | 139.85 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 09:15:00 | 145.70 | 140.41 | 140.18 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 13:15:00 | 141.04 | 141.68 | 141.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 09:15:00 | 140.17 | 141.22 | 141.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 09:15:00 | 134.98 | 134.47 | 136.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-23 09:45:00 | 135.74 | 134.47 | 136.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 134.08 | 134.13 | 135.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:30:00 | 135.00 | 134.13 | 135.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 133.88 | 133.78 | 134.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:30:00 | 134.08 | 133.78 | 134.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 134.85 | 133.03 | 133.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:30:00 | 136.35 | 133.03 | 133.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 134.01 | 133.23 | 133.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 13:15:00 | 133.49 | 133.43 | 133.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 13:45:00 | 133.26 | 133.39 | 133.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 15:15:00 | 133.75 | 133.51 | 133.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2024-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 15:15:00 | 133.75 | 133.51 | 133.48 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 09:15:00 | 132.95 | 133.40 | 133.43 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-08-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 11:15:00 | 133.73 | 133.46 | 133.45 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-08-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 15:15:00 | 133.20 | 133.41 | 133.43 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 09:15:00 | 136.45 | 134.02 | 133.71 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 10:15:00 | 133.75 | 134.92 | 134.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 09:15:00 | 133.20 | 133.94 | 134.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 129.92 | 127.76 | 129.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 129.92 | 127.76 | 129.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 129.92 | 127.76 | 129.33 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2024-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 14:15:00 | 133.29 | 130.40 | 130.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 15:15:00 | 134.14 | 131.15 | 130.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 131.26 | 131.62 | 131.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 14:00:00 | 131.26 | 131.62 | 131.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 131.71 | 131.64 | 131.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:45:00 | 131.36 | 131.64 | 131.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 130.11 | 131.38 | 131.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:45:00 | 129.80 | 131.38 | 131.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 130.13 | 131.13 | 130.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:45:00 | 130.07 | 131.13 | 130.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 11:15:00 | 129.81 | 130.87 | 130.89 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 131.95 | 131.05 | 130.93 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-09-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 15:15:00 | 130.54 | 130.96 | 130.97 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 09:15:00 | 131.36 | 131.04 | 131.00 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 11:15:00 | 130.18 | 130.86 | 130.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 14:15:00 | 129.73 | 130.57 | 130.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 10:15:00 | 130.49 | 130.44 | 130.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 10:15:00 | 130.49 | 130.44 | 130.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 130.49 | 130.44 | 130.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:30:00 | 130.88 | 130.44 | 130.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 11:15:00 | 130.12 | 130.37 | 130.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 12:15:00 | 130.01 | 130.37 | 130.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 14:15:00 | 129.83 | 130.35 | 130.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 11:30:00 | 129.94 | 130.19 | 130.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 12:45:00 | 129.98 | 130.09 | 130.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 128.71 | 129.45 | 129.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 10:15:00 | 128.23 | 129.45 | 129.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 15:15:00 | 128.00 | 128.46 | 129.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 10:00:00 | 128.47 | 128.39 | 128.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 14:15:00 | 130.20 | 129.33 | 129.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-09-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 14:15:00 | 130.20 | 129.33 | 129.27 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 15:15:00 | 128.70 | 129.36 | 129.40 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 12:15:00 | 129.50 | 129.41 | 129.41 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 09:15:00 | 127.10 | 129.04 | 129.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 09:15:00 | 126.60 | 127.64 | 128.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 09:15:00 | 128.00 | 126.81 | 127.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 09:15:00 | 128.00 | 126.81 | 127.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 128.00 | 126.81 | 127.46 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2024-09-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 12:15:00 | 131.63 | 128.37 | 128.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 14:15:00 | 134.75 | 130.19 | 128.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 13:15:00 | 131.53 | 131.62 | 130.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-30 13:45:00 | 132.25 | 131.62 | 130.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 131.15 | 131.57 | 130.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 12:45:00 | 131.12 | 131.57 | 130.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 130.84 | 131.43 | 130.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 15:00:00 | 130.84 | 131.43 | 130.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 130.72 | 131.29 | 130.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 09:15:00 | 128.97 | 131.29 | 130.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 128.50 | 130.73 | 130.72 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 10:15:00 | 128.35 | 130.25 | 130.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 11:15:00 | 128.04 | 129.81 | 130.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 120.45 | 120.45 | 123.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 11:00:00 | 120.45 | 120.45 | 123.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 123.32 | 120.97 | 122.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 12:00:00 | 122.00 | 121.49 | 122.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 09:30:00 | 121.97 | 121.48 | 121.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 09:45:00 | 121.81 | 119.89 | 120.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 10:15:00 | 121.88 | 120.29 | 120.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2024-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 10:15:00 | 121.88 | 120.29 | 120.28 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 15:15:00 | 119.60 | 120.61 | 120.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 117.15 | 118.93 | 119.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 14:15:00 | 123.20 | 119.28 | 119.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 14:15:00 | 123.20 | 119.28 | 119.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 123.20 | 119.28 | 119.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 15:00:00 | 123.20 | 119.28 | 119.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2024-10-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 15:15:00 | 123.20 | 120.07 | 119.79 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-10-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 15:15:00 | 118.82 | 119.73 | 119.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 116.35 | 119.05 | 119.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 09:15:00 | 112.55 | 111.93 | 113.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-24 10:00:00 | 112.55 | 111.93 | 113.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 112.79 | 112.10 | 113.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:30:00 | 113.84 | 112.10 | 113.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 109.29 | 108.97 | 109.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 09:30:00 | 109.20 | 108.97 | 109.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 13:15:00 | 109.74 | 109.14 | 109.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 14:00:00 | 109.74 | 109.14 | 109.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 111.38 | 109.59 | 109.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 15:00:00 | 111.38 | 109.59 | 109.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2024-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 15:15:00 | 111.30 | 109.93 | 109.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 112.85 | 110.51 | 110.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 111.25 | 113.28 | 112.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 111.25 | 113.28 | 112.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 111.25 | 113.28 | 112.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 111.25 | 113.28 | 112.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 110.88 | 112.80 | 112.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 110.88 | 112.80 | 112.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2024-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 13:15:00 | 111.95 | 112.28 | 112.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 15:15:00 | 110.95 | 111.90 | 112.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 111.92 | 111.21 | 111.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 13:15:00 | 111.92 | 111.21 | 111.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 111.92 | 111.21 | 111.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:00:00 | 111.92 | 111.21 | 111.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 113.57 | 111.68 | 111.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:45:00 | 114.45 | 111.68 | 111.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2024-11-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 15:15:00 | 112.93 | 111.93 | 111.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 114.19 | 112.38 | 112.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 14:15:00 | 115.04 | 115.38 | 114.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 15:00:00 | 115.04 | 115.38 | 114.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 113.59 | 114.89 | 114.35 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2024-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 12:15:00 | 112.78 | 113.87 | 113.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 14:15:00 | 111.63 | 113.17 | 113.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 111.13 | 110.49 | 111.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 111.13 | 110.49 | 111.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 111.13 | 110.49 | 111.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:30:00 | 110.94 | 110.49 | 111.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 110.87 | 110.56 | 111.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:00:00 | 110.43 | 110.54 | 111.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 12:15:00 | 104.91 | 107.24 | 109.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 09:15:00 | 106.15 | 106.02 | 107.81 | SL hit (close>ema200) qty=0.50 sl=106.02 alert=retest2 |

### Cycle 53 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 107.10 | 106.05 | 105.92 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 103.45 | 105.45 | 105.69 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2024-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 10:15:00 | 108.00 | 105.58 | 105.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 110.99 | 107.67 | 106.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 09:15:00 | 115.31 | 115.95 | 114.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-29 10:00:00 | 115.31 | 115.95 | 114.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 114.86 | 115.50 | 114.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 12:45:00 | 114.89 | 115.50 | 114.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 115.09 | 115.27 | 114.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:15:00 | 114.12 | 115.27 | 114.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 113.87 | 114.99 | 114.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:00:00 | 113.87 | 114.99 | 114.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 114.19 | 114.83 | 114.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:30:00 | 114.10 | 114.83 | 114.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2024-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 13:15:00 | 113.81 | 114.36 | 114.40 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2024-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 09:15:00 | 115.97 | 114.51 | 114.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 14:15:00 | 116.70 | 115.48 | 114.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 09:15:00 | 116.93 | 117.07 | 116.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 09:45:00 | 116.66 | 117.07 | 116.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 120.75 | 117.82 | 117.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 10:30:00 | 121.80 | 118.67 | 117.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 09:15:00 | 122.46 | 119.86 | 118.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 15:15:00 | 119.30 | 120.12 | 120.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2024-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 15:15:00 | 119.30 | 120.12 | 120.16 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 12:15:00 | 120.29 | 120.18 | 120.17 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 15:15:00 | 120.00 | 120.17 | 120.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 09:15:00 | 118.95 | 119.92 | 120.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 12:15:00 | 117.89 | 117.78 | 118.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 13:00:00 | 117.89 | 117.78 | 118.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 118.36 | 118.03 | 118.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 12:45:00 | 117.75 | 118.00 | 118.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 14:30:00 | 117.65 | 117.85 | 118.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 11:30:00 | 117.72 | 117.96 | 118.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 14:15:00 | 117.77 | 118.19 | 118.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 15:15:00 | 117.90 | 118.10 | 118.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 09:15:00 | 116.90 | 118.10 | 118.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 111.86 | 115.72 | 116.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 111.83 | 115.72 | 116.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 111.88 | 115.72 | 116.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-20 11:15:00 | 114.82 | 114.53 | 115.33 | SL hit (close>ema200) qty=0.50 sl=114.53 alert=retest2 |

### Cycle 61 — BUY (started 2025-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 10:15:00 | 110.25 | 106.96 | 106.79 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 105.13 | 108.27 | 108.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 11:15:00 | 104.86 | 107.58 | 108.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 94.20 | 93.47 | 96.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 94.20 | 93.47 | 96.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 95.35 | 94.12 | 95.24 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 97.76 | 95.56 | 95.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 99.56 | 97.54 | 96.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 100.65 | 100.75 | 99.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:00:00 | 100.65 | 100.75 | 99.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 100.65 | 100.64 | 99.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 100.66 | 100.64 | 99.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 99.88 | 100.41 | 99.82 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 95.77 | 99.07 | 99.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 94.44 | 97.59 | 98.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 97.75 | 96.61 | 97.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 97.75 | 96.61 | 97.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 97.75 | 96.61 | 97.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 97.75 | 96.61 | 97.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 98.05 | 96.90 | 97.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:45:00 | 98.37 | 96.90 | 97.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 98.14 | 97.26 | 97.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 13:00:00 | 98.14 | 97.26 | 97.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 97.91 | 97.39 | 97.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 14:15:00 | 97.31 | 97.39 | 97.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:15:00 | 97.87 | 97.73 | 97.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 92.44 | 95.54 | 96.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 92.98 | 95.54 | 96.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 12:15:00 | 91.98 | 91.37 | 93.18 | SL hit (close>ema200) qty=0.50 sl=91.37 alert=retest2 |

### Cycle 65 — BUY (started 2025-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 11:15:00 | 93.76 | 92.37 | 92.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 12:15:00 | 94.93 | 92.88 | 92.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 96.30 | 98.40 | 96.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 96.30 | 98.40 | 96.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 96.30 | 98.40 | 96.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 96.30 | 98.40 | 96.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 97.86 | 98.29 | 96.75 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 11:15:00 | 93.46 | 95.98 | 96.14 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 97.63 | 95.88 | 95.72 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-02-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 12:15:00 | 95.76 | 96.29 | 96.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 13:15:00 | 95.08 | 96.04 | 96.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 15:15:00 | 90.45 | 90.33 | 91.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 09:15:00 | 91.35 | 90.33 | 91.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 91.70 | 90.60 | 91.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:45:00 | 90.22 | 91.51 | 91.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 10:00:00 | 90.37 | 89.01 | 89.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 10:30:00 | 90.42 | 89.23 | 89.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 11:15:00 | 90.51 | 89.48 | 89.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 90.51 | 89.48 | 89.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 91.10 | 90.05 | 89.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 13:15:00 | 93.22 | 93.31 | 92.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 14:00:00 | 93.22 | 93.31 | 92.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 90.79 | 92.69 | 92.19 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2025-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 12:15:00 | 90.95 | 91.84 | 91.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 13:15:00 | 90.63 | 91.60 | 91.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 13:15:00 | 90.88 | 90.81 | 91.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-25 13:30:00 | 90.68 | 90.81 | 91.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 83.94 | 83.03 | 84.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 83.94 | 83.03 | 84.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 83.26 | 83.07 | 84.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 11:15:00 | 83.16 | 83.07 | 84.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 14:30:00 | 83.05 | 82.84 | 83.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 85.50 | 83.41 | 83.99 | SL hit (close>static) qty=1.00 sl=84.64 alert=retest2 |

### Cycle 71 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 86.14 | 84.64 | 84.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 13:15:00 | 86.34 | 84.98 | 84.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 12:15:00 | 87.61 | 87.62 | 86.81 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 13:45:00 | 87.82 | 87.70 | 86.92 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 09:30:00 | 88.05 | 87.77 | 87.14 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 10:45:00 | 87.86 | 87.71 | 87.17 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 87.08 | 87.58 | 87.16 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-10 11:15:00 | 87.08 | 87.58 | 87.16 | SL hit (close<ema400) qty=1.00 sl=87.16 alert=retest1 |

### Cycle 72 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 85.52 | 86.78 | 86.89 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 10:15:00 | 86.72 | 86.34 | 86.33 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 12:15:00 | 86.12 | 86.31 | 86.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 14:15:00 | 85.49 | 86.11 | 86.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 86.32 | 86.03 | 86.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 86.32 | 86.03 | 86.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 86.32 | 86.03 | 86.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 86.50 | 86.03 | 86.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 86.10 | 86.04 | 86.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:45:00 | 86.39 | 86.04 | 86.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 86.09 | 86.05 | 86.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 11:30:00 | 86.15 | 86.05 | 86.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 13:15:00 | 86.42 | 86.10 | 86.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 14:00:00 | 86.42 | 86.10 | 86.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 86.34 | 86.15 | 86.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 14:45:00 | 86.38 | 86.15 | 86.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 15:15:00 | 86.39 | 86.19 | 86.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 87.15 | 86.39 | 86.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 94.74 | 96.05 | 94.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:00:00 | 94.74 | 96.05 | 94.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 94.52 | 95.74 | 94.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:00:00 | 94.52 | 95.74 | 94.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 93.93 | 95.38 | 94.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 93.44 | 95.38 | 94.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 93.61 | 95.03 | 94.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:30:00 | 93.80 | 95.03 | 94.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 93.43 | 94.38 | 94.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 94.47 | 94.38 | 94.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 11:15:00 | 93.40 | 94.13 | 94.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 11:15:00 | 93.40 | 94.13 | 94.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 12:15:00 | 93.05 | 93.91 | 94.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 92.50 | 92.11 | 92.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 92.50 | 92.11 | 92.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 92.50 | 92.11 | 92.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 92.50 | 92.11 | 92.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 94.72 | 92.62 | 92.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:45:00 | 95.52 | 92.62 | 92.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 94.03 | 92.90 | 92.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:30:00 | 94.94 | 92.90 | 92.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 11:15:00 | 94.01 | 93.12 | 93.08 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 13:15:00 | 92.24 | 92.94 | 93.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 14:15:00 | 91.60 | 92.67 | 92.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 09:15:00 | 92.42 | 92.41 | 92.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 92.42 | 92.41 | 92.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 92.42 | 92.41 | 92.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:45:00 | 92.74 | 92.41 | 92.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 91.00 | 92.12 | 92.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 09:15:00 | 90.64 | 91.65 | 92.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 12:15:00 | 92.81 | 91.99 | 92.12 | SL hit (close>static) qty=1.00 sl=92.60 alert=retest2 |

### Cycle 79 — BUY (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 14:15:00 | 93.27 | 92.35 | 92.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 15:15:00 | 93.45 | 92.57 | 92.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 92.32 | 94.31 | 93.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 92.32 | 94.31 | 93.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 92.32 | 94.31 | 93.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 92.32 | 94.31 | 93.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 93.45 | 94.14 | 93.64 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2025-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 13:15:00 | 92.11 | 93.26 | 93.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 86.50 | 91.74 | 92.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 89.80 | 89.42 | 90.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 91.26 | 89.42 | 90.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 90.35 | 89.60 | 90.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 89.77 | 89.88 | 90.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 89.50 | 90.53 | 90.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 14:00:00 | 89.85 | 90.14 | 90.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 11:15:00 | 91.57 | 90.75 | 90.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 91.57 | 90.75 | 90.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 94.23 | 91.73 | 91.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 93.78 | 94.53 | 93.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 93.78 | 94.53 | 93.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 93.78 | 94.53 | 93.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 10:00:00 | 93.78 | 94.53 | 93.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 94.60 | 94.54 | 93.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:15:00 | 95.06 | 94.54 | 93.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:30:00 | 95.11 | 94.58 | 94.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 10:15:00 | 94.44 | 97.98 | 98.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 94.44 | 97.98 | 98.35 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 09:15:00 | 98.20 | 97.59 | 97.54 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 13:15:00 | 97.02 | 97.51 | 97.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 14:15:00 | 96.79 | 97.37 | 97.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 95.57 | 95.34 | 96.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-02 10:00:00 | 95.57 | 95.34 | 96.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 95.20 | 95.31 | 96.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:15:00 | 94.57 | 95.31 | 96.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 89.84 | 91.24 | 92.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-07 11:15:00 | 91.75 | 91.24 | 92.40 | SL hit (close>ema200) qty=0.50 sl=91.24 alert=retest2 |

### Cycle 85 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 94.03 | 90.64 | 90.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 94.46 | 91.93 | 91.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 103.49 | 104.51 | 102.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 103.49 | 104.51 | 102.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 103.49 | 104.51 | 102.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 103.26 | 104.51 | 102.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 103.88 | 104.38 | 102.93 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 09:15:00 | 101.20 | 102.51 | 102.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 99.57 | 101.66 | 102.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 10:15:00 | 101.20 | 101.17 | 101.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 11:00:00 | 101.20 | 101.17 | 101.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 101.20 | 100.95 | 101.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:15:00 | 100.69 | 100.95 | 101.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 99.65 | 100.69 | 101.16 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 11:15:00 | 101.55 | 101.06 | 101.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 13:15:00 | 102.17 | 101.38 | 101.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 101.40 | 101.55 | 101.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 101.40 | 101.55 | 101.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 101.40 | 101.55 | 101.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 101.40 | 101.55 | 101.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 103.00 | 101.84 | 101.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 12:30:00 | 103.26 | 102.20 | 101.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 10:30:00 | 103.36 | 102.78 | 102.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 97.71 | 101.48 | 101.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 97.71 | 101.48 | 101.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 10:15:00 | 97.02 | 100.59 | 101.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 12:15:00 | 98.30 | 97.68 | 98.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-02 13:00:00 | 98.30 | 97.68 | 98.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 97.54 | 97.20 | 97.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:00:00 | 97.54 | 97.20 | 97.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 98.55 | 97.47 | 97.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:45:00 | 98.45 | 97.47 | 97.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 98.50 | 97.68 | 97.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:30:00 | 98.79 | 97.68 | 97.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 14:15:00 | 98.66 | 98.04 | 98.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 103.21 | 99.19 | 98.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 14:15:00 | 103.49 | 103.75 | 102.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 14:30:00 | 103.53 | 103.75 | 102.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 102.85 | 104.12 | 103.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 103.01 | 104.12 | 103.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 103.39 | 103.97 | 103.73 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 102.45 | 103.55 | 103.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 10:15:00 | 101.90 | 103.22 | 103.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 98.79 | 98.46 | 99.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 13:00:00 | 98.79 | 98.46 | 99.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 98.95 | 98.81 | 99.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 99.08 | 98.81 | 99.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 95.76 | 94.70 | 95.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:45:00 | 95.43 | 94.70 | 95.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 95.69 | 94.90 | 95.84 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 97.23 | 96.29 | 96.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 12:15:00 | 97.86 | 96.72 | 96.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 98.31 | 98.48 | 97.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 13:30:00 | 98.27 | 98.48 | 97.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 99.77 | 100.35 | 99.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:45:00 | 99.72 | 100.35 | 99.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 99.40 | 100.16 | 99.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 13:00:00 | 99.40 | 100.16 | 99.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 99.88 | 100.11 | 99.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 13:30:00 | 99.57 | 100.11 | 99.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 100.04 | 100.39 | 100.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:00:00 | 100.04 | 100.39 | 100.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 99.95 | 100.31 | 99.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:30:00 | 99.98 | 100.31 | 99.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 99.42 | 100.13 | 99.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 99.42 | 100.13 | 99.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 100.11 | 100.12 | 99.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 14:00:00 | 100.32 | 100.10 | 100.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 10:15:00 | 99.36 | 99.96 | 99.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 99.36 | 99.96 | 99.98 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 13:15:00 | 100.31 | 99.98 | 99.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 14:15:00 | 100.38 | 100.06 | 100.01 | Break + close above crossover candle high |

### Cycle 94 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 99.15 | 99.93 | 99.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 10:15:00 | 98.85 | 99.71 | 99.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 99.30 | 99.21 | 99.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 99.30 | 99.21 | 99.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 99.30 | 99.21 | 99.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:45:00 | 99.36 | 99.21 | 99.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 99.65 | 99.30 | 99.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:30:00 | 99.56 | 99.30 | 99.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 99.49 | 99.34 | 99.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:15:00 | 99.62 | 99.34 | 99.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 99.15 | 99.30 | 99.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 13:15:00 | 98.77 | 99.30 | 99.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:45:00 | 99.10 | 98.51 | 98.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:45:00 | 99.02 | 98.74 | 98.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 10:30:00 | 98.99 | 98.74 | 98.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 99.30 | 98.61 | 98.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 14:45:00 | 99.36 | 98.61 | 98.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 99.16 | 98.72 | 98.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 99.19 | 98.72 | 98.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 99.00 | 98.77 | 98.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 99.00 | 98.77 | 98.75 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 15:15:00 | 98.38 | 98.75 | 98.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 10:15:00 | 97.78 | 98.51 | 98.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 11:15:00 | 98.02 | 97.92 | 98.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 11:15:00 | 98.02 | 97.92 | 98.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 98.02 | 97.92 | 98.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:45:00 | 98.06 | 97.92 | 98.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 98.67 | 98.07 | 98.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:45:00 | 98.75 | 98.07 | 98.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 98.04 | 98.07 | 98.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 14:15:00 | 97.96 | 98.07 | 98.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 09:45:00 | 97.97 | 98.01 | 98.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:15:00 | 97.89 | 98.05 | 98.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 15:15:00 | 98.45 | 98.19 | 98.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 15:15:00 | 98.45 | 98.19 | 98.19 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 09:15:00 | 97.85 | 98.12 | 98.15 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 98.56 | 98.25 | 98.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 99.01 | 98.45 | 98.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 11:15:00 | 98.59 | 98.64 | 98.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 11:15:00 | 98.59 | 98.64 | 98.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 98.59 | 98.64 | 98.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 13:15:00 | 98.80 | 98.63 | 98.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 14:15:00 | 98.76 | 98.63 | 98.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 98.29 | 98.81 | 98.73 | SL hit (close<static) qty=1.00 sl=98.35 alert=retest2 |

### Cycle 100 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 97.97 | 98.64 | 98.66 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 11:15:00 | 98.95 | 98.70 | 98.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 12:15:00 | 99.22 | 98.81 | 98.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 14:15:00 | 98.91 | 98.93 | 98.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 14:15:00 | 98.91 | 98.93 | 98.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 98.91 | 98.93 | 98.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 15:00:00 | 98.91 | 98.93 | 98.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 98.64 | 98.89 | 98.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:30:00 | 99.40 | 98.89 | 98.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 11:45:00 | 99.25 | 98.96 | 98.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 15:15:00 | 99.18 | 99.10 | 98.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 09:45:00 | 99.14 | 99.07 | 99.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 10:15:00 | 98.76 | 99.01 | 99.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 10:15:00 | 98.76 | 99.01 | 99.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 11:15:00 | 98.57 | 98.92 | 98.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 15:15:00 | 98.89 | 98.84 | 98.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 15:15:00 | 98.89 | 98.84 | 98.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 98.89 | 98.84 | 98.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 98.88 | 98.84 | 98.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 98.50 | 98.77 | 98.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 11:00:00 | 98.18 | 98.65 | 98.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 12:15:00 | 98.27 | 98.60 | 98.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:30:00 | 97.80 | 98.42 | 98.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 11:15:00 | 97.32 | 96.32 | 96.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 11:15:00 | 97.32 | 96.32 | 96.28 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 94.78 | 96.37 | 96.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 93.86 | 95.21 | 95.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 13:15:00 | 92.90 | 92.82 | 93.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 14:00:00 | 92.90 | 92.82 | 93.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 93.59 | 93.06 | 93.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:30:00 | 93.84 | 93.06 | 93.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 93.57 | 93.16 | 93.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:45:00 | 93.59 | 93.16 | 93.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 93.69 | 93.27 | 93.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 12:45:00 | 93.43 | 93.31 | 93.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:30:00 | 93.46 | 92.55 | 92.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 10:15:00 | 93.19 | 92.68 | 92.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 10:15:00 | 93.19 | 92.68 | 92.67 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 91.89 | 92.56 | 92.64 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 09:15:00 | 93.62 | 92.77 | 92.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 10:15:00 | 94.03 | 93.02 | 92.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-11 12:15:00 | 93.02 | 93.10 | 92.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 12:15:00 | 93.02 | 93.10 | 92.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 93.02 | 93.10 | 92.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:45:00 | 93.00 | 93.10 | 92.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 93.15 | 93.11 | 92.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 13:45:00 | 93.10 | 93.11 | 92.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 93.25 | 93.14 | 92.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 15:15:00 | 93.22 | 93.14 | 92.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 93.22 | 93.15 | 92.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 96.82 | 93.15 | 92.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 14:15:00 | 93.42 | 94.82 | 94.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 14:15:00 | 93.42 | 94.82 | 94.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 92.45 | 94.16 | 94.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 92.97 | 92.70 | 93.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-18 10:00:00 | 92.97 | 92.70 | 93.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 92.67 | 92.67 | 93.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 14:45:00 | 93.04 | 92.67 | 93.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 93.39 | 92.82 | 93.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:00:00 | 93.39 | 92.82 | 93.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 93.84 | 93.02 | 93.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:00:00 | 93.84 | 93.02 | 93.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 12:15:00 | 94.17 | 93.36 | 93.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 14:15:00 | 94.85 | 93.80 | 93.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 12:15:00 | 97.75 | 97.79 | 96.59 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 09:15:00 | 97.96 | 97.65 | 96.82 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 101.18 | 99.82 | 99.03 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-26 14:15:00 | 99.30 | 99.88 | 99.41 | SL hit (close<ema400) qty=1.00 sl=99.41 alert=retest1 |

### Cycle 110 — SELL (started 2025-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 12:15:00 | 97.92 | 98.93 | 99.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 13:15:00 | 97.50 | 98.64 | 98.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 15:15:00 | 94.35 | 94.21 | 95.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-02 09:15:00 | 93.75 | 94.21 | 95.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 93.85 | 93.98 | 94.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 12:15:00 | 93.61 | 93.93 | 94.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 13:00:00 | 93.41 | 93.83 | 94.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 09:45:00 | 93.39 | 92.84 | 92.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 15:15:00 | 94.20 | 93.22 | 93.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2025-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 15:15:00 | 94.20 | 93.22 | 93.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 09:15:00 | 94.33 | 93.44 | 93.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 12:15:00 | 93.73 | 93.73 | 93.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 12:45:00 | 93.82 | 93.73 | 93.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 93.49 | 93.69 | 93.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:00:00 | 93.49 | 93.69 | 93.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 93.65 | 93.68 | 93.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:30:00 | 93.55 | 93.68 | 93.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 93.66 | 93.67 | 93.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 94.15 | 93.67 | 93.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 11:15:00 | 93.27 | 93.59 | 93.48 | SL hit (close<static) qty=1.00 sl=93.45 alert=retest2 |

### Cycle 112 — SELL (started 2025-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 14:15:00 | 93.01 | 93.36 | 93.40 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 94.75 | 93.59 | 93.49 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 93.61 | 94.22 | 94.22 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 14:15:00 | 94.40 | 94.17 | 94.16 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 13:15:00 | 93.80 | 94.16 | 94.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 10:15:00 | 92.82 | 93.75 | 93.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 93.09 | 93.03 | 93.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 10:00:00 | 93.09 | 93.03 | 93.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 92.69 | 92.70 | 93.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 12:00:00 | 92.27 | 92.60 | 92.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 12:45:00 | 92.28 | 92.52 | 92.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 91.38 | 90.89 | 90.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 91.38 | 90.89 | 90.88 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-10-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 13:15:00 | 90.71 | 90.96 | 90.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 11:15:00 | 90.58 | 90.77 | 90.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 90.13 | 90.06 | 90.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 11:00:00 | 90.13 | 90.06 | 90.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 89.80 | 90.01 | 90.24 | EMA400 retest candle locked (from downside) |

### Cycle 119 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 90.57 | 90.33 | 90.30 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 14:15:00 | 90.06 | 90.26 | 90.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 13:15:00 | 90.00 | 90.15 | 90.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 89.42 | 89.15 | 89.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 89.42 | 89.15 | 89.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 89.42 | 89.15 | 89.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 89.28 | 89.15 | 89.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 89.60 | 89.24 | 89.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 89.60 | 89.24 | 89.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 89.66 | 89.32 | 89.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:00:00 | 89.66 | 89.32 | 89.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 89.31 | 89.32 | 89.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:30:00 | 89.59 | 89.32 | 89.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 89.69 | 89.34 | 89.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 89.65 | 89.34 | 89.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 89.51 | 89.37 | 89.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:00:00 | 89.26 | 89.35 | 89.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 14:15:00 | 90.01 | 88.93 | 89.08 | SL hit (close>static) qty=1.00 sl=89.90 alert=retest2 |

### Cycle 121 — BUY (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 15:15:00 | 90.80 | 89.31 | 89.23 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 12:15:00 | 89.07 | 89.18 | 89.19 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 89.93 | 89.34 | 89.26 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 89.02 | 89.33 | 89.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 11:15:00 | 88.91 | 89.25 | 89.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 88.56 | 88.46 | 88.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 88.56 | 88.46 | 88.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 88.56 | 88.46 | 88.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 12:15:00 | 88.10 | 88.48 | 88.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 13:00:00 | 88.07 | 88.40 | 88.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 13:30:00 | 88.05 | 88.34 | 88.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 89.95 | 88.61 | 88.65 | SL hit (close>static) qty=1.00 sl=89.10 alert=retest2 |

### Cycle 125 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 88.95 | 88.73 | 88.70 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 10:15:00 | 88.37 | 88.68 | 88.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 88.03 | 88.33 | 88.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 88.59 | 88.22 | 88.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 88.59 | 88.22 | 88.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 88.59 | 88.22 | 88.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 10:45:00 | 88.26 | 88.22 | 88.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 13:15:00 | 83.85 | 84.58 | 85.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 11:15:00 | 83.00 | 82.89 | 83.59 | SL hit (close>ema200) qty=0.50 sl=82.89 alert=retest2 |

### Cycle 127 — BUY (started 2025-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 15:15:00 | 83.68 | 83.09 | 83.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 09:15:00 | 84.10 | 83.29 | 83.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 12:15:00 | 83.34 | 83.40 | 83.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-17 13:00:00 | 83.34 | 83.40 | 83.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 83.40 | 83.40 | 83.26 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 82.18 | 83.14 | 83.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 81.75 | 82.31 | 82.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 14:15:00 | 82.10 | 82.04 | 82.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 15:00:00 | 82.10 | 82.04 | 82.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 82.06 | 82.02 | 82.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 15:15:00 | 81.75 | 81.98 | 82.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 12:15:00 | 77.66 | 77.99 | 78.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-02 10:15:00 | 78.32 | 77.98 | 78.16 | SL hit (close>ema200) qty=0.50 sl=77.98 alert=retest2 |

### Cycle 129 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 73.64 | 73.19 | 73.19 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 09:15:00 | 72.41 | 73.20 | 73.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 71.43 | 72.06 | 72.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 14:15:00 | 71.73 | 71.70 | 71.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 15:00:00 | 71.73 | 71.70 | 71.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 72.38 | 70.86 | 71.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 72.38 | 70.86 | 71.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 72.86 | 71.26 | 71.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 73.31 | 71.89 | 71.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 11:15:00 | 74.83 | 74.84 | 73.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 11:45:00 | 74.77 | 74.84 | 73.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 75.13 | 74.92 | 74.33 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 73.32 | 74.29 | 74.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 73.20 | 74.07 | 74.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 14:15:00 | 72.98 | 72.93 | 73.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 15:00:00 | 72.98 | 72.93 | 73.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 73.67 | 73.08 | 73.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 74.94 | 73.08 | 73.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 74.80 | 73.42 | 73.49 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 74.65 | 73.67 | 73.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 11:15:00 | 75.15 | 73.96 | 73.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 84.54 | 85.68 | 82.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 82.88 | 84.31 | 83.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 82.88 | 84.31 | 83.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:45:00 | 82.90 | 84.31 | 83.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 82.99 | 84.04 | 83.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:30:00 | 82.80 | 84.04 | 83.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 82.92 | 83.82 | 83.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:00:00 | 82.92 | 83.82 | 83.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 82.84 | 83.62 | 83.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:45:00 | 82.74 | 83.62 | 83.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 82.85 | 83.28 | 83.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 82.58 | 83.28 | 83.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 81.78 | 82.98 | 83.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 81.10 | 82.60 | 82.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 77.98 | 77.92 | 79.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:30:00 | 77.71 | 77.92 | 79.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 80.28 | 78.53 | 79.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 15:00:00 | 80.28 | 78.53 | 79.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 81.00 | 79.02 | 79.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 80.00 | 79.02 | 79.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 79.81 | 79.43 | 79.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:45:00 | 79.64 | 79.43 | 79.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 13:15:00 | 79.29 | 79.40 | 79.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:15:00 | 79.14 | 79.40 | 79.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:15:00 | 79.07 | 79.43 | 79.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 15:15:00 | 75.18 | 77.69 | 78.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 11:15:00 | 75.12 | 76.58 | 77.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-20 15:15:00 | 71.23 | 72.91 | 74.58 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 135 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 72.83 | 71.86 | 71.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 73.32 | 72.15 | 71.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 11:15:00 | 72.55 | 73.09 | 72.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 11:15:00 | 72.55 | 73.09 | 72.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 72.55 | 73.09 | 72.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:00:00 | 72.55 | 73.09 | 72.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 72.65 | 73.00 | 72.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 09:30:00 | 72.95 | 73.00 | 72.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 12:15:00 | 71.88 | 72.66 | 72.66 | SL hit (close<static) qty=1.00 sl=72.34 alert=retest2 |

### Cycle 136 — SELL (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 13:15:00 | 72.25 | 72.58 | 72.62 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 09:15:00 | 74.00 | 72.85 | 72.73 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 71.61 | 72.58 | 72.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 68.60 | 71.08 | 71.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 70.55 | 70.19 | 71.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 70.55 | 70.19 | 71.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 71.85 | 70.63 | 71.14 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2026-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 13:15:00 | 72.01 | 71.38 | 71.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 15:15:00 | 72.20 | 71.66 | 71.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 72.92 | 73.16 | 72.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 72.92 | 73.16 | 72.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 72.75 | 73.08 | 72.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 72.75 | 73.08 | 72.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 72.52 | 72.89 | 72.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:45:00 | 72.50 | 72.89 | 72.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 72.38 | 72.78 | 72.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:00:00 | 72.38 | 72.78 | 72.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 72.24 | 72.68 | 72.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 15:00:00 | 72.24 | 72.68 | 72.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 72.10 | 72.56 | 72.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 72.00 | 72.56 | 72.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 71.87 | 72.42 | 72.43 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 72.69 | 72.39 | 72.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 14:15:00 | 73.33 | 72.91 | 72.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 13:15:00 | 72.87 | 73.01 | 72.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 13:15:00 | 72.87 | 73.01 | 72.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 72.87 | 73.01 | 72.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:00:00 | 72.87 | 73.01 | 72.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 72.72 | 72.95 | 72.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 15:00:00 | 72.72 | 72.95 | 72.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 72.80 | 72.92 | 72.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:15:00 | 72.44 | 72.92 | 72.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 72.12 | 72.76 | 72.77 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 14:15:00 | 77.59 | 73.54 | 73.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 09:15:00 | 79.99 | 75.54 | 74.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 76.74 | 77.69 | 76.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 76.74 | 77.69 | 76.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 76.74 | 77.69 | 76.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 76.67 | 77.69 | 76.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 76.54 | 77.11 | 76.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 15:00:00 | 76.54 | 77.11 | 76.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 76.46 | 76.93 | 76.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:45:00 | 75.99 | 76.93 | 76.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 76.90 | 76.93 | 76.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 11:15:00 | 77.48 | 76.93 | 76.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 13:15:00 | 77.20 | 77.00 | 76.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 13:45:00 | 77.39 | 77.09 | 76.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 12:15:00 | 76.61 | 77.06 | 77.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 12:15:00 | 76.61 | 77.06 | 77.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 76.01 | 76.63 | 76.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 14:15:00 | 75.36 | 75.34 | 75.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 14:45:00 | 75.47 | 75.34 | 75.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 74.58 | 74.13 | 74.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:15:00 | 74.72 | 74.13 | 74.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 75.18 | 74.34 | 74.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:15:00 | 75.27 | 74.34 | 74.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 75.32 | 74.54 | 74.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:45:00 | 75.38 | 74.54 | 74.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — BUY (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 11:15:00 | 75.41 | 74.71 | 74.71 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 11:15:00 | 74.01 | 74.72 | 74.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 13:15:00 | 73.82 | 74.44 | 74.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 70.30 | 69.14 | 70.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 70.30 | 69.14 | 70.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 70.30 | 69.14 | 70.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:45:00 | 70.28 | 69.14 | 70.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 70.14 | 69.34 | 70.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 11:15:00 | 69.95 | 69.34 | 70.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 72.85 | 70.41 | 70.53 | SL hit (close>static) qty=1.00 sl=70.45 alert=retest2 |

### Cycle 147 — BUY (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 15:15:00 | 72.50 | 70.83 | 70.71 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 69.04 | 70.83 | 70.92 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 71.21 | 70.14 | 70.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 10:15:00 | 71.95 | 70.74 | 70.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 71.87 | 71.91 | 71.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-13 10:00:00 | 71.87 | 71.91 | 71.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 12:15:00 | 70.80 | 71.71 | 71.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 13:00:00 | 70.80 | 71.71 | 71.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 13:15:00 | 70.50 | 71.47 | 71.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 14:00:00 | 70.50 | 71.47 | 71.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — SELL (started 2026-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 15:15:00 | 69.94 | 70.94 | 71.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 69.06 | 70.46 | 70.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 69.85 | 69.29 | 69.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 09:15:00 | 69.85 | 69.29 | 69.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 69.85 | 69.29 | 69.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 70.05 | 69.29 | 69.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 70.32 | 69.49 | 69.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 70.32 | 69.49 | 69.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 70.97 | 69.85 | 69.84 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 69.20 | 69.88 | 69.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 68.67 | 69.64 | 69.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 70.95 | 69.78 | 69.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 70.95 | 69.78 | 69.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 70.95 | 69.78 | 69.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 70.55 | 69.78 | 69.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 71.33 | 70.09 | 69.98 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 67.25 | 69.71 | 69.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 67.10 | 69.19 | 69.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 67.64 | 67.18 | 68.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 13:00:00 | 67.64 | 67.18 | 68.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 68.95 | 67.50 | 67.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 68.76 | 67.50 | 67.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 68.92 | 67.79 | 67.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:45:00 | 68.85 | 67.79 | 67.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 66.10 | 64.92 | 65.81 | EMA400 retest candle locked (from downside) |

### Cycle 155 — BUY (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 15:15:00 | 67.25 | 66.30 | 66.20 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 65.34 | 66.11 | 66.12 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 67.45 | 66.25 | 66.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 67.74 | 67.07 | 66.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 12:15:00 | 67.50 | 67.52 | 67.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 13:00:00 | 67.50 | 67.52 | 67.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 72.77 | 72.03 | 71.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 73.09 | 72.03 | 71.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:30:00 | 72.92 | 72.54 | 71.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 73.83 | 72.48 | 71.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-22 14:15:00 | 80.40 | 78.14 | 77.51 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 158 — SELL (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 12:15:00 | 77.30 | 78.33 | 78.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 13:15:00 | 77.13 | 78.09 | 78.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 15:15:00 | 78.04 | 78.04 | 78.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:15:00 | 78.66 | 78.04 | 78.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 159 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 79.55 | 78.34 | 78.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 10:15:00 | 80.84 | 78.84 | 78.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 80.62 | 80.63 | 79.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 14:00:00 | 80.62 | 80.63 | 79.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 80.82 | 81.22 | 80.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 80.82 | 81.22 | 80.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 80.81 | 81.14 | 80.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:45:00 | 80.73 | 81.14 | 80.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 80.34 | 80.98 | 80.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 80.49 | 80.98 | 80.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 78.94 | 80.57 | 80.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 78.94 | 80.57 | 80.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 79.16 | 80.29 | 80.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 12:15:00 | 78.68 | 79.17 | 79.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 14:15:00 | 79.97 | 79.31 | 79.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 14:15:00 | 79.97 | 79.31 | 79.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 79.97 | 79.31 | 79.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 15:00:00 | 79.97 | 79.31 | 79.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 79.99 | 79.44 | 79.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 80.00 | 79.44 | 79.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — BUY (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 10:15:00 | 79.96 | 79.58 | 79.57 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 11:15:00 | 79.12 | 79.60 | 79.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 13:15:00 | 78.94 | 79.41 | 79.54 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-16 09:15:00 | 132.95 | 2024-05-21 10:15:00 | 139.60 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-05-16 09:15:00 | 132.95 | 2024-05-21 11:15:00 | 146.25 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-06 11:30:00 | 131.25 | 2024-06-07 09:15:00 | 131.55 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2024-06-11 09:15:00 | 133.97 | 2024-06-13 14:15:00 | 133.12 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-06-11 11:15:00 | 134.40 | 2024-06-13 14:15:00 | 133.12 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-06-13 11:45:00 | 133.40 | 2024-06-13 14:15:00 | 133.12 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2024-06-26 14:15:00 | 130.45 | 2024-06-27 09:15:00 | 132.62 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-07-23 12:15:00 | 133.70 | 2024-07-23 14:15:00 | 147.75 | STOP_HIT | 1.00 | -10.51% |
| SELL | retest2 | 2024-08-06 14:00:00 | 137.89 | 2024-08-09 12:15:00 | 140.00 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-08-07 15:00:00 | 139.85 | 2024-08-09 12:15:00 | 140.00 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2024-08-08 09:45:00 | 139.70 | 2024-08-09 12:15:00 | 140.00 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2024-08-08 10:30:00 | 140.04 | 2024-08-09 12:15:00 | 140.00 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2024-08-09 10:15:00 | 140.01 | 2024-08-09 12:15:00 | 140.00 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2024-08-09 11:45:00 | 140.25 | 2024-08-09 12:15:00 | 140.00 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2024-08-29 13:15:00 | 133.49 | 2024-08-29 15:15:00 | 133.75 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2024-08-29 13:45:00 | 133.26 | 2024-08-29 15:15:00 | 133.75 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2024-09-17 12:15:00 | 130.01 | 2024-09-20 14:15:00 | 130.20 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2024-09-17 14:15:00 | 129.83 | 2024-09-20 14:15:00 | 130.20 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2024-09-18 11:30:00 | 129.94 | 2024-09-20 14:15:00 | 130.20 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2024-09-18 12:45:00 | 129.98 | 2024-09-20 14:15:00 | 130.20 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2024-09-19 10:15:00 | 128.23 | 2024-09-20 14:15:00 | 130.20 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-09-19 15:15:00 | 128.00 | 2024-09-20 14:15:00 | 130.20 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-09-20 10:00:00 | 128.47 | 2024-09-20 14:15:00 | 130.20 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2024-10-09 12:00:00 | 122.00 | 2024-10-15 10:15:00 | 121.88 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2024-10-10 09:30:00 | 121.97 | 2024-10-15 10:15:00 | 121.88 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2024-10-15 09:45:00 | 121.81 | 2024-10-15 10:15:00 | 121.88 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2024-11-12 12:00:00 | 110.43 | 2024-11-13 12:15:00 | 104.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 12:00:00 | 110.43 | 2024-11-14 09:15:00 | 106.15 | STOP_HIT | 0.50 | 3.88% |
| BUY | retest2 | 2024-12-06 10:30:00 | 121.80 | 2024-12-10 15:15:00 | 119.30 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2024-12-09 09:15:00 | 122.46 | 2024-12-10 15:15:00 | 119.30 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2024-12-16 12:45:00 | 117.75 | 2024-12-19 09:15:00 | 111.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 14:30:00 | 117.65 | 2024-12-19 09:15:00 | 111.83 | PARTIAL | 0.50 | 4.94% |
| SELL | retest2 | 2024-12-17 11:30:00 | 117.72 | 2024-12-19 09:15:00 | 111.88 | PARTIAL | 0.50 | 4.96% |
| SELL | retest2 | 2024-12-16 12:45:00 | 117.75 | 2024-12-20 11:15:00 | 114.82 | STOP_HIT | 0.50 | 2.49% |
| SELL | retest2 | 2024-12-16 14:30:00 | 117.65 | 2024-12-20 11:15:00 | 114.82 | STOP_HIT | 0.50 | 2.41% |
| SELL | retest2 | 2024-12-17 11:30:00 | 117.72 | 2024-12-20 11:15:00 | 114.82 | STOP_HIT | 0.50 | 2.46% |
| SELL | retest2 | 2024-12-17 14:15:00 | 117.77 | 2024-12-20 14:15:00 | 111.77 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2024-12-18 09:15:00 | 116.90 | 2024-12-20 15:15:00 | 111.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 14:15:00 | 117.77 | 2024-12-24 09:15:00 | 110.93 | STOP_HIT | 0.50 | 5.81% |
| SELL | retest2 | 2024-12-18 09:15:00 | 116.90 | 2024-12-24 09:15:00 | 110.93 | STOP_HIT | 0.50 | 5.11% |
| SELL | retest2 | 2025-01-23 14:15:00 | 97.31 | 2025-01-27 09:15:00 | 92.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 09:15:00 | 97.87 | 2025-01-27 09:15:00 | 92.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 14:15:00 | 97.31 | 2025-01-28 12:15:00 | 91.98 | STOP_HIT | 0.50 | 5.48% |
| SELL | retest2 | 2025-01-24 09:15:00 | 97.87 | 2025-01-28 12:15:00 | 91.98 | STOP_HIT | 0.50 | 6.02% |
| SELL | retest2 | 2025-02-14 10:45:00 | 90.22 | 2025-02-19 11:15:00 | 90.51 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-02-19 10:00:00 | 90.37 | 2025-02-19 11:15:00 | 90.51 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-02-19 10:30:00 | 90.42 | 2025-02-19 11:15:00 | 90.51 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-03-04 11:15:00 | 83.16 | 2025-03-05 09:15:00 | 85.50 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-03-04 14:30:00 | 83.05 | 2025-03-05 09:15:00 | 85.50 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest1 | 2025-03-07 13:45:00 | 87.82 | 2025-03-10 11:15:00 | 87.08 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest1 | 2025-03-10 09:30:00 | 88.05 | 2025-03-10 11:15:00 | 87.08 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest1 | 2025-03-10 10:45:00 | 87.86 | 2025-03-10 11:15:00 | 87.08 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-03-26 09:15:00 | 94.47 | 2025-03-26 11:15:00 | 93.40 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-04-02 09:15:00 | 90.64 | 2025-04-02 12:15:00 | 92.81 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2025-04-08 10:30:00 | 89.77 | 2025-04-11 11:15:00 | 91.57 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-04-09 09:15:00 | 89.50 | 2025-04-11 11:15:00 | 91.57 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-04-09 14:00:00 | 89.85 | 2025-04-11 11:15:00 | 91.57 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-04-17 11:15:00 | 95.06 | 2025-04-25 10:15:00 | 94.44 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-04-21 09:30:00 | 95.11 | 2025-04-25 10:15:00 | 94.44 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-05-02 11:15:00 | 94.57 | 2025-05-07 09:15:00 | 89.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-02 11:15:00 | 94.57 | 2025-05-07 11:15:00 | 91.75 | STOP_HIT | 0.50 | 2.98% |
| BUY | retest2 | 2025-05-27 12:30:00 | 103.26 | 2025-05-30 09:15:00 | 97.71 | STOP_HIT | 1.00 | -5.37% |
| BUY | retest2 | 2025-05-28 10:30:00 | 103.36 | 2025-05-30 09:15:00 | 97.71 | STOP_HIT | 1.00 | -5.47% |
| BUY | retest2 | 2025-06-30 14:00:00 | 100.32 | 2025-07-01 10:15:00 | 99.36 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-07-03 13:15:00 | 98.77 | 2025-07-09 09:15:00 | 99.00 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-07-07 09:45:00 | 99.10 | 2025-07-09 09:15:00 | 99.00 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-07-07 11:45:00 | 99.02 | 2025-07-09 09:15:00 | 99.00 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2025-07-08 10:30:00 | 98.99 | 2025-07-09 09:15:00 | 99.00 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-07-11 14:15:00 | 97.96 | 2025-07-14 15:15:00 | 98.45 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-07-14 09:45:00 | 97.97 | 2025-07-14 15:15:00 | 98.45 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-07-14 11:15:00 | 97.89 | 2025-07-14 15:15:00 | 98.45 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-07-16 13:15:00 | 98.80 | 2025-07-18 09:15:00 | 98.29 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-07-16 14:15:00 | 98.76 | 2025-07-18 09:15:00 | 98.29 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-07-21 10:30:00 | 99.40 | 2025-07-23 10:15:00 | 98.76 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-07-21 11:45:00 | 99.25 | 2025-07-23 10:15:00 | 98.76 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-07-21 15:15:00 | 99.18 | 2025-07-23 10:15:00 | 98.76 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-07-23 09:45:00 | 99.14 | 2025-07-23 10:15:00 | 98.76 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-07-24 11:00:00 | 98.18 | 2025-07-30 11:15:00 | 97.32 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2025-07-24 12:15:00 | 98.27 | 2025-07-30 11:15:00 | 97.32 | STOP_HIT | 1.00 | 0.97% |
| SELL | retest2 | 2025-07-25 09:30:00 | 97.80 | 2025-07-30 11:15:00 | 97.32 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2025-08-05 12:45:00 | 93.43 | 2025-08-08 10:15:00 | 93.19 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2025-08-08 09:30:00 | 93.46 | 2025-08-08 10:15:00 | 93.19 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-08-12 09:15:00 | 96.82 | 2025-08-13 14:15:00 | 93.42 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest1 | 2025-08-22 09:15:00 | 97.96 | 2025-08-26 14:15:00 | 99.30 | STOP_HIT | 1.00 | 1.37% |
| SELL | retest2 | 2025-09-04 12:15:00 | 93.61 | 2025-09-10 15:15:00 | 94.20 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-09-04 13:00:00 | 93.41 | 2025-09-10 15:15:00 | 94.20 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-09-10 09:45:00 | 93.39 | 2025-09-10 15:15:00 | 94.20 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-09-12 09:15:00 | 94.15 | 2025-09-12 11:15:00 | 93.27 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-09-25 12:00:00 | 92.27 | 2025-10-03 09:15:00 | 91.38 | STOP_HIT | 1.00 | 0.96% |
| SELL | retest2 | 2025-09-25 12:45:00 | 92.28 | 2025-10-03 09:15:00 | 91.38 | STOP_HIT | 1.00 | 0.98% |
| SELL | retest2 | 2025-10-16 11:00:00 | 89.26 | 2025-10-17 14:15:00 | 90.01 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-10-28 12:15:00 | 88.10 | 2025-10-29 09:15:00 | 89.95 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-10-28 13:00:00 | 88.07 | 2025-10-29 09:15:00 | 89.95 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-10-28 13:30:00 | 88.05 | 2025-10-29 09:15:00 | 89.95 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-11-03 10:45:00 | 88.26 | 2025-11-10 13:15:00 | 83.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 10:45:00 | 88.26 | 2025-11-12 11:15:00 | 83.00 | STOP_HIT | 0.50 | 5.96% |
| SELL | retest2 | 2025-11-20 15:15:00 | 81.75 | 2025-12-01 12:15:00 | 77.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 15:15:00 | 81.75 | 2025-12-02 10:15:00 | 78.32 | STOP_HIT | 0.50 | 4.20% |
| SELL | retest2 | 2026-01-13 14:15:00 | 79.14 | 2026-01-16 15:15:00 | 75.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 09:15:00 | 79.07 | 2026-01-19 11:15:00 | 75.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 14:15:00 | 79.14 | 2026-01-20 15:15:00 | 71.23 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-14 09:15:00 | 79.07 | 2026-01-20 15:15:00 | 71.16 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-01-30 09:30:00 | 72.95 | 2026-01-30 12:15:00 | 71.88 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2026-02-16 11:15:00 | 77.48 | 2026-02-18 12:15:00 | 76.61 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2026-02-16 13:15:00 | 77.20 | 2026-02-18 12:15:00 | 76.61 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2026-02-16 13:45:00 | 77.39 | 2026-02-18 12:15:00 | 76.61 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2026-03-05 11:15:00 | 69.95 | 2026-03-05 14:15:00 | 72.85 | STOP_HIT | 1.00 | -4.15% |
| BUY | retest2 | 2026-04-13 10:15:00 | 73.09 | 2026-04-22 14:15:00 | 80.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 13:30:00 | 72.92 | 2026-04-22 14:15:00 | 80.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-15 09:15:00 | 73.83 | 2026-04-22 14:15:00 | 81.21 | TARGET_HIT | 1.00 | 10.00% |
