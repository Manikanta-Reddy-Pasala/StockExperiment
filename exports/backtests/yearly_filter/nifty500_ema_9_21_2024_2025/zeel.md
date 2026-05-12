# Zee Entertainment Enterprises Ltd. (ZEEL)

## Backtest Summary

- **Window:** 2024-03-13 10:15:00 → 2026-05-08 15:15:00 (3709 bars)
- **Last close:** 95.22
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 137 |
| ALERT1 | 94 |
| ALERT2 | 93 |
| ALERT2_SKIP | 40 |
| ALERT3 | 250 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 111 |
| PARTIAL | 23 |
| TARGET_HIT | 9 |
| STOP_HIT | 107 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 136 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 56 / 80
- **Target hits / Stop hits / Partials:** 9 / 104 / 23
- **Avg / median % per leg:** 0.86% / -0.74%
- **Sum % (uncompounded):** 117.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 42 | 2 | 4.8% | 1 | 41 | 0 | -1.18% | -49.5% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.95% | -2.0% |
| BUY @ 3rd Alert (retest2) | 41 | 2 | 4.9% | 1 | 40 | 0 | -1.16% | -47.5% |
| SELL (all) | 94 | 54 | 57.4% | 8 | 63 | 23 | 1.78% | 166.9% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.07% | 0.1% |
| SELL @ 3rd Alert (retest2) | 93 | 53 | 57.0% | 8 | 62 | 23 | 1.79% | 166.8% |
| retest1 (combined) | 2 | 1 | 50.0% | 0 | 2 | 0 | -0.94% | -1.9% |
| retest2 (combined) | 134 | 55 | 41.0% | 9 | 102 | 23 | 0.89% | 119.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 13:15:00 | 132.55 | 132.05 | 132.05 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 14:15:00 | 131.40 | 131.92 | 131.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 15:15:00 | 130.45 | 131.63 | 131.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 09:15:00 | 132.30 | 131.76 | 131.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 09:15:00 | 132.30 | 131.76 | 131.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 132.30 | 131.76 | 131.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 09:45:00 | 132.40 | 131.76 | 131.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 10:15:00 | 131.05 | 131.62 | 131.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 10:45:00 | 131.85 | 131.62 | 131.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 132.55 | 131.69 | 131.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 14:00:00 | 132.55 | 131.69 | 131.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 14:15:00 | 133.45 | 132.04 | 131.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 09:15:00 | 134.70 | 132.73 | 132.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 14:15:00 | 148.55 | 149.61 | 147.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-23 14:45:00 | 148.40 | 149.61 | 147.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 151.20 | 151.27 | 149.67 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2024-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 11:15:00 | 147.25 | 149.18 | 149.39 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 14:15:00 | 151.20 | 149.52 | 149.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 09:15:00 | 153.20 | 150.44 | 149.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 14:15:00 | 151.15 | 151.34 | 150.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-29 15:00:00 | 151.15 | 151.34 | 150.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 15:15:00 | 150.85 | 151.24 | 150.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:15:00 | 151.95 | 151.24 | 150.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 151.95 | 151.38 | 150.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-30 10:15:00 | 153.40 | 151.38 | 150.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-30 12:30:00 | 152.65 | 152.29 | 151.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-30 14:00:00 | 152.80 | 152.39 | 151.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 09:15:00 | 152.85 | 152.21 | 151.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 150.25 | 151.82 | 151.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 10:00:00 | 150.25 | 151.82 | 151.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-05-31 10:15:00 | 148.80 | 151.21 | 151.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 10:15:00 | 148.80 | 151.21 | 151.23 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 156.15 | 151.75 | 151.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 11:15:00 | 157.25 | 152.85 | 151.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 152.70 | 154.66 | 153.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 152.70 | 154.66 | 153.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 152.70 | 154.66 | 153.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 150.40 | 154.66 | 153.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 141.90 | 152.11 | 152.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 140.70 | 149.83 | 151.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 144.15 | 143.77 | 146.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 11:00:00 | 144.15 | 143.77 | 146.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 145.60 | 144.30 | 146.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:45:00 | 146.30 | 144.30 | 146.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 144.85 | 144.41 | 146.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:45:00 | 145.00 | 144.41 | 146.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 148.25 | 145.18 | 146.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:15:00 | 152.75 | 145.18 | 146.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 154.30 | 147.00 | 147.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:45:00 | 153.40 | 147.00 | 147.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 153.90 | 148.38 | 147.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 10:15:00 | 156.30 | 153.24 | 151.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 12:15:00 | 164.73 | 164.96 | 162.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 12:45:00 | 164.59 | 164.96 | 162.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 164.27 | 165.16 | 163.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 11:45:00 | 167.01 | 165.16 | 163.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 14:30:00 | 165.84 | 165.10 | 164.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-14 10:15:00 | 163.21 | 164.59 | 164.05 | SL hit (close<static) qty=1.00 sl=163.55 alert=retest2 |

### Cycle 10 — SELL (started 2024-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 10:15:00 | 162.90 | 163.69 | 163.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 11:15:00 | 161.69 | 163.29 | 163.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 13:15:00 | 156.42 | 156.31 | 158.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-20 14:00:00 | 156.42 | 156.31 | 158.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 154.75 | 150.97 | 151.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 12:00:00 | 154.75 | 150.97 | 151.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 155.75 | 151.93 | 152.22 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2024-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 13:15:00 | 155.80 | 152.70 | 152.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 09:15:00 | 158.25 | 154.59 | 153.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 11:15:00 | 155.05 | 155.21 | 154.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-27 12:00:00 | 155.05 | 155.21 | 154.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 151.35 | 154.31 | 153.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:00:00 | 151.35 | 154.31 | 153.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 151.57 | 153.76 | 153.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 151.57 | 153.76 | 153.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2024-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 15:15:00 | 151.00 | 153.21 | 153.37 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 13:15:00 | 153.61 | 153.04 | 152.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 10:15:00 | 156.00 | 153.72 | 153.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 12:15:00 | 153.27 | 153.82 | 153.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 12:15:00 | 153.27 | 153.82 | 153.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 153.27 | 153.82 | 153.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 153.27 | 153.82 | 153.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 153.71 | 153.80 | 153.47 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2024-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 10:15:00 | 152.80 | 153.30 | 153.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-03 11:15:00 | 151.60 | 152.96 | 153.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-05 09:15:00 | 150.73 | 150.36 | 151.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 09:15:00 | 150.73 | 150.36 | 151.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 150.73 | 150.36 | 151.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 12:00:00 | 149.67 | 150.17 | 150.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 11:00:00 | 149.69 | 150.53 | 150.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-09 09:15:00 | 152.38 | 150.67 | 150.70 | SL hit (close>static) qty=1.00 sl=152.10 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 10:15:00 | 152.50 | 151.04 | 150.86 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 147.65 | 150.40 | 150.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 10:15:00 | 146.70 | 149.66 | 150.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 15:15:00 | 147.30 | 147.18 | 148.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-12 09:15:00 | 151.39 | 147.18 | 148.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 17 — BUY (started 2024-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 09:15:00 | 155.11 | 148.76 | 148.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 11:15:00 | 157.32 | 151.72 | 150.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 11:15:00 | 157.77 | 158.17 | 155.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-16 12:00:00 | 157.77 | 158.17 | 155.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 156.23 | 157.68 | 156.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:00:00 | 156.23 | 157.68 | 156.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 155.75 | 157.30 | 156.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:45:00 | 155.10 | 157.30 | 156.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 155.41 | 156.92 | 156.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 145.65 | 156.92 | 156.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 09:15:00 | 145.74 | 154.68 | 155.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 10:15:00 | 142.75 | 152.30 | 153.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 14:15:00 | 134.90 | 134.86 | 137.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-23 14:30:00 | 134.69 | 134.86 | 137.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 137.30 | 135.34 | 137.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:45:00 | 138.01 | 135.34 | 137.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 137.73 | 135.82 | 137.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:45:00 | 138.03 | 135.82 | 137.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 11:15:00 | 137.70 | 136.19 | 137.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 12:15:00 | 137.30 | 136.19 | 137.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 14:30:00 | 136.70 | 136.61 | 137.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 09:45:00 | 136.97 | 136.99 | 137.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-25 10:15:00 | 141.95 | 137.98 | 137.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 10:15:00 | 141.95 | 137.98 | 137.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 12:15:00 | 143.50 | 141.55 | 140.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 09:15:00 | 147.00 | 147.59 | 146.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-01 09:45:00 | 146.62 | 147.59 | 146.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 144.93 | 147.06 | 146.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:45:00 | 144.16 | 147.06 | 146.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 144.06 | 146.46 | 145.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:00:00 | 144.06 | 146.46 | 145.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2024-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 13:15:00 | 142.64 | 145.28 | 145.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 14:15:00 | 141.65 | 144.56 | 145.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 13:15:00 | 143.60 | 142.86 | 143.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 13:15:00 | 143.60 | 142.86 | 143.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 13:15:00 | 143.60 | 142.86 | 143.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 13:45:00 | 144.33 | 142.86 | 143.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 14:15:00 | 142.67 | 142.82 | 143.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 09:15:00 | 137.50 | 142.80 | 143.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 10:15:00 | 140.91 | 138.02 | 137.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 10:15:00 | 140.91 | 138.02 | 137.92 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 15:15:00 | 137.65 | 138.36 | 138.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 09:15:00 | 136.75 | 138.04 | 138.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-13 10:15:00 | 136.97 | 136.91 | 137.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-13 11:00:00 | 136.97 | 136.91 | 137.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 11:15:00 | 136.93 | 136.91 | 137.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 11:45:00 | 137.32 | 136.91 | 137.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 135.10 | 134.85 | 135.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:45:00 | 135.69 | 134.85 | 135.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 13:15:00 | 135.20 | 134.99 | 135.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 13:30:00 | 135.34 | 134.99 | 135.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 135.85 | 135.01 | 135.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:30:00 | 135.78 | 135.01 | 135.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 10:15:00 | 135.75 | 135.16 | 135.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 12:15:00 | 135.34 | 135.25 | 135.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 12:45:00 | 135.39 | 135.28 | 135.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 09:45:00 | 135.42 | 135.21 | 135.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 15:00:00 | 135.33 | 135.30 | 135.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 15:15:00 | 135.23 | 135.29 | 135.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:15:00 | 138.89 | 135.29 | 135.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-21 09:15:00 | 138.67 | 135.97 | 135.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2024-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 09:15:00 | 138.67 | 135.97 | 135.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 10:15:00 | 140.50 | 136.87 | 136.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 14:15:00 | 139.21 | 139.57 | 138.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 15:00:00 | 139.21 | 139.57 | 138.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 138.15 | 139.32 | 138.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:45:00 | 138.60 | 139.32 | 138.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 138.21 | 139.09 | 138.57 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2024-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 14:15:00 | 136.64 | 138.03 | 138.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 10:15:00 | 135.96 | 137.27 | 137.77 | Break + close below crossover candle low |

### Cycle 25 — BUY (started 2024-08-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 14:15:00 | 150.11 | 138.61 | 137.68 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-08-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 15:15:00 | 140.55 | 142.55 | 142.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 09:15:00 | 139.44 | 141.41 | 141.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 10:15:00 | 136.91 | 136.74 | 138.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 10:15:00 | 136.91 | 136.74 | 138.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 136.91 | 136.74 | 138.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 13:30:00 | 136.22 | 136.51 | 137.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 09:30:00 | 135.80 | 135.82 | 136.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 11:15:00 | 140.36 | 135.83 | 135.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2024-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 11:15:00 | 140.36 | 135.83 | 135.56 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 09:15:00 | 135.55 | 136.46 | 136.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-12 10:15:00 | 134.78 | 136.13 | 136.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-13 09:15:00 | 136.65 | 135.52 | 135.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-13 09:15:00 | 136.65 | 135.52 | 135.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 136.65 | 135.52 | 135.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:30:00 | 137.34 | 135.52 | 135.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 12:15:00 | 136.50 | 135.99 | 136.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 12:45:00 | 136.55 | 135.99 | 136.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2024-09-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 13:15:00 | 138.00 | 136.39 | 136.21 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 10:15:00 | 135.22 | 136.01 | 136.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 14:15:00 | 134.85 | 135.41 | 135.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 14:15:00 | 128.34 | 127.66 | 129.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 14:15:00 | 128.34 | 127.66 | 129.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 128.34 | 127.66 | 129.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 14:15:00 | 127.24 | 127.60 | 128.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 14:45:00 | 127.19 | 127.56 | 128.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 10:15:00 | 134.10 | 128.81 | 128.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2024-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 10:15:00 | 134.10 | 128.81 | 128.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-25 15:15:00 | 134.65 | 131.81 | 130.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 12:15:00 | 135.78 | 136.14 | 134.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-27 13:00:00 | 135.78 | 136.14 | 134.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 134.16 | 135.51 | 134.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:30:00 | 133.75 | 135.51 | 134.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 134.26 | 135.26 | 134.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:30:00 | 133.61 | 135.26 | 134.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 137.40 | 135.77 | 135.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 13:30:00 | 135.20 | 135.77 | 135.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 141.90 | 139.39 | 137.77 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2024-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 09:15:00 | 135.87 | 137.45 | 137.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 14:15:00 | 131.90 | 135.04 | 136.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 128.95 | 128.34 | 130.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 11:00:00 | 128.95 | 128.34 | 130.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 12:15:00 | 130.70 | 128.36 | 129.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 13:00:00 | 130.70 | 128.36 | 129.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 13:15:00 | 130.69 | 128.83 | 129.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 14:15:00 | 129.49 | 128.83 | 129.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 11:00:00 | 129.82 | 129.41 | 129.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 13:15:00 | 129.74 | 129.60 | 129.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-10 13:15:00 | 129.75 | 129.63 | 129.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 13:15:00 | 129.75 | 129.63 | 129.62 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 14:15:00 | 129.22 | 129.55 | 129.59 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 09:15:00 | 130.60 | 129.77 | 129.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 15:15:00 | 131.22 | 130.44 | 130.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 09:15:00 | 129.63 | 130.28 | 130.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 09:15:00 | 129.63 | 130.28 | 130.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 129.63 | 130.28 | 130.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:00:00 | 129.63 | 130.28 | 130.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 130.05 | 130.23 | 130.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 11:45:00 | 130.80 | 130.28 | 130.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 12:15:00 | 129.22 | 130.07 | 130.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 12:15:00 | 129.22 | 130.07 | 130.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 09:15:00 | 128.80 | 129.63 | 129.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 13:15:00 | 127.09 | 126.68 | 127.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 14:15:00 | 127.13 | 126.68 | 127.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 132.50 | 127.85 | 127.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 15:00:00 | 132.50 | 127.85 | 127.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2024-10-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 15:15:00 | 130.50 | 128.38 | 128.19 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 14:15:00 | 126.27 | 128.09 | 128.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 125.90 | 127.32 | 127.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 09:15:00 | 126.00 | 124.46 | 125.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-24 09:15:00 | 126.00 | 124.46 | 125.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 126.00 | 124.46 | 125.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:00:00 | 126.00 | 124.46 | 125.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 124.93 | 124.55 | 125.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 11:45:00 | 124.88 | 124.62 | 125.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 12:30:00 | 124.67 | 124.63 | 125.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 14:15:00 | 124.38 | 124.71 | 125.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 10:15:00 | 118.64 | 120.38 | 121.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 10:15:00 | 118.44 | 120.38 | 121.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 10:15:00 | 118.16 | 120.38 | 121.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-30 09:15:00 | 121.25 | 119.74 | 120.52 | SL hit (close>ema200) qty=0.50 sl=119.74 alert=retest2 |

### Cycle 39 — BUY (started 2024-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 14:15:00 | 121.98 | 121.06 | 120.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 15:15:00 | 122.25 | 121.30 | 121.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 12:15:00 | 121.32 | 121.52 | 121.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 12:15:00 | 121.32 | 121.52 | 121.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 121.32 | 121.52 | 121.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:45:00 | 121.22 | 121.52 | 121.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 13:15:00 | 121.05 | 121.42 | 121.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 14:00:00 | 121.05 | 121.42 | 121.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 122.50 | 121.64 | 121.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 14:30:00 | 121.33 | 121.64 | 121.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 120.54 | 121.98 | 121.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 120.54 | 121.98 | 121.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 120.31 | 121.65 | 121.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 120.31 | 121.65 | 121.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2024-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 11:15:00 | 120.61 | 121.44 | 121.46 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-11-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 13:15:00 | 121.91 | 121.41 | 121.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 122.37 | 121.61 | 121.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 09:15:00 | 124.04 | 124.47 | 123.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 09:15:00 | 124.04 | 124.47 | 123.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 124.04 | 124.47 | 123.62 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2024-11-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 15:15:00 | 122.48 | 123.25 | 123.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 121.56 | 122.91 | 123.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 11:15:00 | 116.30 | 116.29 | 117.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 11:30:00 | 115.90 | 116.29 | 117.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 43 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 123.10 | 117.19 | 117.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 10:15:00 | 125.35 | 118.82 | 117.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-21 09:15:00 | 119.21 | 121.23 | 119.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 09:15:00 | 119.21 | 121.23 | 119.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 119.21 | 121.23 | 119.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:00:00 | 119.21 | 121.23 | 119.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 119.68 | 120.92 | 119.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 12:15:00 | 119.85 | 120.63 | 119.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 13:30:00 | 120.00 | 120.33 | 119.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-21 15:15:00 | 117.99 | 119.61 | 119.54 | SL hit (close<static) qty=1.00 sl=118.51 alert=retest2 |

### Cycle 44 — SELL (started 2024-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 09:15:00 | 118.37 | 119.36 | 119.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-22 15:15:00 | 116.95 | 118.23 | 118.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 119.42 | 118.47 | 118.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 09:15:00 | 119.42 | 118.47 | 118.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 119.42 | 118.47 | 118.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 12:45:00 | 118.40 | 118.75 | 118.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-26 11:15:00 | 121.30 | 119.32 | 119.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2024-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 11:15:00 | 121.30 | 119.32 | 119.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 11:15:00 | 122.97 | 121.32 | 120.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 09:15:00 | 141.58 | 142.47 | 140.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-09 09:15:00 | 141.58 | 142.47 | 140.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 141.58 | 142.47 | 140.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:00:00 | 141.58 | 142.47 | 140.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 14:15:00 | 141.30 | 141.80 | 141.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 14:30:00 | 141.00 | 141.80 | 141.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 15:15:00 | 140.86 | 141.61 | 141.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:15:00 | 141.25 | 141.61 | 141.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 141.11 | 141.51 | 141.02 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2024-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 14:15:00 | 139.74 | 140.63 | 140.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 09:15:00 | 138.79 | 140.09 | 140.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 09:15:00 | 137.30 | 134.02 | 135.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 09:15:00 | 137.30 | 134.02 | 135.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 137.30 | 134.02 | 135.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 137.74 | 134.02 | 135.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 135.75 | 134.37 | 135.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 13:15:00 | 135.60 | 134.90 | 135.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 14:45:00 | 135.66 | 135.21 | 135.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 09:15:00 | 136.95 | 135.68 | 135.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 09:15:00 | 136.95 | 135.68 | 135.63 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 09:15:00 | 133.12 | 135.36 | 135.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 11:15:00 | 132.10 | 134.34 | 135.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 09:15:00 | 125.28 | 125.24 | 126.04 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 15:15:00 | 124.44 | 125.15 | 125.69 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 124.35 | 122.27 | 123.00 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-01 09:15:00 | 124.35 | 122.27 | 123.00 | SL hit (close>ema400) qty=1.00 sl=123.00 alert=retest1 |

### Cycle 49 — BUY (started 2025-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 10:15:00 | 124.49 | 123.38 | 123.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 09:15:00 | 128.25 | 124.67 | 123.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 14:15:00 | 125.78 | 126.08 | 125.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 15:00:00 | 125.78 | 126.08 | 125.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 125.38 | 125.94 | 125.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 125.10 | 125.94 | 125.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 124.90 | 125.73 | 125.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:00:00 | 124.90 | 125.73 | 125.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 123.66 | 125.32 | 124.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 123.66 | 125.32 | 124.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 123.70 | 125.00 | 124.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 123.77 | 125.00 | 124.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 12:15:00 | 123.62 | 124.72 | 124.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 15:15:00 | 122.30 | 123.81 | 124.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 124.47 | 123.94 | 124.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 124.47 | 123.94 | 124.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 124.47 | 123.94 | 124.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 125.22 | 123.94 | 124.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 124.92 | 124.14 | 124.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:00:00 | 124.92 | 124.14 | 124.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 124.15 | 124.14 | 124.34 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 13:15:00 | 125.48 | 124.64 | 124.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-07 14:15:00 | 129.61 | 125.64 | 125.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 14:15:00 | 133.00 | 133.26 | 131.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-09 14:30:00 | 132.85 | 133.26 | 131.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 127.61 | 131.97 | 131.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 127.61 | 131.97 | 131.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 129.01 | 131.38 | 130.90 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2025-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 12:15:00 | 128.40 | 130.32 | 130.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 13:15:00 | 127.20 | 129.70 | 130.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 122.60 | 122.43 | 124.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 122.60 | 122.43 | 124.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 124.36 | 122.86 | 124.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 14:45:00 | 124.63 | 122.86 | 124.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 124.20 | 123.13 | 124.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:15:00 | 124.30 | 123.13 | 124.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 123.86 | 123.27 | 124.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 13:00:00 | 123.29 | 123.48 | 124.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 13:15:00 | 123.33 | 123.41 | 123.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 14:45:00 | 123.10 | 123.29 | 123.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 09:30:00 | 122.83 | 123.08 | 123.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 120.99 | 121.84 | 122.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 10:45:00 | 120.76 | 121.29 | 121.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 12:15:00 | 120.58 | 121.20 | 121.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 13:45:00 | 120.75 | 121.04 | 121.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 11:15:00 | 117.13 | 119.42 | 120.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 11:15:00 | 117.16 | 119.42 | 120.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 11:15:00 | 116.94 | 119.42 | 120.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 11:15:00 | 116.69 | 119.42 | 120.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-23 09:15:00 | 119.62 | 118.67 | 119.63 | SL hit (close>ema200) qty=0.50 sl=118.67 alert=retest2 |

### Cycle 53 — BUY (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 13:15:00 | 107.70 | 106.42 | 106.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 14:15:00 | 108.95 | 106.93 | 106.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 09:15:00 | 107.18 | 107.29 | 106.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 09:15:00 | 107.18 | 107.29 | 106.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 107.18 | 107.29 | 106.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 12:00:00 | 108.42 | 107.47 | 107.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 12:45:00 | 108.39 | 107.62 | 107.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 09:15:00 | 108.50 | 107.73 | 107.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 15:15:00 | 109.50 | 109.12 | 108.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 109.50 | 109.19 | 108.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:15:00 | 107.99 | 109.19 | 108.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 108.54 | 109.06 | 108.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:15:00 | 107.80 | 109.06 | 108.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 108.53 | 108.96 | 108.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:30:00 | 109.09 | 108.87 | 108.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 12:45:00 | 110.02 | 109.08 | 108.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 14:45:00 | 108.88 | 109.10 | 108.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 107.07 | 108.74 | 108.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 107.07 | 108.74 | 108.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 10:15:00 | 106.66 | 108.32 | 108.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 101.47 | 100.33 | 101.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 101.47 | 100.33 | 101.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 101.47 | 100.33 | 101.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 101.83 | 100.33 | 101.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 12:15:00 | 101.20 | 100.77 | 101.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 12:45:00 | 101.55 | 100.77 | 101.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 96.59 | 97.25 | 98.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:30:00 | 98.49 | 97.25 | 98.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 98.61 | 97.32 | 98.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 10:00:00 | 98.61 | 97.32 | 98.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 10:15:00 | 97.87 | 97.43 | 98.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 10:45:00 | 98.47 | 97.43 | 98.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 11:15:00 | 97.82 | 97.51 | 98.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 11:45:00 | 97.81 | 97.51 | 98.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 13:15:00 | 98.10 | 97.49 | 97.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 13:45:00 | 97.87 | 97.49 | 97.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 98.22 | 97.63 | 97.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 14:30:00 | 98.95 | 97.63 | 97.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 100.35 | 98.25 | 98.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 15:15:00 | 100.50 | 99.83 | 99.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 10:15:00 | 99.34 | 99.78 | 99.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 10:15:00 | 99.34 | 99.78 | 99.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 99.34 | 99.78 | 99.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 99.34 | 99.78 | 99.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 98.41 | 99.51 | 99.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 12:00:00 | 98.41 | 99.51 | 99.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 12:15:00 | 98.15 | 99.24 | 99.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:00:00 | 98.15 | 99.24 | 99.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2025-02-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 13:15:00 | 97.72 | 98.93 | 99.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 15:15:00 | 97.51 | 98.55 | 98.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 99.27 | 97.40 | 97.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 09:15:00 | 99.27 | 97.40 | 97.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 99.27 | 97.40 | 97.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 10:00:00 | 99.27 | 97.40 | 97.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 98.84 | 97.69 | 97.95 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2025-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 12:15:00 | 99.35 | 98.27 | 98.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-25 14:15:00 | 99.74 | 98.73 | 98.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-27 10:15:00 | 98.60 | 98.99 | 98.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 10:15:00 | 98.60 | 98.99 | 98.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 10:15:00 | 98.60 | 98.99 | 98.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 11:00:00 | 98.60 | 98.99 | 98.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 11:15:00 | 97.06 | 98.60 | 98.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 12:00:00 | 97.06 | 98.60 | 98.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 12:15:00 | 97.15 | 98.31 | 98.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 94.25 | 97.06 | 97.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 91.83 | 91.61 | 93.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-04 10:00:00 | 91.83 | 91.61 | 93.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 92.36 | 91.79 | 92.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 14:30:00 | 92.85 | 91.79 | 92.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 93.52 | 92.26 | 92.81 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 98.12 | 94.00 | 93.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 100.04 | 97.18 | 95.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 14:15:00 | 98.25 | 98.39 | 96.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 15:00:00 | 98.25 | 98.39 | 96.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 104.80 | 105.60 | 104.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 11:00:00 | 104.80 | 105.60 | 104.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 103.85 | 105.25 | 104.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 12:00:00 | 103.85 | 105.25 | 104.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 103.72 | 104.95 | 104.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 12:45:00 | 103.47 | 104.95 | 104.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 104.80 | 104.82 | 104.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 15:00:00 | 104.80 | 104.82 | 104.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 104.33 | 104.72 | 104.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:15:00 | 102.99 | 104.72 | 104.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 104.15 | 104.61 | 104.47 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 10:15:00 | 102.93 | 104.27 | 104.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 11:15:00 | 102.20 | 103.86 | 104.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 100.48 | 100.37 | 101.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-18 09:45:00 | 100.30 | 100.37 | 101.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 12:15:00 | 102.49 | 100.92 | 101.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 13:00:00 | 102.49 | 100.92 | 101.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 13:15:00 | 102.31 | 101.20 | 101.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 14:45:00 | 102.17 | 101.55 | 101.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-19 09:15:00 | 102.97 | 102.00 | 101.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 102.97 | 102.00 | 101.88 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-19 14:15:00 | 100.20 | 101.80 | 101.89 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 09:15:00 | 104.19 | 102.03 | 101.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-21 10:15:00 | 107.60 | 104.17 | 103.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 14:15:00 | 105.30 | 105.85 | 105.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 15:00:00 | 105.30 | 105.85 | 105.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 15:15:00 | 105.36 | 105.75 | 105.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 09:30:00 | 104.77 | 105.40 | 104.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 103.84 | 105.09 | 104.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:45:00 | 103.95 | 105.09 | 104.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2025-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 12:15:00 | 103.35 | 104.55 | 104.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 14:15:00 | 102.70 | 104.05 | 104.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 09:15:00 | 101.49 | 101.47 | 102.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 09:15:00 | 101.49 | 101.47 | 102.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 101.49 | 101.47 | 102.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 11:30:00 | 100.16 | 101.12 | 101.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:00:00 | 100.09 | 99.86 | 100.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 09:15:00 | 100.15 | 100.76 | 100.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 11:15:00 | 101.18 | 100.92 | 100.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 11:15:00 | 101.18 | 100.92 | 100.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 12:15:00 | 101.88 | 101.11 | 101.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 11:15:00 | 104.79 | 105.10 | 103.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-04 12:00:00 | 104.79 | 105.10 | 103.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 13:15:00 | 103.09 | 104.56 | 103.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 14:00:00 | 103.09 | 104.56 | 103.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 14:15:00 | 104.99 | 104.65 | 103.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 14:30:00 | 102.37 | 104.65 | 103.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 95.60 | 102.83 | 103.15 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 14:15:00 | 107.54 | 101.19 | 100.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 14:15:00 | 108.41 | 106.34 | 105.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 09:15:00 | 116.66 | 117.13 | 114.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 10:00:00 | 116.66 | 117.13 | 114.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 115.19 | 116.50 | 115.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 09:30:00 | 114.70 | 115.88 | 115.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 112.44 | 115.19 | 114.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 112.08 | 115.19 | 114.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 11:15:00 | 112.75 | 114.71 | 114.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 110.20 | 113.51 | 114.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 110.21 | 109.86 | 111.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 10:00:00 | 110.21 | 109.86 | 111.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 108.82 | 109.74 | 110.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 10:15:00 | 108.49 | 109.74 | 110.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 11:00:00 | 108.37 | 107.87 | 108.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 13:15:00 | 108.50 | 107.63 | 107.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 13:15:00 | 108.50 | 107.63 | 107.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 14:15:00 | 108.75 | 107.85 | 107.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 106.60 | 107.71 | 107.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 106.60 | 107.71 | 107.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 106.60 | 107.71 | 107.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 10:00:00 | 106.60 | 107.71 | 107.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 10:15:00 | 106.50 | 107.47 | 107.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 105.94 | 107.03 | 107.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 107.66 | 106.79 | 107.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 10:15:00 | 107.66 | 106.79 | 107.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 107.66 | 106.79 | 107.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:00:00 | 107.66 | 106.79 | 107.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 107.76 | 106.99 | 107.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:30:00 | 108.57 | 106.99 | 107.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 12:15:00 | 108.53 | 107.30 | 107.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 13:15:00 | 108.82 | 107.60 | 107.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 14:15:00 | 111.33 | 111.47 | 109.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 15:00:00 | 111.33 | 111.47 | 109.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 110.43 | 111.15 | 110.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 14:30:00 | 115.05 | 112.25 | 110.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-15 09:15:00 | 126.56 | 124.15 | 121.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 124.51 | 126.73 | 126.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 123.00 | 125.99 | 126.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 14:15:00 | 124.27 | 124.26 | 125.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 15:00:00 | 124.27 | 124.26 | 125.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 125.45 | 124.50 | 125.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:45:00 | 126.14 | 124.50 | 125.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 126.90 | 124.98 | 125.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:45:00 | 127.00 | 124.98 | 125.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2025-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 12:15:00 | 126.92 | 125.70 | 125.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 14:15:00 | 127.16 | 126.20 | 125.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 15:15:00 | 127.21 | 127.96 | 127.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 127.50 | 127.87 | 127.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 127.50 | 127.87 | 127.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:45:00 | 126.65 | 127.87 | 127.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 127.40 | 127.78 | 127.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 10:45:00 | 127.50 | 127.78 | 127.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 127.19 | 127.64 | 127.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 13:00:00 | 127.19 | 127.64 | 127.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 127.60 | 127.63 | 127.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 15:15:00 | 128.00 | 127.60 | 127.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 14:15:00 | 126.99 | 128.10 | 127.83 | SL hit (close<static) qty=1.00 sl=127.18 alert=retest2 |

### Cycle 74 — SELL (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 09:15:00 | 126.72 | 127.57 | 127.62 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 11:15:00 | 128.09 | 127.73 | 127.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 14:15:00 | 128.60 | 127.89 | 127.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 15:15:00 | 127.75 | 127.87 | 127.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 15:15:00 | 127.75 | 127.87 | 127.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 127.75 | 127.87 | 127.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:30:00 | 127.77 | 127.78 | 127.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 128.07 | 127.84 | 127.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:30:00 | 127.26 | 127.84 | 127.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 128.10 | 127.89 | 127.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 15:00:00 | 128.24 | 128.01 | 127.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 126.82 | 127.78 | 127.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 126.82 | 127.78 | 127.80 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 14:15:00 | 131.07 | 128.20 | 127.94 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 14:15:00 | 128.10 | 129.02 | 129.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 15:15:00 | 127.85 | 128.79 | 128.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 11:15:00 | 128.71 | 128.66 | 128.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 11:15:00 | 128.71 | 128.66 | 128.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 128.71 | 128.66 | 128.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 11:45:00 | 129.15 | 128.66 | 128.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 12:15:00 | 127.75 | 128.48 | 128.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 13:15:00 | 127.55 | 128.48 | 128.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 15:15:00 | 127.57 | 128.17 | 128.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 10:00:00 | 127.67 | 127.16 | 127.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 10:30:00 | 127.35 | 127.09 | 127.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 13:15:00 | 127.50 | 127.21 | 127.52 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-10 09:15:00 | 131.65 | 128.16 | 127.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2025-06-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 09:15:00 | 131.65 | 128.16 | 127.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-12 09:15:00 | 135.72 | 132.87 | 131.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 15:15:00 | 134.18 | 134.44 | 133.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 09:15:00 | 134.07 | 134.44 | 133.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 136.37 | 134.83 | 133.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:30:00 | 136.83 | 135.14 | 133.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 14:30:00 | 136.82 | 135.75 | 134.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 15:00:00 | 136.96 | 135.75 | 134.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 10:45:00 | 136.85 | 136.49 | 135.12 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 137.34 | 138.63 | 137.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 14:30:00 | 137.59 | 138.63 | 137.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 137.19 | 138.34 | 137.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:15:00 | 137.15 | 138.34 | 137.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 136.82 | 138.04 | 137.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:15:00 | 136.05 | 138.04 | 137.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 135.27 | 137.48 | 137.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:00:00 | 135.27 | 137.48 | 137.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-18 11:15:00 | 135.22 | 137.03 | 137.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 135.22 | 137.03 | 137.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 13:15:00 | 134.16 | 136.11 | 136.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 11:15:00 | 130.31 | 129.82 | 131.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 12:00:00 | 130.31 | 129.82 | 131.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 133.64 | 130.75 | 131.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 133.64 | 130.75 | 131.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 133.86 | 131.37 | 131.98 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 09:15:00 | 142.26 | 133.55 | 132.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 10:15:00 | 144.52 | 135.74 | 133.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 144.68 | 144.94 | 141.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 13:45:00 | 144.80 | 144.94 | 141.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 143.61 | 144.85 | 143.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:00:00 | 143.61 | 144.85 | 143.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 143.75 | 144.63 | 143.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:45:00 | 143.46 | 144.63 | 143.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 143.95 | 144.49 | 143.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 14:30:00 | 144.18 | 144.33 | 143.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 144.06 | 144.26 | 143.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 13:15:00 | 144.20 | 144.43 | 144.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 145.49 | 144.36 | 144.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 144.28 | 144.34 | 144.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:45:00 | 144.20 | 144.34 | 144.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 145.61 | 144.59 | 144.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 11:15:00 | 147.02 | 144.59 | 144.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 14:30:00 | 146.24 | 145.63 | 144.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 10:15:00 | 143.09 | 144.91 | 144.79 | SL hit (close<static) qty=1.00 sl=143.15 alert=retest2 |

### Cycle 82 — SELL (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 11:15:00 | 143.28 | 144.59 | 144.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 14:15:00 | 142.13 | 143.76 | 144.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 11:15:00 | 142.91 | 142.69 | 143.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-02 12:00:00 | 142.91 | 142.69 | 143.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 141.63 | 141.65 | 142.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:45:00 | 141.93 | 141.65 | 142.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 142.50 | 141.75 | 142.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:30:00 | 143.20 | 141.75 | 142.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 142.32 | 141.86 | 142.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:30:00 | 142.49 | 141.86 | 142.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 143.92 | 142.27 | 142.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 15:00:00 | 143.92 | 142.27 | 142.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 15:15:00 | 144.50 | 142.72 | 142.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 09:15:00 | 146.88 | 143.55 | 143.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 09:15:00 | 144.69 | 145.69 | 144.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 09:15:00 | 144.69 | 145.69 | 144.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 144.69 | 145.69 | 144.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 144.69 | 145.69 | 144.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 144.85 | 145.52 | 144.71 | EMA400 retest candle locked (from upside) |

### Cycle 84 — SELL (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 10:15:00 | 142.26 | 144.09 | 144.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 11:15:00 | 141.55 | 143.58 | 144.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 13:15:00 | 143.30 | 143.22 | 143.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 14:00:00 | 143.30 | 143.22 | 143.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 145.80 | 143.74 | 143.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 145.80 | 143.74 | 143.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 144.71 | 143.93 | 144.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 144.93 | 143.93 | 144.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 140.25 | 141.98 | 142.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:30:00 | 141.12 | 141.98 | 142.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 141.54 | 141.69 | 142.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:30:00 | 142.87 | 141.69 | 142.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 141.75 | 141.70 | 142.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 15:00:00 | 141.75 | 141.70 | 142.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 143.92 | 139.51 | 140.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:00:00 | 143.92 | 139.51 | 140.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 142.80 | 140.16 | 140.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:30:00 | 143.57 | 140.16 | 140.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 12:15:00 | 143.00 | 141.15 | 140.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 145.04 | 142.59 | 141.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 143.75 | 143.89 | 143.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 10:00:00 | 143.75 | 143.89 | 143.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 143.78 | 143.84 | 143.44 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 142.04 | 143.05 | 143.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 14:15:00 | 141.54 | 142.38 | 142.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 143.30 | 142.51 | 142.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 143.30 | 142.51 | 142.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 143.30 | 142.51 | 142.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:00:00 | 143.30 | 142.51 | 142.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 142.50 | 142.51 | 142.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 14:15:00 | 141.95 | 142.48 | 142.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 15:15:00 | 141.40 | 142.41 | 142.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:30:00 | 141.55 | 141.75 | 142.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 14:15:00 | 134.85 | 138.96 | 140.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 14:15:00 | 134.33 | 138.96 | 140.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 14:15:00 | 134.47 | 138.96 | 140.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-07-24 09:15:00 | 127.75 | 130.71 | 134.29 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 87 — BUY (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 15:15:00 | 119.40 | 118.34 | 118.26 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 10:15:00 | 117.54 | 118.08 | 118.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 12:15:00 | 117.11 | 117.80 | 118.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 09:15:00 | 113.80 | 113.80 | 115.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 09:45:00 | 113.64 | 113.80 | 115.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 114.08 | 113.70 | 114.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 13:30:00 | 114.08 | 113.70 | 114.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 114.48 | 113.86 | 114.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 114.48 | 113.86 | 114.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 114.96 | 114.08 | 114.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 113.64 | 114.08 | 114.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 112.91 | 113.84 | 114.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:45:00 | 112.80 | 113.70 | 114.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:00:00 | 112.76 | 113.42 | 114.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 09:30:00 | 112.47 | 112.99 | 113.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 10:45:00 | 112.75 | 112.87 | 113.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 113.37 | 112.68 | 113.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 15:00:00 | 113.37 | 112.68 | 113.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 112.40 | 112.62 | 113.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:15:00 | 113.59 | 112.62 | 113.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 113.90 | 112.88 | 113.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:45:00 | 113.92 | 112.88 | 113.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 113.64 | 113.03 | 113.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:45:00 | 113.97 | 113.03 | 113.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-12 12:15:00 | 115.19 | 113.65 | 113.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 12:15:00 | 115.19 | 113.65 | 113.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 13:15:00 | 117.26 | 114.73 | 114.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 13:15:00 | 116.72 | 116.72 | 116.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 13:30:00 | 116.95 | 116.72 | 116.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 116.09 | 116.60 | 116.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 15:00:00 | 116.09 | 116.60 | 116.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 115.98 | 116.47 | 116.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 09:15:00 | 116.84 | 116.47 | 116.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 10:15:00 | 118.70 | 119.91 | 119.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 118.70 | 119.91 | 119.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 117.26 | 118.48 | 118.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 119.02 | 118.59 | 118.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 119.02 | 118.59 | 118.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 119.02 | 118.59 | 118.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 119.02 | 118.59 | 118.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 119.04 | 118.68 | 118.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:45:00 | 118.96 | 118.68 | 118.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 119.21 | 118.79 | 118.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 13:00:00 | 119.21 | 118.79 | 118.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 118.01 | 118.63 | 118.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 14:15:00 | 117.44 | 118.63 | 118.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 10:15:00 | 117.04 | 116.42 | 116.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 10:15:00 | 117.04 | 116.42 | 116.36 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 114.70 | 116.05 | 116.22 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 11:15:00 | 116.44 | 116.02 | 115.99 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 11:15:00 | 115.56 | 115.97 | 116.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 14:15:00 | 115.03 | 115.76 | 115.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 116.45 | 115.81 | 115.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 116.45 | 115.81 | 115.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 116.45 | 115.81 | 115.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 11:45:00 | 115.61 | 115.85 | 115.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 09:15:00 | 117.20 | 115.54 | 115.65 | SL hit (close>static) qty=1.00 sl=117.18 alert=retest2 |

### Cycle 95 — BUY (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 10:15:00 | 117.20 | 115.87 | 115.80 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 13:15:00 | 115.33 | 116.05 | 116.11 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 12:15:00 | 116.29 | 115.89 | 115.88 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 115.34 | 115.81 | 115.87 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 09:15:00 | 116.67 | 115.93 | 115.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 11:15:00 | 118.41 | 116.46 | 116.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 14:15:00 | 116.27 | 116.86 | 116.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 14:15:00 | 116.27 | 116.86 | 116.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 116.27 | 116.86 | 116.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 116.27 | 116.86 | 116.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 116.62 | 116.81 | 116.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 09:15:00 | 118.49 | 116.81 | 116.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 10:15:00 | 116.81 | 117.99 | 118.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 10:15:00 | 116.81 | 117.99 | 118.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 11:15:00 | 116.58 | 117.71 | 117.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 113.94 | 113.76 | 114.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 09:30:00 | 114.14 | 113.76 | 114.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 113.83 | 112.80 | 113.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 114.24 | 112.80 | 113.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 113.92 | 113.03 | 113.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:30:00 | 113.88 | 113.03 | 113.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 114.79 | 113.61 | 113.53 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 13:15:00 | 113.39 | 113.63 | 113.63 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 14:15:00 | 113.88 | 113.68 | 113.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 15:15:00 | 114.15 | 113.77 | 113.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 113.57 | 113.77 | 113.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 10:15:00 | 113.57 | 113.77 | 113.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 113.57 | 113.77 | 113.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:00:00 | 113.57 | 113.77 | 113.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 113.55 | 113.73 | 113.70 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2025-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 12:15:00 | 113.31 | 113.64 | 113.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 13:15:00 | 112.94 | 113.50 | 113.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 15:15:00 | 110.19 | 110.13 | 110.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 09:15:00 | 110.11 | 110.13 | 110.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 111.80 | 110.44 | 110.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 12:00:00 | 111.80 | 110.44 | 110.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 111.60 | 110.67 | 110.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 111.20 | 111.05 | 111.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 10:15:00 | 110.66 | 110.09 | 110.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 110.66 | 110.09 | 110.05 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 15:15:00 | 108.41 | 109.84 | 109.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 09:15:00 | 106.21 | 109.11 | 109.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 104.99 | 104.55 | 105.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-21 13:45:00 | 104.95 | 104.55 | 105.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 106.10 | 104.89 | 105.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:45:00 | 106.09 | 104.89 | 105.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 105.98 | 105.11 | 105.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 11:45:00 | 105.57 | 105.21 | 105.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 13:00:00 | 105.78 | 105.32 | 105.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 13:45:00 | 105.74 | 105.41 | 105.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 15:00:00 | 105.50 | 105.43 | 105.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 105.79 | 105.50 | 105.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 106.24 | 105.50 | 105.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 105.80 | 105.56 | 105.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 10:15:00 | 105.27 | 105.56 | 105.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 15:15:00 | 100.29 | 101.14 | 101.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 15:15:00 | 100.49 | 101.14 | 101.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 15:15:00 | 100.45 | 101.14 | 101.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 15:15:00 | 100.22 | 101.14 | 101.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 15:15:00 | 100.01 | 101.14 | 101.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-03 11:15:00 | 101.02 | 101.00 | 101.66 | SL hit (close>ema200) qty=0.50 sl=101.00 alert=retest2 |

### Cycle 107 — BUY (started 2025-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 15:15:00 | 101.90 | 101.61 | 101.58 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 100.10 | 101.30 | 101.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 10:15:00 | 100.05 | 101.05 | 101.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 11:15:00 | 98.19 | 98.08 | 98.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-11 11:30:00 | 98.25 | 98.08 | 98.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 98.87 | 98.19 | 98.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 15:00:00 | 98.87 | 98.19 | 98.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 99.10 | 98.38 | 98.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:15:00 | 99.26 | 98.38 | 98.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 101.90 | 99.28 | 99.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 11:15:00 | 102.64 | 99.96 | 99.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 11:15:00 | 101.36 | 101.95 | 100.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 12:00:00 | 101.36 | 101.95 | 100.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 100.77 | 101.71 | 100.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 13:00:00 | 100.77 | 101.71 | 100.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 100.48 | 101.46 | 100.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 100.48 | 101.46 | 100.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 100.10 | 101.19 | 100.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 100.10 | 101.19 | 100.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 100.19 | 100.75 | 100.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:30:00 | 99.83 | 100.75 | 100.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 100.23 | 100.63 | 100.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 14:15:00 | 99.99 | 100.41 | 100.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 11:15:00 | 99.10 | 99.02 | 99.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 11:30:00 | 99.03 | 99.02 | 99.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 98.53 | 98.30 | 98.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 13:45:00 | 97.40 | 98.04 | 98.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 97.31 | 98.06 | 98.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 12:45:00 | 97.40 | 97.76 | 98.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 98.77 | 97.94 | 97.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 98.77 | 97.94 | 97.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 12:15:00 | 102.15 | 99.22 | 98.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 10:15:00 | 100.24 | 100.44 | 99.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 10:45:00 | 100.24 | 100.44 | 99.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 99.85 | 100.17 | 99.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 99.40 | 100.17 | 99.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 99.18 | 99.98 | 99.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:45:00 | 99.38 | 99.98 | 99.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 99.19 | 99.82 | 99.67 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2025-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 13:15:00 | 98.23 | 99.32 | 99.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 15:15:00 | 96.99 | 98.70 | 99.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 13:15:00 | 98.95 | 98.34 | 98.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 13:15:00 | 98.95 | 98.34 | 98.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 98.95 | 98.34 | 98.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 14:00:00 | 98.95 | 98.34 | 98.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 99.92 | 98.65 | 98.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 15:00:00 | 99.92 | 98.65 | 98.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 99.99 | 98.92 | 98.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 09:15:00 | 98.97 | 98.92 | 98.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 14:15:00 | 94.02 | 95.64 | 96.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 94.34 | 93.55 | 94.67 | SL hit (close>ema200) qty=0.50 sl=93.55 alert=retest2 |

### Cycle 113 — BUY (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 15:15:00 | 94.25 | 94.07 | 94.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 09:15:00 | 94.45 | 94.14 | 94.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 10:15:00 | 93.80 | 94.08 | 94.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 10:15:00 | 93.80 | 94.08 | 94.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 93.80 | 94.08 | 94.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 93.80 | 94.08 | 94.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 11:15:00 | 94.00 | 94.06 | 94.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 10:15:00 | 93.46 | 93.80 | 93.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 09:15:00 | 93.23 | 93.20 | 93.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 93.23 | 93.20 | 93.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 93.23 | 93.20 | 93.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 10:30:00 | 92.69 | 93.13 | 93.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 11:00:00 | 92.82 | 93.13 | 93.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 12:00:00 | 92.65 | 93.03 | 93.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 10:15:00 | 92.80 | 91.15 | 91.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 92.27 | 91.37 | 91.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 11:15:00 | 92.18 | 91.37 | 91.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 12:15:00 | 92.28 | 91.71 | 91.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-12-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 12:15:00 | 92.28 | 91.71 | 91.65 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 09:15:00 | 91.44 | 91.85 | 91.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 91.00 | 91.38 | 91.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 11:15:00 | 90.17 | 90.10 | 90.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 11:45:00 | 90.16 | 90.10 | 90.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 91.13 | 90.28 | 90.43 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 10:15:00 | 91.58 | 90.54 | 90.54 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2026-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 14:15:00 | 90.50 | 91.20 | 91.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 89.87 | 90.76 | 90.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 89.92 | 89.57 | 89.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 89.92 | 89.57 | 89.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 89.92 | 89.57 | 89.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:30:00 | 90.13 | 89.57 | 89.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 90.26 | 89.71 | 89.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:30:00 | 90.43 | 89.71 | 89.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 90.50 | 89.87 | 90.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:45:00 | 90.64 | 89.87 | 90.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 13:15:00 | 90.54 | 90.12 | 90.12 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 10:15:00 | 89.79 | 90.10 | 90.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 12:15:00 | 89.56 | 89.94 | 90.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 84.55 | 83.27 | 84.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 84.55 | 83.27 | 84.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 84.55 | 83.27 | 84.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 84.90 | 83.27 | 84.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 84.75 | 83.56 | 84.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:45:00 | 84.80 | 83.56 | 84.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 84.38 | 83.73 | 84.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:15:00 | 83.07 | 84.46 | 84.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 78.92 | 81.76 | 83.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 82.75 | 80.44 | 81.46 | SL hit (close>ema200) qty=0.50 sl=80.44 alert=retest2 |

### Cycle 121 — BUY (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 13:15:00 | 83.41 | 82.09 | 82.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 83.92 | 82.46 | 82.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 82.63 | 82.80 | 82.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 10:00:00 | 82.63 | 82.80 | 82.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 82.10 | 82.66 | 82.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 81.94 | 82.66 | 82.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 81.98 | 82.53 | 82.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:00:00 | 81.98 | 82.53 | 82.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 82.24 | 82.47 | 82.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 15:00:00 | 82.41 | 82.43 | 82.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 09:30:00 | 82.40 | 82.59 | 82.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 12:30:00 | 82.55 | 83.37 | 83.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 81.99 | 82.89 | 82.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 81.99 | 82.89 | 82.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 80.30 | 82.37 | 82.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 81.08 | 80.95 | 81.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 81.08 | 80.95 | 81.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 82.15 | 81.19 | 81.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 82.93 | 81.19 | 81.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 82.60 | 81.47 | 81.86 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 83.14 | 82.26 | 82.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 13:15:00 | 83.76 | 83.17 | 82.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 92.81 | 93.18 | 91.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:30:00 | 92.39 | 93.18 | 91.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 93.40 | 93.04 | 92.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 92.10 | 93.04 | 92.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 92.27 | 92.91 | 92.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:00:00 | 92.27 | 92.91 | 92.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 91.65 | 92.66 | 92.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 13:00:00 | 91.65 | 92.66 | 92.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 91.99 | 92.52 | 92.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 13:30:00 | 91.65 | 92.52 | 92.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 93.63 | 92.75 | 92.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 10:00:00 | 94.71 | 93.26 | 92.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 11:15:00 | 94.79 | 93.54 | 92.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 10:00:00 | 95.05 | 95.01 | 94.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 11:15:00 | 94.01 | 94.68 | 94.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 11:15:00 | 94.01 | 94.68 | 94.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 09:15:00 | 93.81 | 94.38 | 94.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 89.12 | 88.97 | 90.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 10:00:00 | 89.12 | 88.97 | 90.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 87.95 | 87.38 | 87.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 85.40 | 87.38 | 87.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:15:00 | 81.13 | 82.12 | 83.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 82.15 | 81.87 | 83.06 | SL hit (close>ema200) qty=0.50 sl=81.87 alert=retest2 |

### Cycle 125 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 82.41 | 81.11 | 81.04 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 79.68 | 81.07 | 81.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 79.55 | 80.61 | 80.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 10:15:00 | 76.39 | 75.71 | 76.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-18 11:15:00 | 76.44 | 75.71 | 76.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 76.37 | 75.84 | 76.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:30:00 | 76.70 | 75.84 | 76.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 12:15:00 | 77.09 | 76.09 | 76.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:00:00 | 77.09 | 76.09 | 76.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 13:15:00 | 77.06 | 76.28 | 76.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:30:00 | 77.42 | 76.28 | 76.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 71.28 | 70.16 | 71.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 15:00:00 | 71.28 | 70.16 | 71.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 71.25 | 70.38 | 71.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 76.36 | 70.38 | 71.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 75.70 | 71.44 | 71.60 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 76.75 | 72.50 | 72.07 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 12:15:00 | 72.83 | 73.45 | 73.52 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 76.36 | 73.51 | 73.46 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 11:15:00 | 72.91 | 74.20 | 74.22 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 75.94 | 74.07 | 73.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 78.12 | 74.88 | 74.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 79.08 | 79.17 | 77.83 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 81.48 | 79.17 | 77.83 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 80.12 | 81.18 | 79.90 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 14:15:00 | 79.89 | 80.57 | 80.06 | SL hit (close<ema400) qty=1.00 sl=80.06 alert=retest1 |

### Cycle 132 — SELL (started 2026-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 12:15:00 | 79.45 | 80.75 | 80.85 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 14:15:00 | 81.11 | 80.75 | 80.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 09:15:00 | 81.63 | 80.97 | 80.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 14:15:00 | 86.48 | 86.90 | 85.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 15:00:00 | 86.48 | 86.90 | 85.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 89.18 | 90.41 | 88.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:30:00 | 88.81 | 90.41 | 88.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 87.86 | 89.90 | 88.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:00:00 | 87.86 | 89.90 | 88.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 87.80 | 89.48 | 88.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:45:00 | 87.48 | 89.48 | 88.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 91.77 | 92.29 | 91.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:45:00 | 91.46 | 92.29 | 91.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 91.50 | 92.13 | 91.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 11:45:00 | 91.74 | 92.13 | 91.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 91.70 | 92.04 | 91.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 12:30:00 | 91.50 | 92.04 | 91.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 91.08 | 91.85 | 91.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 91.08 | 91.85 | 91.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 90.60 | 91.60 | 91.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:00:00 | 90.60 | 91.60 | 91.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 88.70 | 90.83 | 91.05 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 15:15:00 | 90.89 | 90.57 | 90.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 91.66 | 90.79 | 90.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 11:15:00 | 90.60 | 90.85 | 90.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 11:15:00 | 90.60 | 90.85 | 90.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 90.60 | 90.85 | 90.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:00:00 | 90.60 | 90.85 | 90.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 90.91 | 90.86 | 90.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 91.54 | 90.72 | 90.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:45:00 | 91.29 | 90.72 | 90.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 10:15:00 | 90.11 | 90.60 | 90.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 10:15:00 | 90.11 | 90.60 | 90.63 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 92.54 | 90.99 | 90.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 12:15:00 | 94.07 | 91.60 | 91.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 15:15:00 | 94.42 | 94.46 | 93.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 09:15:00 | 94.73 | 94.46 | 93.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-14 14:30:00 | 132.10 | 2024-05-15 13:15:00 | 132.55 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2024-05-30 10:15:00 | 153.40 | 2024-05-31 10:15:00 | 148.80 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2024-05-30 12:30:00 | 152.65 | 2024-05-31 10:15:00 | 148.80 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2024-05-30 14:00:00 | 152.80 | 2024-05-31 10:15:00 | 148.80 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2024-05-31 09:15:00 | 152.85 | 2024-05-31 10:15:00 | 148.80 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2024-06-13 11:45:00 | 167.01 | 2024-06-14 10:15:00 | 163.21 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2024-06-13 14:30:00 | 165.84 | 2024-06-14 10:15:00 | 163.21 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-07-05 12:00:00 | 149.67 | 2024-07-09 09:15:00 | 152.38 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-07-08 11:00:00 | 149.69 | 2024-07-09 09:15:00 | 152.38 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2024-07-24 12:15:00 | 137.30 | 2024-07-25 10:15:00 | 141.95 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2024-07-24 14:30:00 | 136.70 | 2024-07-25 10:15:00 | 141.95 | STOP_HIT | 1.00 | -3.84% |
| SELL | retest2 | 2024-07-25 09:45:00 | 136.97 | 2024-07-25 10:15:00 | 141.95 | STOP_HIT | 1.00 | -3.64% |
| SELL | retest2 | 2024-08-05 09:15:00 | 137.50 | 2024-08-08 10:15:00 | 140.91 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2024-08-19 12:15:00 | 135.34 | 2024-08-21 09:15:00 | 138.67 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2024-08-19 12:45:00 | 135.39 | 2024-08-21 09:15:00 | 138.67 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2024-08-20 09:45:00 | 135.42 | 2024-08-21 09:15:00 | 138.67 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2024-08-20 15:00:00 | 135.33 | 2024-08-21 09:15:00 | 138.67 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2024-09-05 13:30:00 | 136.22 | 2024-09-10 11:15:00 | 140.36 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2024-09-06 09:30:00 | 135.80 | 2024-09-10 11:15:00 | 140.36 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2024-09-24 14:15:00 | 127.24 | 2024-09-25 10:15:00 | 134.10 | STOP_HIT | 1.00 | -5.39% |
| SELL | retest2 | 2024-09-24 14:45:00 | 127.19 | 2024-09-25 10:15:00 | 134.10 | STOP_HIT | 1.00 | -5.43% |
| SELL | retest2 | 2024-10-09 14:15:00 | 129.49 | 2024-10-10 13:15:00 | 129.75 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2024-10-10 11:00:00 | 129.82 | 2024-10-10 13:15:00 | 129.75 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2024-10-10 13:15:00 | 129.74 | 2024-10-10 13:15:00 | 129.75 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2024-10-15 11:45:00 | 130.80 | 2024-10-15 12:15:00 | 129.22 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-10-24 11:45:00 | 124.88 | 2024-10-29 10:15:00 | 118.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 12:30:00 | 124.67 | 2024-10-29 10:15:00 | 118.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 14:15:00 | 124.38 | 2024-10-29 10:15:00 | 118.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 11:45:00 | 124.88 | 2024-10-30 09:15:00 | 121.25 | STOP_HIT | 0.50 | 2.91% |
| SELL | retest2 | 2024-10-24 12:30:00 | 124.67 | 2024-10-30 09:15:00 | 121.25 | STOP_HIT | 0.50 | 2.74% |
| SELL | retest2 | 2024-10-24 14:15:00 | 124.38 | 2024-10-30 09:15:00 | 121.25 | STOP_HIT | 0.50 | 2.52% |
| BUY | retest2 | 2024-11-21 12:15:00 | 119.85 | 2024-11-21 15:15:00 | 117.99 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-11-21 13:30:00 | 120.00 | 2024-11-21 15:15:00 | 117.99 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2024-11-25 12:45:00 | 118.40 | 2024-11-26 11:15:00 | 121.30 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2024-12-16 13:15:00 | 135.60 | 2024-12-17 09:15:00 | 136.95 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-12-16 14:45:00 | 135.66 | 2024-12-17 09:15:00 | 136.95 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest1 | 2024-12-27 15:15:00 | 124.44 | 2025-01-01 09:15:00 | 124.35 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2025-01-01 11:15:00 | 123.72 | 2025-01-02 10:15:00 | 124.49 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-01-15 13:00:00 | 123.29 | 2025-01-22 11:15:00 | 117.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-16 13:15:00 | 123.33 | 2025-01-22 11:15:00 | 117.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-16 14:45:00 | 123.10 | 2025-01-22 11:15:00 | 116.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-17 09:30:00 | 122.83 | 2025-01-22 11:15:00 | 116.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-15 13:00:00 | 123.29 | 2025-01-23 09:15:00 | 119.62 | STOP_HIT | 0.50 | 2.98% |
| SELL | retest2 | 2025-01-16 13:15:00 | 123.33 | 2025-01-23 09:15:00 | 119.62 | STOP_HIT | 0.50 | 3.01% |
| SELL | retest2 | 2025-01-16 14:45:00 | 123.10 | 2025-01-23 09:15:00 | 119.62 | STOP_HIT | 0.50 | 2.83% |
| SELL | retest2 | 2025-01-17 09:30:00 | 122.83 | 2025-01-23 09:15:00 | 119.62 | STOP_HIT | 0.50 | 2.61% |
| SELL | retest2 | 2025-01-21 10:45:00 | 120.76 | 2025-01-27 09:15:00 | 114.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 12:15:00 | 120.58 | 2025-01-27 09:15:00 | 114.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 13:45:00 | 120.75 | 2025-01-27 09:15:00 | 114.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 13:45:00 | 120.40 | 2025-01-27 09:15:00 | 114.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 13:30:00 | 117.10 | 2025-01-27 10:15:00 | 111.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 10:45:00 | 120.76 | 2025-01-27 14:15:00 | 108.68 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-21 12:15:00 | 120.58 | 2025-01-27 14:15:00 | 108.67 | TARGET_HIT | 0.50 | 9.87% |
| SELL | retest2 | 2025-01-21 13:45:00 | 120.75 | 2025-01-27 15:15:00 | 108.52 | TARGET_HIT | 0.50 | 10.13% |
| SELL | retest2 | 2025-01-23 13:45:00 | 120.40 | 2025-01-27 15:15:00 | 108.36 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-24 13:30:00 | 117.10 | 2025-01-28 10:15:00 | 105.39 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-04 12:00:00 | 108.42 | 2025-02-10 09:15:00 | 107.07 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-02-04 12:45:00 | 108.39 | 2025-02-10 09:15:00 | 107.07 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-02-05 09:15:00 | 108.50 | 2025-02-10 09:15:00 | 107.07 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-02-06 15:15:00 | 109.50 | 2025-02-10 09:15:00 | 107.07 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-02-07 11:30:00 | 109.09 | 2025-02-10 09:15:00 | 107.07 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-02-07 12:45:00 | 110.02 | 2025-02-10 09:15:00 | 107.07 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2025-02-07 14:45:00 | 108.88 | 2025-02-10 09:15:00 | 107.07 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-03-18 14:45:00 | 102.17 | 2025-03-19 09:15:00 | 102.97 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-03-28 11:30:00 | 100.16 | 2025-04-02 11:15:00 | 101.18 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-04-01 11:00:00 | 100.09 | 2025-04-02 11:15:00 | 101.18 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-04-02 09:15:00 | 100.15 | 2025-04-02 11:15:00 | 101.18 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-04-29 10:15:00 | 108.49 | 2025-05-05 13:15:00 | 108.50 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-04-30 11:00:00 | 108.37 | 2025-05-05 13:15:00 | 108.50 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2025-05-09 14:30:00 | 115.05 | 2025-05-15 09:15:00 | 126.56 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-26 15:15:00 | 128.00 | 2025-05-27 14:15:00 | 126.99 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-05-27 14:30:00 | 128.00 | 2025-05-27 15:15:00 | 126.50 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-05-29 15:00:00 | 128.24 | 2025-05-30 09:15:00 | 126.82 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-06-05 13:15:00 | 127.55 | 2025-06-10 09:15:00 | 131.65 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2025-06-05 15:15:00 | 127.57 | 2025-06-10 09:15:00 | 131.65 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2025-06-09 10:00:00 | 127.67 | 2025-06-10 09:15:00 | 131.65 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2025-06-09 10:30:00 | 127.35 | 2025-06-10 09:15:00 | 131.65 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest2 | 2025-06-13 10:30:00 | 136.83 | 2025-06-18 11:15:00 | 135.22 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-06-13 14:30:00 | 136.82 | 2025-06-18 11:15:00 | 135.22 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-06-13 15:00:00 | 136.96 | 2025-06-18 11:15:00 | 135.22 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-06-16 10:45:00 | 136.85 | 2025-06-18 11:15:00 | 135.22 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-06-26 14:30:00 | 144.18 | 2025-07-01 10:15:00 | 143.09 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-06-27 09:15:00 | 144.06 | 2025-07-01 10:15:00 | 143.09 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-06-27 13:15:00 | 144.20 | 2025-07-01 10:15:00 | 143.09 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-06-30 09:15:00 | 145.49 | 2025-07-01 10:15:00 | 143.09 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-06-30 11:15:00 | 147.02 | 2025-07-01 10:15:00 | 143.09 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2025-06-30 14:30:00 | 146.24 | 2025-07-01 10:15:00 | 143.09 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-07-21 14:15:00 | 141.95 | 2025-07-22 14:15:00 | 134.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-21 15:15:00 | 141.40 | 2025-07-22 14:15:00 | 134.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 09:30:00 | 141.55 | 2025-07-22 14:15:00 | 134.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-21 14:15:00 | 141.95 | 2025-07-24 09:15:00 | 127.75 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-21 15:15:00 | 141.40 | 2025-07-24 09:15:00 | 127.26 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-22 09:30:00 | 141.55 | 2025-07-24 09:15:00 | 127.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-08 10:45:00 | 112.80 | 2025-08-12 12:15:00 | 115.19 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-08-08 14:00:00 | 112.76 | 2025-08-12 12:15:00 | 115.19 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-08-11 09:30:00 | 112.47 | 2025-08-12 12:15:00 | 115.19 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-08-11 10:45:00 | 112.75 | 2025-08-12 12:15:00 | 115.19 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-08-19 09:15:00 | 116.84 | 2025-08-26 10:15:00 | 118.70 | STOP_HIT | 1.00 | 1.59% |
| SELL | retest2 | 2025-08-29 14:15:00 | 117.44 | 2025-09-04 10:15:00 | 117.04 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2025-09-10 11:45:00 | 115.61 | 2025-09-11 09:15:00 | 117.20 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-09-22 09:15:00 | 118.49 | 2025-09-24 10:15:00 | 116.81 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-10-13 09:15:00 | 111.20 | 2025-10-16 10:15:00 | 110.66 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2025-10-23 11:45:00 | 105.57 | 2025-10-31 15:15:00 | 100.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-23 13:00:00 | 105.78 | 2025-10-31 15:15:00 | 100.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-23 13:45:00 | 105.74 | 2025-10-31 15:15:00 | 100.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-23 15:00:00 | 105.50 | 2025-10-31 15:15:00 | 100.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-24 10:15:00 | 105.27 | 2025-10-31 15:15:00 | 100.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-23 11:45:00 | 105.57 | 2025-11-03 11:15:00 | 101.02 | STOP_HIT | 0.50 | 4.31% |
| SELL | retest2 | 2025-10-23 13:00:00 | 105.78 | 2025-11-03 11:15:00 | 101.02 | STOP_HIT | 0.50 | 4.50% |
| SELL | retest2 | 2025-10-23 13:45:00 | 105.74 | 2025-11-03 11:15:00 | 101.02 | STOP_HIT | 0.50 | 4.46% |
| SELL | retest2 | 2025-10-23 15:00:00 | 105.50 | 2025-11-03 11:15:00 | 101.02 | STOP_HIT | 0.50 | 4.25% |
| SELL | retest2 | 2025-10-24 10:15:00 | 105.27 | 2025-11-03 11:15:00 | 101.02 | STOP_HIT | 0.50 | 4.04% |
| SELL | retest2 | 2025-11-24 13:45:00 | 97.40 | 2025-11-27 09:15:00 | 98.77 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-11-25 09:15:00 | 97.31 | 2025-11-27 09:15:00 | 98.77 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-11-25 12:45:00 | 97.40 | 2025-11-27 09:15:00 | 98.77 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-12-04 09:15:00 | 98.97 | 2025-12-08 14:15:00 | 94.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 09:15:00 | 98.97 | 2025-12-10 09:15:00 | 94.34 | STOP_HIT | 0.50 | 4.68% |
| SELL | retest2 | 2025-12-17 10:30:00 | 92.69 | 2025-12-22 12:15:00 | 92.28 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2025-12-17 11:00:00 | 92.82 | 2025-12-22 12:15:00 | 92.28 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2025-12-17 12:00:00 | 92.65 | 2025-12-22 12:15:00 | 92.28 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2025-12-22 10:15:00 | 92.80 | 2025-12-22 12:15:00 | 92.28 | STOP_HIT | 1.00 | 0.56% |
| SELL | retest2 | 2025-12-22 11:15:00 | 92.18 | 2025-12-22 12:15:00 | 92.28 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2026-01-23 09:15:00 | 83.07 | 2026-01-27 09:15:00 | 78.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 09:15:00 | 83.07 | 2026-01-28 09:15:00 | 82.75 | STOP_HIT | 0.50 | 0.39% |
| BUY | retest2 | 2026-01-29 15:00:00 | 82.41 | 2026-02-01 14:15:00 | 81.99 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2026-01-30 09:30:00 | 82.40 | 2026-02-01 14:15:00 | 81.99 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2026-02-01 12:30:00 | 82.55 | 2026-02-01 14:15:00 | 81.99 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2026-02-13 10:00:00 | 94.71 | 2026-02-18 11:15:00 | 94.01 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2026-02-13 11:15:00 | 94.79 | 2026-02-18 11:15:00 | 94.01 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2026-02-17 10:00:00 | 95.05 | 2026-02-18 11:15:00 | 94.01 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-03-02 09:15:00 | 85.40 | 2026-03-05 11:15:00 | 81.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 85.40 | 2026-03-05 14:15:00 | 82.15 | STOP_HIT | 0.50 | 3.81% |
| BUY | retest1 | 2026-04-10 09:15:00 | 81.48 | 2026-04-13 14:15:00 | 79.89 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2026-04-15 09:15:00 | 81.64 | 2026-04-16 12:15:00 | 79.45 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2026-05-06 09:15:00 | 91.54 | 2026-05-06 10:15:00 | 90.11 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2026-05-06 09:45:00 | 91.29 | 2026-05-06 10:15:00 | 90.11 | STOP_HIT | 1.00 | -1.29% |
