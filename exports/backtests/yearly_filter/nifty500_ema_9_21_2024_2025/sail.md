# Steel Authority of India Ltd. (SAIL)

## Backtest Summary

- **Window:** 2024-03-13 10:15:00 → 2026-05-08 15:15:00 (3709 bars)
- **Last close:** 184.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 140 |
| ALERT1 | 102 |
| ALERT2 | 100 |
| ALERT2_SKIP | 45 |
| ALERT3 | 257 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 95 |
| PARTIAL | 15 |
| TARGET_HIT | 7 |
| STOP_HIT | 94 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 112 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 41 / 71
- **Target hits / Stop hits / Partials:** 7 / 92 / 13
- **Avg / median % per leg:** 0.39% / -0.80%
- **Sum % (uncompounded):** 43.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 50 | 14 | 28.0% | 5 | 45 | 0 | -0.07% | -3.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 50 | 14 | 28.0% | 5 | 45 | 0 | -0.07% | -3.5% |
| SELL (all) | 62 | 27 | 43.5% | 2 | 47 | 13 | 0.76% | 47.0% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.40% | -5.6% |
| SELL @ 3rd Alert (retest2) | 58 | 27 | 46.6% | 2 | 43 | 13 | 0.91% | 52.6% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.40% | -5.6% |
| retest2 (combined) | 108 | 41 | 38.0% | 7 | 88 | 13 | 0.45% | 49.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 160.25 | 157.15 | 156.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 12:15:00 | 163.15 | 158.89 | 157.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 11:15:00 | 165.25 | 165.56 | 163.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 11:15:00 | 165.25 | 165.56 | 163.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 165.25 | 165.56 | 163.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 11:30:00 | 164.20 | 165.56 | 163.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 162.90 | 164.84 | 163.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:30:00 | 162.50 | 164.84 | 163.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 163.80 | 164.63 | 163.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 166.20 | 164.44 | 163.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 12:15:00 | 167.65 | 168.94 | 169.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 12:15:00 | 167.65 | 168.94 | 169.09 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 09:15:00 | 171.00 | 169.30 | 169.19 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 10:15:00 | 167.15 | 169.23 | 169.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 14:15:00 | 166.70 | 168.07 | 168.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 12:15:00 | 163.70 | 163.60 | 165.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-29 12:45:00 | 163.40 | 163.60 | 165.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 165.95 | 160.49 | 160.91 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 164.50 | 161.30 | 161.24 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 149.75 | 160.49 | 161.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 133.10 | 155.01 | 158.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 12:15:00 | 141.20 | 140.86 | 147.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 13:00:00 | 141.20 | 140.86 | 147.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 145.10 | 142.44 | 147.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 15:00:00 | 145.10 | 142.44 | 147.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 149.10 | 143.89 | 147.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 150.30 | 143.89 | 147.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 151.80 | 145.48 | 147.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 11:00:00 | 151.80 | 145.48 | 147.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 13:15:00 | 148.65 | 146.92 | 147.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 13:30:00 | 149.60 | 146.92 | 147.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 14:15:00 | 146.85 | 146.90 | 147.64 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-06-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 12:15:00 | 149.80 | 148.13 | 148.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 14:15:00 | 152.75 | 149.35 | 148.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 150.67 | 150.90 | 149.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 15:00:00 | 150.67 | 150.90 | 149.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 151.01 | 150.88 | 150.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:30:00 | 150.30 | 150.88 | 150.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 14:15:00 | 151.06 | 151.30 | 150.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 151.96 | 151.32 | 150.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 09:15:00 | 152.09 | 151.11 | 150.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-13 12:15:00 | 150.32 | 150.82 | 150.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 12:15:00 | 150.32 | 150.82 | 150.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-13 13:15:00 | 149.90 | 150.63 | 150.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-14 09:15:00 | 151.16 | 150.46 | 150.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-14 09:15:00 | 151.16 | 150.46 | 150.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 151.16 | 150.46 | 150.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-14 10:00:00 | 151.16 | 150.46 | 150.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 10:15:00 | 151.31 | 150.63 | 150.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-14 10:45:00 | 151.35 | 150.63 | 150.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-06-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 11:15:00 | 151.22 | 150.75 | 150.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-14 12:15:00 | 151.81 | 150.96 | 150.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-18 13:15:00 | 152.81 | 152.96 | 152.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-18 13:30:00 | 152.91 | 152.96 | 152.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 151.08 | 152.72 | 152.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 151.17 | 152.72 | 152.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 150.59 | 152.29 | 152.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:30:00 | 150.37 | 152.29 | 152.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 12:15:00 | 150.66 | 151.80 | 151.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 13:15:00 | 150.52 | 151.54 | 151.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 09:15:00 | 152.51 | 151.27 | 151.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 09:15:00 | 152.51 | 151.27 | 151.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 152.51 | 151.27 | 151.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:00:00 | 152.51 | 151.27 | 151.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 153.44 | 151.70 | 151.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 11:00:00 | 153.44 | 151.70 | 151.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2024-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 11:15:00 | 154.00 | 152.16 | 151.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 15:15:00 | 154.25 | 153.17 | 152.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 12:15:00 | 155.50 | 155.55 | 154.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-21 13:00:00 | 155.50 | 155.55 | 154.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 150.90 | 154.72 | 154.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:30:00 | 149.16 | 154.72 | 154.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2024-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 11:15:00 | 152.05 | 153.73 | 153.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 12:15:00 | 151.15 | 153.21 | 153.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 09:15:00 | 145.77 | 145.47 | 147.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-27 09:45:00 | 146.65 | 145.47 | 147.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 149.01 | 144.63 | 145.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:30:00 | 149.26 | 144.63 | 145.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 149.25 | 145.55 | 146.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:30:00 | 150.19 | 145.55 | 146.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2024-06-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 12:15:00 | 149.00 | 146.56 | 146.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 15:15:00 | 149.20 | 147.60 | 147.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 09:15:00 | 148.74 | 148.78 | 148.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 09:15:00 | 148.74 | 148.78 | 148.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 148.74 | 148.78 | 148.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:45:00 | 148.66 | 148.78 | 148.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 148.96 | 148.82 | 148.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:30:00 | 148.25 | 148.82 | 148.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 146.89 | 148.43 | 148.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:00:00 | 146.89 | 148.43 | 148.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 146.28 | 148.00 | 147.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 146.28 | 148.00 | 147.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2024-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 13:15:00 | 146.63 | 147.73 | 147.79 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 12:15:00 | 149.90 | 148.04 | 147.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 14:15:00 | 151.69 | 149.04 | 148.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 14:15:00 | 151.00 | 151.30 | 150.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-04 15:00:00 | 151.00 | 151.30 | 150.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 151.40 | 151.30 | 150.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:30:00 | 150.93 | 151.30 | 150.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 149.94 | 154.72 | 154.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 149.94 | 154.72 | 154.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 10:15:00 | 149.37 | 153.65 | 154.18 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 09:15:00 | 152.59 | 152.06 | 151.98 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 14:15:00 | 150.99 | 151.99 | 152.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 15:15:00 | 150.83 | 151.76 | 151.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 14:15:00 | 143.26 | 143.09 | 144.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 15:00:00 | 143.26 | 143.09 | 144.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 141.25 | 142.74 | 144.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 137.66 | 142.54 | 144.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 13:00:00 | 140.12 | 142.05 | 143.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 13:30:00 | 140.06 | 141.84 | 143.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 14:15:00 | 140.41 | 141.84 | 143.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 146.09 | 142.54 | 143.32 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 146.09 | 142.54 | 143.32 | SL hit (close>static) qty=1.00 sl=144.59 alert=retest2 |

### Cycle 19 — BUY (started 2024-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 11:15:00 | 147.49 | 144.04 | 143.90 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 11:15:00 | 142.35 | 144.24 | 144.32 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 10:15:00 | 146.85 | 144.34 | 144.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 13:15:00 | 147.47 | 145.49 | 144.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 13:15:00 | 147.37 | 147.61 | 146.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 14:00:00 | 147.37 | 147.61 | 146.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 148.51 | 147.85 | 146.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 10:30:00 | 148.63 | 147.98 | 147.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 14:45:00 | 148.77 | 147.96 | 147.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:15:00 | 148.80 | 147.98 | 147.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 10:15:00 | 147.04 | 149.30 | 149.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2024-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 10:15:00 | 147.04 | 149.30 | 149.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 14:15:00 | 146.24 | 147.85 | 148.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 140.07 | 139.50 | 142.80 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 10:30:00 | 138.61 | 139.48 | 142.49 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 11:30:00 | 139.00 | 139.54 | 142.24 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 12:30:00 | 138.96 | 139.41 | 141.94 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 139.30 | 138.11 | 140.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:45:00 | 140.09 | 138.11 | 140.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 139.97 | 138.66 | 140.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:30:00 | 139.50 | 138.66 | 140.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 140.48 | 139.03 | 140.11 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-07 13:15:00 | 140.48 | 139.03 | 140.11 | SL hit (close>ema400) qty=1.00 sl=140.11 alert=retest1 |

### Cycle 23 — BUY (started 2024-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 11:15:00 | 130.46 | 128.96 | 128.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 12:15:00 | 131.41 | 129.45 | 129.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 09:15:00 | 133.70 | 134.18 | 132.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 10:00:00 | 133.70 | 134.18 | 132.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 133.79 | 134.01 | 133.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:30:00 | 133.98 | 134.01 | 133.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 134.31 | 134.00 | 133.55 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2024-08-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 15:15:00 | 131.68 | 133.05 | 133.23 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 10:15:00 | 134.30 | 133.52 | 133.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 12:15:00 | 135.70 | 134.07 | 133.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-27 14:15:00 | 135.91 | 136.52 | 135.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-27 15:00:00 | 135.91 | 136.52 | 135.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 135.90 | 136.40 | 135.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:15:00 | 136.08 | 136.40 | 135.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 135.27 | 136.17 | 135.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 10:00:00 | 135.27 | 136.17 | 135.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 134.92 | 135.92 | 135.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 11:00:00 | 134.92 | 135.92 | 135.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 11:15:00 | 134.86 | 135.71 | 135.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 11:30:00 | 134.75 | 135.71 | 135.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 13:15:00 | 134.81 | 135.41 | 135.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 14:00:00 | 134.81 | 135.41 | 135.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2024-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 14:15:00 | 134.29 | 135.19 | 135.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 15:15:00 | 133.93 | 134.94 | 135.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 14:15:00 | 134.10 | 133.63 | 134.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 14:15:00 | 134.10 | 133.63 | 134.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 134.10 | 133.63 | 134.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 15:00:00 | 134.10 | 133.63 | 134.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 134.75 | 133.85 | 134.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 134.71 | 133.85 | 134.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 135.10 | 134.10 | 134.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:45:00 | 135.10 | 134.10 | 134.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 135.23 | 134.33 | 134.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:45:00 | 135.15 | 134.33 | 134.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 133.64 | 134.29 | 134.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 09:15:00 | 132.97 | 134.22 | 134.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 09:30:00 | 132.83 | 133.05 | 133.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 12:15:00 | 126.32 | 128.23 | 129.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 12:15:00 | 126.19 | 128.23 | 129.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-09 15:15:00 | 128.40 | 127.99 | 129.00 | SL hit (close>ema200) qty=0.50 sl=127.99 alert=retest2 |

### Cycle 27 — BUY (started 2024-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 14:15:00 | 130.43 | 128.85 | 128.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 133.75 | 130.12 | 129.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 130.82 | 132.34 | 131.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 09:15:00 | 130.82 | 132.34 | 131.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 130.82 | 132.34 | 131.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:00:00 | 130.82 | 132.34 | 131.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 131.45 | 132.16 | 131.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 12:30:00 | 132.18 | 132.02 | 131.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 11:15:00 | 130.89 | 131.54 | 131.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 11:15:00 | 130.89 | 131.54 | 131.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 12:15:00 | 129.77 | 131.18 | 131.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 127.36 | 127.10 | 128.47 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-20 15:00:00 | 126.11 | 126.99 | 127.92 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 127.70 | 127.00 | 127.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 10:15:00 | 127.94 | 127.00 | 127.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 128.76 | 127.35 | 127.84 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-23 10:15:00 | 128.76 | 127.35 | 127.84 | SL hit (close>ema400) qty=1.00 sl=127.84 alert=retest1 |

### Cycle 29 — BUY (started 2024-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 13:15:00 | 129.73 | 128.45 | 128.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 09:15:00 | 134.60 | 130.02 | 129.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 14:15:00 | 134.19 | 134.23 | 132.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 14:45:00 | 134.22 | 134.23 | 132.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 134.00 | 134.16 | 133.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 10:45:00 | 135.47 | 134.46 | 133.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 12:15:00 | 137.66 | 139.62 | 139.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 12:15:00 | 137.66 | 139.62 | 139.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 136.27 | 138.57 | 138.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 10:15:00 | 132.68 | 132.03 | 133.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-09 11:00:00 | 132.68 | 132.03 | 133.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 131.70 | 131.44 | 132.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 11:00:00 | 130.55 | 131.26 | 132.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-11 09:15:00 | 133.01 | 131.14 | 131.76 | SL hit (close>static) qty=1.00 sl=132.95 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 14:15:00 | 134.35 | 132.51 | 132.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 09:15:00 | 134.69 | 133.17 | 132.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 09:15:00 | 131.93 | 133.64 | 133.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 09:15:00 | 131.93 | 133.64 | 133.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 131.93 | 133.64 | 133.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:00:00 | 131.93 | 133.64 | 133.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 132.45 | 133.40 | 133.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:30:00 | 132.20 | 133.40 | 133.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2024-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 12:15:00 | 130.62 | 132.56 | 132.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 10:15:00 | 130.00 | 131.19 | 131.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 11:15:00 | 129.01 | 128.42 | 129.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 11:15:00 | 129.01 | 128.42 | 129.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 129.01 | 128.42 | 129.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:00:00 | 129.01 | 128.42 | 129.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 130.19 | 128.77 | 129.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:30:00 | 129.90 | 128.77 | 129.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 129.20 | 128.86 | 129.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 14:15:00 | 129.00 | 128.86 | 129.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 09:30:00 | 129.06 | 128.73 | 129.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 122.55 | 124.60 | 126.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 122.61 | 124.60 | 126.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-25 09:15:00 | 116.10 | 116.72 | 119.26 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 116.50 | 115.26 | 115.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 11:15:00 | 117.15 | 115.63 | 115.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 115.90 | 115.91 | 115.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 14:00:00 | 115.90 | 115.91 | 115.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 116.95 | 116.14 | 115.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:45:00 | 118.20 | 116.39 | 116.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 09:15:00 | 114.44 | 116.00 | 115.86 | SL hit (close<static) qty=1.00 sl=115.15 alert=retest2 |

### Cycle 34 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 113.10 | 115.42 | 115.61 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-11-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 11:15:00 | 116.45 | 115.39 | 115.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 12:15:00 | 116.84 | 115.68 | 115.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 14:15:00 | 123.31 | 123.32 | 121.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 15:00:00 | 123.31 | 123.32 | 121.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 119.32 | 122.50 | 121.26 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 118.06 | 120.20 | 120.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 116.10 | 118.75 | 119.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 112.81 | 112.75 | 114.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 112.81 | 112.75 | 114.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 112.81 | 112.75 | 114.29 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 114.13 | 113.53 | 113.49 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 111.34 | 113.06 | 113.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 109.73 | 112.14 | 112.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 111.65 | 111.38 | 112.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 09:30:00 | 111.08 | 111.38 | 112.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 11:15:00 | 112.30 | 111.63 | 112.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 12:00:00 | 112.30 | 111.63 | 112.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 113.20 | 111.94 | 112.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 13:00:00 | 113.20 | 111.94 | 112.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2024-11-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 14:15:00 | 112.87 | 112.30 | 112.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 114.43 | 112.80 | 112.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 09:15:00 | 115.06 | 115.33 | 114.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 09:15:00 | 115.06 | 115.33 | 114.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 115.06 | 115.33 | 114.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:30:00 | 114.91 | 115.33 | 114.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 115.95 | 116.16 | 115.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 13:45:00 | 115.98 | 116.16 | 115.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 116.21 | 116.17 | 115.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:30:00 | 115.84 | 116.17 | 115.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 116.64 | 116.37 | 115.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 11:15:00 | 117.01 | 116.37 | 115.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 13:15:00 | 116.88 | 116.38 | 116.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 09:30:00 | 116.96 | 117.02 | 116.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-11 09:15:00 | 128.57 | 126.92 | 125.73 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 10:15:00 | 123.06 | 126.19 | 126.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 14:15:00 | 121.00 | 122.66 | 123.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 10:15:00 | 119.21 | 119.04 | 120.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-20 10:45:00 | 119.24 | 119.04 | 120.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 120.53 | 119.34 | 120.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 12:00:00 | 120.53 | 119.34 | 120.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 119.31 | 119.33 | 120.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 14:00:00 | 118.77 | 119.22 | 119.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 14:30:00 | 118.86 | 118.56 | 119.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-23 14:15:00 | 121.60 | 119.85 | 119.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2024-12-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 14:15:00 | 121.60 | 119.85 | 119.74 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 10:15:00 | 118.94 | 119.70 | 119.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-24 11:15:00 | 118.64 | 119.49 | 119.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 12:15:00 | 119.55 | 119.50 | 119.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 12:15:00 | 119.55 | 119.50 | 119.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 12:15:00 | 119.55 | 119.50 | 119.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 13:00:00 | 119.55 | 119.50 | 119.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 13:15:00 | 119.38 | 119.48 | 119.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 14:30:00 | 118.55 | 119.38 | 119.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 09:30:00 | 118.21 | 118.93 | 119.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 14:15:00 | 112.62 | 113.97 | 115.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 14:15:00 | 112.30 | 113.97 | 115.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-31 10:15:00 | 113.40 | 113.31 | 114.89 | SL hit (close>ema200) qty=0.50 sl=113.31 alert=retest2 |

### Cycle 43 — BUY (started 2025-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 09:15:00 | 114.90 | 113.88 | 113.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 11:15:00 | 115.55 | 114.38 | 114.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 13:15:00 | 114.53 | 114.55 | 114.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 14:00:00 | 114.53 | 114.55 | 114.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 114.31 | 114.50 | 114.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 114.31 | 114.50 | 114.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 114.10 | 114.42 | 114.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 112.30 | 114.42 | 114.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 112.62 | 114.06 | 114.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 112.12 | 114.06 | 114.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 110.38 | 113.33 | 113.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 15:15:00 | 110.15 | 111.46 | 112.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 111.53 | 111.47 | 112.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 09:30:00 | 112.08 | 111.47 | 112.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 111.61 | 111.63 | 112.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 15:15:00 | 111.29 | 111.63 | 112.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 105.73 | 108.28 | 109.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-10 10:15:00 | 108.57 | 108.34 | 109.41 | SL hit (close>ema200) qty=0.50 sl=108.34 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 10:15:00 | 107.39 | 105.74 | 105.60 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-01-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 15:15:00 | 105.21 | 105.57 | 105.59 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 106.79 | 105.81 | 105.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 108.36 | 107.14 | 106.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 10:15:00 | 109.00 | 109.28 | 108.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 10:15:00 | 109.00 | 109.28 | 108.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 109.00 | 109.28 | 108.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 109.00 | 109.28 | 108.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 109.75 | 109.37 | 108.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 109.23 | 109.37 | 108.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 107.89 | 109.28 | 108.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:45:00 | 108.20 | 109.28 | 108.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 107.10 | 108.84 | 108.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 11:00:00 | 107.10 | 108.84 | 108.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 11:15:00 | 106.20 | 108.32 | 108.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 13:15:00 | 105.68 | 107.42 | 108.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 108.40 | 107.54 | 107.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 108.40 | 107.54 | 107.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 108.40 | 107.54 | 107.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 108.40 | 107.54 | 107.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 108.82 | 107.80 | 108.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:45:00 | 109.23 | 107.80 | 108.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 12:15:00 | 109.52 | 108.26 | 108.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-24 11:15:00 | 110.02 | 109.06 | 108.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 13:15:00 | 108.54 | 109.04 | 108.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 13:15:00 | 108.54 | 109.04 | 108.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 13:15:00 | 108.54 | 109.04 | 108.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:00:00 | 108.54 | 109.04 | 108.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 14:15:00 | 108.24 | 108.88 | 108.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:45:00 | 108.40 | 108.88 | 108.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 15:15:00 | 107.71 | 108.65 | 108.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 09:15:00 | 105.27 | 108.65 | 108.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 105.15 | 107.95 | 108.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 11:15:00 | 104.25 | 106.75 | 107.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 104.06 | 104.00 | 105.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:00:00 | 104.06 | 104.00 | 105.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 104.95 | 103.94 | 104.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 104.95 | 103.94 | 104.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 103.86 | 103.93 | 104.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 11:15:00 | 103.42 | 103.93 | 104.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 12:00:00 | 103.61 | 103.86 | 104.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 09:15:00 | 106.10 | 104.50 | 104.71 | SL hit (close>static) qty=1.00 sl=105.14 alert=retest2 |

### Cycle 51 — BUY (started 2025-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 11:15:00 | 105.37 | 104.91 | 104.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 10:15:00 | 107.89 | 105.76 | 105.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 106.63 | 107.04 | 106.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 106.63 | 107.04 | 106.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 106.63 | 107.04 | 106.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 106.24 | 107.04 | 106.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 104.96 | 106.62 | 106.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 104.96 | 106.62 | 106.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 106.53 | 106.60 | 106.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:45:00 | 106.86 | 106.54 | 106.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 101.82 | 105.56 | 105.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 101.82 | 105.56 | 105.88 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 13:15:00 | 106.11 | 105.00 | 104.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 14:15:00 | 106.92 | 105.38 | 105.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 11:15:00 | 107.69 | 108.01 | 107.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 12:00:00 | 107.69 | 108.01 | 107.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 108.01 | 108.01 | 107.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:45:00 | 107.91 | 108.01 | 107.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 107.31 | 107.87 | 107.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:45:00 | 107.50 | 107.87 | 107.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 108.31 | 107.91 | 107.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 09:15:00 | 108.72 | 107.91 | 107.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:30:00 | 109.29 | 108.48 | 107.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 105.64 | 108.89 | 108.40 | SL hit (close<static) qty=1.00 sl=107.33 alert=retest2 |

### Cycle 54 — SELL (started 2025-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 11:15:00 | 105.27 | 107.67 | 107.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 15:15:00 | 104.96 | 106.18 | 107.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 102.70 | 101.84 | 103.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 10:45:00 | 102.66 | 101.84 | 103.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 103.59 | 102.19 | 103.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:00:00 | 103.59 | 102.19 | 103.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 104.79 | 102.71 | 103.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 104.79 | 102.71 | 103.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 104.87 | 103.14 | 103.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:30:00 | 105.60 | 103.14 | 103.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 09:15:00 | 109.82 | 105.21 | 104.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 11:15:00 | 110.64 | 107.06 | 105.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-14 10:15:00 | 107.65 | 108.52 | 107.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-14 10:15:00 | 107.65 | 108.52 | 107.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 10:15:00 | 107.65 | 108.52 | 107.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 11:00:00 | 107.65 | 108.52 | 107.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 11:15:00 | 106.62 | 108.14 | 107.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 12:00:00 | 106.62 | 108.14 | 107.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 12:15:00 | 106.14 | 107.74 | 107.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 13:00:00 | 106.14 | 107.74 | 107.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 13:15:00 | 104.66 | 107.12 | 106.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 14:00:00 | 104.66 | 107.12 | 106.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2025-02-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 15:15:00 | 105.44 | 106.53 | 106.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-17 09:15:00 | 103.51 | 105.93 | 106.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 14:15:00 | 105.43 | 104.91 | 105.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 14:15:00 | 105.43 | 104.91 | 105.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 105.43 | 104.91 | 105.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 105.43 | 104.91 | 105.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 103.20 | 104.64 | 105.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 12:45:00 | 102.16 | 103.79 | 104.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 09:15:00 | 106.05 | 104.31 | 104.67 | SL hit (close>static) qty=1.00 sl=105.75 alert=retest2 |

### Cycle 57 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 106.89 | 105.09 | 104.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 107.95 | 106.38 | 105.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 09:15:00 | 110.00 | 111.48 | 109.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 09:15:00 | 110.00 | 111.48 | 109.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 110.00 | 111.48 | 109.83 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2025-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 15:15:00 | 108.00 | 109.24 | 109.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 09:15:00 | 107.80 | 108.95 | 109.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 106.46 | 105.88 | 106.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-27 15:00:00 | 106.46 | 105.88 | 106.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 104.82 | 104.45 | 105.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 09:15:00 | 105.51 | 104.45 | 105.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 103.74 | 104.31 | 105.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 11:30:00 | 103.40 | 104.11 | 105.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-03 15:15:00 | 107.23 | 105.48 | 105.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-03-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 15:15:00 | 107.23 | 105.48 | 105.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 09:15:00 | 107.78 | 105.94 | 105.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-04 12:15:00 | 106.25 | 106.44 | 106.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 12:15:00 | 106.25 | 106.44 | 106.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 12:15:00 | 106.25 | 106.44 | 106.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-04 13:00:00 | 106.25 | 106.44 | 106.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 13:15:00 | 107.17 | 106.59 | 106.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-04 13:30:00 | 106.23 | 106.59 | 106.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 108.02 | 106.88 | 106.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-04 14:45:00 | 107.13 | 106.88 | 106.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 110.77 | 111.85 | 110.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:00:00 | 110.77 | 111.85 | 110.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 110.92 | 111.67 | 110.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 13:30:00 | 111.28 | 111.51 | 110.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 14:15:00 | 111.33 | 111.51 | 110.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:30:00 | 111.70 | 111.41 | 110.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 11:15:00 | 109.80 | 111.03 | 110.89 | SL hit (close<static) qty=1.00 sl=110.20 alert=retest2 |

### Cycle 60 — SELL (started 2025-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 12:15:00 | 109.35 | 110.69 | 110.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 13:15:00 | 108.12 | 110.18 | 110.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 14:15:00 | 107.99 | 107.34 | 108.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 14:15:00 | 107.99 | 107.34 | 108.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 107.99 | 107.34 | 108.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 15:00:00 | 107.99 | 107.34 | 108.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 108.77 | 107.63 | 108.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 107.85 | 107.63 | 108.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 106.30 | 107.36 | 108.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:15:00 | 106.10 | 107.36 | 108.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 11:00:00 | 106.00 | 107.09 | 108.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 14:45:00 | 106.06 | 107.13 | 107.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 09:45:00 | 106.24 | 106.69 | 107.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 13:15:00 | 106.98 | 106.38 | 106.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 14:00:00 | 106.98 | 106.38 | 106.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 105.90 | 106.28 | 106.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 14:45:00 | 106.67 | 106.28 | 106.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 107.64 | 106.59 | 106.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:45:00 | 107.61 | 106.59 | 106.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-18 10:15:00 | 108.39 | 106.95 | 106.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 108.39 | 106.95 | 106.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 109.04 | 107.37 | 107.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 115.61 | 116.68 | 115.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:15:00 | 115.80 | 116.68 | 115.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 115.73 | 116.49 | 115.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:30:00 | 115.22 | 116.49 | 115.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 114.89 | 116.17 | 115.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:45:00 | 114.94 | 116.17 | 115.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 115.65 | 116.07 | 115.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 13:45:00 | 116.21 | 116.07 | 115.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 15:15:00 | 114.30 | 115.51 | 115.34 | SL hit (close<static) qty=1.00 sl=114.51 alert=retest2 |

### Cycle 62 — SELL (started 2025-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 11:15:00 | 114.37 | 115.22 | 115.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 12:15:00 | 113.89 | 114.95 | 115.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 10:15:00 | 114.80 | 114.04 | 114.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 10:15:00 | 114.80 | 114.04 | 114.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 114.80 | 114.04 | 114.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:45:00 | 115.24 | 114.04 | 114.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 11:15:00 | 115.20 | 114.27 | 114.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 12:00:00 | 115.20 | 114.27 | 114.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 114.29 | 114.29 | 114.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 13:30:00 | 114.30 | 114.29 | 114.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2025-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 14:15:00 | 115.05 | 114.44 | 114.41 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 10:15:00 | 113.93 | 114.41 | 114.41 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 11:15:00 | 114.95 | 114.51 | 114.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 12:15:00 | 116.29 | 114.87 | 114.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 12:15:00 | 117.69 | 117.82 | 116.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-03 13:00:00 | 117.69 | 117.82 | 116.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 112.65 | 117.06 | 116.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 112.65 | 117.06 | 116.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 114.19 | 116.49 | 116.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 103.48 | 111.84 | 114.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-09 12:15:00 | 104.73 | 104.08 | 106.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-09 13:00:00 | 104.73 | 104.08 | 106.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 108.95 | 105.32 | 106.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 09:45:00 | 108.95 | 105.32 | 106.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 109.00 | 106.53 | 106.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 111.80 | 108.44 | 107.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 113.43 | 113.93 | 112.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:00:00 | 113.43 | 113.93 | 112.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 113.51 | 114.14 | 113.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:15:00 | 113.50 | 114.14 | 113.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 114.60 | 114.23 | 113.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 10:15:00 | 115.52 | 114.23 | 113.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 10:15:00 | 115.11 | 115.98 | 115.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 10:45:00 | 114.95 | 115.86 | 115.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 10:30:00 | 114.89 | 116.62 | 116.48 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 11:15:00 | 114.80 | 116.26 | 116.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 114.80 | 116.26 | 116.32 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 11:15:00 | 117.25 | 116.19 | 116.17 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 13:15:00 | 115.54 | 116.20 | 116.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 10:15:00 | 114.98 | 115.64 | 115.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 116.11 | 114.75 | 115.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 116.11 | 114.75 | 115.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 116.11 | 114.75 | 115.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:45:00 | 116.09 | 114.75 | 115.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 114.60 | 114.72 | 115.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:15:00 | 113.88 | 114.72 | 115.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 09:15:00 | 113.90 | 114.32 | 114.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:15:00 | 113.83 | 114.29 | 114.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 108.19 | 110.45 | 111.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 108.20 | 110.45 | 111.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 108.14 | 110.45 | 111.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 114.32 | 110.14 | 110.76 | SL hit (close>ema200) qty=0.50 sl=110.14 alert=retest2 |

### Cycle 71 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 114.73 | 111.86 | 111.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 117.38 | 112.96 | 112.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 116.45 | 116.50 | 114.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:45:00 | 116.38 | 116.50 | 114.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 122.09 | 122.22 | 121.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 121.60 | 122.22 | 121.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 122.63 | 123.27 | 122.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 122.63 | 123.27 | 122.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 122.68 | 123.15 | 122.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:30:00 | 123.03 | 123.33 | 122.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 127.10 | 128.95 | 128.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 09:15:00 | 127.10 | 128.95 | 128.97 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 12:15:00 | 129.69 | 128.95 | 128.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 14:15:00 | 130.54 | 129.42 | 129.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 130.55 | 130.77 | 130.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 13:00:00 | 130.55 | 130.77 | 130.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 130.33 | 130.68 | 130.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:45:00 | 130.18 | 130.68 | 130.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 129.90 | 130.53 | 130.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 15:00:00 | 129.90 | 130.53 | 130.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 130.19 | 130.46 | 130.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 130.55 | 130.62 | 130.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 133.65 | 134.12 | 133.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 09:15:00 | 134.19 | 134.12 | 133.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 11:00:00 | 134.11 | 134.18 | 133.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 12:15:00 | 133.10 | 133.87 | 133.68 | SL hit (close<static) qty=1.00 sl=133.55 alert=retest2 |

### Cycle 74 — SELL (started 2025-06-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 15:15:00 | 132.81 | 133.43 | 133.50 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-06-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 11:15:00 | 134.04 | 133.54 | 133.53 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 14:15:00 | 133.25 | 133.50 | 133.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 15:15:00 | 132.86 | 133.37 | 133.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 129.65 | 129.12 | 130.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 129.65 | 129.12 | 130.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 129.01 | 129.03 | 129.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 129.85 | 129.03 | 129.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 126.05 | 125.55 | 126.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:45:00 | 126.00 | 125.55 | 126.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 127.21 | 125.88 | 126.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 127.21 | 125.88 | 126.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 127.11 | 126.13 | 126.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:30:00 | 127.15 | 126.13 | 126.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 126.55 | 126.26 | 126.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:30:00 | 126.61 | 126.26 | 126.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 127.53 | 126.52 | 126.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 127.53 | 126.52 | 126.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 127.27 | 126.67 | 126.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 126.71 | 126.67 | 126.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 11:15:00 | 126.94 | 126.72 | 126.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 126.94 | 126.72 | 126.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 12:15:00 | 128.22 | 127.02 | 126.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 14:15:00 | 126.71 | 127.05 | 126.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 14:15:00 | 126.71 | 127.05 | 126.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 126.71 | 127.05 | 126.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:30:00 | 126.80 | 127.05 | 126.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 126.54 | 126.95 | 126.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 128.87 | 126.95 | 126.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 10:15:00 | 130.56 | 131.54 | 131.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 130.56 | 131.54 | 131.57 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 11:15:00 | 131.79 | 131.59 | 131.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 12:15:00 | 132.28 | 131.73 | 131.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 14:15:00 | 136.48 | 136.71 | 135.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-03 14:30:00 | 136.85 | 136.71 | 135.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 135.93 | 136.46 | 135.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 10:45:00 | 136.56 | 136.49 | 135.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 11:15:00 | 134.33 | 136.06 | 135.49 | SL hit (close<static) qty=1.00 sl=135.07 alert=retest2 |

### Cycle 80 — SELL (started 2025-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 15:15:00 | 134.80 | 135.17 | 135.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 09:15:00 | 133.50 | 134.84 | 135.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 15:15:00 | 133.66 | 133.62 | 134.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 09:15:00 | 134.58 | 133.62 | 134.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 134.49 | 133.80 | 134.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:45:00 | 134.47 | 133.80 | 134.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 134.26 | 133.89 | 134.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:30:00 | 134.64 | 133.89 | 134.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 134.16 | 133.94 | 134.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:30:00 | 134.20 | 133.94 | 134.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 134.78 | 134.11 | 134.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:30:00 | 134.78 | 134.11 | 134.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 134.53 | 134.19 | 134.30 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2025-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 15:15:00 | 135.11 | 134.51 | 134.44 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 09:15:00 | 132.44 | 134.10 | 134.25 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 11:15:00 | 134.32 | 134.02 | 133.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 15:15:00 | 134.90 | 134.40 | 134.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 13:15:00 | 134.41 | 134.71 | 134.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 13:15:00 | 134.41 | 134.71 | 134.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 134.41 | 134.71 | 134.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:00:00 | 134.41 | 134.71 | 134.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 134.50 | 134.67 | 134.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 15:00:00 | 134.50 | 134.67 | 134.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 133.59 | 134.46 | 134.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:15:00 | 134.19 | 134.46 | 134.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 135.00 | 134.56 | 134.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 10:15:00 | 135.30 | 134.56 | 134.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 12:15:00 | 133.36 | 134.23 | 134.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 12:15:00 | 133.36 | 134.23 | 134.32 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 14:15:00 | 135.00 | 134.31 | 134.27 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 09:15:00 | 132.94 | 134.09 | 134.18 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 09:15:00 | 134.94 | 133.80 | 133.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 14:15:00 | 136.59 | 134.88 | 134.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 12:15:00 | 136.71 | 137.09 | 136.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 12:15:00 | 136.71 | 137.09 | 136.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 136.71 | 137.09 | 136.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:30:00 | 137.05 | 137.09 | 136.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 136.81 | 137.04 | 136.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:45:00 | 136.92 | 137.04 | 136.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 136.10 | 136.85 | 136.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 136.10 | 136.85 | 136.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 135.77 | 136.63 | 136.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:15:00 | 135.72 | 136.63 | 136.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 135.30 | 136.15 | 136.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:30:00 | 135.30 | 136.15 | 136.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2025-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 11:15:00 | 134.41 | 135.80 | 135.94 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 13:15:00 | 136.38 | 135.74 | 135.70 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 132.18 | 135.10 | 135.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 130.79 | 134.24 | 135.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 15:15:00 | 126.14 | 126.03 | 127.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-30 09:15:00 | 125.19 | 126.03 | 127.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 123.95 | 122.12 | 123.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:45:00 | 123.43 | 122.12 | 123.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 123.51 | 122.40 | 123.28 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 125.44 | 123.97 | 123.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 15:15:00 | 125.70 | 124.31 | 123.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 15:15:00 | 124.89 | 124.97 | 124.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 15:15:00 | 124.89 | 124.97 | 124.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 124.89 | 124.97 | 124.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:30:00 | 124.38 | 124.74 | 124.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 123.03 | 124.40 | 124.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 123.03 | 124.40 | 124.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 123.01 | 124.12 | 124.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 121.65 | 123.24 | 123.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 123.35 | 122.37 | 123.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 123.35 | 122.37 | 123.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 123.35 | 122.37 | 123.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 123.35 | 122.37 | 123.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 123.75 | 122.65 | 123.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 123.20 | 122.65 | 123.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 11:15:00 | 123.05 | 122.00 | 121.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 11:15:00 | 123.05 | 122.00 | 121.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 123.80 | 122.57 | 122.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 11:15:00 | 122.48 | 122.71 | 122.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 11:15:00 | 122.48 | 122.71 | 122.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 122.48 | 122.71 | 122.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 122.48 | 122.71 | 122.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 122.41 | 122.65 | 122.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:30:00 | 122.43 | 122.65 | 122.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 122.35 | 122.59 | 122.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:00:00 | 122.35 | 122.59 | 122.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 124.15 | 122.90 | 122.57 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 121.17 | 122.29 | 122.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 12:15:00 | 121.05 | 122.04 | 122.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 122.72 | 121.56 | 121.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 122.72 | 121.56 | 121.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 122.72 | 121.56 | 121.90 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2025-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 13:15:00 | 122.72 | 122.12 | 122.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 123.44 | 122.48 | 122.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 122.86 | 122.90 | 122.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 09:15:00 | 122.86 | 122.90 | 122.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 122.86 | 122.90 | 122.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:00:00 | 122.86 | 122.90 | 122.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 125.18 | 123.36 | 122.87 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 09:15:00 | 122.05 | 123.27 | 123.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 120.87 | 122.37 | 122.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 119.85 | 119.46 | 120.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:30:00 | 120.02 | 119.46 | 120.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 119.60 | 119.22 | 119.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 119.60 | 119.22 | 119.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 119.90 | 119.35 | 119.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 119.90 | 119.35 | 119.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 120.40 | 119.56 | 119.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:30:00 | 120.40 | 119.56 | 119.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 121.12 | 120.17 | 120.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 122.50 | 120.79 | 120.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 14:15:00 | 127.34 | 127.77 | 126.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 15:00:00 | 127.34 | 127.77 | 126.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 130.47 | 131.14 | 129.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:30:00 | 130.73 | 131.14 | 129.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 129.81 | 130.71 | 129.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:00:00 | 129.81 | 130.71 | 129.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 130.16 | 130.60 | 129.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 10:00:00 | 131.15 | 130.31 | 129.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:30:00 | 130.75 | 130.21 | 130.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 10:15:00 | 132.01 | 132.46 | 132.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 10:15:00 | 132.01 | 132.46 | 132.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 12:15:00 | 131.60 | 132.19 | 132.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 13:15:00 | 132.27 | 132.20 | 132.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 13:15:00 | 132.27 | 132.20 | 132.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 132.27 | 132.20 | 132.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:30:00 | 132.54 | 132.20 | 132.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 132.75 | 132.31 | 132.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 15:00:00 | 132.75 | 132.31 | 132.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 15:15:00 | 133.54 | 132.56 | 132.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 09:15:00 | 135.49 | 133.15 | 132.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 135.50 | 136.09 | 135.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 14:30:00 | 135.32 | 136.09 | 135.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 136.24 | 136.03 | 135.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 09:30:00 | 137.83 | 136.90 | 136.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 13:30:00 | 137.85 | 137.20 | 136.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 09:45:00 | 138.37 | 137.09 | 136.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 11:15:00 | 137.71 | 137.20 | 136.70 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 136.49 | 137.09 | 136.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:00:00 | 136.49 | 137.09 | 136.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 136.52 | 136.98 | 136.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:15:00 | 134.76 | 136.98 | 136.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-26 09:15:00 | 134.31 | 136.45 | 136.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 09:15:00 | 134.31 | 136.45 | 136.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 10:15:00 | 133.18 | 135.79 | 136.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 133.22 | 132.93 | 134.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 133.22 | 132.93 | 134.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 134.55 | 132.97 | 133.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:00:00 | 134.55 | 132.97 | 133.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 133.93 | 133.16 | 133.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 14:00:00 | 133.28 | 133.19 | 133.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 135.55 | 134.04 | 133.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 09:15:00 | 135.55 | 134.04 | 133.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 10:15:00 | 137.22 | 135.37 | 134.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 14:15:00 | 134.95 | 135.72 | 135.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 14:15:00 | 134.95 | 135.72 | 135.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 134.95 | 135.72 | 135.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 15:00:00 | 134.95 | 135.72 | 135.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 134.95 | 135.57 | 135.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:15:00 | 134.00 | 135.57 | 135.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 136.02 | 135.66 | 135.15 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 11:15:00 | 132.17 | 134.52 | 134.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 11:15:00 | 131.76 | 132.72 | 133.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 14:15:00 | 132.76 | 132.46 | 133.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-07 15:00:00 | 132.76 | 132.46 | 133.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 132.60 | 132.48 | 133.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 131.90 | 132.48 | 133.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 09:15:00 | 134.79 | 132.28 | 132.51 | SL hit (close>static) qty=1.00 sl=133.13 alert=retest2 |

### Cycle 103 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 136.28 | 133.08 | 132.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 14:15:00 | 136.63 | 134.34 | 133.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 09:15:00 | 133.56 | 134.53 | 133.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 133.56 | 134.53 | 133.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 133.56 | 134.53 | 133.82 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 13:15:00 | 132.27 | 133.45 | 133.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 14:15:00 | 132.20 | 133.20 | 133.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 14:15:00 | 132.50 | 131.90 | 132.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 14:15:00 | 132.50 | 131.90 | 132.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 132.50 | 131.90 | 132.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 15:00:00 | 132.50 | 131.90 | 132.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 132.65 | 132.05 | 132.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:15:00 | 133.41 | 132.05 | 132.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 131.97 | 132.03 | 132.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 10:15:00 | 131.70 | 132.03 | 132.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 12:15:00 | 131.79 | 130.92 | 130.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2025-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 12:15:00 | 131.79 | 130.92 | 130.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 13:15:00 | 132.42 | 131.22 | 131.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 15:15:00 | 131.20 | 131.26 | 131.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 15:15:00 | 131.20 | 131.26 | 131.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 131.20 | 131.26 | 131.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 131.63 | 131.26 | 131.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 130.90 | 131.19 | 131.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:15:00 | 130.43 | 131.19 | 131.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 131.34 | 131.22 | 131.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:00:00 | 131.34 | 131.22 | 131.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 12:15:00 | 129.38 | 130.97 | 131.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 13:15:00 | 129.16 | 130.61 | 130.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 12:15:00 | 130.10 | 129.60 | 130.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 12:15:00 | 130.10 | 129.60 | 130.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 130.10 | 129.60 | 130.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:00:00 | 130.10 | 129.60 | 130.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 129.91 | 129.66 | 130.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:45:00 | 130.15 | 129.66 | 130.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 129.83 | 129.70 | 130.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 15:15:00 | 129.00 | 129.72 | 129.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 09:15:00 | 131.44 | 129.95 | 129.97 | SL hit (close>static) qty=1.00 sl=130.12 alert=retest2 |

### Cycle 107 — BUY (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 11:15:00 | 130.38 | 130.04 | 130.01 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-10-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 13:15:00 | 129.90 | 129.97 | 129.98 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 09:15:00 | 130.49 | 130.01 | 129.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 10:15:00 | 131.79 | 130.37 | 130.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 137.89 | 138.61 | 135.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 10:15:00 | 136.70 | 138.22 | 136.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 136.70 | 138.22 | 136.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:45:00 | 136.45 | 138.22 | 136.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 136.15 | 137.81 | 136.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:45:00 | 135.83 | 137.81 | 136.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 136.27 | 137.50 | 136.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:15:00 | 136.05 | 137.50 | 136.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 136.34 | 137.27 | 136.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:30:00 | 136.02 | 137.27 | 136.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 137.76 | 137.32 | 136.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 10:30:00 | 138.53 | 137.49 | 136.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 139.62 | 138.24 | 137.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 09:15:00 | 139.05 | 138.13 | 137.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 136.78 | 137.67 | 137.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 136.78 | 137.67 | 137.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 135.77 | 137.31 | 137.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 137.96 | 137.42 | 137.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 11:15:00 | 137.96 | 137.42 | 137.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 137.96 | 137.42 | 137.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:45:00 | 138.12 | 137.42 | 137.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2025-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 12:15:00 | 141.30 | 138.20 | 137.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 09:15:00 | 142.20 | 139.96 | 138.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 142.15 | 142.88 | 141.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-11 09:45:00 | 142.90 | 142.88 | 141.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 143.35 | 143.67 | 143.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 15:00:00 | 143.35 | 143.67 | 143.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 143.00 | 143.53 | 143.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 144.25 | 143.53 | 143.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 10:00:00 | 144.02 | 143.63 | 143.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 13:00:00 | 143.87 | 143.94 | 143.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 11:15:00 | 142.72 | 143.70 | 143.54 | SL hit (close<static) qty=1.00 sl=142.90 alert=retest2 |

### Cycle 112 — SELL (started 2025-11-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 13:15:00 | 141.84 | 143.28 | 143.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 15:15:00 | 141.29 | 141.89 | 142.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 09:15:00 | 139.84 | 139.29 | 140.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 09:45:00 | 139.50 | 139.29 | 140.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 140.15 | 139.46 | 140.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:30:00 | 140.31 | 139.46 | 140.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 140.15 | 139.60 | 140.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:30:00 | 140.30 | 139.60 | 140.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 139.95 | 139.67 | 140.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:30:00 | 140.60 | 139.67 | 140.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 140.12 | 139.76 | 140.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 15:00:00 | 140.12 | 139.76 | 140.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 139.80 | 139.77 | 140.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 139.50 | 139.77 | 140.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 138.57 | 139.53 | 140.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 13:00:00 | 138.18 | 139.02 | 139.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 14:30:00 | 138.40 | 138.84 | 139.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-25 13:15:00 | 131.27 | 132.52 | 134.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-25 13:15:00 | 131.48 | 132.52 | 134.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 15:15:00 | 132.60 | 132.48 | 133.73 | SL hit (close>ema200) qty=0.50 sl=132.48 alert=retest2 |

### Cycle 113 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 136.80 | 134.51 | 134.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 137.31 | 135.65 | 135.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 12:15:00 | 136.13 | 136.28 | 135.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 13:00:00 | 136.13 | 136.28 | 135.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 135.10 | 136.11 | 135.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:00:00 | 135.10 | 136.11 | 135.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 136.86 | 136.26 | 135.87 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2025-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 14:15:00 | 134.80 | 135.71 | 135.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 09:15:00 | 133.61 | 134.96 | 135.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 14:15:00 | 132.07 | 131.93 | 132.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 15:00:00 | 132.07 | 131.93 | 132.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 133.81 | 132.32 | 132.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:00:00 | 133.81 | 132.32 | 132.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 133.39 | 132.53 | 133.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:45:00 | 133.50 | 132.53 | 133.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 132.26 | 132.58 | 132.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:45:00 | 131.88 | 132.56 | 132.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 131.61 | 132.42 | 132.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:45:00 | 131.51 | 132.15 | 132.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 09:15:00 | 131.16 | 129.58 | 129.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 131.16 | 129.58 | 129.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 132.44 | 130.96 | 130.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 10:15:00 | 130.65 | 131.27 | 130.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 10:15:00 | 130.65 | 131.27 | 130.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 130.65 | 131.27 | 130.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 130.65 | 131.27 | 130.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 131.17 | 131.25 | 130.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 13:15:00 | 131.80 | 131.25 | 130.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 130.25 | 131.45 | 131.07 | SL hit (close<static) qty=1.00 sl=130.50 alert=retest2 |

### Cycle 116 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 128.95 | 130.66 | 130.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 128.50 | 129.93 | 130.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 12:15:00 | 129.79 | 129.71 | 130.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 13:15:00 | 129.85 | 129.73 | 130.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 129.85 | 129.73 | 130.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:00:00 | 129.85 | 129.73 | 130.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 130.28 | 129.84 | 130.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:00:00 | 130.28 | 129.84 | 130.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 129.81 | 129.84 | 130.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 129.22 | 129.84 | 130.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 13:00:00 | 129.54 | 129.64 | 129.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 09:30:00 | 129.47 | 126.61 | 127.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 10:15:00 | 130.46 | 127.38 | 127.72 | SL hit (close>static) qty=1.00 sl=130.32 alert=retest2 |

### Cycle 117 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 130.53 | 128.01 | 127.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 12:15:00 | 131.11 | 128.63 | 128.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 13:15:00 | 132.52 | 132.58 | 131.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 14:00:00 | 132.52 | 132.58 | 131.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 131.28 | 132.26 | 131.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 132.55 | 132.26 | 131.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 132.00 | 132.20 | 131.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 137.23 | 132.21 | 131.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-05 14:15:00 | 150.95 | 148.79 | 147.29 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 146.46 | 147.80 | 147.93 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2026-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 13:15:00 | 147.89 | 146.46 | 146.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 14:15:00 | 149.30 | 147.03 | 146.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 11:15:00 | 147.32 | 147.87 | 147.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-13 11:45:00 | 147.59 | 147.87 | 147.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 147.39 | 147.77 | 147.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 13:00:00 | 147.39 | 147.77 | 147.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 13:15:00 | 147.00 | 147.62 | 147.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 13:30:00 | 147.38 | 147.62 | 147.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 147.70 | 147.63 | 147.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 09:15:00 | 148.52 | 147.65 | 147.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 148.69 | 149.60 | 149.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 148.69 | 149.60 | 149.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 147.59 | 148.69 | 149.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 147.41 | 147.23 | 148.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 09:45:00 | 147.32 | 147.23 | 148.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 151.55 | 147.36 | 147.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 153.29 | 147.36 | 147.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 10:15:00 | 151.17 | 148.12 | 147.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 156.04 | 151.30 | 149.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 13:15:00 | 150.43 | 151.77 | 150.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 13:15:00 | 150.43 | 151.77 | 150.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 150.43 | 151.77 | 150.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:00:00 | 150.43 | 151.77 | 150.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 149.12 | 151.24 | 150.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:45:00 | 149.14 | 151.24 | 150.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 148.89 | 150.77 | 150.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 149.94 | 150.77 | 150.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 11:15:00 | 150.64 | 154.06 | 154.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 11:15:00 | 150.64 | 154.06 | 154.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 12:15:00 | 148.19 | 152.88 | 153.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 147.95 | 146.61 | 148.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 147.95 | 146.61 | 148.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 147.95 | 146.61 | 148.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 147.95 | 146.61 | 148.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 149.49 | 147.19 | 148.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 150.75 | 147.19 | 148.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 150.23 | 147.79 | 148.82 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 154.84 | 150.34 | 149.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 157.07 | 153.27 | 151.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 155.11 | 155.79 | 153.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 155.11 | 155.79 | 153.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 155.11 | 155.79 | 153.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 11:15:00 | 156.00 | 155.81 | 154.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 157.08 | 160.28 | 160.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 157.08 | 160.28 | 160.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-17 10:15:00 | 154.25 | 157.38 | 158.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 14:15:00 | 157.26 | 156.87 | 157.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 15:00:00 | 157.26 | 156.87 | 157.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 160.32 | 157.60 | 158.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:30:00 | 159.96 | 157.60 | 158.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 159.75 | 158.03 | 158.18 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 11:15:00 | 159.75 | 158.37 | 158.32 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 156.20 | 158.16 | 158.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 155.80 | 157.69 | 158.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 10:15:00 | 158.09 | 157.56 | 157.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 10:15:00 | 158.09 | 157.56 | 157.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 158.09 | 157.56 | 157.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:45:00 | 158.26 | 157.56 | 157.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 159.01 | 157.85 | 158.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:45:00 | 158.84 | 157.85 | 158.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 158.69 | 158.02 | 158.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:30:00 | 159.41 | 158.02 | 158.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 14:15:00 | 158.93 | 158.26 | 158.23 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 10:15:00 | 156.70 | 157.93 | 158.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 13:15:00 | 156.11 | 157.23 | 157.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 09:15:00 | 157.73 | 157.21 | 157.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 09:15:00 | 157.73 | 157.21 | 157.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 157.73 | 157.21 | 157.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 157.81 | 157.21 | 157.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 157.00 | 157.17 | 157.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 10:30:00 | 158.32 | 157.17 | 157.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 156.28 | 156.99 | 157.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 12:30:00 | 156.03 | 156.58 | 157.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 14:15:00 | 160.43 | 157.33 | 157.41 | SL hit (close>static) qty=1.00 sl=157.41 alert=retest2 |

### Cycle 129 — BUY (started 2026-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 15:15:00 | 160.01 | 157.86 | 157.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 166.66 | 159.62 | 158.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 10:15:00 | 163.19 | 163.85 | 161.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 11:00:00 | 163.19 | 163.85 | 161.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 165.60 | 165.71 | 164.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 165.60 | 165.71 | 164.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 166.82 | 165.82 | 164.66 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 157.88 | 163.92 | 164.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 10:15:00 | 154.54 | 162.05 | 163.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 158.68 | 157.49 | 160.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 10:00:00 | 158.68 | 157.49 | 160.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 159.29 | 157.85 | 159.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:45:00 | 159.75 | 157.85 | 159.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 149.62 | 149.79 | 152.36 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 13:15:00 | 154.22 | 152.38 | 152.20 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 10:15:00 | 151.54 | 152.17 | 152.20 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 11:15:00 | 152.67 | 152.27 | 152.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 13:15:00 | 153.58 | 152.55 | 152.37 | Break + close above crossover candle high |

### Cycle 134 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 147.44 | 151.79 | 152.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 145.88 | 150.61 | 151.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 14:15:00 | 149.60 | 148.95 | 150.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-13 15:00:00 | 149.60 | 148.95 | 150.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 149.10 | 148.98 | 150.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 09:15:00 | 147.96 | 148.98 | 150.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 148.45 | 147.15 | 148.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 151.68 | 149.07 | 148.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 151.68 | 149.07 | 148.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 14:15:00 | 153.49 | 149.96 | 149.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 15:15:00 | 153.45 | 153.57 | 151.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 09:15:00 | 152.80 | 153.57 | 151.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 152.85 | 153.42 | 152.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 10:30:00 | 153.94 | 153.49 | 152.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 155.88 | 153.16 | 152.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 147.36 | 153.91 | 153.77 | SL hit (close<static) qty=1.00 sl=151.00 alert=retest2 |

### Cycle 136 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 147.24 | 152.58 | 153.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 11:15:00 | 145.34 | 151.13 | 152.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 146.88 | 145.95 | 148.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 13:00:00 | 146.88 | 145.95 | 148.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 149.90 | 146.76 | 147.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 150.40 | 146.76 | 147.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 149.71 | 147.35 | 148.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:30:00 | 150.21 | 147.35 | 148.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 150.81 | 148.61 | 148.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 14:15:00 | 151.91 | 149.27 | 148.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 149.43 | 149.62 | 149.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 10:15:00 | 149.42 | 149.62 | 149.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 148.48 | 149.39 | 149.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 148.48 | 149.39 | 149.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 146.52 | 148.82 | 148.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 146.52 | 148.82 | 148.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 146.58 | 148.37 | 148.60 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 10:15:00 | 152.54 | 149.19 | 148.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-30 11:15:00 | 156.15 | 150.58 | 149.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 14:15:00 | 151.20 | 151.53 | 150.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-30 14:45:00 | 151.56 | 151.53 | 150.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 151.75 | 154.22 | 152.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 151.75 | 154.22 | 152.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 151.15 | 153.61 | 152.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:45:00 | 150.76 | 153.61 | 152.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 155.70 | 154.46 | 153.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 12:00:00 | 157.17 | 155.22 | 154.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-17 14:15:00 | 172.89 | 171.46 | 169.84 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 186.19 | 186.93 | 186.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 14:15:00 | 184.69 | 186.30 | 186.67 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-14 09:45:00 | 159.15 | 2024-05-14 10:15:00 | 160.25 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2024-05-17 09:15:00 | 166.20 | 2024-05-23 12:15:00 | 167.65 | STOP_HIT | 1.00 | 0.87% |
| BUY | retest2 | 2024-06-12 09:15:00 | 151.96 | 2024-06-13 12:15:00 | 150.32 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-06-13 09:15:00 | 152.09 | 2024-06-13 12:15:00 | 150.32 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-07-23 12:15:00 | 137.66 | 2024-07-24 09:15:00 | 146.09 | STOP_HIT | 1.00 | -6.12% |
| SELL | retest2 | 2024-07-23 13:00:00 | 140.12 | 2024-07-24 09:15:00 | 146.09 | STOP_HIT | 1.00 | -4.26% |
| SELL | retest2 | 2024-07-23 13:30:00 | 140.06 | 2024-07-24 09:15:00 | 146.09 | STOP_HIT | 1.00 | -4.31% |
| SELL | retest2 | 2024-07-23 14:15:00 | 140.41 | 2024-07-24 09:15:00 | 146.09 | STOP_HIT | 1.00 | -4.05% |
| BUY | retest2 | 2024-07-30 10:30:00 | 148.63 | 2024-08-02 10:15:00 | 147.04 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2024-07-30 14:45:00 | 148.77 | 2024-08-02 10:15:00 | 147.04 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-07-31 09:15:00 | 148.80 | 2024-08-02 10:15:00 | 147.04 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest1 | 2024-08-06 10:30:00 | 138.61 | 2024-08-07 13:15:00 | 140.48 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest1 | 2024-08-06 11:30:00 | 139.00 | 2024-08-07 13:15:00 | 140.48 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest1 | 2024-08-06 12:30:00 | 138.96 | 2024-08-07 13:15:00 | 140.48 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-08-08 09:15:00 | 139.18 | 2024-08-09 12:15:00 | 132.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-08 09:15:00 | 139.18 | 2024-08-12 13:15:00 | 132.22 | STOP_HIT | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-02 09:15:00 | 132.97 | 2024-09-09 12:15:00 | 126.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-03 09:30:00 | 132.83 | 2024-09-09 12:15:00 | 126.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-02 09:15:00 | 132.97 | 2024-09-09 15:15:00 | 128.40 | STOP_HIT | 0.50 | 3.44% |
| SELL | retest2 | 2024-09-03 09:30:00 | 132.83 | 2024-09-09 15:15:00 | 128.40 | STOP_HIT | 0.50 | 3.34% |
| BUY | retest2 | 2024-09-17 12:30:00 | 132.18 | 2024-09-18 11:15:00 | 130.89 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest1 | 2024-09-20 15:00:00 | 126.11 | 2024-09-23 10:15:00 | 128.76 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2024-09-26 10:45:00 | 135.47 | 2024-10-03 12:15:00 | 137.66 | STOP_HIT | 1.00 | 1.62% |
| SELL | retest2 | 2024-10-10 11:00:00 | 130.55 | 2024-10-11 09:15:00 | 133.01 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-10-18 14:15:00 | 129.00 | 2024-10-22 14:15:00 | 122.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 09:30:00 | 129.06 | 2024-10-22 14:15:00 | 122.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 14:15:00 | 129.00 | 2024-10-25 09:15:00 | 116.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-21 09:30:00 | 129.06 | 2024-10-25 09:15:00 | 116.15 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-11-01 18:45:00 | 118.20 | 2024-11-04 09:15:00 | 114.44 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2024-11-29 11:15:00 | 117.01 | 2024-12-11 09:15:00 | 128.57 | TARGET_HIT | 1.00 | 9.88% |
| BUY | retest2 | 2024-11-29 13:15:00 | 116.88 | 2024-12-11 09:15:00 | 128.66 | TARGET_HIT | 1.00 | 10.08% |
| BUY | retest2 | 2024-12-02 09:30:00 | 116.96 | 2024-12-12 12:15:00 | 128.71 | TARGET_HIT | 1.00 | 10.05% |
| SELL | retest2 | 2024-12-20 14:00:00 | 118.77 | 2024-12-23 14:15:00 | 121.60 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2024-12-20 14:30:00 | 118.86 | 2024-12-23 14:15:00 | 121.60 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2024-12-24 14:30:00 | 118.55 | 2024-12-30 14:15:00 | 112.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-26 09:30:00 | 118.21 | 2024-12-30 14:15:00 | 112.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-24 14:30:00 | 118.55 | 2024-12-31 10:15:00 | 113.40 | STOP_HIT | 0.50 | 4.34% |
| SELL | retest2 | 2024-12-26 09:30:00 | 118.21 | 2024-12-31 10:15:00 | 113.40 | STOP_HIT | 0.50 | 4.07% |
| SELL | retest2 | 2025-01-07 15:15:00 | 111.29 | 2025-01-10 09:15:00 | 105.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-07 15:15:00 | 111.29 | 2025-01-10 10:15:00 | 108.57 | STOP_HIT | 0.50 | 2.44% |
| SELL | retest2 | 2025-01-29 11:15:00 | 103.42 | 2025-01-30 09:15:00 | 106.10 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-01-29 12:00:00 | 103.61 | 2025-01-30 09:15:00 | 106.10 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-02-01 14:45:00 | 106.86 | 2025-02-03 09:15:00 | 101.82 | STOP_HIT | 1.00 | -4.72% |
| BUY | retest2 | 2025-02-07 09:15:00 | 108.72 | 2025-02-10 09:15:00 | 105.64 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2025-02-07 10:30:00 | 109.29 | 2025-02-10 09:15:00 | 105.64 | STOP_HIT | 1.00 | -3.34% |
| SELL | retest2 | 2025-02-18 12:45:00 | 102.16 | 2025-02-19 09:15:00 | 106.05 | STOP_HIT | 1.00 | -3.81% |
| SELL | retest2 | 2025-03-03 11:30:00 | 103.40 | 2025-03-03 15:15:00 | 107.23 | STOP_HIT | 1.00 | -3.70% |
| BUY | retest2 | 2025-03-07 13:30:00 | 111.28 | 2025-03-10 11:15:00 | 109.80 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-03-07 14:15:00 | 111.33 | 2025-03-10 11:15:00 | 109.80 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-03-10 09:30:00 | 111.70 | 2025-03-10 11:15:00 | 109.80 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-03-10 11:30:00 | 111.47 | 2025-03-10 12:15:00 | 109.35 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-03-12 10:15:00 | 106.10 | 2025-03-18 10:15:00 | 108.39 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-03-12 11:00:00 | 106.00 | 2025-03-18 10:15:00 | 108.39 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-03-13 14:45:00 | 106.06 | 2025-03-18 10:15:00 | 108.39 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-03-17 09:45:00 | 106.24 | 2025-03-18 10:15:00 | 108.39 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-03-25 13:45:00 | 116.21 | 2025-03-25 15:15:00 | 114.30 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-04-21 10:15:00 | 115.52 | 2025-04-25 11:15:00 | 114.80 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-04-23 10:15:00 | 115.11 | 2025-04-25 11:15:00 | 114.80 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-04-23 10:45:00 | 114.95 | 2025-04-25 11:15:00 | 114.80 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-04-25 10:30:00 | 114.89 | 2025-04-25 11:15:00 | 114.80 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-05-02 11:15:00 | 113.88 | 2025-05-09 09:15:00 | 108.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 09:15:00 | 113.90 | 2025-05-09 09:15:00 | 108.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 11:15:00 | 113.83 | 2025-05-09 09:15:00 | 108.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-02 11:15:00 | 113.88 | 2025-05-12 09:15:00 | 114.32 | STOP_HIT | 0.50 | -0.39% |
| SELL | retest2 | 2025-05-06 09:15:00 | 113.90 | 2025-05-12 09:15:00 | 114.32 | STOP_HIT | 0.50 | -0.37% |
| SELL | retest2 | 2025-05-06 11:15:00 | 113.83 | 2025-05-12 09:15:00 | 114.32 | STOP_HIT | 0.50 | -0.43% |
| SELL | retest2 | 2025-05-12 10:30:00 | 114.46 | 2025-05-12 11:15:00 | 114.73 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-05-21 09:30:00 | 123.03 | 2025-06-02 09:15:00 | 127.10 | STOP_HIT | 1.00 | 3.31% |
| BUY | retest2 | 2025-06-10 09:15:00 | 134.19 | 2025-06-10 12:15:00 | 133.10 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-06-10 11:00:00 | 134.11 | 2025-06-10 12:15:00 | 133.10 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-06-23 09:15:00 | 126.71 | 2025-06-23 11:15:00 | 126.94 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2025-06-24 09:15:00 | 128.87 | 2025-07-01 10:15:00 | 130.56 | STOP_HIT | 1.00 | 1.31% |
| BUY | retest2 | 2025-07-04 10:45:00 | 136.56 | 2025-07-04 11:15:00 | 134.33 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-07-14 10:15:00 | 135.30 | 2025-07-14 12:15:00 | 133.36 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-08-08 09:15:00 | 123.20 | 2025-08-12 11:15:00 | 123.05 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2025-09-10 10:00:00 | 131.15 | 2025-09-18 10:15:00 | 132.01 | STOP_HIT | 1.00 | 0.66% |
| BUY | retest2 | 2025-09-11 09:30:00 | 130.75 | 2025-09-18 10:15:00 | 132.01 | STOP_HIT | 1.00 | 0.96% |
| BUY | retest2 | 2025-09-24 09:30:00 | 137.83 | 2025-09-26 09:15:00 | 134.31 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-09-24 13:30:00 | 137.85 | 2025-09-26 09:15:00 | 134.31 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2025-09-25 09:45:00 | 138.37 | 2025-09-26 09:15:00 | 134.31 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2025-09-25 11:15:00 | 137.71 | 2025-09-26 09:15:00 | 134.31 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-09-30 14:00:00 | 133.28 | 2025-10-01 09:15:00 | 135.55 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-10-08 09:15:00 | 131.90 | 2025-10-09 09:15:00 | 134.79 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-10-14 10:15:00 | 131.70 | 2025-10-16 12:15:00 | 131.79 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-10-23 15:15:00 | 129.00 | 2025-10-24 09:15:00 | 131.44 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-11-03 10:30:00 | 138.53 | 2025-11-06 11:15:00 | 136.78 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-11-04 09:15:00 | 139.62 | 2025-11-06 11:15:00 | 136.78 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-11-06 09:15:00 | 139.05 | 2025-11-06 11:15:00 | 136.78 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-11-13 09:15:00 | 144.25 | 2025-11-14 11:15:00 | 142.72 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-11-13 10:00:00 | 144.02 | 2025-11-14 11:15:00 | 142.72 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-11-13 13:00:00 | 143.87 | 2025-11-14 11:15:00 | 142.72 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-11-20 13:00:00 | 138.18 | 2025-11-25 13:15:00 | 131.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 14:30:00 | 138.40 | 2025-11-25 13:15:00 | 131.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 13:00:00 | 138.18 | 2025-11-25 15:15:00 | 132.60 | STOP_HIT | 0.50 | 4.04% |
| SELL | retest2 | 2025-11-20 14:30:00 | 138.40 | 2025-11-25 15:15:00 | 132.60 | STOP_HIT | 0.50 | 4.19% |
| SELL | retest2 | 2025-12-04 13:45:00 | 131.88 | 2025-12-12 09:15:00 | 131.16 | STOP_HIT | 1.00 | 0.55% |
| SELL | retest2 | 2025-12-05 09:15:00 | 131.61 | 2025-12-12 09:15:00 | 131.16 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2025-12-08 09:45:00 | 131.51 | 2025-12-12 09:15:00 | 131.16 | STOP_HIT | 1.00 | 0.27% |
| BUY | retest2 | 2025-12-15 13:15:00 | 131.80 | 2025-12-16 09:15:00 | 130.25 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-12-18 09:15:00 | 129.22 | 2025-12-22 10:15:00 | 130.46 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-12-18 13:00:00 | 129.54 | 2025-12-22 10:15:00 | 130.46 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-12-22 09:30:00 | 129.47 | 2025-12-22 10:15:00 | 130.46 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-12-29 09:15:00 | 137.23 | 2026-01-05 14:15:00 | 150.95 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-14 09:15:00 | 148.52 | 2026-01-19 10:15:00 | 148.69 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2026-01-27 09:15:00 | 149.94 | 2026-01-30 11:15:00 | 150.64 | STOP_HIT | 1.00 | 0.47% |
| BUY | retest2 | 2026-02-05 11:15:00 | 156.00 | 2026-02-13 09:15:00 | 157.08 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2026-02-24 12:30:00 | 156.03 | 2026-02-24 14:15:00 | 160.43 | STOP_HIT | 1.00 | -2.82% |
| SELL | retest2 | 2026-03-16 09:15:00 | 147.96 | 2026-03-17 13:15:00 | 151.68 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2026-03-17 11:15:00 | 148.45 | 2026-03-17 13:15:00 | 151.68 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2026-03-19 10:30:00 | 153.94 | 2026-03-23 09:15:00 | 147.36 | STOP_HIT | 1.00 | -4.27% |
| BUY | retest2 | 2026-03-20 09:15:00 | 155.88 | 2026-03-23 09:15:00 | 147.36 | STOP_HIT | 1.00 | -5.47% |
| BUY | retest2 | 2026-04-06 12:00:00 | 157.17 | 2026-04-17 14:15:00 | 172.89 | TARGET_HIT | 1.00 | 10.00% |
