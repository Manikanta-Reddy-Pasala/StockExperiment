# Oil & Natural Gas Corporation Ltd. (ONGC)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 279.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 212 |
| ALERT1 | 143 |
| ALERT2 | 141 |
| ALERT2_SKIP | 63 |
| ALERT3 | 412 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 173 |
| PARTIAL | 12 |
| TARGET_HIT | 7 |
| STOP_HIT | 169 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 188 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 62 / 126
- **Target hits / Stop hits / Partials:** 7 / 169 / 12
- **Avg / median % per leg:** 0.27% / -0.73%
- **Sum % (uncompounded):** 51.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 102 | 36 | 35.3% | 7 | 95 | 0 | 0.24% | 24.7% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.12% | -0.1% |
| BUY @ 3rd Alert (retest2) | 101 | 36 | 35.6% | 7 | 94 | 0 | 0.25% | 24.8% |
| SELL (all) | 86 | 26 | 30.2% | 0 | 74 | 12 | 0.31% | 26.9% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 2 | 1 | 1.63% | 4.9% |
| SELL @ 3rd Alert (retest2) | 83 | 24 | 28.9% | 0 | 72 | 11 | 0.27% | 22.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 3 | 1 | 1.19% | 4.8% |
| retest2 (combined) | 184 | 60 | 32.6% | 7 | 166 | 11 | 0.25% | 46.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 11:15:00 | 166.55 | 166.25 | 166.22 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 13:15:00 | 165.60 | 166.09 | 166.15 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 09:15:00 | 167.75 | 166.31 | 166.22 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-05-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 14:15:00 | 166.50 | 166.85 | 166.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-19 09:15:00 | 165.00 | 166.38 | 166.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-22 09:15:00 | 165.85 | 165.56 | 166.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 09:15:00 | 165.85 | 165.56 | 166.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 165.85 | 165.56 | 166.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 09:45:00 | 166.00 | 165.56 | 166.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 10:15:00 | 166.20 | 165.69 | 166.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 11:00:00 | 166.20 | 165.69 | 166.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 11:15:00 | 166.00 | 165.75 | 166.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 11:45:00 | 166.15 | 165.75 | 166.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 12:15:00 | 165.50 | 165.70 | 165.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-22 15:15:00 | 165.00 | 165.75 | 165.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-23 09:15:00 | 166.50 | 165.78 | 165.93 | SL hit (close>static) qty=1.00 sl=166.00 alert=retest2 |

### Cycle 5 — BUY (started 2023-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 14:15:00 | 166.40 | 165.83 | 165.76 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2023-05-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 11:15:00 | 165.45 | 165.73 | 165.74 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-05-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 15:15:00 | 166.50 | 165.84 | 165.78 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-26 09:15:00 | 162.95 | 165.26 | 165.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-29 09:15:00 | 157.55 | 162.85 | 164.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-30 09:15:00 | 161.50 | 160.41 | 161.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-30 09:45:00 | 161.30 | 160.41 | 161.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 09:15:00 | 155.30 | 154.52 | 155.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-05 11:00:00 | 154.55 | 154.52 | 155.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-06 09:45:00 | 154.50 | 155.03 | 155.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-08 09:15:00 | 156.30 | 154.35 | 154.36 | SL hit (close>static) qty=1.00 sl=155.95 alert=retest2 |

### Cycle 9 — BUY (started 2023-06-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-08 10:15:00 | 156.25 | 154.73 | 154.54 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2023-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 12:15:00 | 154.10 | 154.84 | 154.86 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 12:15:00 | 155.15 | 154.77 | 154.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 09:15:00 | 155.75 | 155.11 | 154.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-16 11:15:00 | 157.35 | 157.46 | 156.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-16 11:45:00 | 157.50 | 157.46 | 156.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 14:15:00 | 156.70 | 157.32 | 157.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-16 15:00:00 | 156.70 | 157.32 | 157.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 15:15:00 | 157.10 | 157.28 | 157.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 09:15:00 | 157.30 | 157.28 | 157.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 09:15:00 | 157.75 | 157.37 | 157.10 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 14:15:00 | 157.20 | 157.30 | 157.31 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 09:15:00 | 157.85 | 157.36 | 157.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-21 10:15:00 | 158.60 | 157.61 | 157.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 11:15:00 | 159.25 | 159.30 | 158.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-22 12:00:00 | 159.25 | 159.30 | 158.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 12:15:00 | 158.10 | 159.06 | 158.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 13:00:00 | 158.10 | 159.06 | 158.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 13:15:00 | 159.40 | 159.13 | 158.61 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2023-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 11:15:00 | 157.55 | 158.29 | 158.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 14:15:00 | 157.05 | 157.84 | 158.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 13:15:00 | 157.50 | 157.38 | 157.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 13:15:00 | 157.50 | 157.38 | 157.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 13:15:00 | 157.50 | 157.38 | 157.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-26 14:45:00 | 156.95 | 157.33 | 157.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-27 13:15:00 | 158.25 | 157.86 | 157.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2023-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 13:15:00 | 158.25 | 157.86 | 157.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 10:15:00 | 158.90 | 158.09 | 157.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 14:15:00 | 158.25 | 158.62 | 158.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-28 14:15:00 | 158.25 | 158.62 | 158.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 14:15:00 | 158.25 | 158.62 | 158.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 15:00:00 | 158.25 | 158.62 | 158.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 15:15:00 | 158.35 | 158.57 | 158.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-30 11:30:00 | 160.00 | 159.15 | 158.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-10 14:15:00 | 162.75 | 163.42 | 163.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2023-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 14:15:00 | 162.75 | 163.42 | 163.49 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 10:15:00 | 164.20 | 163.64 | 163.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-12 09:15:00 | 168.10 | 164.91 | 164.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-13 14:15:00 | 167.90 | 167.91 | 166.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-13 15:00:00 | 167.90 | 167.91 | 166.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 167.35 | 168.52 | 167.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 09:30:00 | 167.55 | 168.52 | 167.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 10:15:00 | 167.55 | 168.32 | 167.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 11:15:00 | 167.40 | 168.32 | 167.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2023-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-17 14:15:00 | 166.40 | 167.42 | 167.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-19 11:15:00 | 166.20 | 166.96 | 167.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-19 15:15:00 | 166.85 | 166.81 | 167.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-20 09:15:00 | 166.70 | 166.81 | 167.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 09:15:00 | 166.50 | 166.75 | 166.99 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2023-07-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 13:15:00 | 167.80 | 167.07 | 167.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-21 09:15:00 | 168.90 | 167.48 | 167.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-24 15:15:00 | 170.50 | 170.82 | 169.78 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 09:15:00 | 172.20 | 170.82 | 169.78 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 10:15:00 | 172.00 | 172.51 | 172.01 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-07-27 10:15:00 | 172.00 | 172.51 | 172.01 | SL hit (close<ema400) qty=1.00 sl=172.01 alert=retest1 |

### Cycle 20 — SELL (started 2023-07-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 15:15:00 | 171.00 | 171.71 | 171.77 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-28 09:15:00 | 173.20 | 172.01 | 171.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-28 10:15:00 | 174.40 | 172.49 | 172.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-28 13:15:00 | 171.75 | 172.56 | 172.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-28 13:15:00 | 171.75 | 172.56 | 172.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 13:15:00 | 171.75 | 172.56 | 172.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-28 14:00:00 | 171.75 | 172.56 | 172.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 14:15:00 | 171.75 | 172.40 | 172.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-28 14:30:00 | 171.50 | 172.40 | 172.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 15:15:00 | 171.40 | 172.20 | 172.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-31 09:15:00 | 172.45 | 172.20 | 172.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-03 11:15:00 | 173.05 | 174.70 | 174.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2023-08-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 11:15:00 | 173.05 | 174.70 | 174.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-03 12:15:00 | 172.55 | 174.27 | 174.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-04 12:15:00 | 173.35 | 173.01 | 173.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-04 13:00:00 | 173.35 | 173.01 | 173.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 13:15:00 | 173.50 | 173.11 | 173.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 13:30:00 | 173.65 | 173.11 | 173.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 14:15:00 | 173.30 | 173.15 | 173.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-07 10:00:00 | 172.45 | 173.03 | 173.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-07 15:00:00 | 172.75 | 173.00 | 173.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-08 09:45:00 | 172.80 | 173.14 | 173.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-08 11:45:00 | 172.80 | 173.13 | 173.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 14:15:00 | 173.35 | 173.13 | 173.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-08 15:00:00 | 173.35 | 173.13 | 173.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 15:15:00 | 173.05 | 173.11 | 173.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-09 09:15:00 | 173.85 | 173.11 | 173.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 09:15:00 | 173.75 | 173.24 | 173.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-09 09:30:00 | 174.30 | 173.24 | 173.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-08-09 10:15:00 | 174.05 | 173.40 | 173.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2023-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 10:15:00 | 174.05 | 173.40 | 173.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-09 12:15:00 | 174.40 | 173.66 | 173.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-11 09:15:00 | 177.50 | 177.76 | 176.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-11 09:45:00 | 177.55 | 177.76 | 176.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 15:15:00 | 178.20 | 178.29 | 177.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-16 09:15:00 | 177.85 | 178.29 | 177.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 09:15:00 | 177.25 | 178.08 | 177.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-16 10:30:00 | 178.25 | 178.10 | 177.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-16 13:45:00 | 178.20 | 178.19 | 177.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-17 09:30:00 | 178.25 | 178.34 | 178.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-17 10:45:00 | 178.20 | 178.25 | 177.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 11:15:00 | 177.65 | 178.13 | 177.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-17 12:00:00 | 177.65 | 178.13 | 177.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 12:15:00 | 177.80 | 178.06 | 177.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-17 14:30:00 | 178.25 | 178.01 | 177.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-17 15:15:00 | 177.45 | 177.90 | 177.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2023-08-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-17 15:15:00 | 177.45 | 177.90 | 177.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 09:15:00 | 175.65 | 177.45 | 177.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 10:15:00 | 175.90 | 175.85 | 176.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-21 11:00:00 | 175.90 | 175.85 | 176.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 15:15:00 | 175.90 | 175.89 | 176.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 09:15:00 | 176.85 | 175.89 | 176.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 09:15:00 | 177.65 | 176.25 | 176.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 10:00:00 | 177.65 | 176.25 | 176.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 10:15:00 | 176.55 | 176.31 | 176.43 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2023-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 12:15:00 | 176.80 | 176.52 | 176.51 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-22 13:15:00 | 176.45 | 176.50 | 176.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-22 14:15:00 | 176.10 | 176.42 | 176.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-24 09:15:00 | 176.50 | 175.89 | 176.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-24 09:15:00 | 176.50 | 175.89 | 176.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 09:15:00 | 176.50 | 175.89 | 176.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-24 10:00:00 | 176.50 | 175.89 | 176.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 10:15:00 | 175.20 | 175.75 | 175.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-24 11:15:00 | 174.80 | 175.75 | 175.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-28 11:15:00 | 175.50 | 174.88 | 174.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2023-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 11:15:00 | 175.50 | 174.88 | 174.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 09:15:00 | 176.30 | 175.62 | 175.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 14:15:00 | 175.45 | 175.94 | 175.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 14:15:00 | 175.45 | 175.94 | 175.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 14:15:00 | 175.45 | 175.94 | 175.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-30 15:00:00 | 175.45 | 175.94 | 175.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 15:15:00 | 175.70 | 175.89 | 175.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 09:15:00 | 175.50 | 175.89 | 175.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 175.15 | 175.74 | 175.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 10:00:00 | 175.15 | 175.74 | 175.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 10:15:00 | 174.90 | 175.57 | 175.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 10:30:00 | 174.70 | 175.57 | 175.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2023-08-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 11:15:00 | 174.55 | 175.37 | 175.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 15:15:00 | 173.85 | 174.73 | 175.09 | Break + close below crossover candle low |

### Cycle 29 — BUY (started 2023-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 09:15:00 | 179.15 | 175.61 | 175.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 10:15:00 | 183.55 | 177.20 | 176.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-05 13:15:00 | 183.30 | 183.75 | 182.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-05 13:45:00 | 182.95 | 183.75 | 182.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 12:15:00 | 183.20 | 183.60 | 182.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 12:45:00 | 183.00 | 183.60 | 182.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 13:15:00 | 182.40 | 183.36 | 182.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 14:00:00 | 182.40 | 183.36 | 182.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 14:15:00 | 183.15 | 183.32 | 182.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-07 10:30:00 | 183.65 | 183.15 | 182.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-07 11:15:00 | 182.10 | 182.94 | 182.73 | SL hit (close<static) qty=1.00 sl=182.35 alert=retest2 |

### Cycle 30 — SELL (started 2023-09-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-07 13:15:00 | 181.75 | 182.47 | 182.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-07 15:15:00 | 181.10 | 182.10 | 182.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-08 09:15:00 | 183.60 | 182.40 | 182.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-08 09:15:00 | 183.60 | 182.40 | 182.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 09:15:00 | 183.60 | 182.40 | 182.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-08 09:45:00 | 183.50 | 182.40 | 182.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 10:15:00 | 182.85 | 182.49 | 182.50 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2023-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-08 11:15:00 | 183.30 | 182.65 | 182.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-08 13:15:00 | 184.80 | 183.16 | 182.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-11 11:15:00 | 183.40 | 183.79 | 183.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-11 11:15:00 | 183.40 | 183.79 | 183.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 11:15:00 | 183.40 | 183.79 | 183.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-11 12:00:00 | 183.40 | 183.79 | 183.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 12:15:00 | 182.80 | 183.59 | 183.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-11 13:00:00 | 182.80 | 183.59 | 183.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 13:15:00 | 182.85 | 183.44 | 183.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-11 13:30:00 | 182.80 | 183.44 | 183.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 15:15:00 | 183.55 | 183.46 | 183.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:30:00 | 182.90 | 183.39 | 183.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 183.45 | 183.40 | 183.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 10:30:00 | 183.10 | 183.40 | 183.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 11:15:00 | 181.85 | 183.09 | 183.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 14:15:00 | 180.35 | 182.30 | 182.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 10:15:00 | 182.95 | 182.19 | 182.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 10:15:00 | 182.95 | 182.19 | 182.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 10:15:00 | 182.95 | 182.19 | 182.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 10:45:00 | 182.60 | 182.19 | 182.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 11:15:00 | 182.95 | 182.34 | 182.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 11:30:00 | 183.00 | 182.34 | 182.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2023-09-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 14:15:00 | 183.85 | 182.94 | 182.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 09:15:00 | 185.40 | 183.62 | 183.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 12:15:00 | 187.10 | 187.17 | 185.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-15 12:30:00 | 187.10 | 187.17 | 185.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 14:15:00 | 186.60 | 186.93 | 186.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-15 14:30:00 | 185.95 | 186.93 | 186.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 09:15:00 | 186.65 | 186.82 | 186.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-20 09:15:00 | 188.90 | 186.71 | 186.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-20 14:30:00 | 188.00 | 187.47 | 186.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-21 14:15:00 | 186.70 | 186.88 | 186.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2023-09-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 14:15:00 | 186.70 | 186.88 | 186.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-22 09:15:00 | 185.50 | 186.59 | 186.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 11:15:00 | 186.70 | 186.55 | 186.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 11:15:00 | 186.70 | 186.55 | 186.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 11:15:00 | 186.70 | 186.55 | 186.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 12:00:00 | 186.70 | 186.55 | 186.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 12:15:00 | 186.35 | 186.51 | 186.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-22 13:15:00 | 186.00 | 186.51 | 186.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-25 09:45:00 | 185.90 | 185.92 | 186.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-26 13:15:00 | 187.50 | 186.07 | 186.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2023-09-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 13:15:00 | 187.50 | 186.07 | 186.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-26 14:15:00 | 187.70 | 186.40 | 186.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-27 09:15:00 | 186.45 | 186.66 | 186.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-27 09:15:00 | 186.45 | 186.66 | 186.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 186.45 | 186.66 | 186.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-27 09:30:00 | 186.85 | 186.66 | 186.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 10:15:00 | 187.05 | 186.73 | 186.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-27 11:30:00 | 187.40 | 186.88 | 186.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-27 12:45:00 | 187.35 | 186.97 | 186.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-27 15:00:00 | 187.25 | 187.10 | 186.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-28 09:15:00 | 187.50 | 187.12 | 186.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 186.55 | 187.54 | 187.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 15:00:00 | 186.55 | 187.54 | 187.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 15:15:00 | 188.20 | 187.67 | 187.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-29 09:15:00 | 190.45 | 187.67 | 187.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-29 11:30:00 | 189.60 | 188.45 | 187.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-29 12:00:00 | 189.65 | 188.45 | 187.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-03 09:15:00 | 185.00 | 189.12 | 188.54 | SL hit (close<static) qty=1.00 sl=186.30 alert=retest2 |

### Cycle 36 — SELL (started 2023-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 11:15:00 | 184.55 | 187.51 | 187.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 10:15:00 | 183.60 | 185.06 | 186.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 10:15:00 | 183.50 | 183.46 | 184.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-05 11:00:00 | 183.50 | 183.46 | 184.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 182.60 | 181.87 | 182.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-09 15:00:00 | 181.55 | 182.26 | 182.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-10 10:15:00 | 183.65 | 182.80 | 182.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2023-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 10:15:00 | 183.65 | 182.80 | 182.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 11:15:00 | 184.10 | 183.72 | 183.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 13:15:00 | 184.50 | 184.81 | 184.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 13:15:00 | 184.50 | 184.81 | 184.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 13:15:00 | 184.50 | 184.81 | 184.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 14:00:00 | 184.50 | 184.81 | 184.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 14:15:00 | 184.95 | 184.84 | 184.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 15:15:00 | 185.15 | 184.84 | 184.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-19 15:15:00 | 185.90 | 186.22 | 186.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2023-10-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 15:15:00 | 185.90 | 186.22 | 186.24 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-20 09:15:00 | 186.75 | 186.33 | 186.29 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2023-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 14:15:00 | 186.15 | 186.28 | 186.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 09:15:00 | 185.50 | 186.08 | 186.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 14:15:00 | 184.75 | 184.73 | 185.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-25 15:00:00 | 184.75 | 184.73 | 185.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 11:15:00 | 183.10 | 182.32 | 183.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 12:00:00 | 183.10 | 182.32 | 183.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 12:15:00 | 183.60 | 182.58 | 183.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 12:45:00 | 183.40 | 182.58 | 183.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 13:15:00 | 183.75 | 182.81 | 183.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 14:00:00 | 183.75 | 182.81 | 183.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 14:15:00 | 184.75 | 183.20 | 183.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 15:00:00 | 184.75 | 183.20 | 183.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2023-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 15:15:00 | 185.00 | 183.56 | 183.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 11:15:00 | 185.95 | 184.55 | 184.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 09:15:00 | 186.55 | 186.61 | 185.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-31 09:45:00 | 186.80 | 186.61 | 185.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 10:15:00 | 185.15 | 186.32 | 185.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 11:00:00 | 185.15 | 186.32 | 185.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 11:15:00 | 185.10 | 186.07 | 185.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 11:45:00 | 185.30 | 186.07 | 185.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 12:15:00 | 185.20 | 185.90 | 185.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 12:45:00 | 185.15 | 185.90 | 185.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 13:15:00 | 185.15 | 185.75 | 185.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 13:30:00 | 184.50 | 185.75 | 185.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 14:15:00 | 186.25 | 185.85 | 185.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-01 09:15:00 | 187.50 | 185.88 | 185.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-02 09:15:00 | 186.60 | 186.48 | 186.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-02 14:45:00 | 186.40 | 186.47 | 186.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-03 09:30:00 | 186.85 | 186.74 | 186.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 09:15:00 | 194.55 | 194.13 | 192.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 11:15:00 | 195.40 | 194.29 | 192.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 12:00:00 | 195.25 | 194.48 | 193.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 12:45:00 | 195.05 | 194.65 | 193.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-10 10:30:00 | 195.40 | 194.08 | 193.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 09:15:00 | 195.40 | 195.39 | 194.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-13 09:45:00 | 195.05 | 195.39 | 194.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 13:15:00 | 194.95 | 195.56 | 195.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-13 13:30:00 | 194.90 | 195.56 | 195.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 14:15:00 | 195.90 | 195.63 | 195.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-13 14:30:00 | 195.55 | 195.63 | 195.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 15:15:00 | 195.65 | 195.63 | 195.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-15 09:15:00 | 198.85 | 195.63 | 195.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-17 12:15:00 | 195.95 | 198.72 | 198.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2023-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 12:15:00 | 195.95 | 198.72 | 198.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-21 09:15:00 | 192.05 | 196.65 | 197.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-22 15:15:00 | 191.20 | 191.13 | 192.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-23 09:15:00 | 190.35 | 191.13 | 192.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 189.85 | 190.48 | 191.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 09:30:00 | 191.40 | 190.48 | 191.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 10:15:00 | 190.65 | 189.60 | 190.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 10:30:00 | 190.65 | 189.60 | 190.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 11:15:00 | 190.75 | 189.83 | 190.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 11:45:00 | 191.25 | 189.83 | 190.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 13:15:00 | 192.15 | 190.53 | 190.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 14:00:00 | 192.15 | 190.53 | 190.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2023-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 14:15:00 | 194.00 | 191.22 | 190.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 11:15:00 | 194.75 | 193.11 | 192.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 13:15:00 | 194.75 | 194.96 | 194.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-01 14:00:00 | 194.75 | 194.96 | 194.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 13:15:00 | 201.00 | 200.92 | 199.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 14:30:00 | 201.35 | 201.18 | 199.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-06 15:00:00 | 201.95 | 200.97 | 200.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-07 09:15:00 | 197.50 | 200.50 | 200.13 | SL hit (close<static) qty=1.00 sl=199.15 alert=retest2 |

### Cycle 44 — SELL (started 2023-12-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-07 11:15:00 | 198.10 | 199.54 | 199.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 12:15:00 | 195.30 | 198.04 | 198.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-11 09:15:00 | 199.85 | 197.54 | 198.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-11 09:15:00 | 199.85 | 197.54 | 198.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 199.85 | 197.54 | 198.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-11 09:45:00 | 199.60 | 197.54 | 198.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 10:15:00 | 199.45 | 197.92 | 198.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-11 10:30:00 | 199.05 | 197.92 | 198.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2023-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 13:15:00 | 200.00 | 198.83 | 198.68 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2023-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 15:15:00 | 194.35 | 198.01 | 198.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-13 09:15:00 | 193.70 | 195.83 | 196.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-14 10:15:00 | 194.60 | 194.07 | 195.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-14 11:00:00 | 194.60 | 194.07 | 195.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 11:15:00 | 194.90 | 194.23 | 195.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 11:30:00 | 195.10 | 194.23 | 195.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 12:15:00 | 195.65 | 194.52 | 195.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 12:30:00 | 196.00 | 194.52 | 195.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 13:15:00 | 195.45 | 194.70 | 195.20 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2023-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 09:15:00 | 199.50 | 196.03 | 195.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 14:15:00 | 201.05 | 198.47 | 197.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 12:15:00 | 197.65 | 198.95 | 197.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-18 12:15:00 | 197.65 | 198.95 | 197.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 12:15:00 | 197.65 | 198.95 | 197.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 13:00:00 | 197.65 | 198.95 | 197.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 13:15:00 | 198.60 | 198.88 | 198.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-18 14:15:00 | 198.70 | 198.88 | 198.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-29 13:15:00 | 204.95 | 205.97 | 206.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2023-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-29 13:15:00 | 204.95 | 205.97 | 206.10 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-02 11:15:00 | 209.30 | 206.47 | 206.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 09:15:00 | 212.60 | 209.12 | 208.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-09 09:15:00 | 217.50 | 217.63 | 215.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-09 10:00:00 | 217.50 | 217.63 | 215.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 14:15:00 | 216.80 | 217.27 | 216.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-09 14:30:00 | 217.05 | 217.27 | 216.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 15:15:00 | 216.80 | 217.18 | 216.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 09:15:00 | 210.20 | 217.18 | 216.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 211.85 | 216.11 | 216.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 09:30:00 | 212.05 | 216.11 | 216.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2024-01-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 10:15:00 | 212.05 | 215.30 | 215.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-10 11:15:00 | 210.90 | 214.42 | 215.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-11 15:15:00 | 212.00 | 211.92 | 212.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-12 09:15:00 | 213.30 | 211.92 | 212.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 09:15:00 | 214.40 | 212.42 | 213.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-12 10:00:00 | 214.40 | 212.42 | 213.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2024-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 10:15:00 | 219.60 | 213.85 | 213.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-12 11:15:00 | 220.85 | 215.25 | 214.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 12:15:00 | 232.00 | 233.19 | 227.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-16 13:00:00 | 232.00 | 233.19 | 227.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 14:15:00 | 232.25 | 233.50 | 231.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 14:45:00 | 231.50 | 233.50 | 231.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 15:15:00 | 232.80 | 233.36 | 231.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-18 09:15:00 | 230.00 | 233.36 | 231.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 228.10 | 232.31 | 231.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-18 09:45:00 | 225.00 | 232.31 | 231.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 10:15:00 | 231.45 | 232.13 | 231.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-18 10:30:00 | 229.80 | 232.13 | 231.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 11:15:00 | 230.80 | 231.87 | 231.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-18 11:30:00 | 230.20 | 231.87 | 231.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 12:15:00 | 231.80 | 231.85 | 231.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-18 12:30:00 | 230.95 | 231.85 | 231.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 13:15:00 | 232.75 | 232.03 | 231.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-18 13:30:00 | 231.80 | 232.03 | 231.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 239.65 | 241.29 | 239.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:00:00 | 239.65 | 241.29 | 239.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 236.95 | 240.43 | 239.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:30:00 | 235.75 | 240.43 | 239.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 11:15:00 | 234.10 | 239.16 | 238.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 12:00:00 | 234.10 | 239.16 | 238.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2024-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 12:15:00 | 232.90 | 237.91 | 238.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 14:15:00 | 230.10 | 235.47 | 236.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 12:15:00 | 233.10 | 232.57 | 234.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-24 13:00:00 | 233.10 | 232.57 | 234.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 14:15:00 | 234.00 | 232.93 | 234.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 15:00:00 | 234.00 | 232.93 | 234.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 15:15:00 | 234.85 | 233.31 | 234.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:15:00 | 235.90 | 233.31 | 234.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 235.30 | 233.71 | 234.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:45:00 | 235.20 | 233.71 | 234.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 10:15:00 | 233.60 | 233.69 | 234.48 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2024-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 09:15:00 | 244.80 | 236.40 | 235.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 10:15:00 | 250.65 | 239.25 | 236.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 14:15:00 | 249.55 | 253.74 | 248.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 14:15:00 | 249.55 | 253.74 | 248.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 14:15:00 | 249.55 | 253.74 | 248.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 15:00:00 | 249.55 | 253.74 | 248.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 15:15:00 | 249.20 | 252.84 | 248.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 09:15:00 | 252.00 | 252.84 | 248.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-31 10:15:00 | 247.65 | 251.64 | 248.98 | SL hit (close<static) qty=1.00 sl=248.65 alert=retest2 |

### Cycle 54 — SELL (started 2024-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 14:15:00 | 247.25 | 248.99 | 249.08 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2024-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 09:15:00 | 252.90 | 249.60 | 249.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-02 10:15:00 | 255.15 | 250.71 | 249.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-07 11:15:00 | 269.35 | 269.89 | 265.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-07 11:30:00 | 268.75 | 269.89 | 265.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 266.15 | 271.15 | 269.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 10:00:00 | 266.15 | 271.15 | 269.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 10:15:00 | 261.45 | 269.21 | 268.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 11:00:00 | 261.45 | 269.21 | 268.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2024-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 11:15:00 | 265.50 | 268.47 | 268.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 09:15:00 | 259.10 | 266.25 | 267.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 10:15:00 | 261.10 | 260.84 | 263.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-13 10:45:00 | 261.80 | 260.84 | 263.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 09:15:00 | 260.65 | 259.97 | 261.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 10:00:00 | 260.65 | 259.97 | 261.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 11:15:00 | 262.50 | 260.59 | 261.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 11:30:00 | 262.90 | 260.59 | 261.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 12:15:00 | 265.50 | 261.57 | 262.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 12:45:00 | 265.95 | 261.57 | 262.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2024-02-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 13:15:00 | 267.25 | 262.71 | 262.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 14:15:00 | 268.70 | 263.91 | 263.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 09:15:00 | 273.20 | 274.45 | 270.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 09:15:00 | 273.20 | 274.45 | 270.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 09:15:00 | 273.20 | 274.45 | 270.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 09:45:00 | 271.45 | 274.45 | 270.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 09:15:00 | 276.15 | 275.04 | 272.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-19 13:30:00 | 278.05 | 275.84 | 273.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 09:15:00 | 280.25 | 275.37 | 274.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-21 09:30:00 | 279.00 | 276.62 | 275.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-21 11:00:00 | 278.10 | 276.92 | 275.70 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 14:15:00 | 274.15 | 276.79 | 276.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 15:00:00 | 274.15 | 276.79 | 276.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 15:15:00 | 273.25 | 276.08 | 275.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 09:15:00 | 270.55 | 276.08 | 275.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-02-22 09:15:00 | 271.30 | 275.13 | 275.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2024-02-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 09:15:00 | 271.30 | 275.13 | 275.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 09:15:00 | 267.35 | 269.75 | 271.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-27 14:15:00 | 269.80 | 268.58 | 269.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 14:15:00 | 269.80 | 268.58 | 269.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 14:15:00 | 269.80 | 268.58 | 269.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-27 15:00:00 | 269.80 | 268.58 | 269.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 269.75 | 269.00 | 269.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 10:30:00 | 267.70 | 268.60 | 269.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-01 09:15:00 | 272.15 | 266.40 | 266.79 | SL hit (close>static) qty=1.00 sl=270.30 alert=retest2 |

### Cycle 59 — BUY (started 2024-03-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 10:15:00 | 269.95 | 267.11 | 267.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-02 11:15:00 | 272.25 | 270.29 | 268.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 09:15:00 | 275.00 | 279.87 | 277.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 09:15:00 | 275.00 | 279.87 | 277.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 275.00 | 279.87 | 277.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:00:00 | 275.00 | 279.87 | 277.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 10:15:00 | 276.85 | 279.27 | 277.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-06 13:45:00 | 278.75 | 278.40 | 277.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-06 14:15:00 | 279.75 | 278.40 | 277.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-07 12:45:00 | 279.45 | 279.71 | 278.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-07 14:45:00 | 279.35 | 279.34 | 278.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 15:15:00 | 279.00 | 279.28 | 278.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 09:15:00 | 278.55 | 279.28 | 278.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 09:15:00 | 278.30 | 279.08 | 278.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 09:45:00 | 277.45 | 279.08 | 278.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 10:15:00 | 275.90 | 278.44 | 278.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 11:00:00 | 275.90 | 278.44 | 278.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-03-11 11:15:00 | 274.50 | 277.66 | 278.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2024-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 11:15:00 | 274.50 | 277.66 | 278.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 14:15:00 | 273.65 | 276.13 | 277.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 13:15:00 | 259.35 | 259.08 | 263.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 13:30:00 | 258.20 | 259.08 | 263.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 14:15:00 | 263.60 | 259.98 | 263.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 15:00:00 | 263.60 | 259.98 | 263.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 15:15:00 | 263.95 | 260.78 | 263.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 09:15:00 | 261.70 | 260.78 | 263.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 257.15 | 260.05 | 262.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 10:15:00 | 254.60 | 260.05 | 262.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-18 15:15:00 | 262.10 | 260.68 | 260.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2024-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 15:15:00 | 262.10 | 260.68 | 260.56 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 09:15:00 | 258.85 | 260.31 | 260.41 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2024-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 12:15:00 | 261.95 | 260.30 | 260.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-20 14:15:00 | 263.95 | 261.41 | 260.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-21 11:15:00 | 262.95 | 263.14 | 261.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-21 12:00:00 | 262.95 | 263.14 | 261.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 09:15:00 | 261.95 | 262.93 | 262.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-22 14:00:00 | 263.65 | 262.80 | 262.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-22 15:15:00 | 263.30 | 262.87 | 262.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-26 09:45:00 | 264.80 | 263.56 | 262.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 15:15:00 | 265.30 | 264.67 | 264.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 15:15:00 | 265.30 | 264.80 | 264.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 12:30:00 | 266.65 | 265.28 | 264.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-04 12:15:00 | 268.00 | 271.44 | 271.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2024-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 12:15:00 | 268.00 | 271.44 | 271.73 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2024-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-09 09:15:00 | 271.75 | 269.81 | 269.75 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 13:15:00 | 268.00 | 269.49 | 269.64 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 09:15:00 | 272.00 | 269.85 | 269.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 12:15:00 | 272.65 | 271.05 | 270.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 09:15:00 | 271.35 | 271.51 | 270.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 09:15:00 | 271.35 | 271.51 | 270.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 271.35 | 271.51 | 270.85 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2024-04-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 12:15:00 | 268.45 | 270.47 | 270.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 13:15:00 | 266.45 | 269.67 | 270.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-15 09:15:00 | 275.50 | 269.68 | 269.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-15 09:15:00 | 275.50 | 269.68 | 269.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 275.50 | 269.68 | 269.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 10:00:00 | 275.50 | 269.68 | 269.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2024-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-15 10:15:00 | 277.00 | 271.14 | 270.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-15 11:15:00 | 279.10 | 272.73 | 271.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-16 14:15:00 | 283.35 | 283.60 | 279.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-16 14:45:00 | 283.05 | 283.60 | 279.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 13:15:00 | 278.10 | 281.98 | 280.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-18 14:00:00 | 278.10 | 281.98 | 280.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 14:15:00 | 273.65 | 280.31 | 279.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-18 15:00:00 | 273.65 | 280.31 | 279.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2024-04-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 15:15:00 | 275.00 | 279.25 | 279.45 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 10:15:00 | 277.50 | 276.85 | 276.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 11:15:00 | 278.10 | 277.10 | 276.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 14:15:00 | 282.70 | 283.05 | 281.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-26 15:00:00 | 282.70 | 283.05 | 281.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 14:15:00 | 283.35 | 284.04 | 283.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 15:00:00 | 283.35 | 284.04 | 283.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 15:15:00 | 282.60 | 283.76 | 283.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 09:15:00 | 280.85 | 283.76 | 283.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 09:15:00 | 283.25 | 283.65 | 283.22 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2024-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 13:15:00 | 281.75 | 282.86 | 282.94 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2024-05-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-03 09:15:00 | 290.40 | 284.28 | 283.55 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 09:15:00 | 277.00 | 284.01 | 284.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 12:15:00 | 274.55 | 279.11 | 281.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 276.80 | 276.68 | 279.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 10:00:00 | 276.80 | 276.68 | 279.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 12:15:00 | 279.20 | 277.47 | 278.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 13:00:00 | 279.20 | 277.47 | 278.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 13:15:00 | 277.00 | 277.38 | 278.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 14:15:00 | 276.70 | 277.38 | 278.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 09:15:00 | 274.80 | 277.20 | 278.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-13 10:15:00 | 262.86 | 267.89 | 270.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-13 13:15:00 | 268.00 | 267.70 | 269.71 | SL hit (close>ema200) qty=0.50 sl=267.70 alert=retest2 |

### Cycle 75 — BUY (started 2024-05-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 15:15:00 | 274.50 | 270.22 | 269.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 09:15:00 | 276.25 | 273.65 | 272.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 13:15:00 | 274.35 | 274.41 | 273.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 14:00:00 | 274.35 | 274.41 | 273.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 280.85 | 279.78 | 278.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 280.30 | 279.78 | 278.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 12:15:00 | 278.25 | 279.33 | 278.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 13:00:00 | 278.25 | 279.33 | 278.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 13:15:00 | 277.40 | 278.94 | 278.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 13:30:00 | 277.45 | 278.94 | 278.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 277.15 | 282.02 | 281.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 10:00:00 | 277.15 | 282.02 | 281.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2024-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 10:15:00 | 276.90 | 281.00 | 281.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 11:15:00 | 274.60 | 277.43 | 278.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 15:15:00 | 272.90 | 272.69 | 274.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-30 09:15:00 | 272.65 | 272.69 | 274.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 271.30 | 272.41 | 274.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 11:30:00 | 270.80 | 271.83 | 273.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 12:00:00 | 270.30 | 271.83 | 273.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 279.65 | 269.01 | 269.74 | SL hit (close>static) qty=1.00 sl=275.10 alert=retest2 |

### Cycle 77 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 278.85 | 270.98 | 270.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 12:15:00 | 284.10 | 274.82 | 272.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 268.75 | 277.17 | 274.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 268.75 | 277.17 | 274.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 268.75 | 277.17 | 274.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 265.85 | 277.17 | 274.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 255.70 | 272.87 | 272.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 228.45 | 263.99 | 268.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 12:15:00 | 246.05 | 244.71 | 253.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 13:00:00 | 246.05 | 244.71 | 253.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 256.90 | 248.07 | 252.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 257.75 | 248.07 | 252.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 259.20 | 250.30 | 253.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 11:00:00 | 259.20 | 250.30 | 253.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 13:15:00 | 251.75 | 251.27 | 252.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 13:30:00 | 253.75 | 251.27 | 252.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 15:15:00 | 253.50 | 251.86 | 252.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 09:15:00 | 255.60 | 251.86 | 252.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 256.50 | 252.79 | 253.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 09:30:00 | 256.10 | 252.79 | 253.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 10:15:00 | 255.40 | 253.31 | 253.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 10:45:00 | 256.00 | 253.31 | 253.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2024-06-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 11:15:00 | 255.60 | 253.77 | 253.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 13:15:00 | 257.65 | 254.93 | 254.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 259.20 | 259.43 | 257.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 14:30:00 | 259.40 | 259.43 | 257.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 275.80 | 275.70 | 274.52 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2024-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 12:15:00 | 272.65 | 274.40 | 274.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 13:15:00 | 271.65 | 273.85 | 274.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 10:15:00 | 273.25 | 272.93 | 273.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 10:15:00 | 273.25 | 272.93 | 273.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 273.25 | 272.93 | 273.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:45:00 | 273.60 | 272.93 | 273.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 11:15:00 | 272.75 | 272.90 | 273.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 13:30:00 | 271.45 | 272.52 | 273.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 09:15:00 | 271.50 | 272.35 | 273.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 09:15:00 | 273.95 | 272.67 | 273.11 | SL hit (close>static) qty=1.00 sl=273.80 alert=retest2 |

### Cycle 81 — BUY (started 2024-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 09:15:00 | 271.75 | 267.73 | 267.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 10:15:00 | 275.30 | 269.25 | 268.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 11:15:00 | 272.50 | 272.90 | 271.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-01 12:00:00 | 272.50 | 272.90 | 271.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 272.70 | 273.86 | 272.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 272.70 | 273.86 | 272.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 274.00 | 273.89 | 272.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 14:15:00 | 275.15 | 273.89 | 272.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 15:15:00 | 275.70 | 273.92 | 272.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 15:00:00 | 274.80 | 275.04 | 274.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 11:15:00 | 274.60 | 274.83 | 274.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 275.20 | 274.90 | 274.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 12:15:00 | 275.65 | 274.90 | 274.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 13:15:00 | 276.10 | 274.94 | 274.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 15:15:00 | 277.80 | 275.21 | 274.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-08 14:15:00 | 302.67 | 292.55 | 286.74 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2024-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 12:15:00 | 319.95 | 322.00 | 322.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 09:15:00 | 315.35 | 320.19 | 321.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 15:15:00 | 314.90 | 314.69 | 317.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-24 09:15:00 | 316.50 | 314.69 | 317.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 320.20 | 315.79 | 317.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 320.20 | 315.79 | 317.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 322.90 | 317.21 | 318.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:45:00 | 324.25 | 317.21 | 318.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2024-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 12:15:00 | 321.60 | 318.91 | 318.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 09:15:00 | 324.85 | 320.51 | 319.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 12:15:00 | 330.70 | 331.03 | 327.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-26 12:45:00 | 330.60 | 331.03 | 327.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 336.55 | 333.91 | 331.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:15:00 | 339.25 | 334.31 | 333.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 14:30:00 | 341.10 | 338.90 | 336.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 14:15:00 | 330.40 | 334.97 | 335.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2024-08-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 14:15:00 | 330.40 | 334.97 | 335.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 318.35 | 330.91 | 333.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 09:15:00 | 327.15 | 314.21 | 318.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 09:15:00 | 327.15 | 314.21 | 318.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 327.15 | 314.21 | 318.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:00:00 | 327.15 | 314.21 | 318.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 327.85 | 316.94 | 319.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:30:00 | 328.25 | 316.94 | 319.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2024-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 12:15:00 | 329.00 | 321.20 | 320.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 13:15:00 | 331.60 | 328.24 | 325.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 12:15:00 | 330.35 | 330.61 | 328.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-12 13:00:00 | 330.35 | 330.61 | 328.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 334.95 | 336.49 | 333.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:30:00 | 333.10 | 336.49 | 333.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 334.00 | 335.99 | 333.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:15:00 | 333.55 | 335.99 | 333.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 336.05 | 336.00 | 334.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 331.65 | 336.00 | 334.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 331.10 | 335.02 | 333.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:00:00 | 331.10 | 335.02 | 333.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 11:15:00 | 333.50 | 334.72 | 333.76 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2024-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 14:15:00 | 327.45 | 332.56 | 332.95 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2024-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 10:15:00 | 334.65 | 332.46 | 332.26 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 10:15:00 | 330.20 | 332.79 | 332.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 13:15:00 | 328.80 | 331.19 | 332.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 09:15:00 | 326.40 | 322.26 | 324.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 09:15:00 | 326.40 | 322.26 | 324.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 326.40 | 322.26 | 324.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 10:00:00 | 326.40 | 322.26 | 324.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 325.65 | 322.94 | 324.42 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2024-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 13:15:00 | 328.00 | 325.35 | 325.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 10:15:00 | 329.10 | 326.95 | 326.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-27 15:15:00 | 328.30 | 328.67 | 327.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-28 09:15:00 | 325.25 | 328.67 | 327.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 326.40 | 328.21 | 327.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:30:00 | 327.05 | 328.21 | 327.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 326.70 | 327.91 | 327.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 10:45:00 | 326.15 | 327.91 | 327.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 12:15:00 | 327.55 | 327.72 | 327.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 14:30:00 | 328.60 | 327.72 | 327.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 09:15:00 | 326.55 | 327.37 | 327.29 | SL hit (close<static) qty=1.00 sl=326.70 alert=retest2 |

### Cycle 90 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 326.40 | 327.17 | 327.21 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 12:15:00 | 327.85 | 327.34 | 327.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-29 14:15:00 | 329.30 | 327.72 | 327.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-30 15:15:00 | 330.30 | 330.40 | 329.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 09:15:00 | 329.50 | 330.40 | 329.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 328.60 | 330.04 | 329.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:30:00 | 328.50 | 330.04 | 329.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 330.60 | 330.15 | 329.35 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2024-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 13:15:00 | 327.10 | 328.89 | 328.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 14:15:00 | 326.25 | 328.36 | 328.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 10:15:00 | 288.50 | 288.23 | 292.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-12 10:30:00 | 288.25 | 288.23 | 292.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 294.35 | 290.12 | 292.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 15:00:00 | 294.35 | 290.12 | 292.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 15:15:00 | 293.00 | 290.69 | 292.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:15:00 | 293.00 | 290.69 | 292.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 12:15:00 | 292.20 | 292.41 | 292.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 12:30:00 | 292.80 | 292.41 | 292.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 294.20 | 292.46 | 292.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 11:15:00 | 292.10 | 292.59 | 292.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 11:45:00 | 292.15 | 292.40 | 292.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 13:00:00 | 292.20 | 292.36 | 292.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 14:15:00 | 291.80 | 292.36 | 292.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 14:15:00 | 292.65 | 292.42 | 292.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 15:00:00 | 292.65 | 292.42 | 292.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 15:15:00 | 292.75 | 292.48 | 292.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:15:00 | 294.20 | 292.48 | 292.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-17 09:15:00 | 293.50 | 292.69 | 292.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 09:15:00 | 293.50 | 292.69 | 292.66 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 11:15:00 | 292.05 | 292.96 | 292.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 12:15:00 | 289.05 | 292.18 | 292.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 15:15:00 | 286.55 | 286.50 | 288.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 09:15:00 | 287.05 | 286.50 | 288.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 285.85 | 286.37 | 288.41 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 293.95 | 288.85 | 288.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 11:15:00 | 294.90 | 290.06 | 289.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 297.25 | 297.68 | 295.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 10:00:00 | 297.25 | 297.68 | 295.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 293.70 | 297.21 | 296.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:30:00 | 292.20 | 297.21 | 296.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 294.85 | 296.73 | 296.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:30:00 | 294.65 | 296.73 | 296.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2024-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 12:15:00 | 294.25 | 295.48 | 295.55 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 14:15:00 | 297.55 | 295.31 | 295.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 11:15:00 | 299.90 | 296.44 | 295.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 15:15:00 | 296.80 | 297.51 | 296.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 15:15:00 | 296.80 | 297.51 | 296.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 296.80 | 297.51 | 296.65 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2024-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 12:15:00 | 295.00 | 296.16 | 296.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 14:15:00 | 292.20 | 295.02 | 295.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-03 09:15:00 | 295.15 | 294.62 | 295.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 09:15:00 | 295.15 | 294.62 | 295.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 295.15 | 294.62 | 295.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 11:30:00 | 293.00 | 294.31 | 295.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 10:15:00 | 298.50 | 294.98 | 294.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2024-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 10:15:00 | 298.50 | 294.98 | 294.93 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 09:15:00 | 290.90 | 294.54 | 294.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 10:15:00 | 284.30 | 292.49 | 293.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 291.90 | 290.03 | 291.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 10:15:00 | 291.90 | 290.03 | 291.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 291.90 | 290.03 | 291.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 291.90 | 290.03 | 291.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 291.65 | 290.35 | 291.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:30:00 | 292.10 | 290.35 | 291.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 292.95 | 290.87 | 291.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:00:00 | 292.95 | 290.87 | 291.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 292.95 | 291.29 | 291.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:45:00 | 292.65 | 291.29 | 291.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2024-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 15:15:00 | 293.45 | 292.07 | 292.05 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 09:15:00 | 289.65 | 291.59 | 291.84 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 11:15:00 | 291.35 | 290.73 | 290.65 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 12:15:00 | 285.60 | 290.05 | 290.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-15 09:15:00 | 282.80 | 287.29 | 288.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 09:15:00 | 284.50 | 284.25 | 286.20 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 10:45:00 | 282.15 | 283.75 | 285.80 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 13:15:00 | 285.70 | 283.91 | 285.33 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-16 13:15:00 | 285.70 | 283.91 | 285.33 | SL hit (close>ema400) qty=1.00 sl=285.33 alert=retest1 |

### Cycle 105 — BUY (started 2024-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 11:15:00 | 267.20 | 264.82 | 264.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 17:15:00 | 269.45 | 266.34 | 265.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 264.55 | 267.21 | 266.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 264.55 | 267.21 | 266.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 264.55 | 267.21 | 266.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 264.55 | 267.21 | 266.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 262.60 | 266.29 | 265.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 262.60 | 266.29 | 265.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 12:15:00 | 261.25 | 264.63 | 265.03 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2024-11-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 10:15:00 | 266.30 | 265.23 | 265.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 268.00 | 266.68 | 265.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 267.75 | 267.96 | 267.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 267.75 | 267.96 | 267.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 267.75 | 267.96 | 267.09 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2024-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 14:15:00 | 265.00 | 266.72 | 266.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 09:15:00 | 262.85 | 265.77 | 266.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 10:15:00 | 260.00 | 258.83 | 260.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 10:15:00 | 260.00 | 258.83 | 260.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 260.00 | 258.83 | 260.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 11:00:00 | 260.00 | 258.83 | 260.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 11:15:00 | 258.85 | 258.83 | 260.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 11:30:00 | 261.05 | 258.83 | 260.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 251.80 | 250.92 | 251.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:00:00 | 249.00 | 250.44 | 251.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:45:00 | 248.05 | 249.87 | 251.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 257.60 | 247.86 | 247.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 257.60 | 247.86 | 247.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 10:15:00 | 258.75 | 250.04 | 248.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 255.65 | 255.74 | 252.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 09:45:00 | 254.85 | 255.74 | 252.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 13:15:00 | 253.65 | 254.62 | 253.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 13:30:00 | 252.90 | 254.62 | 253.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 14:15:00 | 254.45 | 254.59 | 253.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 15:15:00 | 254.80 | 253.51 | 253.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 14:15:00 | 251.95 | 253.44 | 253.38 | SL hit (close<static) qty=1.00 sl=253.10 alert=retest2 |

### Cycle 110 — SELL (started 2024-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 09:15:00 | 253.10 | 253.30 | 253.32 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2024-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 10:15:00 | 254.20 | 253.48 | 253.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 12:15:00 | 255.90 | 254.13 | 253.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 09:15:00 | 255.00 | 255.38 | 254.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 09:15:00 | 255.00 | 255.38 | 254.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 255.00 | 255.38 | 254.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 10:15:00 | 256.20 | 255.38 | 254.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 12:30:00 | 256.45 | 255.41 | 254.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 14:00:00 | 256.35 | 255.60 | 254.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 09:15:00 | 257.80 | 260.19 | 260.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 09:15:00 | 257.80 | 260.19 | 260.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 09:15:00 | 256.90 | 258.82 | 259.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 09:15:00 | 257.30 | 257.01 | 258.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 09:15:00 | 257.30 | 257.01 | 258.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 257.30 | 257.01 | 258.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 14:15:00 | 256.05 | 256.72 | 257.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 10:00:00 | 256.05 | 256.66 | 257.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 12:45:00 | 255.35 | 255.95 | 256.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 14:30:00 | 255.80 | 254.35 | 255.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 255.00 | 254.48 | 255.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 253.20 | 254.48 | 255.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 253.00 | 254.19 | 254.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 10:15:00 | 252.55 | 254.19 | 254.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 243.25 | 244.61 | 247.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 243.25 | 244.61 | 247.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 242.58 | 244.61 | 247.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 243.01 | 244.61 | 247.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 239.92 | 244.61 | 247.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-20 10:15:00 | 243.75 | 242.98 | 244.84 | SL hit (close>ema200) qty=0.50 sl=242.98 alert=retest2 |

### Cycle 113 — BUY (started 2024-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 14:15:00 | 239.40 | 237.39 | 237.37 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 14:15:00 | 237.00 | 237.47 | 237.48 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 09:15:00 | 240.96 | 238.08 | 237.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 10:15:00 | 242.59 | 238.98 | 238.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 10:15:00 | 252.20 | 254.38 | 249.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 11:00:00 | 252.20 | 254.38 | 249.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 11:15:00 | 265.00 | 267.45 | 264.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 11:45:00 | 264.82 | 267.45 | 264.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 12:15:00 | 263.76 | 266.71 | 264.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 13:00:00 | 263.76 | 266.71 | 264.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 13:15:00 | 262.52 | 265.87 | 264.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 14:00:00 | 262.52 | 265.87 | 264.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 260.12 | 263.99 | 263.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 260.12 | 263.99 | 263.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 265.85 | 264.36 | 263.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 11:45:00 | 266.25 | 264.39 | 263.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-10 15:15:00 | 263.00 | 263.65 | 263.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-01-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 15:15:00 | 263.00 | 263.65 | 263.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 261.32 | 263.18 | 263.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 261.80 | 259.06 | 260.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 10:15:00 | 261.80 | 259.06 | 260.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 261.80 | 259.06 | 260.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 261.80 | 259.06 | 260.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 260.20 | 259.29 | 260.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 09:15:00 | 258.58 | 259.94 | 260.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-15 09:15:00 | 262.20 | 260.39 | 260.63 | SL hit (close>static) qty=1.00 sl=261.80 alert=retest2 |

### Cycle 117 — BUY (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 10:15:00 | 262.81 | 260.88 | 260.83 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 11:15:00 | 260.40 | 260.78 | 260.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-15 12:15:00 | 259.40 | 260.50 | 260.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-16 09:15:00 | 262.64 | 260.26 | 260.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-16 09:15:00 | 262.64 | 260.26 | 260.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 262.64 | 260.26 | 260.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 09:45:00 | 261.61 | 260.26 | 260.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 264.15 | 261.04 | 260.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 11:15:00 | 265.98 | 263.87 | 262.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 266.68 | 267.56 | 266.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 09:15:00 | 266.68 | 267.56 | 266.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 266.68 | 267.56 | 266.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 266.68 | 267.56 | 266.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 266.73 | 267.39 | 266.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 11:45:00 | 267.89 | 267.42 | 266.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 12:30:00 | 267.28 | 267.47 | 266.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 14:00:00 | 267.30 | 267.43 | 266.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-22 09:15:00 | 263.18 | 266.10 | 266.04 | SL hit (close<static) qty=1.00 sl=264.81 alert=retest2 |

### Cycle 120 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 263.16 | 265.51 | 265.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 262.39 | 264.89 | 265.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 251.78 | 251.69 | 255.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:00:00 | 251.78 | 251.69 | 255.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 15:15:00 | 252.00 | 250.78 | 251.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:15:00 | 254.07 | 250.78 | 251.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 257.73 | 252.17 | 252.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 10:00:00 | 257.73 | 252.17 | 252.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2025-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 10:15:00 | 258.48 | 253.43 | 253.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 10:15:00 | 259.23 | 256.80 | 255.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 258.10 | 259.86 | 258.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 258.10 | 259.86 | 258.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 258.10 | 259.86 | 258.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:00:00 | 258.10 | 259.86 | 258.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 256.05 | 259.10 | 257.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:45:00 | 256.30 | 259.10 | 257.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 257.55 | 258.79 | 257.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 15:00:00 | 257.90 | 258.61 | 257.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 247.35 | 256.21 | 256.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 247.35 | 256.21 | 256.91 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 259.15 | 254.85 | 254.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 14:15:00 | 261.60 | 258.48 | 256.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 10:15:00 | 258.50 | 258.94 | 257.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 11:00:00 | 258.50 | 258.94 | 257.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 256.30 | 258.41 | 257.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:00:00 | 256.30 | 258.41 | 257.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 255.60 | 257.85 | 257.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 255.60 | 257.85 | 257.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 09:15:00 | 248.65 | 255.25 | 256.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 243.80 | 249.32 | 252.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 15:15:00 | 238.50 | 238.17 | 240.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 09:15:00 | 235.90 | 238.17 | 240.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 231.35 | 231.09 | 233.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:45:00 | 233.10 | 231.09 | 233.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 232.50 | 231.44 | 233.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 14:00:00 | 232.50 | 231.44 | 233.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 233.80 | 231.91 | 233.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 233.80 | 231.91 | 233.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 234.00 | 232.33 | 233.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 09:15:00 | 230.45 | 232.33 | 233.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-18 14:15:00 | 236.70 | 233.48 | 233.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 14:15:00 | 236.70 | 233.48 | 233.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 09:15:00 | 237.45 | 234.68 | 234.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 238.75 | 240.10 | 238.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 238.75 | 240.10 | 238.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 238.40 | 239.76 | 238.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 238.40 | 239.76 | 238.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 239.25 | 239.66 | 238.45 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 10:15:00 | 235.90 | 238.30 | 238.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 14:15:00 | 234.30 | 236.56 | 237.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 15:15:00 | 231.20 | 231.12 | 233.01 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-28 09:15:00 | 227.95 | 231.12 | 233.01 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-04 09:15:00 | 216.55 | 224.60 | 226.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 225.32 | 224.74 | 226.27 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-04 10:15:00 | 225.32 | 224.74 | 226.27 | SL hit (close>ema200) qty=0.50 sl=224.74 alert=retest1 |

### Cycle 127 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 229.90 | 226.94 | 226.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 230.81 | 227.71 | 227.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 09:15:00 | 227.10 | 228.36 | 227.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-06 09:15:00 | 227.10 | 228.36 | 227.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 09:15:00 | 227.10 | 228.36 | 227.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 09:45:00 | 227.46 | 228.36 | 227.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 10:15:00 | 229.60 | 228.61 | 227.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 12:30:00 | 230.56 | 229.16 | 228.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 12:15:00 | 226.95 | 230.63 | 230.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2025-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 12:15:00 | 226.95 | 230.63 | 230.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 13:15:00 | 225.09 | 229.52 | 230.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 13:15:00 | 225.34 | 225.09 | 227.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 14:00:00 | 225.34 | 225.09 | 227.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 227.06 | 225.48 | 227.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 15:00:00 | 227.06 | 225.48 | 227.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 226.25 | 225.64 | 226.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 226.33 | 225.64 | 226.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 224.60 | 225.43 | 226.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:45:00 | 223.30 | 225.17 | 226.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 13:45:00 | 224.07 | 224.52 | 225.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 15:15:00 | 224.20 | 224.56 | 225.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 09:15:00 | 228.02 | 226.05 | 226.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2025-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 09:15:00 | 228.02 | 226.05 | 226.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 13:15:00 | 230.05 | 227.47 | 226.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 242.80 | 243.20 | 240.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-25 11:15:00 | 241.63 | 242.69 | 240.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 241.63 | 242.69 | 240.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:45:00 | 242.08 | 242.69 | 240.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 243.00 | 242.75 | 241.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 13:30:00 | 243.73 | 243.07 | 241.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 243.92 | 242.65 | 241.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 10:15:00 | 243.66 | 242.70 | 241.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 12:15:00 | 240.61 | 242.23 | 241.71 | SL hit (close<static) qty=1.00 sl=241.11 alert=retest2 |

### Cycle 130 — SELL (started 2025-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 15:15:00 | 239.90 | 241.18 | 241.31 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 15:15:00 | 242.65 | 241.43 | 241.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 09:15:00 | 252.60 | 243.66 | 242.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 14:15:00 | 246.54 | 248.27 | 245.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-28 15:00:00 | 246.54 | 248.27 | 245.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 245.81 | 248.93 | 248.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:45:00 | 245.48 | 248.93 | 248.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 246.26 | 248.39 | 248.06 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 11:15:00 | 244.28 | 247.57 | 247.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-03 12:15:00 | 243.34 | 246.72 | 247.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 223.23 | 221.22 | 226.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 11:00:00 | 223.23 | 221.22 | 226.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 225.42 | 222.67 | 226.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:45:00 | 225.99 | 222.67 | 226.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 226.54 | 224.01 | 226.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 15:00:00 | 226.54 | 224.01 | 226.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 15:15:00 | 226.47 | 224.50 | 226.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 221.00 | 224.50 | 226.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 228.12 | 224.01 | 224.83 | SL hit (close>static) qty=1.00 sl=227.16 alert=retest2 |

### Cycle 133 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 229.44 | 225.72 | 225.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 12:15:00 | 230.70 | 226.71 | 225.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 14:15:00 | 247.80 | 248.56 | 245.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 15:00:00 | 247.80 | 248.56 | 245.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 245.75 | 247.85 | 246.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 245.66 | 247.85 | 246.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 247.65 | 247.81 | 246.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 11:30:00 | 246.79 | 247.81 | 246.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 248.53 | 249.56 | 248.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 13:30:00 | 248.76 | 249.56 | 248.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 249.36 | 249.52 | 248.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 14:45:00 | 248.22 | 249.52 | 248.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 246.17 | 248.85 | 248.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 246.17 | 248.85 | 248.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 246.41 | 248.36 | 248.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:30:00 | 246.02 | 248.36 | 248.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 246.75 | 248.04 | 248.13 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2025-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 10:15:00 | 249.66 | 248.04 | 247.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 11:15:00 | 251.15 | 248.66 | 248.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 10:15:00 | 249.88 | 250.06 | 249.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 10:15:00 | 249.88 | 250.06 | 249.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 249.88 | 250.06 | 249.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 10:45:00 | 249.60 | 250.06 | 249.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 249.25 | 249.90 | 249.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 12:00:00 | 249.25 | 249.90 | 249.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 12:15:00 | 246.92 | 249.30 | 249.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 13:00:00 | 246.92 | 249.30 | 249.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2025-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 13:15:00 | 245.70 | 248.58 | 248.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 14:15:00 | 245.52 | 247.97 | 248.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 245.44 | 244.92 | 246.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 245.44 | 244.92 | 246.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 245.44 | 244.92 | 246.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:00:00 | 245.44 | 244.92 | 246.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 240.42 | 240.01 | 241.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 10:15:00 | 239.33 | 240.01 | 241.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:00:00 | 238.89 | 239.79 | 241.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 10:15:00 | 241.58 | 237.02 | 236.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 241.58 | 237.02 | 236.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 243.10 | 239.04 | 237.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 241.30 | 242.03 | 240.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 12:45:00 | 241.33 | 242.03 | 240.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 240.89 | 241.59 | 240.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 243.60 | 241.47 | 240.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 10:15:00 | 245.75 | 247.74 | 248.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 10:15:00 | 245.75 | 247.74 | 248.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 12:15:00 | 243.68 | 246.54 | 247.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 12:15:00 | 243.96 | 243.68 | 245.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 13:00:00 | 243.96 | 243.68 | 245.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 246.49 | 244.40 | 244.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:30:00 | 246.35 | 244.40 | 244.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2025-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 13:15:00 | 246.39 | 245.39 | 245.32 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 09:15:00 | 243.76 | 245.23 | 245.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 11:15:00 | 242.40 | 243.94 | 244.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 243.39 | 243.17 | 243.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 243.39 | 243.17 | 243.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 243.39 | 243.17 | 243.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 10:30:00 | 242.07 | 242.90 | 243.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 12:00:00 | 242.22 | 242.76 | 243.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 12:30:00 | 242.18 | 242.82 | 243.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 13:45:00 | 242.20 | 242.70 | 243.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 243.18 | 242.79 | 243.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 243.18 | 242.79 | 243.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 244.45 | 243.13 | 243.45 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-05-29 15:15:00 | 244.45 | 243.13 | 243.45 | SL hit (close>static) qty=1.00 sl=244.29 alert=retest2 |

### Cycle 141 — BUY (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 11:15:00 | 238.73 | 238.19 | 238.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 13:15:00 | 239.47 | 238.59 | 238.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 247.65 | 248.41 | 246.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 14:00:00 | 247.65 | 248.41 | 246.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 252.77 | 254.29 | 252.25 | EMA400 retest candle locked (from upside) |

### Cycle 142 — SELL (started 2025-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 13:15:00 | 250.54 | 251.88 | 252.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 15:15:00 | 250.00 | 251.27 | 251.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 11:15:00 | 251.21 | 250.58 | 251.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 11:15:00 | 251.21 | 250.58 | 251.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 251.21 | 250.58 | 251.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:00:00 | 251.21 | 250.58 | 251.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 250.17 | 250.49 | 251.13 | EMA400 retest candle locked (from downside) |

### Cycle 143 — BUY (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 10:15:00 | 252.63 | 251.40 | 251.39 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 11:15:00 | 251.08 | 251.43 | 251.44 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 251.82 | 251.51 | 251.47 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 13:15:00 | 251.01 | 251.41 | 251.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-24 09:15:00 | 246.67 | 250.41 | 250.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 12:15:00 | 243.07 | 242.96 | 244.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-26 13:00:00 | 243.07 | 242.96 | 244.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 244.67 | 243.41 | 244.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 15:00:00 | 244.67 | 243.41 | 244.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 245.02 | 243.73 | 244.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:15:00 | 244.50 | 243.73 | 244.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 244.29 | 243.84 | 244.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 10:45:00 | 243.16 | 243.91 | 244.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 14:00:00 | 243.58 | 243.97 | 244.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 14:30:00 | 242.92 | 243.72 | 244.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 11:00:00 | 243.56 | 243.90 | 244.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 244.31 | 243.77 | 244.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 15:00:00 | 244.31 | 243.77 | 244.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 244.00 | 243.82 | 244.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:15:00 | 244.40 | 243.82 | 244.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 243.27 | 243.71 | 243.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 10:15:00 | 242.50 | 243.71 | 243.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 14:30:00 | 242.72 | 243.23 | 243.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 15:15:00 | 243.19 | 243.23 | 243.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 11:30:00 | 242.00 | 242.85 | 243.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 244.60 | 242.35 | 242.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 244.60 | 242.35 | 242.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 245.94 | 243.07 | 243.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 245.94 | 243.07 | 243.04 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 13:15:00 | 242.00 | 243.53 | 243.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 15:15:00 | 241.40 | 242.83 | 243.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 13:15:00 | 242.29 | 242.17 | 242.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 14:00:00 | 242.29 | 242.17 | 242.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 243.45 | 242.43 | 242.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 243.45 | 242.43 | 242.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 243.30 | 242.60 | 242.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 243.56 | 242.60 | 242.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 243.44 | 242.98 | 243.01 | EMA400 retest candle locked (from downside) |

### Cycle 149 — BUY (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 11:15:00 | 243.98 | 243.18 | 243.10 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 15:15:00 | 243.00 | 243.22 | 243.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 241.55 | 242.89 | 243.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 242.44 | 242.16 | 242.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 242.44 | 242.16 | 242.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 242.44 | 242.16 | 242.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:30:00 | 242.18 | 242.16 | 242.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 242.59 | 242.25 | 242.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:00:00 | 242.59 | 242.25 | 242.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 243.49 | 242.50 | 242.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:00:00 | 243.49 | 242.50 | 242.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 243.10 | 242.62 | 242.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:15:00 | 244.15 | 242.62 | 242.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 13:15:00 | 244.23 | 242.94 | 242.80 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 14:15:00 | 242.74 | 243.18 | 243.24 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 10:15:00 | 244.25 | 243.40 | 243.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 09:15:00 | 244.75 | 243.98 | 243.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 09:15:00 | 245.10 | 245.46 | 244.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 245.10 | 245.46 | 244.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 245.10 | 245.46 | 244.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:45:00 | 244.94 | 245.46 | 244.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 245.00 | 245.28 | 244.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:30:00 | 244.64 | 245.28 | 244.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 245.19 | 245.26 | 244.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:30:00 | 245.29 | 245.26 | 244.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 245.00 | 245.17 | 244.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 14:45:00 | 244.85 | 245.17 | 244.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 244.94 | 245.12 | 244.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 244.87 | 245.12 | 244.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 244.80 | 245.06 | 244.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:30:00 | 245.18 | 245.06 | 244.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 244.73 | 244.99 | 244.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:00:00 | 244.73 | 244.99 | 244.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 245.51 | 245.10 | 244.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 12:15:00 | 245.93 | 245.10 | 244.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 15:00:00 | 245.73 | 245.44 | 245.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 10:45:00 | 245.80 | 245.44 | 245.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 11:15:00 | 244.29 | 245.21 | 245.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 244.29 | 245.21 | 245.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 242.61 | 244.54 | 244.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 10:15:00 | 241.90 | 240.68 | 241.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 10:15:00 | 241.90 | 240.68 | 241.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 241.90 | 240.68 | 241.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:00:00 | 241.90 | 240.68 | 241.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 241.50 | 240.85 | 241.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:30:00 | 241.73 | 240.85 | 241.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 241.32 | 240.94 | 241.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:30:00 | 241.07 | 240.94 | 241.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 241.26 | 241.01 | 241.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:45:00 | 241.58 | 241.01 | 241.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 241.66 | 241.14 | 241.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 241.66 | 241.14 | 241.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 241.10 | 241.13 | 241.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 242.08 | 241.13 | 241.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 240.29 | 240.96 | 241.42 | EMA400 retest candle locked (from downside) |

### Cycle 155 — BUY (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 12:15:00 | 242.90 | 241.76 | 241.70 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 239.80 | 241.43 | 241.59 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 13:15:00 | 242.41 | 241.66 | 241.65 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 241.00 | 241.53 | 241.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 15:15:00 | 240.25 | 241.27 | 241.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 13:15:00 | 237.84 | 237.83 | 239.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-01 14:00:00 | 237.84 | 237.83 | 239.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 234.00 | 233.26 | 234.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 234.00 | 233.26 | 234.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 233.55 | 233.32 | 233.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 234.49 | 233.32 | 233.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 233.58 | 233.37 | 233.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 09:30:00 | 232.52 | 233.21 | 233.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 12:15:00 | 232.80 | 233.07 | 233.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 09:15:00 | 234.36 | 233.60 | 233.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 09:15:00 | 234.36 | 233.60 | 233.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 11:15:00 | 235.28 | 233.95 | 233.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 236.01 | 237.63 | 236.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 236.01 | 237.63 | 236.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 236.01 | 237.63 | 236.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:30:00 | 235.03 | 237.63 | 236.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 236.25 | 237.36 | 236.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:30:00 | 235.76 | 237.36 | 236.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 237.10 | 237.30 | 236.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 12:45:00 | 237.22 | 237.26 | 236.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 13:30:00 | 237.36 | 237.30 | 236.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 14:00:00 | 237.46 | 237.30 | 236.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 12:30:00 | 237.32 | 237.10 | 236.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 237.35 | 237.54 | 237.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:45:00 | 236.94 | 237.54 | 237.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 238.20 | 237.88 | 237.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 14:30:00 | 238.32 | 237.89 | 237.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:15:00 | 238.50 | 237.96 | 237.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 11:15:00 | 236.75 | 238.19 | 238.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 236.75 | 238.19 | 238.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 12:15:00 | 236.62 | 237.88 | 238.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 14:15:00 | 236.58 | 236.55 | 237.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 14:15:00 | 236.58 | 236.55 | 237.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 236.58 | 236.55 | 237.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 235.18 | 236.63 | 237.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 11:15:00 | 236.35 | 234.54 | 234.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 236.35 | 234.54 | 234.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 13:15:00 | 237.20 | 235.27 | 234.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 239.30 | 239.52 | 237.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 14:00:00 | 239.30 | 239.52 | 237.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 239.07 | 239.78 | 238.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:00:00 | 239.07 | 239.78 | 238.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 239.02 | 239.63 | 238.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:30:00 | 238.96 | 239.63 | 238.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 238.94 | 239.49 | 238.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:45:00 | 238.60 | 239.49 | 238.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 236.71 | 238.88 | 238.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:00:00 | 236.71 | 238.88 | 238.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 236.88 | 238.48 | 238.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:30:00 | 236.66 | 238.48 | 238.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 237.25 | 238.23 | 238.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 236.17 | 237.57 | 237.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 234.01 | 233.44 | 234.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 234.01 | 233.44 | 234.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 234.01 | 233.44 | 234.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:00:00 | 234.01 | 233.44 | 234.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 233.75 | 232.25 | 232.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:30:00 | 234.03 | 232.25 | 232.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 234.38 | 232.68 | 232.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:45:00 | 234.51 | 232.68 | 232.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — BUY (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 12:15:00 | 233.78 | 233.05 | 232.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 09:15:00 | 234.65 | 233.68 | 233.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 11:15:00 | 233.45 | 233.64 | 233.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 12:00:00 | 233.45 | 233.64 | 233.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 232.86 | 233.48 | 233.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 13:00:00 | 232.86 | 233.48 | 233.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 233.45 | 233.48 | 233.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 14:15:00 | 233.82 | 233.48 | 233.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 09:15:00 | 234.04 | 233.39 | 233.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 232.45 | 233.20 | 233.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — SELL (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 09:15:00 | 232.45 | 233.20 | 233.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 13:15:00 | 232.20 | 232.72 | 232.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 234.43 | 232.95 | 233.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 234.43 | 232.95 | 233.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 234.43 | 232.95 | 233.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:00:00 | 234.43 | 232.95 | 233.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — BUY (started 2025-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 10:15:00 | 234.69 | 233.30 | 233.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 12:15:00 | 234.83 | 233.83 | 233.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 09:15:00 | 235.55 | 236.04 | 235.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 10:00:00 | 235.55 | 236.04 | 235.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 235.03 | 235.75 | 235.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:00:00 | 235.03 | 235.75 | 235.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 234.60 | 235.52 | 235.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 234.60 | 235.52 | 235.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 235.00 | 235.42 | 235.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 15:00:00 | 235.62 | 235.46 | 235.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 10:15:00 | 235.58 | 235.44 | 235.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 12:15:00 | 242.45 | 244.36 | 244.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — SELL (started 2025-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 12:15:00 | 242.45 | 244.36 | 244.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 241.68 | 243.55 | 243.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 243.45 | 243.07 | 243.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 11:00:00 | 243.45 | 243.07 | 243.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 244.00 | 243.26 | 243.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 244.00 | 243.26 | 243.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 243.62 | 243.33 | 243.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 13:15:00 | 243.10 | 243.33 | 243.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 246.35 | 243.88 | 243.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 246.35 | 243.88 | 243.78 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 243.45 | 244.21 | 244.29 | EMA200 below EMA400 |

### Cycle 169 — BUY (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 09:15:00 | 248.33 | 244.86 | 244.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 14:15:00 | 248.85 | 247.74 | 247.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-21 14:15:00 | 247.85 | 247.95 | 247.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 14:15:00 | 247.85 | 247.95 | 247.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 247.85 | 247.95 | 247.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 247.85 | 247.95 | 247.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 253.05 | 254.14 | 252.44 | EMA400 retest candle locked (from upside) |

### Cycle 170 — SELL (started 2025-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 12:15:00 | 250.85 | 252.26 | 252.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 14:15:00 | 250.37 | 251.65 | 252.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 254.00 | 251.94 | 252.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 254.00 | 251.94 | 252.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 254.00 | 251.94 | 252.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 254.00 | 251.94 | 252.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 171 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 253.90 | 252.33 | 252.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 15:15:00 | 256.51 | 254.23 | 253.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 12:15:00 | 254.05 | 254.43 | 253.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 13:00:00 | 254.05 | 254.43 | 253.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 254.27 | 254.40 | 253.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:30:00 | 253.75 | 254.40 | 253.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 254.54 | 254.43 | 253.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:45:00 | 253.95 | 254.43 | 253.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 253.87 | 254.36 | 253.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:30:00 | 253.56 | 254.36 | 253.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 255.34 | 254.55 | 254.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:30:00 | 254.47 | 254.55 | 254.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 257.00 | 255.73 | 254.93 | EMA400 retest candle locked (from upside) |

### Cycle 172 — SELL (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 13:15:00 | 253.40 | 255.27 | 255.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 14:15:00 | 252.30 | 254.67 | 255.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 10:15:00 | 254.20 | 254.14 | 254.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 11:00:00 | 254.20 | 254.14 | 254.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 252.25 | 252.34 | 253.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:30:00 | 252.90 | 252.34 | 253.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 252.40 | 252.45 | 253.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 252.95 | 252.45 | 253.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 254.45 | 252.83 | 253.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 255.20 | 252.83 | 253.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 251.70 | 252.60 | 253.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 12:15:00 | 251.15 | 252.38 | 252.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 14:45:00 | 251.00 | 251.88 | 252.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:30:00 | 250.30 | 251.27 | 252.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 11:15:00 | 253.50 | 251.38 | 251.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — BUY (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 11:15:00 | 253.50 | 251.38 | 251.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 15:15:00 | 254.20 | 252.93 | 252.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 249.60 | 252.26 | 251.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 249.60 | 252.26 | 251.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 249.60 | 252.26 | 251.89 | EMA400 retest candle locked (from upside) |

### Cycle 174 — SELL (started 2025-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 12:15:00 | 251.00 | 251.67 | 251.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 247.75 | 250.59 | 251.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 248.90 | 248.55 | 249.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 248.90 | 248.55 | 249.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 248.90 | 248.55 | 249.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:30:00 | 248.70 | 248.55 | 249.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 247.85 | 248.20 | 248.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:30:00 | 249.10 | 248.20 | 248.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 248.20 | 247.59 | 248.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:00:00 | 248.20 | 247.59 | 248.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 249.25 | 247.93 | 248.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:00:00 | 249.25 | 247.93 | 248.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — BUY (started 2025-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 14:15:00 | 249.05 | 248.45 | 248.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 10:15:00 | 250.10 | 248.95 | 248.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 12:15:00 | 248.70 | 248.97 | 248.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 12:15:00 | 248.70 | 248.97 | 248.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 12:15:00 | 248.70 | 248.97 | 248.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 13:00:00 | 248.70 | 248.97 | 248.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 249.60 | 249.10 | 248.81 | EMA400 retest candle locked (from upside) |

### Cycle 176 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 247.85 | 248.49 | 248.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 15:15:00 | 246.35 | 247.43 | 247.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 10:15:00 | 247.00 | 246.19 | 246.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 10:15:00 | 247.00 | 246.19 | 246.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 247.00 | 246.19 | 246.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:45:00 | 247.00 | 246.19 | 246.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 247.30 | 246.41 | 246.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:45:00 | 247.20 | 246.41 | 246.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 247.10 | 246.11 | 246.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 247.10 | 246.11 | 246.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 247.15 | 246.32 | 246.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:30:00 | 247.00 | 246.32 | 246.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 247.40 | 246.75 | 246.73 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 09:15:00 | 245.15 | 246.73 | 246.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 10:15:00 | 244.90 | 246.37 | 246.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 244.50 | 243.81 | 244.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 244.50 | 243.81 | 244.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 244.50 | 243.81 | 244.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:30:00 | 245.13 | 243.81 | 244.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 244.89 | 244.03 | 244.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:30:00 | 244.97 | 244.03 | 244.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 244.22 | 244.06 | 244.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:30:00 | 244.34 | 244.06 | 244.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 244.30 | 244.11 | 244.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:15:00 | 244.30 | 244.11 | 244.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 244.97 | 244.28 | 244.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:00:00 | 244.97 | 244.28 | 244.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 244.98 | 244.42 | 244.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:30:00 | 244.80 | 244.42 | 244.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 244.35 | 244.41 | 244.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 245.97 | 244.41 | 244.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 244.14 | 244.35 | 244.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 14:15:00 | 243.12 | 244.02 | 244.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 242.25 | 243.93 | 244.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 09:15:00 | 230.96 | 236.84 | 237.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 09:15:00 | 230.14 | 236.84 | 237.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-15 14:15:00 | 235.33 | 235.06 | 236.47 | SL hit (close>ema200) qty=0.50 sl=235.06 alert=retest2 |

### Cycle 179 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 234.99 | 233.08 | 232.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 236.74 | 234.60 | 233.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 235.48 | 235.62 | 234.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 15:00:00 | 235.48 | 235.62 | 234.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 235.25 | 235.54 | 234.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:45:00 | 234.96 | 235.54 | 234.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 234.83 | 235.40 | 234.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:45:00 | 234.75 | 235.40 | 234.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 234.09 | 235.14 | 234.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:00:00 | 234.09 | 235.14 | 234.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 233.52 | 234.82 | 234.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:30:00 | 233.60 | 234.82 | 234.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 233.80 | 234.61 | 234.64 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 09:15:00 | 235.26 | 234.61 | 234.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 10:15:00 | 238.55 | 235.40 | 234.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 12:15:00 | 235.45 | 235.70 | 235.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-29 13:00:00 | 235.45 | 235.70 | 235.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 234.85 | 235.53 | 235.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 234.85 | 235.53 | 235.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 234.69 | 235.36 | 235.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:45:00 | 234.85 | 235.36 | 235.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 234.99 | 235.23 | 235.08 | EMA400 retest candle locked (from upside) |

### Cycle 182 — SELL (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 11:15:00 | 233.94 | 234.80 | 234.90 | EMA200 below EMA400 |

### Cycle 183 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 236.70 | 235.02 | 234.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 11:15:00 | 238.59 | 236.09 | 235.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 11:15:00 | 238.65 | 238.70 | 237.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 11:45:00 | 238.90 | 238.70 | 237.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 237.72 | 238.28 | 237.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 14:45:00 | 237.88 | 238.28 | 237.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 236.83 | 239.84 | 239.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:00:00 | 236.83 | 239.84 | 239.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 238.20 | 239.51 | 238.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:30:00 | 235.97 | 239.51 | 238.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 237.62 | 238.61 | 238.65 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 09:15:00 | 241.05 | 238.93 | 238.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 14:15:00 | 241.88 | 240.33 | 239.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 11:15:00 | 240.18 | 240.70 | 240.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 12:00:00 | 240.18 | 240.70 | 240.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 239.93 | 240.55 | 240.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:15:00 | 239.24 | 240.55 | 240.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 238.60 | 240.16 | 239.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:30:00 | 238.40 | 240.16 | 239.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 239.14 | 239.96 | 239.83 | EMA400 retest candle locked (from upside) |

### Cycle 186 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 236.20 | 239.08 | 239.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 234.48 | 238.16 | 239.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 235.79 | 234.14 | 236.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 235.79 | 234.14 | 236.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 235.79 | 234.14 | 236.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 09:30:00 | 231.96 | 233.83 | 235.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 09:15:00 | 240.05 | 235.79 | 235.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — BUY (started 2026-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 09:15:00 | 240.05 | 235.79 | 235.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 11:15:00 | 241.58 | 237.67 | 236.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 246.15 | 247.05 | 243.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 246.15 | 247.05 | 243.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 246.15 | 247.05 | 243.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 10:15:00 | 247.48 | 247.05 | 243.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 11:15:00 | 247.00 | 246.97 | 244.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 14:45:00 | 247.44 | 246.34 | 244.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 14:15:00 | 243.10 | 244.36 | 244.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — SELL (started 2026-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 14:15:00 | 243.10 | 244.36 | 244.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 10:15:00 | 241.24 | 243.29 | 243.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 242.00 | 241.67 | 242.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 242.00 | 241.67 | 242.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 242.00 | 241.67 | 242.64 | EMA400 retest candle locked (from downside) |

### Cycle 189 — BUY (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 09:15:00 | 245.51 | 242.84 | 242.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 09:15:00 | 247.02 | 245.32 | 244.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 270.44 | 271.62 | 265.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 10:15:00 | 270.00 | 271.62 | 265.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 265.50 | 269.57 | 267.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 265.50 | 269.57 | 267.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 260.55 | 267.77 | 267.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 263.70 | 267.77 | 267.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 259.75 | 266.16 | 266.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 254.55 | 263.84 | 265.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 255.50 | 253.82 | 257.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 11:45:00 | 255.80 | 253.82 | 257.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 256.80 | 254.91 | 256.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 14:45:00 | 257.10 | 254.91 | 256.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 15:15:00 | 256.75 | 255.28 | 256.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 09:15:00 | 265.85 | 255.28 | 256.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 268.00 | 257.82 | 257.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 13:15:00 | 268.70 | 263.83 | 261.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 265.80 | 267.39 | 265.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 265.80 | 267.39 | 265.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 265.80 | 267.39 | 265.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:30:00 | 265.15 | 267.39 | 265.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 267.90 | 267.49 | 265.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:30:00 | 265.35 | 267.49 | 265.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 266.00 | 267.43 | 266.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 266.00 | 267.43 | 266.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 267.60 | 267.46 | 266.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 13:00:00 | 268.80 | 267.72 | 266.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 09:15:00 | 269.70 | 267.13 | 266.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 10:15:00 | 268.60 | 269.37 | 268.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 13:00:00 | 268.65 | 269.17 | 268.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 269.00 | 269.14 | 268.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 14:15:00 | 271.55 | 269.14 | 268.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 266.90 | 273.41 | 272.28 | SL hit (close<static) qty=1.00 sl=268.20 alert=retest2 |

### Cycle 192 — SELL (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 12:15:00 | 269.20 | 271.16 | 271.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 267.15 | 269.82 | 270.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 271.15 | 269.71 | 270.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 09:15:00 | 271.15 | 269.71 | 270.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 271.15 | 269.71 | 270.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:00:00 | 271.15 | 269.71 | 270.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 270.90 | 269.95 | 270.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:30:00 | 271.50 | 269.95 | 270.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 270.55 | 270.07 | 270.53 | EMA400 retest candle locked (from downside) |

### Cycle 193 — BUY (started 2026-02-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 15:15:00 | 271.50 | 270.87 | 270.81 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 09:15:00 | 270.00 | 270.69 | 270.73 | EMA200 below EMA400 |

### Cycle 195 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 272.10 | 270.77 | 270.70 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 09:15:00 | 264.20 | 269.63 | 270.20 | EMA200 below EMA400 |

### Cycle 197 — BUY (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 12:15:00 | 272.70 | 268.69 | 268.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 13:15:00 | 275.35 | 270.02 | 269.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 09:15:00 | 273.80 | 275.85 | 273.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 273.80 | 275.85 | 273.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 273.80 | 275.85 | 273.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:45:00 | 273.65 | 275.85 | 273.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 273.80 | 275.44 | 273.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:45:00 | 273.15 | 275.44 | 273.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 272.70 | 274.89 | 273.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:00:00 | 272.70 | 274.89 | 273.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 272.65 | 274.44 | 273.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:30:00 | 273.40 | 274.44 | 273.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 273.80 | 274.31 | 273.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 14:15:00 | 274.45 | 274.31 | 273.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 14:45:00 | 274.80 | 274.61 | 273.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 10:30:00 | 274.10 | 274.81 | 274.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 13:30:00 | 274.90 | 275.05 | 274.39 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 276.70 | 275.88 | 274.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 15:00:00 | 280.05 | 277.86 | 276.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 10:30:00 | 279.20 | 278.90 | 277.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 13:00:00 | 280.50 | 281.28 | 280.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 14:00:00 | 279.00 | 280.82 | 280.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 14:15:00 | 277.15 | 280.09 | 280.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 15:00:00 | 277.15 | 280.09 | 280.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-04 15:15:00 | 275.70 | 279.21 | 279.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — SELL (started 2026-03-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 15:15:00 | 275.70 | 279.21 | 279.67 | EMA200 below EMA400 |

### Cycle 199 — BUY (started 2026-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 09:15:00 | 284.80 | 280.33 | 280.13 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 15:15:00 | 276.75 | 280.19 | 280.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 09:15:00 | 273.80 | 278.91 | 279.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 11:15:00 | 277.80 | 277.78 | 279.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 12:00:00 | 277.80 | 277.78 | 279.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 279.60 | 278.14 | 279.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 12:45:00 | 279.75 | 278.14 | 279.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 279.45 | 278.40 | 279.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 13:45:00 | 279.85 | 278.40 | 279.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 278.85 | 278.49 | 279.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:30:00 | 277.00 | 278.48 | 279.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 263.15 | 265.86 | 267.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-17 10:15:00 | 261.80 | 261.59 | 264.03 | SL hit (close>ema200) qty=0.50 sl=261.59 alert=retest2 |

### Cycle 201 — BUY (started 2026-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 14:15:00 | 265.00 | 264.13 | 264.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-19 10:15:00 | 269.50 | 265.44 | 264.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-20 12:15:00 | 268.60 | 268.73 | 267.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-20 12:30:00 | 269.10 | 268.73 | 267.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 266.35 | 268.25 | 267.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:45:00 | 266.20 | 268.25 | 267.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 265.45 | 267.69 | 267.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 14:45:00 | 264.95 | 267.69 | 267.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 266.80 | 267.27 | 266.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 11:00:00 | 266.80 | 267.27 | 266.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 11:15:00 | 267.05 | 267.23 | 266.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 12:15:00 | 265.80 | 267.23 | 266.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 12:15:00 | 266.15 | 267.01 | 266.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 12:30:00 | 266.90 | 267.01 | 266.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — SELL (started 2026-03-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 13:15:00 | 266.05 | 266.82 | 266.83 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2026-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 09:15:00 | 270.20 | 267.16 | 266.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-27 09:15:00 | 276.90 | 271.29 | 269.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-01 13:15:00 | 286.85 | 287.01 | 283.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-01 13:30:00 | 286.40 | 287.01 | 283.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 285.40 | 286.98 | 284.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 09:30:00 | 284.65 | 286.98 | 284.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 285.40 | 286.67 | 284.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:45:00 | 285.00 | 286.67 | 284.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 12:15:00 | 285.55 | 286.24 | 284.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 12:30:00 | 285.20 | 286.24 | 284.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 285.40 | 286.42 | 285.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:15:00 | 284.70 | 286.42 | 285.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 281.15 | 285.36 | 284.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 11:00:00 | 281.15 | 285.36 | 284.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 11:15:00 | 281.85 | 284.66 | 284.59 | EMA400 retest candle locked (from upside) |

### Cycle 204 — SELL (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 12:15:00 | 283.10 | 284.35 | 284.46 | EMA200 below EMA400 |

### Cycle 205 — BUY (started 2026-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 12:15:00 | 285.55 | 284.25 | 284.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 13:15:00 | 286.35 | 284.67 | 284.38 | Break + close above crossover candle high |

### Cycle 206 — SELL (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-08 09:15:00 | 277.85 | 283.83 | 284.12 | EMA200 below EMA400 |

### Cycle 207 — BUY (started 2026-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 14:15:00 | 285.55 | 284.23 | 284.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 09:15:00 | 288.20 | 285.28 | 284.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 13:15:00 | 286.00 | 287.82 | 286.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 13:15:00 | 286.00 | 287.82 | 286.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 13:15:00 | 286.00 | 287.82 | 286.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:00:00 | 286.00 | 287.82 | 286.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 286.85 | 287.63 | 286.94 | EMA400 retest candle locked (from upside) |

### Cycle 208 — SELL (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 10:15:00 | 284.45 | 286.16 | 286.38 | EMA200 below EMA400 |

### Cycle 209 — BUY (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 14:15:00 | 287.80 | 286.41 | 286.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 11:15:00 | 288.60 | 287.12 | 286.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-15 13:15:00 | 287.10 | 287.42 | 286.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-15 14:00:00 | 287.10 | 287.42 | 286.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 287.30 | 287.39 | 287.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 14:30:00 | 287.05 | 287.39 | 287.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 15:15:00 | 287.50 | 287.41 | 287.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 09:15:00 | 286.60 | 287.41 | 287.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 285.95 | 287.12 | 286.95 | EMA400 retest candle locked (from upside) |

### Cycle 210 — SELL (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 10:15:00 | 283.90 | 286.48 | 286.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-16 11:15:00 | 283.05 | 285.79 | 286.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 11:15:00 | 284.45 | 283.86 | 284.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-17 11:45:00 | 284.15 | 283.86 | 284.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 13:15:00 | 284.75 | 284.14 | 284.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 13:30:00 | 285.25 | 284.14 | 284.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 14:15:00 | 283.95 | 284.10 | 284.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 14:45:00 | 285.35 | 284.10 | 284.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 15:15:00 | 284.00 | 284.08 | 284.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:15:00 | 284.25 | 284.08 | 284.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 283.10 | 283.89 | 284.51 | EMA400 retest candle locked (from downside) |

### Cycle 211 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 285.50 | 283.93 | 283.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 293.95 | 287.25 | 286.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 14:15:00 | 300.65 | 301.36 | 296.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 15:00:00 | 300.65 | 301.36 | 296.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 297.45 | 300.09 | 298.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:00:00 | 297.45 | 300.09 | 298.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 300.10 | 300.10 | 298.35 | EMA400 retest candle locked (from upside) |

### Cycle 212 — SELL (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 11:15:00 | 294.15 | 296.88 | 297.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 09:15:00 | 290.20 | 293.75 | 295.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-07 10:15:00 | 284.50 | 284.24 | 287.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-07 10:30:00 | 285.10 | 284.24 | 287.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-22 15:15:00 | 165.00 | 2023-05-23 09:15:00 | 166.50 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2023-05-23 11:45:00 | 164.95 | 2023-05-24 14:15:00 | 166.40 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2023-06-05 11:00:00 | 154.55 | 2023-06-08 09:15:00 | 156.30 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2023-06-06 09:45:00 | 154.50 | 2023-06-08 09:15:00 | 156.30 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2023-06-26 14:45:00 | 156.95 | 2023-06-27 13:15:00 | 158.25 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2023-06-30 11:30:00 | 160.00 | 2023-07-10 14:15:00 | 162.75 | STOP_HIT | 1.00 | 1.72% |
| BUY | retest1 | 2023-07-25 09:15:00 | 172.20 | 2023-07-27 10:15:00 | 172.00 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2023-07-31 09:15:00 | 172.45 | 2023-08-03 11:15:00 | 173.05 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2023-08-07 10:00:00 | 172.45 | 2023-08-09 10:15:00 | 174.05 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2023-08-07 15:00:00 | 172.75 | 2023-08-09 10:15:00 | 174.05 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2023-08-08 09:45:00 | 172.80 | 2023-08-09 10:15:00 | 174.05 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2023-08-08 11:45:00 | 172.80 | 2023-08-09 10:15:00 | 174.05 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2023-08-16 10:30:00 | 178.25 | 2023-08-17 15:15:00 | 177.45 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2023-08-16 13:45:00 | 178.20 | 2023-08-17 15:15:00 | 177.45 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2023-08-17 09:30:00 | 178.25 | 2023-08-17 15:15:00 | 177.45 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2023-08-17 10:45:00 | 178.20 | 2023-08-17 15:15:00 | 177.45 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2023-08-17 14:30:00 | 178.25 | 2023-08-17 15:15:00 | 177.45 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2023-08-24 11:15:00 | 174.80 | 2023-08-28 11:15:00 | 175.50 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2023-09-07 10:30:00 | 183.65 | 2023-09-07 11:15:00 | 182.10 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2023-09-20 09:15:00 | 188.90 | 2023-09-21 14:15:00 | 186.70 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2023-09-20 14:30:00 | 188.00 | 2023-09-21 14:15:00 | 186.70 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2023-09-22 13:15:00 | 186.00 | 2023-09-26 13:15:00 | 187.50 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2023-09-25 09:45:00 | 185.90 | 2023-09-26 13:15:00 | 187.50 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2023-09-27 11:30:00 | 187.40 | 2023-10-03 09:15:00 | 185.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2023-09-27 12:45:00 | 187.35 | 2023-10-03 09:15:00 | 185.00 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2023-09-27 15:00:00 | 187.25 | 2023-10-03 09:15:00 | 185.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2023-09-28 09:15:00 | 187.50 | 2023-10-03 09:15:00 | 185.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2023-09-29 09:15:00 | 190.45 | 2023-10-03 09:15:00 | 185.00 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2023-09-29 11:30:00 | 189.60 | 2023-10-03 09:15:00 | 185.00 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2023-09-29 12:00:00 | 189.65 | 2023-10-03 09:15:00 | 185.00 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2023-10-09 15:00:00 | 181.55 | 2023-10-10 10:15:00 | 183.65 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2023-10-13 15:15:00 | 185.15 | 2023-10-19 15:15:00 | 185.90 | STOP_HIT | 1.00 | 0.41% |
| BUY | retest2 | 2023-11-01 09:15:00 | 187.50 | 2023-11-17 12:15:00 | 195.95 | STOP_HIT | 1.00 | 4.51% |
| BUY | retest2 | 2023-11-02 09:15:00 | 186.60 | 2023-11-17 12:15:00 | 195.95 | STOP_HIT | 1.00 | 5.01% |
| BUY | retest2 | 2023-11-02 14:45:00 | 186.40 | 2023-11-17 12:15:00 | 195.95 | STOP_HIT | 1.00 | 5.12% |
| BUY | retest2 | 2023-11-03 09:30:00 | 186.85 | 2023-11-17 12:15:00 | 195.95 | STOP_HIT | 1.00 | 4.87% |
| BUY | retest2 | 2023-11-08 11:15:00 | 195.40 | 2023-11-17 12:15:00 | 195.95 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2023-11-08 12:00:00 | 195.25 | 2023-11-17 12:15:00 | 195.95 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2023-11-08 12:45:00 | 195.05 | 2023-11-17 12:15:00 | 195.95 | STOP_HIT | 1.00 | 0.46% |
| BUY | retest2 | 2023-11-10 10:30:00 | 195.40 | 2023-11-17 12:15:00 | 195.95 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2023-11-15 09:15:00 | 198.85 | 2023-11-17 12:15:00 | 195.95 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2023-12-05 14:30:00 | 201.35 | 2023-12-07 09:15:00 | 197.50 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2023-12-06 15:00:00 | 201.95 | 2023-12-07 09:15:00 | 197.50 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2023-12-18 14:15:00 | 198.70 | 2023-12-29 13:15:00 | 204.95 | STOP_HIT | 1.00 | 3.15% |
| BUY | retest2 | 2024-01-31 09:15:00 | 252.00 | 2024-01-31 10:15:00 | 247.65 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-01-31 11:30:00 | 250.60 | 2024-01-31 12:15:00 | 248.60 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-01-31 12:15:00 | 250.80 | 2024-01-31 12:15:00 | 248.60 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-01-31 14:15:00 | 250.55 | 2024-02-01 10:15:00 | 248.30 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-01-31 15:15:00 | 253.15 | 2024-02-01 10:15:00 | 248.30 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2024-02-19 13:30:00 | 278.05 | 2024-02-22 09:15:00 | 271.30 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2024-02-20 09:15:00 | 280.25 | 2024-02-22 09:15:00 | 271.30 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2024-02-21 09:30:00 | 279.00 | 2024-02-22 09:15:00 | 271.30 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2024-02-21 11:00:00 | 278.10 | 2024-02-22 09:15:00 | 271.30 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2024-02-28 10:30:00 | 267.70 | 2024-03-01 09:15:00 | 272.15 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-03-06 13:45:00 | 278.75 | 2024-03-11 11:15:00 | 274.50 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-03-06 14:15:00 | 279.75 | 2024-03-11 11:15:00 | 274.50 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2024-03-07 12:45:00 | 279.45 | 2024-03-11 11:15:00 | 274.50 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-03-07 14:45:00 | 279.35 | 2024-03-11 11:15:00 | 274.50 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-03-15 10:15:00 | 254.60 | 2024-03-18 15:15:00 | 262.10 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2024-03-22 14:00:00 | 263.65 | 2024-04-04 12:15:00 | 268.00 | STOP_HIT | 1.00 | 1.65% |
| BUY | retest2 | 2024-03-22 15:15:00 | 263.30 | 2024-04-04 12:15:00 | 268.00 | STOP_HIT | 1.00 | 1.79% |
| BUY | retest2 | 2024-03-26 09:45:00 | 264.80 | 2024-04-04 12:15:00 | 268.00 | STOP_HIT | 1.00 | 1.21% |
| BUY | retest2 | 2024-03-27 15:15:00 | 265.30 | 2024-04-04 12:15:00 | 268.00 | STOP_HIT | 1.00 | 1.02% |
| BUY | retest2 | 2024-03-28 12:30:00 | 266.65 | 2024-04-04 12:15:00 | 268.00 | STOP_HIT | 1.00 | 0.51% |
| SELL | retest2 | 2024-05-08 14:15:00 | 276.70 | 2024-05-13 10:15:00 | 262.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-08 14:15:00 | 276.70 | 2024-05-13 13:15:00 | 268.00 | STOP_HIT | 0.50 | 3.14% |
| SELL | retest2 | 2024-05-09 09:15:00 | 274.80 | 2024-05-14 15:15:00 | 274.50 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2024-05-30 11:30:00 | 270.80 | 2024-06-03 09:15:00 | 279.65 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2024-05-30 12:00:00 | 270.30 | 2024-06-03 09:15:00 | 279.65 | STOP_HIT | 1.00 | -3.46% |
| SELL | retest2 | 2024-06-20 13:30:00 | 271.45 | 2024-06-21 09:15:00 | 273.95 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2024-06-21 09:15:00 | 271.50 | 2024-06-21 09:15:00 | 273.95 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-06-21 12:30:00 | 271.45 | 2024-06-28 09:15:00 | 271.75 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2024-06-21 14:15:00 | 270.90 | 2024-06-28 09:15:00 | 271.75 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2024-06-27 09:15:00 | 266.25 | 2024-06-28 09:15:00 | 271.75 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2024-06-27 10:30:00 | 267.10 | 2024-06-28 09:15:00 | 271.75 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-06-27 15:15:00 | 267.25 | 2024-06-28 09:15:00 | 271.75 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2024-07-02 14:15:00 | 275.15 | 2024-07-08 14:15:00 | 302.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-02 15:15:00 | 275.70 | 2024-07-08 14:15:00 | 302.28 | TARGET_HIT | 1.00 | 9.64% |
| BUY | retest2 | 2024-07-03 15:00:00 | 274.80 | 2024-07-08 14:15:00 | 302.06 | TARGET_HIT | 1.00 | 9.92% |
| BUY | retest2 | 2024-07-04 11:15:00 | 274.60 | 2024-07-11 10:15:00 | 303.27 | TARGET_HIT | 1.00 | 10.44% |
| BUY | retest2 | 2024-07-04 12:15:00 | 275.65 | 2024-07-11 10:15:00 | 303.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-04 13:15:00 | 276.10 | 2024-07-11 10:15:00 | 303.71 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-04 15:15:00 | 277.80 | 2024-07-11 11:15:00 | 305.58 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-01 09:15:00 | 339.25 | 2024-08-02 14:15:00 | 330.40 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2024-08-01 14:30:00 | 341.10 | 2024-08-02 14:15:00 | 330.40 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest2 | 2024-08-28 14:30:00 | 328.60 | 2024-08-29 09:15:00 | 326.55 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-09-16 11:15:00 | 292.10 | 2024-09-17 09:15:00 | 293.50 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2024-09-16 11:45:00 | 292.15 | 2024-09-17 09:15:00 | 293.50 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2024-09-16 13:00:00 | 292.20 | 2024-09-17 09:15:00 | 293.50 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2024-09-16 14:15:00 | 291.80 | 2024-09-17 09:15:00 | 293.50 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-10-03 11:30:00 | 293.00 | 2024-10-04 10:15:00 | 298.50 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest1 | 2024-10-16 10:45:00 | 282.15 | 2024-10-16 13:15:00 | 285.70 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-10-17 09:30:00 | 285.00 | 2024-10-22 14:15:00 | 270.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 12:00:00 | 284.70 | 2024-10-22 14:15:00 | 270.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 09:30:00 | 285.00 | 2024-10-23 12:15:00 | 272.25 | STOP_HIT | 0.50 | 4.47% |
| SELL | retest2 | 2024-10-17 12:00:00 | 284.70 | 2024-10-23 12:15:00 | 272.25 | STOP_HIT | 0.50 | 4.37% |
| SELL | retest2 | 2024-11-19 14:00:00 | 249.00 | 2024-11-25 09:15:00 | 257.60 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2024-11-19 14:45:00 | 248.05 | 2024-11-25 09:15:00 | 257.60 | STOP_HIT | 1.00 | -3.85% |
| BUY | retest2 | 2024-11-27 15:15:00 | 254.80 | 2024-11-28 14:15:00 | 251.95 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-12-02 10:15:00 | 256.20 | 2024-12-09 09:15:00 | 257.80 | STOP_HIT | 1.00 | 0.62% |
| BUY | retest2 | 2024-12-02 12:30:00 | 256.45 | 2024-12-09 09:15:00 | 257.80 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest2 | 2024-12-02 14:00:00 | 256.35 | 2024-12-09 09:15:00 | 257.80 | STOP_HIT | 1.00 | 0.57% |
| SELL | retest2 | 2024-12-11 14:15:00 | 256.05 | 2024-12-19 09:15:00 | 243.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 10:00:00 | 256.05 | 2024-12-19 09:15:00 | 243.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 12:45:00 | 255.35 | 2024-12-19 09:15:00 | 242.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-13 14:30:00 | 255.80 | 2024-12-19 09:15:00 | 243.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 10:15:00 | 252.55 | 2024-12-19 09:15:00 | 239.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-11 14:15:00 | 256.05 | 2024-12-20 10:15:00 | 243.75 | STOP_HIT | 0.50 | 4.80% |
| SELL | retest2 | 2024-12-12 10:00:00 | 256.05 | 2024-12-20 10:15:00 | 243.75 | STOP_HIT | 0.50 | 4.80% |
| SELL | retest2 | 2024-12-12 12:45:00 | 255.35 | 2024-12-20 10:15:00 | 243.75 | STOP_HIT | 0.50 | 4.54% |
| SELL | retest2 | 2024-12-13 14:30:00 | 255.80 | 2024-12-20 10:15:00 | 243.75 | STOP_HIT | 0.50 | 4.71% |
| SELL | retest2 | 2024-12-16 10:15:00 | 252.55 | 2024-12-20 10:15:00 | 243.75 | STOP_HIT | 0.50 | 3.48% |
| BUY | retest2 | 2025-01-10 11:45:00 | 266.25 | 2025-01-10 15:15:00 | 263.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-01-15 09:15:00 | 258.58 | 2025-01-15 09:15:00 | 262.20 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-01-21 11:45:00 | 267.89 | 2025-01-22 09:15:00 | 263.18 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-01-21 12:30:00 | 267.28 | 2025-01-22 09:15:00 | 263.18 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-01-21 14:00:00 | 267.30 | 2025-01-22 09:15:00 | 263.18 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-02-01 15:00:00 | 257.90 | 2025-02-03 09:15:00 | 247.35 | STOP_HIT | 1.00 | -4.09% |
| SELL | retest2 | 2025-02-18 09:15:00 | 230.45 | 2025-02-18 14:15:00 | 236.70 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest1 | 2025-02-28 09:15:00 | 227.95 | 2025-03-04 09:15:00 | 216.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-28 09:15:00 | 227.95 | 2025-03-04 10:15:00 | 225.32 | STOP_HIT | 0.50 | 1.15% |
| BUY | retest2 | 2025-03-06 12:30:00 | 230.56 | 2025-03-10 12:15:00 | 226.95 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-03-12 10:45:00 | 223.30 | 2025-03-17 09:15:00 | 228.02 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-03-12 13:45:00 | 224.07 | 2025-03-17 09:15:00 | 228.02 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-03-12 15:15:00 | 224.20 | 2025-03-17 09:15:00 | 228.02 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-03-25 13:30:00 | 243.73 | 2025-03-26 12:15:00 | 240.61 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-03-26 09:15:00 | 243.92 | 2025-03-26 12:15:00 | 240.61 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-03-26 10:15:00 | 243.66 | 2025-03-26 12:15:00 | 240.61 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-04-09 09:15:00 | 221.00 | 2025-04-11 09:15:00 | 228.12 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2025-05-06 10:15:00 | 239.33 | 2025-05-12 10:15:00 | 241.58 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-05-06 11:00:00 | 238.89 | 2025-05-12 10:15:00 | 241.58 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-05-14 09:15:00 | 243.60 | 2025-05-22 10:15:00 | 245.75 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2025-05-29 10:30:00 | 242.07 | 2025-05-29 15:15:00 | 244.45 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-05-29 12:00:00 | 242.22 | 2025-05-29 15:15:00 | 244.45 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-05-29 12:30:00 | 242.18 | 2025-05-29 15:15:00 | 244.45 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-05-29 13:45:00 | 242.20 | 2025-05-29 15:15:00 | 244.45 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-05-30 10:15:00 | 241.40 | 2025-06-06 11:15:00 | 238.73 | STOP_HIT | 1.00 | 1.11% |
| SELL | retest2 | 2025-06-27 10:45:00 | 243.16 | 2025-07-03 10:15:00 | 245.94 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-06-27 14:00:00 | 243.58 | 2025-07-03 10:15:00 | 245.94 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-06-27 14:30:00 | 242.92 | 2025-07-03 10:15:00 | 245.94 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-06-30 11:00:00 | 243.56 | 2025-07-03 10:15:00 | 245.94 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-07-01 10:15:00 | 242.50 | 2025-07-03 10:15:00 | 245.94 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-07-01 14:30:00 | 242.72 | 2025-07-03 10:15:00 | 245.94 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-07-01 15:15:00 | 243.19 | 2025-07-03 10:15:00 | 245.94 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-07-02 11:30:00 | 242.00 | 2025-07-03 10:15:00 | 245.94 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-07-22 12:15:00 | 245.93 | 2025-07-24 11:15:00 | 244.29 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-07-23 15:00:00 | 245.73 | 2025-07-24 11:15:00 | 244.29 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-07-24 10:45:00 | 245.80 | 2025-07-24 11:15:00 | 244.29 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-08-11 09:30:00 | 232.52 | 2025-08-12 09:15:00 | 234.36 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-08-11 12:15:00 | 232.80 | 2025-08-12 09:15:00 | 234.36 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-08-14 12:45:00 | 237.22 | 2025-08-22 11:15:00 | 236.75 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2025-08-14 13:30:00 | 237.36 | 2025-08-22 11:15:00 | 236.75 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-08-14 14:00:00 | 237.46 | 2025-08-22 11:15:00 | 236.75 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-08-18 12:30:00 | 237.32 | 2025-08-22 11:15:00 | 236.75 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-08-20 14:30:00 | 238.32 | 2025-08-22 11:15:00 | 236.75 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-08-21 09:15:00 | 238.50 | 2025-08-22 11:15:00 | 236.75 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-08-26 09:15:00 | 235.18 | 2025-09-01 11:15:00 | 236.35 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-09-12 14:15:00 | 233.82 | 2025-09-15 09:15:00 | 232.45 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-09-15 09:15:00 | 234.04 | 2025-09-15 09:15:00 | 232.45 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-09-18 15:00:00 | 235.62 | 2025-10-08 12:15:00 | 242.45 | STOP_HIT | 1.00 | 2.90% |
| BUY | retest2 | 2025-09-19 10:15:00 | 235.58 | 2025-10-08 12:15:00 | 242.45 | STOP_HIT | 1.00 | 2.92% |
| SELL | retest2 | 2025-10-09 13:15:00 | 243.10 | 2025-10-10 09:15:00 | 246.35 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-11-10 12:15:00 | 251.15 | 2025-11-12 11:15:00 | 253.50 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-11-10 14:45:00 | 251.00 | 2025-11-12 11:15:00 | 253.50 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-11-11 09:30:00 | 250.30 | 2025-11-12 11:15:00 | 253.50 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-12-02 14:15:00 | 243.12 | 2025-12-15 09:15:00 | 230.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 09:15:00 | 242.25 | 2025-12-15 09:15:00 | 230.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 14:15:00 | 243.12 | 2025-12-15 14:15:00 | 235.33 | STOP_HIT | 0.50 | 3.20% |
| SELL | retest2 | 2025-12-03 09:15:00 | 242.25 | 2025-12-15 14:15:00 | 235.33 | STOP_HIT | 0.50 | 2.86% |
| SELL | retest2 | 2026-01-12 09:30:00 | 231.96 | 2026-01-13 09:15:00 | 240.05 | STOP_HIT | 1.00 | -3.49% |
| BUY | retest2 | 2026-01-16 10:15:00 | 247.48 | 2026-01-19 14:15:00 | 243.10 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2026-01-16 11:15:00 | 247.00 | 2026-01-19 14:15:00 | 243.10 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2026-01-16 14:45:00 | 247.44 | 2026-01-19 14:15:00 | 243.10 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2026-02-09 13:00:00 | 268.80 | 2026-02-13 09:15:00 | 266.90 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2026-02-10 09:15:00 | 269.70 | 2026-02-13 12:15:00 | 269.20 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2026-02-11 10:15:00 | 268.60 | 2026-02-13 12:15:00 | 269.20 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2026-02-11 13:00:00 | 268.65 | 2026-02-13 12:15:00 | 269.20 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2026-02-11 14:15:00 | 271.55 | 2026-02-13 12:15:00 | 269.20 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2026-02-23 14:15:00 | 274.45 | 2026-03-04 15:15:00 | 275.70 | STOP_HIT | 1.00 | 0.46% |
| BUY | retest2 | 2026-02-23 14:45:00 | 274.80 | 2026-03-04 15:15:00 | 275.70 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2026-02-24 10:30:00 | 274.10 | 2026-03-04 15:15:00 | 275.70 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2026-02-24 13:30:00 | 274.90 | 2026-03-04 15:15:00 | 275.70 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2026-02-26 15:00:00 | 280.05 | 2026-03-04 15:15:00 | 275.70 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2026-02-27 10:30:00 | 279.20 | 2026-03-04 15:15:00 | 275.70 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2026-03-04 13:00:00 | 280.50 | 2026-03-04 15:15:00 | 275.70 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2026-03-04 14:00:00 | 279.00 | 2026-03-04 15:15:00 | 275.70 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2026-03-09 09:30:00 | 277.00 | 2026-03-16 09:15:00 | 263.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-09 09:30:00 | 277.00 | 2026-03-17 10:15:00 | 261.80 | STOP_HIT | 0.50 | 5.49% |
