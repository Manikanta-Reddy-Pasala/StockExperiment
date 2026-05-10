# Paradeep Phosphates Ltd. (PARADEEP)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 125.09
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 78 |
| ALERT1 | 48 |
| ALERT2 | 48 |
| ALERT2_SKIP | 22 |
| ALERT3 | 111 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 44 |
| PARTIAL | 8 |
| TARGET_HIT | 4 |
| STOP_HIT | 42 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 54 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 23 / 31
- **Target hits / Stop hits / Partials:** 4 / 42 / 8
- **Avg / median % per leg:** 0.83% / -1.11%
- **Sum % (uncompounded):** 44.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 7 | 28.0% | 4 | 21 | 0 | 0.32% | 8.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 25 | 7 | 28.0% | 4 | 21 | 0 | 0.32% | 8.0% |
| SELL (all) | 29 | 16 | 55.2% | 0 | 21 | 8 | 1.27% | 36.9% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.87% | -7.7% |
| SELL @ 3rd Alert (retest2) | 27 | 16 | 59.3% | 0 | 19 | 8 | 1.65% | 44.6% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.87% | -7.7% |
| retest2 (combined) | 52 | 23 | 44.2% | 4 | 40 | 8 | 1.01% | 52.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 147.85 | 142.04 | 141.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 150.84 | 143.80 | 142.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 147.89 | 148.14 | 146.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 14:00:00 | 147.89 | 148.14 | 146.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 11:15:00 | 145.66 | 153.44 | 152.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 12:00:00 | 145.66 | 153.44 | 152.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 12:15:00 | 144.88 | 151.73 | 152.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-16 15:15:00 | 144.71 | 148.48 | 150.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-19 11:15:00 | 148.85 | 148.00 | 149.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-19 12:00:00 | 148.85 | 148.00 | 149.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 12:15:00 | 149.63 | 148.32 | 149.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 13:00:00 | 149.63 | 148.32 | 149.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 149.60 | 148.58 | 149.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:15:00 | 150.94 | 148.58 | 149.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 151.60 | 149.18 | 149.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 15:00:00 | 151.60 | 149.18 | 149.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 151.00 | 149.55 | 149.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 149.90 | 149.55 | 149.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-05-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 09:15:00 | 154.60 | 150.56 | 150.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 09:15:00 | 163.05 | 154.03 | 152.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-21 14:15:00 | 155.95 | 155.97 | 154.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-21 14:45:00 | 155.66 | 155.97 | 154.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 15:15:00 | 155.90 | 155.96 | 154.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:30:00 | 153.34 | 155.85 | 154.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 154.45 | 155.51 | 154.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 12:00:00 | 154.45 | 155.51 | 154.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 154.19 | 155.25 | 154.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 12:45:00 | 154.18 | 155.25 | 154.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 154.99 | 155.20 | 154.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 14:45:00 | 155.99 | 155.54 | 154.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 12:15:00 | 155.45 | 155.95 | 155.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-29 09:15:00 | 171.59 | 166.41 | 163.26 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-05-29 09:15:00 | 171.00 | 166.41 | 163.26 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 15:15:00 | 173.00 | 173.98 | 174.03 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 174.82 | 174.15 | 174.10 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 12:15:00 | 172.50 | 174.47 | 174.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 14:15:00 | 171.82 | 173.77 | 174.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 13:15:00 | 172.60 | 172.50 | 173.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-09 13:45:00 | 173.01 | 172.50 | 173.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 172.58 | 172.51 | 173.21 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 10:15:00 | 175.71 | 173.62 | 173.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 12:15:00 | 176.85 | 174.56 | 174.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 12:15:00 | 176.23 | 176.50 | 175.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 13:00:00 | 176.23 | 176.50 | 175.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 174.46 | 176.09 | 175.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 174.50 | 176.09 | 175.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 175.66 | 176.01 | 175.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:30:00 | 177.70 | 176.16 | 175.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 14:15:00 | 173.26 | 175.70 | 175.68 | SL hit (close<static) qty=1.00 sl=174.17 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 173.55 | 175.27 | 175.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 12:15:00 | 172.50 | 174.09 | 174.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 171.31 | 171.25 | 172.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 171.31 | 171.25 | 172.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 173.23 | 171.68 | 172.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:45:00 | 173.94 | 171.68 | 172.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 171.70 | 171.69 | 172.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:15:00 | 171.24 | 171.69 | 172.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 09:15:00 | 162.68 | 165.33 | 166.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 10:15:00 | 165.80 | 165.42 | 166.80 | SL hit (close>ema200) qty=0.50 sl=165.42 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 13:15:00 | 165.99 | 165.49 | 165.44 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 09:15:00 | 162.39 | 164.91 | 165.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 10:15:00 | 161.29 | 164.18 | 164.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 09:15:00 | 161.50 | 159.57 | 161.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 09:15:00 | 161.50 | 159.57 | 161.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 161.50 | 159.57 | 161.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:45:00 | 161.25 | 159.57 | 161.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 161.02 | 159.86 | 161.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 13:00:00 | 160.00 | 160.34 | 161.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 164.64 | 160.99 | 161.11 | SL hit (close>static) qty=1.00 sl=161.79 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 10:15:00 | 162.80 | 161.35 | 161.26 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 12:15:00 | 160.07 | 161.12 | 161.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 13:15:00 | 158.74 | 160.64 | 160.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 10:15:00 | 159.67 | 159.62 | 160.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-02 10:45:00 | 159.75 | 159.62 | 160.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 160.49 | 159.43 | 159.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:45:00 | 160.80 | 159.43 | 159.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 160.31 | 159.61 | 159.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:45:00 | 160.36 | 159.61 | 159.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 13:15:00 | 161.05 | 160.16 | 160.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 14:15:00 | 162.70 | 160.67 | 160.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 161.14 | 161.27 | 160.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 12:00:00 | 161.14 | 161.27 | 160.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 160.59 | 161.12 | 160.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 160.59 | 161.12 | 160.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 159.90 | 160.88 | 160.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 09:15:00 | 161.61 | 160.88 | 160.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 10:45:00 | 160.70 | 160.84 | 160.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 11:15:00 | 158.92 | 160.45 | 160.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-07 11:15:00 | 158.92 | 160.45 | 160.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 11:15:00 | 158.92 | 160.45 | 160.59 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 15:15:00 | 161.05 | 159.97 | 159.97 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 09:15:00 | 159.32 | 159.84 | 159.91 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 13:15:00 | 160.10 | 159.94 | 159.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 14:15:00 | 160.90 | 160.13 | 160.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 10:15:00 | 160.61 | 161.48 | 161.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 10:15:00 | 160.61 | 161.48 | 161.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 160.61 | 161.48 | 161.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 160.61 | 161.48 | 161.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 161.20 | 161.42 | 161.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 12:30:00 | 161.70 | 161.50 | 161.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-15 10:15:00 | 177.87 | 170.41 | 166.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 15:15:00 | 190.45 | 191.00 | 191.06 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 09:15:00 | 200.10 | 192.82 | 191.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 09:15:00 | 217.74 | 201.10 | 196.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 09:15:00 | 210.15 | 213.32 | 206.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-30 09:45:00 | 210.72 | 213.32 | 206.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 206.80 | 211.27 | 206.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:30:00 | 207.39 | 211.27 | 206.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 208.00 | 210.61 | 207.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:00:00 | 208.00 | 210.61 | 207.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 209.20 | 209.89 | 207.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 15:15:00 | 208.00 | 209.89 | 207.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 208.00 | 209.52 | 207.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 211.49 | 209.52 | 207.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 217.68 | 211.15 | 208.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 12:30:00 | 217.99 | 216.35 | 213.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 14:15:00 | 218.31 | 216.60 | 213.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 220.61 | 216.97 | 214.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 12:15:00 | 225.87 | 226.35 | 226.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 12:15:00 | 225.87 | 226.35 | 226.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 12:15:00 | 225.87 | 226.35 | 226.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 12:15:00 | 225.87 | 226.35 | 226.35 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 227.20 | 226.46 | 226.40 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 09:15:00 | 219.89 | 225.54 | 226.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 13:15:00 | 216.63 | 221.03 | 223.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 204.59 | 203.93 | 208.54 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-18 12:00:00 | 199.20 | 202.98 | 207.31 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 207.33 | 202.37 | 205.15 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-19 09:15:00 | 207.33 | 202.37 | 205.15 | SL hit (close>ema400) qty=1.00 sl=205.15 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-08-19 09:45:00 | 205.51 | 202.37 | 205.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 207.99 | 203.49 | 205.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:00:00 | 207.99 | 203.49 | 205.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 206.70 | 204.69 | 205.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:30:00 | 207.59 | 204.69 | 205.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 09:15:00 | 208.84 | 206.51 | 206.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 13:15:00 | 212.10 | 208.93 | 207.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 13:15:00 | 219.79 | 219.99 | 216.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 13:30:00 | 219.86 | 219.99 | 216.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 224.18 | 224.13 | 221.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 10:45:00 | 225.70 | 224.51 | 221.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 09:30:00 | 227.27 | 226.03 | 223.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 10:30:00 | 226.80 | 226.06 | 224.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 09:15:00 | 218.06 | 223.41 | 223.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 09:15:00 | 218.06 | 223.41 | 223.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 09:15:00 | 218.06 | 223.41 | 223.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 09:15:00 | 218.06 | 223.41 | 223.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 12:15:00 | 217.52 | 221.09 | 222.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 225.43 | 220.64 | 221.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 225.43 | 220.64 | 221.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 225.43 | 220.64 | 221.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 225.43 | 220.64 | 221.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 224.75 | 221.46 | 221.95 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 225.80 | 222.33 | 222.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 12:15:00 | 226.60 | 223.19 | 222.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 09:15:00 | 224.18 | 225.25 | 224.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 224.18 | 225.25 | 224.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 224.18 | 225.25 | 224.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:00:00 | 224.18 | 225.25 | 224.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 225.66 | 225.33 | 224.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 11:15:00 | 223.83 | 225.33 | 224.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 11:15:00 | 218.33 | 223.93 | 223.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 11:45:00 | 217.65 | 223.93 | 223.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 12:15:00 | 214.37 | 222.02 | 222.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 15:15:00 | 211.10 | 213.21 | 216.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 13:15:00 | 185.75 | 185.61 | 191.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 14:00:00 | 185.75 | 185.61 | 191.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 185.29 | 183.39 | 186.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 15:00:00 | 181.07 | 183.15 | 185.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 11:15:00 | 172.02 | 178.64 | 182.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-16 11:15:00 | 169.34 | 168.59 | 171.93 | SL hit (close>ema200) qty=0.50 sl=168.59 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 13:15:00 | 173.14 | 171.80 | 171.75 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 10:15:00 | 169.55 | 171.42 | 171.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 12:15:00 | 168.95 | 170.58 | 171.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 11:15:00 | 172.90 | 170.34 | 170.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 11:15:00 | 172.90 | 170.34 | 170.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 172.90 | 170.34 | 170.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:00:00 | 172.90 | 170.34 | 170.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 12:15:00 | 174.90 | 171.25 | 171.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 14:15:00 | 179.43 | 173.52 | 172.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 09:15:00 | 176.62 | 177.78 | 176.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 09:15:00 | 176.62 | 177.78 | 176.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 176.62 | 177.78 | 176.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:30:00 | 176.69 | 177.78 | 176.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 174.67 | 177.16 | 175.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 174.67 | 177.16 | 175.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 175.36 | 176.80 | 175.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 12:45:00 | 175.76 | 176.62 | 175.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 13:15:00 | 173.90 | 176.08 | 175.66 | SL hit (close<static) qty=1.00 sl=174.25 alert=retest2 |

### Cycle 30 — SELL (started 2025-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 15:15:00 | 173.99 | 175.27 | 175.34 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 10:15:00 | 177.51 | 175.72 | 175.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-24 11:15:00 | 180.23 | 176.62 | 175.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 183.27 | 189.34 | 185.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 183.27 | 189.34 | 185.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 183.27 | 189.34 | 185.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:00:00 | 183.27 | 189.34 | 185.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 184.29 | 188.33 | 185.70 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 15:15:00 | 182.00 | 184.62 | 184.66 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 09:15:00 | 193.30 | 186.35 | 185.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 09:15:00 | 197.60 | 193.18 | 189.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-30 11:15:00 | 192.70 | 193.25 | 190.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-30 12:00:00 | 192.70 | 193.25 | 190.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 190.60 | 194.55 | 193.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:00:00 | 190.60 | 194.55 | 193.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 193.29 | 194.29 | 193.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 14:45:00 | 193.61 | 193.54 | 193.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 15:15:00 | 193.65 | 193.54 | 193.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 12:15:00 | 189.65 | 194.79 | 194.65 | SL hit (close<static) qty=1.00 sl=190.52 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-07 12:15:00 | 189.65 | 194.79 | 194.65 | SL hit (close<static) qty=1.00 sl=190.52 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 13:15:00 | 190.16 | 193.86 | 194.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 14:15:00 | 188.80 | 192.85 | 193.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 11:15:00 | 178.51 | 178.37 | 181.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-13 11:30:00 | 179.62 | 178.37 | 181.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 175.75 | 174.55 | 176.22 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 178.26 | 177.11 | 177.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 14:15:00 | 179.31 | 177.96 | 177.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 176.31 | 177.78 | 177.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 176.31 | 177.78 | 177.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 176.31 | 177.78 | 177.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:00:00 | 176.31 | 177.78 | 177.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 176.33 | 177.49 | 177.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 12:15:00 | 177.39 | 177.41 | 177.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 12:15:00 | 175.80 | 177.08 | 177.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 12:15:00 | 175.80 | 177.08 | 177.21 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 14:15:00 | 177.95 | 177.29 | 177.28 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 11:15:00 | 176.85 | 177.23 | 177.28 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 15:15:00 | 178.15 | 177.31 | 177.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 179.12 | 177.67 | 177.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 09:15:00 | 176.39 | 177.49 | 177.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 176.39 | 177.49 | 177.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 176.39 | 177.49 | 177.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 176.39 | 177.49 | 177.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 10:15:00 | 176.14 | 177.22 | 177.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 11:15:00 | 175.20 | 176.81 | 177.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 10:15:00 | 172.99 | 172.94 | 174.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 11:00:00 | 172.99 | 172.94 | 174.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 172.65 | 172.88 | 174.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 12:45:00 | 171.80 | 172.75 | 173.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:45:00 | 171.43 | 170.32 | 171.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 170.70 | 170.43 | 170.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 13:15:00 | 163.21 | 166.52 | 167.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 13:15:00 | 162.86 | 166.52 | 167.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 13:15:00 | 162.16 | 166.52 | 167.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 168.61 | 166.32 | 167.06 | SL hit (close>ema200) qty=0.50 sl=166.32 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 168.61 | 166.32 | 167.06 | SL hit (close>ema200) qty=0.50 sl=166.32 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 168.61 | 166.32 | 167.06 | SL hit (close>ema200) qty=0.50 sl=166.32 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 13:15:00 | 170.40 | 167.82 | 167.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 14:15:00 | 174.39 | 169.13 | 168.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 10:15:00 | 168.44 | 169.67 | 168.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 10:15:00 | 168.44 | 169.67 | 168.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 168.44 | 169.67 | 168.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:00:00 | 168.44 | 169.67 | 168.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 168.55 | 169.44 | 168.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:15:00 | 168.72 | 169.44 | 168.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 168.80 | 169.31 | 168.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 13:00:00 | 168.80 | 169.31 | 168.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 168.45 | 169.14 | 168.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 14:00:00 | 168.45 | 169.14 | 168.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 167.18 | 168.75 | 168.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 15:00:00 | 167.18 | 168.75 | 168.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 15:15:00 | 167.24 | 168.45 | 168.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 09:15:00 | 159.65 | 166.69 | 167.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 161.33 | 161.19 | 163.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 09:15:00 | 161.33 | 161.19 | 163.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 161.33 | 161.19 | 163.71 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 165.05 | 163.11 | 163.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 10:15:00 | 166.52 | 163.79 | 163.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 163.00 | 164.98 | 164.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 163.00 | 164.98 | 164.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 163.00 | 164.98 | 164.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:45:00 | 162.74 | 164.98 | 164.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 162.86 | 164.56 | 164.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:45:00 | 162.89 | 164.56 | 164.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 13:15:00 | 162.97 | 163.83 | 163.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 162.15 | 163.49 | 163.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 156.72 | 153.82 | 155.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 14:15:00 | 156.72 | 153.82 | 155.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 156.72 | 153.82 | 155.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 156.72 | 153.82 | 155.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 158.15 | 154.69 | 156.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:15:00 | 157.68 | 154.69 | 156.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 158.44 | 156.53 | 156.69 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 11:15:00 | 158.39 | 156.90 | 156.84 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 09:15:00 | 154.83 | 156.72 | 156.85 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 09:15:00 | 158.80 | 156.45 | 156.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 09:15:00 | 162.13 | 158.83 | 157.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 12:15:00 | 159.08 | 159.23 | 158.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 13:00:00 | 159.08 | 159.23 | 158.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 164.29 | 160.58 | 159.21 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 153.30 | 158.70 | 159.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 152.71 | 154.18 | 155.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 12:15:00 | 154.11 | 152.92 | 153.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 12:15:00 | 154.11 | 152.92 | 153.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 154.11 | 152.92 | 153.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 154.11 | 152.92 | 153.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 154.25 | 153.19 | 153.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:15:00 | 153.22 | 153.19 | 153.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 155.61 | 153.67 | 154.01 | SL hit (close>static) qty=1.00 sl=154.70 alert=retest2 |

### Cycle 49 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 157.50 | 154.63 | 154.40 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 12:15:00 | 153.74 | 154.99 | 155.08 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 15:15:00 | 156.10 | 155.28 | 155.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 156.90 | 155.60 | 155.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 11:15:00 | 155.58 | 155.68 | 155.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-12 11:45:00 | 155.76 | 155.68 | 155.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 153.97 | 155.53 | 155.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 154.10 | 155.53 | 155.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 10:15:00 | 153.80 | 155.19 | 155.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 11:15:00 | 153.26 | 154.80 | 155.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 13:15:00 | 154.50 | 154.44 | 154.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-15 14:00:00 | 154.50 | 154.44 | 154.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 154.53 | 154.46 | 154.87 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 09:15:00 | 163.16 | 156.33 | 155.65 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 155.19 | 157.06 | 157.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 15:15:00 | 155.00 | 156.13 | 156.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 12:15:00 | 153.44 | 153.25 | 154.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 13:00:00 | 153.44 | 153.25 | 154.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 155.35 | 153.77 | 154.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 155.35 | 153.77 | 154.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 155.50 | 154.12 | 154.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 159.26 | 154.12 | 154.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 161.55 | 155.61 | 155.12 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 159.01 | 159.81 | 159.83 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 09:15:00 | 161.55 | 160.08 | 159.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 09:15:00 | 168.28 | 162.97 | 161.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 12:15:00 | 164.20 | 165.21 | 164.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 12:15:00 | 164.20 | 165.21 | 164.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 164.20 | 165.21 | 164.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:45:00 | 163.85 | 165.21 | 164.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 163.65 | 164.89 | 164.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:30:00 | 163.65 | 164.89 | 164.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 164.16 | 164.75 | 164.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 15:15:00 | 165.10 | 164.75 | 164.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 12:45:00 | 165.00 | 164.73 | 164.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 15:00:00 | 164.95 | 164.80 | 164.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 165.26 | 164.82 | 164.48 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 164.29 | 164.71 | 164.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:30:00 | 164.60 | 164.71 | 164.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 164.85 | 164.74 | 164.50 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 161.73 | 164.23 | 164.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 161.73 | 164.23 | 164.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 161.73 | 164.23 | 164.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 161.73 | 164.23 | 164.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 09:15:00 | 161.73 | 164.23 | 164.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 13:15:00 | 161.11 | 162.85 | 163.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 10:15:00 | 158.40 | 157.90 | 159.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 11:00:00 | 158.40 | 157.90 | 159.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 155.83 | 157.82 | 158.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:15:00 | 155.40 | 157.82 | 158.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 14:15:00 | 147.63 | 150.27 | 153.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 146.45 | 146.03 | 148.85 | SL hit (close>ema200) qty=0.50 sl=146.03 alert=retest2 |

### Cycle 59 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 133.28 | 130.82 | 130.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 137.83 | 133.25 | 132.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 138.90 | 139.23 | 136.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 12:00:00 | 138.90 | 139.23 | 136.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 137.00 | 138.78 | 136.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 137.07 | 138.78 | 136.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 135.64 | 138.15 | 136.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 135.64 | 138.15 | 136.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 135.71 | 137.66 | 136.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:45:00 | 135.47 | 137.66 | 136.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 135.45 | 137.22 | 136.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 130.51 | 137.22 | 136.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 130.70 | 135.92 | 135.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 128.50 | 134.43 | 135.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 15:15:00 | 132.69 | 132.07 | 133.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 15:15:00 | 132.69 | 132.07 | 133.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 132.69 | 132.07 | 133.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 127.66 | 132.07 | 133.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 128.61 | 131.37 | 133.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 14:00:00 | 124.85 | 127.74 | 129.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 15:15:00 | 124.85 | 127.28 | 129.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 14:15:00 | 129.49 | 126.24 | 126.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-09 14:15:00 | 129.49 | 126.24 | 126.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 14:15:00 | 129.49 | 126.24 | 126.17 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 12:15:00 | 124.67 | 126.17 | 126.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 09:15:00 | 122.29 | 125.40 | 125.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 15:15:00 | 123.23 | 123.22 | 124.39 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 09:30:00 | 120.37 | 122.79 | 124.09 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 123.92 | 123.09 | 124.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:00:00 | 123.92 | 123.09 | 124.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 122.68 | 123.01 | 123.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:45:00 | 122.30 | 122.90 | 123.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 11:00:00 | 122.06 | 122.73 | 123.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 11:15:00 | 124.78 | 123.14 | 123.56 | SL hit (close>ema400) qty=1.00 sl=123.56 alert=retest1 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:30:00 | 122.20 | 123.11 | 123.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 10:30:00 | 121.91 | 122.97 | 123.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 123.20 | 122.39 | 122.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 123.40 | 122.39 | 122.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 124.20 | 122.75 | 122.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:00:00 | 124.20 | 122.75 | 122.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 124.25 | 123.05 | 123.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:45:00 | 124.70 | 123.05 | 123.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 124.30 | 123.30 | 123.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 124.30 | 123.30 | 123.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 124.30 | 123.30 | 123.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 124.30 | 123.30 | 123.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 124.30 | 123.30 | 123.17 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 11:15:00 | 120.87 | 122.86 | 123.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 09:15:00 | 120.20 | 121.63 | 122.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 13:15:00 | 118.41 | 118.37 | 119.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 13:30:00 | 118.30 | 118.37 | 119.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 117.13 | 118.08 | 119.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:30:00 | 115.18 | 117.05 | 118.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 11:45:00 | 115.58 | 116.52 | 117.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 15:15:00 | 116.36 | 115.58 | 116.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 11:15:00 | 117.01 | 116.60 | 116.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 11:15:00 | 117.01 | 116.60 | 116.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 11:15:00 | 117.01 | 116.60 | 116.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 11:15:00 | 117.01 | 116.60 | 116.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 12:15:00 | 118.74 | 117.03 | 116.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 117.30 | 118.10 | 117.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 117.30 | 118.10 | 117.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 117.30 | 118.10 | 117.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:30:00 | 118.75 | 118.10 | 117.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 116.78 | 117.83 | 117.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:00:00 | 116.78 | 117.83 | 117.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 118.05 | 117.88 | 117.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:45:00 | 116.60 | 117.88 | 117.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 118.31 | 119.32 | 118.47 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 114.63 | 117.65 | 117.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 14:15:00 | 114.29 | 116.55 | 117.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 109.73 | 109.36 | 111.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 109.73 | 109.36 | 111.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 104.85 | 103.71 | 105.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:15:00 | 107.04 | 103.71 | 105.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 108.12 | 104.59 | 105.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 11:00:00 | 108.12 | 104.59 | 105.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 107.34 | 105.14 | 106.06 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 108.45 | 106.84 | 106.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 110.16 | 107.50 | 106.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 111.44 | 111.63 | 109.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-12 09:30:00 | 111.00 | 111.63 | 109.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 109.66 | 111.95 | 111.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 109.66 | 111.95 | 111.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 108.57 | 111.28 | 110.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:00:00 | 108.57 | 111.28 | 110.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 110.11 | 110.76 | 110.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:15:00 | 108.77 | 110.76 | 110.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-03-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 09:15:00 | 107.35 | 110.08 | 110.35 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 10:15:00 | 113.11 | 110.20 | 110.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 113.90 | 112.57 | 111.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 113.00 | 114.12 | 112.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 113.00 | 114.12 | 112.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 113.00 | 114.12 | 112.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:30:00 | 112.29 | 114.12 | 112.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 112.15 | 113.73 | 112.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 112.15 | 113.73 | 112.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 113.06 | 113.59 | 112.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 12:30:00 | 113.25 | 113.67 | 112.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 14:30:00 | 113.60 | 113.46 | 112.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 15:15:00 | 114.20 | 113.46 | 112.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 113.18 | 113.64 | 113.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 113.85 | 113.68 | 113.30 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-20 15:15:00 | 110.50 | 112.72 | 112.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-20 15:15:00 | 110.50 | 112.72 | 112.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-20 15:15:00 | 110.50 | 112.72 | 112.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-20 15:15:00 | 110.50 | 112.72 | 112.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 15:15:00 | 110.50 | 112.72 | 112.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 108.35 | 111.25 | 112.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 108.83 | 108.26 | 109.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 108.83 | 108.26 | 109.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 108.83 | 108.26 | 109.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:30:00 | 109.79 | 108.26 | 109.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 111.25 | 109.01 | 109.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 110.91 | 109.01 | 109.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 111.27 | 109.46 | 110.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:15:00 | 111.10 | 109.46 | 110.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 111.00 | 109.87 | 110.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 110.75 | 109.87 | 110.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 111.16 | 110.13 | 110.21 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 111.40 | 110.38 | 110.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 14:15:00 | 112.08 | 110.95 | 110.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 110.07 | 110.98 | 110.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 110.07 | 110.98 | 110.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 110.07 | 110.98 | 110.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 110.07 | 110.98 | 110.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 108.88 | 110.56 | 110.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:30:00 | 108.95 | 110.56 | 110.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 109.27 | 110.30 | 110.43 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 14:15:00 | 111.61 | 110.47 | 110.47 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 108.35 | 110.20 | 110.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 107.35 | 109.18 | 109.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 110.79 | 109.18 | 109.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 110.79 | 109.18 | 109.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 110.79 | 109.18 | 109.63 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 115.75 | 110.78 | 110.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 13:15:00 | 116.77 | 112.75 | 111.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 110.65 | 113.20 | 111.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 110.65 | 113.20 | 111.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 110.65 | 113.20 | 111.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 110.65 | 113.20 | 111.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 110.66 | 112.69 | 111.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:15:00 | 111.10 | 112.69 | 111.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-08 11:15:00 | 122.21 | 117.92 | 116.07 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 15:15:00 | 134.30 | 134.82 | 134.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 09:15:00 | 132.29 | 134.31 | 134.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 12:15:00 | 133.75 | 133.45 | 134.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-29 13:00:00 | 133.75 | 133.45 | 134.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 128.46 | 129.31 | 130.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:45:00 | 128.30 | 129.31 | 130.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 11:45:00 | 128.29 | 129.00 | 130.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:15:00 | 121.89 | 125.73 | 128.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:15:00 | 121.88 | 125.73 | 128.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 123.86 | 123.72 | 125.78 | SL hit (close>ema200) qty=0.50 sl=123.72 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 123.86 | 123.72 | 125.78 | SL hit (close>ema200) qty=0.50 sl=123.72 alert=retest2 |

### Cycle 77 — BUY (started 2026-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 14:15:00 | 127.65 | 125.42 | 125.29 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 14:15:00 | 124.78 | 125.46 | 125.49 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-22 14:45:00 | 155.99 | 2025-05-29 09:15:00 | 171.59 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-23 12:15:00 | 155.45 | 2025-05-29 09:15:00 | 171.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-12 09:30:00 | 177.70 | 2025-06-12 14:15:00 | 173.26 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-06-17 11:15:00 | 171.24 | 2025-06-20 09:15:00 | 162.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 11:15:00 | 171.24 | 2025-06-20 10:15:00 | 165.80 | STOP_HIT | 0.50 | 3.18% |
| SELL | retest2 | 2025-06-30 13:00:00 | 160.00 | 2025-07-01 09:15:00 | 164.64 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2025-07-07 09:15:00 | 161.61 | 2025-07-07 11:15:00 | 158.92 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-07-07 10:45:00 | 160.70 | 2025-07-07 11:15:00 | 158.92 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-07-11 12:30:00 | 161.70 | 2025-07-15 10:15:00 | 177.87 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-01 12:30:00 | 217.99 | 2025-08-11 12:15:00 | 225.87 | STOP_HIT | 1.00 | 3.61% |
| BUY | retest2 | 2025-08-01 14:15:00 | 218.31 | 2025-08-11 12:15:00 | 225.87 | STOP_HIT | 1.00 | 3.46% |
| BUY | retest2 | 2025-08-04 09:15:00 | 220.61 | 2025-08-11 12:15:00 | 225.87 | STOP_HIT | 1.00 | 2.38% |
| SELL | retest1 | 2025-08-18 12:00:00 | 199.20 | 2025-08-19 09:15:00 | 207.33 | STOP_HIT | 1.00 | -4.08% |
| BUY | retest2 | 2025-08-26 10:45:00 | 225.70 | 2025-08-29 09:15:00 | 218.06 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2025-08-28 09:30:00 | 227.27 | 2025-08-29 09:15:00 | 218.06 | STOP_HIT | 1.00 | -4.05% |
| BUY | retest2 | 2025-08-28 10:30:00 | 226.80 | 2025-08-29 09:15:00 | 218.06 | STOP_HIT | 1.00 | -3.85% |
| SELL | retest2 | 2025-09-11 15:00:00 | 181.07 | 2025-09-12 11:15:00 | 172.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-11 15:00:00 | 181.07 | 2025-09-16 11:15:00 | 169.34 | STOP_HIT | 0.50 | 6.48% |
| BUY | retest2 | 2025-09-23 12:45:00 | 175.76 | 2025-09-23 13:15:00 | 173.90 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-10-03 14:45:00 | 193.61 | 2025-10-07 12:15:00 | 189.65 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-10-03 15:15:00 | 193.65 | 2025-10-07 12:15:00 | 189.65 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-10-17 12:15:00 | 177.39 | 2025-10-17 12:15:00 | 175.80 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-10-27 12:45:00 | 171.80 | 2025-11-06 13:15:00 | 163.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-29 09:45:00 | 171.43 | 2025-11-06 13:15:00 | 162.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 09:15:00 | 170.70 | 2025-11-06 13:15:00 | 162.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-27 12:45:00 | 171.80 | 2025-11-07 09:15:00 | 168.61 | STOP_HIT | 0.50 | 1.86% |
| SELL | retest2 | 2025-10-29 09:45:00 | 171.43 | 2025-11-07 09:15:00 | 168.61 | STOP_HIT | 0.50 | 1.64% |
| SELL | retest2 | 2025-10-31 09:15:00 | 170.70 | 2025-11-07 09:15:00 | 168.61 | STOP_HIT | 0.50 | 1.22% |
| SELL | retest2 | 2025-12-09 14:15:00 | 153.22 | 2025-12-09 14:15:00 | 155.61 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-12-31 15:15:00 | 165.10 | 2026-01-05 09:15:00 | 161.73 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2026-01-01 12:45:00 | 165.00 | 2026-01-05 09:15:00 | 161.73 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-01-01 15:00:00 | 164.95 | 2026-01-05 09:15:00 | 161.73 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2026-01-02 09:15:00 | 165.26 | 2026-01-05 09:15:00 | 161.73 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2026-01-08 10:15:00 | 155.40 | 2026-01-09 14:15:00 | 147.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 10:15:00 | 155.40 | 2026-01-12 15:15:00 | 146.45 | STOP_HIT | 0.50 | 5.76% |
| SELL | retest2 | 2026-02-05 14:00:00 | 124.85 | 2026-02-09 14:15:00 | 129.49 | STOP_HIT | 1.00 | -3.72% |
| SELL | retest2 | 2026-02-05 15:15:00 | 124.85 | 2026-02-09 14:15:00 | 129.49 | STOP_HIT | 1.00 | -3.72% |
| SELL | retest1 | 2026-02-12 09:30:00 | 120.37 | 2026-02-13 11:15:00 | 124.78 | STOP_HIT | 1.00 | -3.66% |
| SELL | retest2 | 2026-02-13 09:45:00 | 122.30 | 2026-02-17 12:15:00 | 124.30 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2026-02-13 11:00:00 | 122.06 | 2026-02-17 12:15:00 | 124.30 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2026-02-16 09:30:00 | 122.20 | 2026-02-17 12:15:00 | 124.30 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2026-02-16 10:30:00 | 121.91 | 2026-02-17 12:15:00 | 124.30 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2026-02-24 09:30:00 | 115.18 | 2026-02-26 11:15:00 | 117.01 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-02-24 11:45:00 | 115.58 | 2026-02-26 11:15:00 | 117.01 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-02-25 15:15:00 | 116.36 | 2026-02-26 11:15:00 | 117.01 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2026-03-19 12:30:00 | 113.25 | 2026-03-20 15:15:00 | 110.50 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2026-03-19 14:30:00 | 113.60 | 2026-03-20 15:15:00 | 110.50 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2026-03-19 15:15:00 | 114.20 | 2026-03-20 15:15:00 | 110.50 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2026-03-20 12:15:00 | 113.18 | 2026-03-20 15:15:00 | 110.50 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2026-04-02 11:15:00 | 111.10 | 2026-04-08 11:15:00 | 122.21 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-05-04 10:45:00 | 128.30 | 2026-05-05 10:15:00 | 121.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-05-04 11:45:00 | 128.29 | 2026-05-05 10:15:00 | 121.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-05-04 10:45:00 | 128.30 | 2026-05-06 09:15:00 | 123.86 | STOP_HIT | 0.50 | 3.46% |
| SELL | retest2 | 2026-05-04 11:45:00 | 128.29 | 2026-05-06 09:15:00 | 123.86 | STOP_HIT | 0.50 | 3.45% |
