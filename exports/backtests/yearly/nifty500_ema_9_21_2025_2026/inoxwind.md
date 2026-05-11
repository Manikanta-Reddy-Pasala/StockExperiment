# Inox Wind Ltd. (INOXWIND)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 103.65
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 62 |
| ALERT1 | 47 |
| ALERT2 | 46 |
| ALERT2_SKIP | 17 |
| ALERT3 | 121 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 64 |
| PARTIAL | 7 |
| TARGET_HIT | 2 |
| STOP_HIT | 69 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 78 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 23 / 55
- **Target hits / Stop hits / Partials:** 2 / 69 / 7
- **Avg / median % per leg:** 0.05% / -1.17%
- **Sum % (uncompounded):** 3.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 47 | 14 | 29.8% | 2 | 42 | 3 | 0.07% | 3.4% |
| BUY @ 2nd Alert (retest1) | 8 | 7 | 87.5% | 0 | 5 | 3 | 3.30% | 26.4% |
| BUY @ 3rd Alert (retest2) | 39 | 7 | 17.9% | 2 | 37 | 0 | -0.59% | -23.0% |
| SELL (all) | 31 | 9 | 29.0% | 0 | 27 | 4 | 0.01% | 0.3% |
| SELL @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 2 | 0 | 0.10% | 0.2% |
| SELL @ 3rd Alert (retest2) | 29 | 8 | 27.6% | 0 | 25 | 4 | 0.00% | 0.1% |
| retest1 (combined) | 10 | 8 | 80.0% | 0 | 7 | 3 | 2.66% | 26.6% |
| retest2 (combined) | 68 | 15 | 22.1% | 2 | 62 | 4 | -0.34% | -22.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 165.65 | 161.12 | 160.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 166.40 | 162.17 | 161.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 14:15:00 | 182.87 | 183.23 | 180.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 14:45:00 | 182.46 | 183.23 | 180.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 184.57 | 183.44 | 181.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 10:15:00 | 185.02 | 183.44 | 181.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 10:45:00 | 184.75 | 183.56 | 181.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 14:15:00 | 181.37 | 181.96 | 181.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 14:15:00 | 181.37 | 181.96 | 181.96 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 13:15:00 | 183.55 | 182.07 | 181.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 186.02 | 182.83 | 182.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 13:15:00 | 191.43 | 191.44 | 189.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 14:00:00 | 191.43 | 191.44 | 189.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 189.54 | 190.95 | 189.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:30:00 | 189.54 | 190.95 | 189.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 189.46 | 190.65 | 189.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:30:00 | 189.94 | 190.65 | 189.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 189.81 | 190.48 | 189.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 09:15:00 | 191.86 | 189.51 | 189.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 10:00:00 | 190.61 | 191.72 | 190.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 10:15:00 | 188.09 | 190.99 | 190.67 | SL hit (close<static) qty=1.00 sl=189.29 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 11:15:00 | 187.29 | 190.25 | 190.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 12:15:00 | 184.93 | 189.19 | 189.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 11:15:00 | 180.93 | 180.41 | 182.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 12:00:00 | 180.93 | 180.41 | 182.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 183.02 | 181.63 | 182.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 14:30:00 | 181.22 | 182.04 | 182.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 09:15:00 | 181.27 | 181.89 | 182.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 10:30:00 | 180.73 | 181.48 | 182.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 12:45:00 | 181.18 | 181.39 | 181.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 181.42 | 181.29 | 181.83 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-09 12:15:00 | 182.27 | 182.07 | 182.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 12:15:00 | 182.27 | 182.07 | 182.06 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 14:15:00 | 181.88 | 182.05 | 182.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-09 15:15:00 | 181.23 | 181.89 | 181.98 | Break + close below crossover candle low |

### Cycle 7 — BUY (started 2025-06-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 09:15:00 | 182.90 | 182.09 | 182.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 10:15:00 | 183.99 | 182.47 | 182.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 10:15:00 | 183.41 | 183.83 | 183.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 10:15:00 | 183.41 | 183.83 | 183.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 183.41 | 183.83 | 183.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:30:00 | 183.54 | 183.83 | 183.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 183.35 | 183.73 | 183.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:00:00 | 183.35 | 183.73 | 183.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 182.18 | 183.42 | 183.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:00:00 | 182.18 | 183.42 | 183.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 180.81 | 182.90 | 182.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 10:15:00 | 178.96 | 181.65 | 182.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 170.74 | 170.44 | 173.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 13:00:00 | 170.74 | 170.44 | 173.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 171.59 | 170.64 | 172.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 171.94 | 170.64 | 172.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 166.54 | 165.17 | 166.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 166.54 | 165.17 | 166.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 166.79 | 165.49 | 166.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:30:00 | 167.26 | 165.49 | 166.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 167.28 | 165.85 | 166.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:45:00 | 167.46 | 165.85 | 166.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 168.25 | 166.33 | 166.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:45:00 | 168.16 | 166.33 | 166.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 09:15:00 | 167.09 | 166.98 | 166.98 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 10:15:00 | 166.79 | 166.94 | 166.96 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 14:15:00 | 167.68 | 167.00 | 166.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 172.44 | 168.15 | 167.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 171.98 | 172.40 | 171.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 09:15:00 | 171.98 | 172.40 | 171.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 171.98 | 172.40 | 171.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:30:00 | 171.62 | 172.40 | 171.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 171.51 | 172.22 | 171.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:30:00 | 171.41 | 172.22 | 171.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 171.36 | 172.05 | 171.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:15:00 | 171.00 | 172.05 | 171.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 170.92 | 171.82 | 171.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:45:00 | 170.62 | 171.82 | 171.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 170.88 | 171.63 | 171.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 13:30:00 | 170.53 | 171.63 | 171.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 171.33 | 171.57 | 171.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 172.08 | 171.48 | 171.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 11:15:00 | 172.26 | 171.45 | 171.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 170.33 | 172.41 | 172.27 | SL hit (close<static) qty=1.00 sl=170.88 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 170.32 | 171.99 | 172.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 11:15:00 | 169.55 | 171.51 | 171.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 15:15:00 | 170.93 | 170.89 | 171.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-02 09:15:00 | 174.57 | 170.89 | 171.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 174.12 | 171.53 | 171.64 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 10:15:00 | 173.76 | 171.98 | 171.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 11:15:00 | 174.55 | 172.49 | 172.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 13:15:00 | 173.80 | 173.91 | 173.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-03 13:45:00 | 173.70 | 173.91 | 173.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 174.33 | 173.99 | 173.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:30:00 | 173.69 | 173.99 | 173.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 176.37 | 174.43 | 173.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 11:00:00 | 178.05 | 175.15 | 174.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 13:30:00 | 177.39 | 176.02 | 174.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 14:30:00 | 177.82 | 176.32 | 175.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 14:15:00 | 174.19 | 174.83 | 174.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 14:15:00 | 174.19 | 174.83 | 174.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 172.75 | 174.13 | 174.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 174.34 | 173.59 | 174.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 14:15:00 | 174.34 | 173.59 | 174.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 174.34 | 173.59 | 174.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 174.34 | 173.59 | 174.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 174.45 | 173.76 | 174.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 174.54 | 173.76 | 174.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 174.35 | 173.88 | 174.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 11:45:00 | 173.26 | 173.89 | 174.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 12:30:00 | 173.29 | 173.74 | 174.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 14:15:00 | 173.32 | 173.71 | 173.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 176.46 | 174.37 | 174.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 09:15:00 | 176.46 | 174.37 | 174.21 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 11:15:00 | 173.70 | 174.67 | 174.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 13:15:00 | 172.95 | 174.24 | 174.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 14:15:00 | 174.24 | 173.21 | 173.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 14:15:00 | 174.24 | 173.21 | 173.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 174.24 | 173.21 | 173.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 15:00:00 | 174.24 | 173.21 | 173.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 173.98 | 173.36 | 173.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:15:00 | 173.71 | 173.36 | 173.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 165.76 | 164.56 | 166.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:30:00 | 166.05 | 164.56 | 166.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 165.90 | 164.82 | 166.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 11:30:00 | 165.24 | 164.70 | 166.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:15:00 | 156.98 | 159.32 | 160.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 158.96 | 157.51 | 159.08 | SL hit (close>ema200) qty=0.50 sl=157.51 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 15:15:00 | 140.27 | 139.08 | 139.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 147.86 | 140.84 | 139.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 10:15:00 | 143.43 | 143.43 | 142.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 10:30:00 | 142.86 | 143.43 | 142.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 142.95 | 144.12 | 143.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 142.95 | 144.12 | 143.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 142.50 | 143.79 | 143.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:15:00 | 143.85 | 143.79 | 143.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 10:00:00 | 143.10 | 143.91 | 143.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 11:15:00 | 142.31 | 143.41 | 143.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 11:15:00 | 142.31 | 143.41 | 143.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 141.41 | 142.70 | 143.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 14:15:00 | 142.06 | 141.64 | 142.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 14:15:00 | 142.06 | 141.64 | 142.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 142.06 | 141.64 | 142.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 15:00:00 | 142.06 | 141.64 | 142.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 138.45 | 141.06 | 141.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 13:30:00 | 137.82 | 139.68 | 140.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 14:30:00 | 137.48 | 139.14 | 140.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 14:15:00 | 140.73 | 140.07 | 139.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 140.73 | 140.07 | 139.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 15:15:00 | 141.39 | 140.33 | 140.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 15:15:00 | 142.88 | 142.97 | 141.90 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 09:15:00 | 143.85 | 142.97 | 141.90 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 09:45:00 | 144.04 | 143.14 | 142.07 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 11:15:00 | 143.59 | 143.16 | 142.18 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 12:15:00 | 143.53 | 143.22 | 142.29 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 144.20 | 143.37 | 142.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:30:00 | 142.55 | 143.37 | 142.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 144.13 | 144.69 | 144.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:00:00 | 144.13 | 144.69 | 144.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 144.90 | 144.73 | 144.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:30:00 | 144.22 | 144.73 | 144.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 146.52 | 146.90 | 146.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:15:00 | 146.90 | 146.90 | 146.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 146.90 | 146.90 | 146.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 148.87 | 146.90 | 146.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 13:15:00 | 151.04 | 148.55 | 147.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 13:15:00 | 150.77 | 148.55 | 147.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 13:15:00 | 150.71 | 148.55 | 147.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-11 11:15:00 | 148.50 | 148.93 | 148.18 | SL hit (close<ema200) qty=0.50 sl=148.93 alert=retest1 |

### Cycle 20 — SELL (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 09:15:00 | 149.35 | 150.52 | 150.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 11:15:00 | 148.81 | 150.00 | 150.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 09:15:00 | 152.60 | 149.87 | 150.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 152.60 | 149.87 | 150.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 152.60 | 149.87 | 150.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:00:00 | 152.60 | 149.87 | 150.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 151.20 | 150.13 | 150.14 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 11:15:00 | 151.11 | 150.33 | 150.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 15:15:00 | 152.00 | 151.01 | 150.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 152.45 | 152.47 | 151.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 14:45:00 | 152.43 | 152.47 | 151.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 152.99 | 152.50 | 151.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 10:15:00 | 151.95 | 152.50 | 151.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 149.88 | 151.97 | 151.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 149.88 | 151.97 | 151.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 150.00 | 151.58 | 151.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:30:00 | 150.00 | 151.58 | 151.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 12:15:00 | 150.15 | 151.29 | 151.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 14:15:00 | 149.59 | 150.85 | 151.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 146.73 | 146.45 | 148.16 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-25 11:15:00 | 143.93 | 146.10 | 147.85 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 141.75 | 139.92 | 141.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:30:00 | 141.41 | 139.92 | 141.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 139.29 | 139.80 | 141.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 12:30:00 | 138.93 | 139.66 | 141.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 15:00:00 | 138.82 | 139.53 | 141.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 138.41 | 139.50 | 140.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:30:00 | 138.50 | 139.30 | 140.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 140.22 | 139.40 | 140.19 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-30 14:15:00 | 140.22 | 139.40 | 140.19 | SL hit (close>ema400) qty=1.00 sl=140.19 alert=retest1 |

### Cycle 23 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 141.57 | 140.50 | 140.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 142.07 | 140.81 | 140.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 10:15:00 | 140.51 | 141.27 | 140.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 10:15:00 | 140.51 | 141.27 | 140.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 140.51 | 141.27 | 140.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:45:00 | 140.43 | 141.27 | 140.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 140.85 | 141.19 | 140.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 13:00:00 | 141.31 | 141.21 | 140.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 15:00:00 | 141.75 | 141.43 | 141.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 10:15:00 | 139.55 | 141.07 | 141.04 | SL hit (close<static) qty=1.00 sl=140.50 alert=retest2 |

### Cycle 24 — SELL (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 11:15:00 | 139.42 | 140.74 | 140.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 138.52 | 139.60 | 140.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 13:15:00 | 139.76 | 139.45 | 139.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 13:15:00 | 139.76 | 139.45 | 139.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 139.76 | 139.45 | 139.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 13:45:00 | 140.04 | 139.45 | 139.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 141.03 | 139.76 | 139.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 141.03 | 139.76 | 139.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 140.70 | 139.95 | 140.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 143.35 | 139.95 | 140.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 09:15:00 | 142.82 | 140.53 | 140.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 10:15:00 | 145.01 | 141.42 | 140.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 10:15:00 | 148.86 | 149.26 | 147.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 11:00:00 | 148.86 | 149.26 | 147.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 149.13 | 149.37 | 148.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:00:00 | 149.13 | 149.37 | 148.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 147.12 | 148.85 | 148.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:00:00 | 147.12 | 148.85 | 148.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 147.44 | 148.57 | 148.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 14:30:00 | 147.94 | 148.07 | 147.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 149.07 | 147.83 | 147.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 11:15:00 | 147.18 | 147.91 | 148.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 11:15:00 | 147.18 | 147.91 | 148.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 09:15:00 | 146.49 | 147.16 | 147.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 15:15:00 | 146.40 | 146.28 | 146.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 15:15:00 | 146.40 | 146.28 | 146.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 146.40 | 146.28 | 146.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 09:30:00 | 144.85 | 145.95 | 146.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-21 13:15:00 | 149.60 | 147.13 | 146.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 149.60 | 147.13 | 146.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 151.17 | 148.27 | 147.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 15:15:00 | 153.71 | 153.95 | 152.21 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 09:15:00 | 155.19 | 153.95 | 152.21 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 153.21 | 153.64 | 152.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 14:15:00 | 153.50 | 153.64 | 152.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 153.93 | 153.41 | 152.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 10:15:00 | 153.61 | 153.38 | 152.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 10:45:00 | 153.63 | 153.52 | 152.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 153.15 | 153.70 | 153.18 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-28 13:15:00 | 153.15 | 153.70 | 153.18 | SL hit (close<ema400) qty=1.00 sl=153.18 alert=retest1 |

### Cycle 28 — SELL (started 2025-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 15:15:00 | 155.31 | 155.82 | 155.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 151.54 | 154.97 | 155.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 13:15:00 | 150.85 | 149.38 | 150.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 13:15:00 | 150.85 | 149.38 | 150.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 150.85 | 149.38 | 150.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 13:45:00 | 150.80 | 149.38 | 150.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 151.68 | 149.84 | 150.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 15:00:00 | 151.68 | 149.84 | 150.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 152.03 | 150.28 | 150.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:30:00 | 150.91 | 150.35 | 150.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 15:15:00 | 151.87 | 150.90 | 150.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 15:15:00 | 151.87 | 150.90 | 150.82 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 11:15:00 | 150.00 | 150.68 | 150.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 09:15:00 | 148.85 | 150.17 | 150.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 11:15:00 | 150.08 | 150.02 | 150.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-13 12:00:00 | 150.08 | 150.02 | 150.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 148.89 | 148.33 | 149.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 12:45:00 | 148.13 | 148.33 | 149.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 148.70 | 148.43 | 149.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 14:30:00 | 148.53 | 148.43 | 149.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 149.10 | 148.56 | 149.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 148.08 | 148.56 | 149.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 147.63 | 148.38 | 148.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 11:15:00 | 146.65 | 148.12 | 148.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 13:15:00 | 139.32 | 141.05 | 142.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-21 12:15:00 | 137.76 | 137.60 | 139.32 | SL hit (close>ema200) qty=0.50 sl=137.60 alert=retest2 |

### Cycle 31 — BUY (started 2025-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 14:15:00 | 124.71 | 123.08 | 123.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 15:15:00 | 124.85 | 123.43 | 123.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 12:15:00 | 126.05 | 126.27 | 125.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 12:15:00 | 126.05 | 126.27 | 125.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 126.05 | 126.27 | 125.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:45:00 | 125.72 | 126.27 | 125.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 125.35 | 126.10 | 125.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:15:00 | 124.06 | 126.10 | 125.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 124.60 | 125.80 | 125.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:30:00 | 124.02 | 125.80 | 125.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 125.14 | 125.67 | 125.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 11:45:00 | 125.66 | 125.81 | 125.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 13:15:00 | 124.83 | 125.53 | 125.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 13:15:00 | 124.83 | 125.53 | 125.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 14:15:00 | 124.49 | 125.32 | 125.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 12:15:00 | 124.78 | 124.56 | 124.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 13:00:00 | 124.78 | 124.56 | 124.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 125.35 | 124.72 | 124.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 125.35 | 124.72 | 124.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 127.02 | 125.18 | 125.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 127.68 | 125.96 | 125.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 13:15:00 | 126.58 | 126.65 | 126.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 14:15:00 | 126.60 | 126.65 | 126.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 127.97 | 126.88 | 126.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 130.75 | 127.05 | 126.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 14:15:00 | 126.50 | 127.11 | 127.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 126.50 | 127.11 | 127.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 12:15:00 | 124.45 | 126.23 | 126.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 11:15:00 | 122.90 | 122.76 | 123.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 12:00:00 | 122.90 | 122.76 | 123.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 123.64 | 122.90 | 123.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 123.64 | 122.90 | 123.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 123.67 | 123.05 | 123.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 123.91 | 123.05 | 123.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 123.20 | 123.08 | 123.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 10:45:00 | 122.74 | 122.98 | 123.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:00:00 | 122.80 | 122.95 | 123.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 125.13 | 123.32 | 123.42 | SL hit (close>static) qty=1.00 sl=124.14 alert=retest2 |

### Cycle 35 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 127.26 | 124.11 | 123.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 11:15:00 | 127.48 | 124.78 | 124.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 10:15:00 | 126.73 | 126.75 | 125.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 11:00:00 | 126.73 | 126.75 | 125.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 125.73 | 126.47 | 125.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 125.73 | 126.47 | 125.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 125.93 | 126.36 | 125.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:30:00 | 125.79 | 126.36 | 125.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 125.63 | 126.22 | 125.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 124.93 | 126.22 | 125.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 125.26 | 126.03 | 125.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:30:00 | 124.90 | 126.03 | 125.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 123.29 | 125.48 | 125.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 11:15:00 | 121.98 | 124.78 | 125.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 123.06 | 122.75 | 123.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 15:00:00 | 123.06 | 122.75 | 123.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 121.95 | 122.63 | 123.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:30:00 | 121.31 | 122.13 | 123.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 115.24 | 119.16 | 120.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 113.25 | 112.66 | 115.00 | SL hit (close>ema200) qty=0.50 sl=112.66 alert=retest2 |

### Cycle 37 — BUY (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 12:15:00 | 115.28 | 114.78 | 114.77 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2026-01-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 14:15:00 | 114.45 | 114.71 | 114.74 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 11:15:00 | 115.60 | 114.89 | 114.81 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2026-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 12:15:00 | 113.98 | 114.71 | 114.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 13:15:00 | 112.79 | 114.33 | 114.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 106.17 | 106.16 | 108.03 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 10:15:00 | 105.30 | 106.16 | 108.03 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 107.80 | 106.42 | 107.29 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-22 15:15:00 | 107.80 | 106.42 | 107.29 | SL hit (close>ema400) qty=1.00 sl=107.29 alert=retest1 |

### Cycle 41 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 108.20 | 106.04 | 105.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 109.16 | 107.34 | 106.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 107.00 | 107.59 | 106.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 10:00:00 | 107.00 | 107.59 | 106.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 106.50 | 107.37 | 106.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 106.46 | 107.37 | 106.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 106.15 | 107.13 | 106.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 11:45:00 | 106.15 | 107.13 | 106.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 106.98 | 107.10 | 106.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 13:30:00 | 107.43 | 107.02 | 106.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:00:00 | 107.79 | 107.15 | 106.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 09:45:00 | 107.03 | 107.54 | 107.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 10:15:00 | 107.24 | 107.54 | 107.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 106.37 | 107.31 | 107.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 10:45:00 | 106.27 | 107.31 | 107.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-01 11:15:00 | 105.78 | 107.00 | 107.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 105.78 | 107.00 | 107.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 105.45 | 106.69 | 106.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 104.63 | 103.78 | 104.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 104.63 | 103.78 | 104.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 104.75 | 103.97 | 104.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 107.84 | 103.97 | 104.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 106.71 | 104.52 | 105.03 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 107.81 | 105.73 | 105.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 13:15:00 | 109.53 | 107.43 | 106.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 10:15:00 | 108.27 | 108.39 | 107.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:45:00 | 107.79 | 108.39 | 107.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 107.11 | 108.11 | 107.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:00:00 | 107.11 | 108.11 | 107.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 107.01 | 107.89 | 107.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 15:15:00 | 106.71 | 107.89 | 107.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 106.71 | 107.65 | 107.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 105.06 | 107.65 | 107.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 103.87 | 106.90 | 107.09 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 109.58 | 107.07 | 106.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 110.33 | 107.72 | 107.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 15:15:00 | 110.85 | 111.11 | 109.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:15:00 | 109.58 | 111.11 | 109.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 109.01 | 110.69 | 109.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 109.01 | 110.69 | 109.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 109.83 | 110.52 | 109.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 14:45:00 | 110.22 | 110.06 | 109.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 11:15:00 | 108.96 | 109.69 | 109.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 108.96 | 109.69 | 109.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 106.06 | 108.70 | 109.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 12:15:00 | 104.60 | 103.58 | 105.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 12:45:00 | 104.25 | 103.58 | 105.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 103.16 | 103.50 | 105.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 14:15:00 | 101.66 | 103.50 | 105.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 14:15:00 | 96.58 | 98.28 | 99.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-20 11:15:00 | 97.43 | 97.24 | 98.63 | SL hit (close>ema200) qty=0.50 sl=97.24 alert=retest2 |

### Cycle 47 — BUY (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 11:15:00 | 85.11 | 83.42 | 83.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 12:15:00 | 85.87 | 83.91 | 83.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 14:15:00 | 83.71 | 84.08 | 83.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 14:15:00 | 83.71 | 84.08 | 83.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 83.71 | 84.08 | 83.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 15:00:00 | 83.71 | 84.08 | 83.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 83.50 | 83.97 | 83.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 09:15:00 | 82.65 | 83.97 | 83.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 81.14 | 83.40 | 83.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 80.57 | 82.84 | 83.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 13:15:00 | 78.37 | 78.27 | 79.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 13:30:00 | 78.21 | 78.27 | 79.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 79.41 | 78.51 | 79.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 79.71 | 78.51 | 79.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 80.03 | 78.82 | 79.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 80.03 | 78.82 | 79.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 81.40 | 79.63 | 79.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 81.58 | 80.02 | 79.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 78.88 | 80.37 | 80.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 78.88 | 80.37 | 80.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 78.88 | 80.37 | 80.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 78.88 | 80.37 | 80.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 78.99 | 80.09 | 79.93 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 78.72 | 79.63 | 79.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 78.09 | 79.32 | 79.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 81.67 | 79.50 | 79.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 81.67 | 79.50 | 79.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 81.67 | 79.50 | 79.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 82.47 | 79.50 | 79.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 82.80 | 80.16 | 79.86 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 76.83 | 79.92 | 80.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 75.82 | 78.57 | 79.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 78.00 | 77.42 | 78.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:00:00 | 78.00 | 77.42 | 78.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 78.61 | 77.66 | 78.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 78.79 | 77.66 | 78.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 78.35 | 77.80 | 78.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 14:30:00 | 77.89 | 77.83 | 78.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 15:00:00 | 77.97 | 77.83 | 78.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 81.17 | 78.53 | 78.54 | SL hit (close>static) qty=1.00 sl=78.81 alert=retest2 |

### Cycle 53 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 81.55 | 79.13 | 78.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 82.40 | 80.17 | 79.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 79.83 | 80.84 | 80.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 79.83 | 80.84 | 80.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 79.83 | 80.84 | 80.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 79.83 | 80.84 | 80.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 79.49 | 80.57 | 79.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 79.17 | 80.57 | 79.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 79.75 | 80.35 | 79.98 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 77.35 | 79.34 | 79.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 76.62 | 78.09 | 78.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 79.65 | 77.59 | 78.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 79.65 | 77.59 | 78.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 79.65 | 77.59 | 78.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 79.89 | 77.59 | 78.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 79.57 | 77.98 | 78.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:45:00 | 79.72 | 77.98 | 78.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 81.17 | 79.11 | 78.93 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 77.11 | 78.91 | 78.99 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 09:15:00 | 79.25 | 78.96 | 78.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 11:15:00 | 80.59 | 79.37 | 79.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 80.27 | 80.31 | 79.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:00:00 | 80.27 | 80.31 | 79.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 87.01 | 86.70 | 85.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 87.90 | 86.70 | 85.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 15:15:00 | 87.60 | 87.61 | 86.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-16 11:15:00 | 96.36 | 93.44 | 91.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 12:15:00 | 98.90 | 100.77 | 100.94 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 103.94 | 101.03 | 100.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 104.80 | 102.70 | 101.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 10:15:00 | 103.37 | 103.42 | 102.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 11:00:00 | 103.37 | 103.42 | 102.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 102.22 | 103.08 | 102.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:00:00 | 102.22 | 103.08 | 102.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 102.90 | 103.04 | 102.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 104.20 | 103.16 | 102.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 10:45:00 | 104.23 | 103.32 | 102.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 11:30:00 | 103.96 | 103.35 | 102.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 14:15:00 | 102.10 | 102.99 | 102.87 | SL hit (close<static) qty=1.00 sl=102.17 alert=retest2 |

### Cycle 60 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 99.17 | 102.10 | 102.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 99.01 | 101.48 | 102.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 14:15:00 | 100.88 | 100.61 | 101.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 14:15:00 | 100.88 | 100.61 | 101.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 100.88 | 100.61 | 101.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:30:00 | 101.68 | 100.61 | 101.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 98.84 | 100.31 | 101.17 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-05-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 14:15:00 | 103.21 | 101.57 | 101.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 105.44 | 102.61 | 102.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 09:15:00 | 105.87 | 106.18 | 104.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 10:00:00 | 105.87 | 106.18 | 104.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 105.00 | 105.94 | 104.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:00:00 | 105.00 | 105.94 | 104.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 105.98 | 105.95 | 105.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 12:30:00 | 106.13 | 105.97 | 105.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 14:00:00 | 106.30 | 106.04 | 105.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 10:15:00 | 104.53 | 105.97 | 105.51 | SL hit (close<static) qty=1.00 sl=104.63 alert=retest2 |

### Cycle 62 — SELL (started 2026-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 13:15:00 | 103.64 | 105.10 | 105.19 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-21 10:15:00 | 185.02 | 2025-05-22 14:15:00 | 181.37 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-05-21 10:45:00 | 184.75 | 2025-05-22 14:15:00 | 181.37 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-05-30 09:15:00 | 191.86 | 2025-06-02 10:15:00 | 188.09 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-06-02 10:00:00 | 190.61 | 2025-06-02 10:15:00 | 188.09 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-06-05 14:30:00 | 181.22 | 2025-06-09 12:15:00 | 182.27 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-06-06 09:15:00 | 181.27 | 2025-06-09 12:15:00 | 182.27 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-06-06 10:30:00 | 180.73 | 2025-06-09 12:15:00 | 182.27 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-06-06 12:45:00 | 181.18 | 2025-06-09 12:15:00 | 182.27 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-06-27 09:15:00 | 172.08 | 2025-07-01 09:15:00 | 170.33 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-06-27 11:15:00 | 172.26 | 2025-07-01 09:15:00 | 170.33 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-07-04 11:00:00 | 178.05 | 2025-07-07 14:15:00 | 174.19 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2025-07-04 13:30:00 | 177.39 | 2025-07-07 14:15:00 | 174.19 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-07-04 14:30:00 | 177.82 | 2025-07-07 14:15:00 | 174.19 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-07-09 11:45:00 | 173.26 | 2025-07-10 09:15:00 | 176.46 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-07-09 12:30:00 | 173.29 | 2025-07-10 09:15:00 | 176.46 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-07-09 14:15:00 | 173.32 | 2025-07-10 09:15:00 | 176.46 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-07-17 11:30:00 | 165.24 | 2025-07-25 10:15:00 | 156.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-17 11:30:00 | 165.24 | 2025-07-28 09:15:00 | 158.96 | STOP_HIT | 0.50 | 3.80% |
| BUY | retest2 | 2025-08-22 09:15:00 | 143.85 | 2025-08-25 11:15:00 | 142.31 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-08-25 10:00:00 | 143.10 | 2025-08-25 11:15:00 | 142.31 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-08-28 13:30:00 | 137.82 | 2025-09-01 14:15:00 | 140.73 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-08-28 14:30:00 | 137.48 | 2025-09-01 14:15:00 | 140.73 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest1 | 2025-09-03 09:15:00 | 143.85 | 2025-09-10 13:15:00 | 151.04 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-09-03 09:45:00 | 144.04 | 2025-09-10 13:15:00 | 150.77 | PARTIAL | 0.50 | 4.67% |
| BUY | retest1 | 2025-09-03 11:15:00 | 143.59 | 2025-09-10 13:15:00 | 150.71 | PARTIAL | 0.50 | 4.96% |
| BUY | retest1 | 2025-09-03 09:15:00 | 143.85 | 2025-09-11 11:15:00 | 148.50 | STOP_HIT | 0.50 | 3.23% |
| BUY | retest1 | 2025-09-03 09:45:00 | 144.04 | 2025-09-11 11:15:00 | 148.50 | STOP_HIT | 0.50 | 3.10% |
| BUY | retest1 | 2025-09-03 11:15:00 | 143.59 | 2025-09-11 11:15:00 | 148.50 | STOP_HIT | 0.50 | 3.42% |
| BUY | retest1 | 2025-09-03 12:15:00 | 143.53 | 2025-09-12 10:15:00 | 148.30 | STOP_HIT | 1.00 | 3.32% |
| BUY | retest2 | 2025-09-10 09:15:00 | 148.87 | 2025-09-18 09:15:00 | 149.35 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest1 | 2025-09-25 11:15:00 | 143.93 | 2025-09-30 14:15:00 | 140.22 | STOP_HIT | 1.00 | 2.58% |
| SELL | retest2 | 2025-09-29 12:30:00 | 138.93 | 2025-10-01 12:15:00 | 141.57 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-09-29 15:00:00 | 138.82 | 2025-10-01 12:15:00 | 141.57 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-09-30 09:15:00 | 138.41 | 2025-10-01 12:15:00 | 141.57 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-09-30 11:30:00 | 138.50 | 2025-10-01 12:15:00 | 141.57 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-10-03 13:00:00 | 141.31 | 2025-10-06 10:15:00 | 139.55 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-10-03 15:00:00 | 141.75 | 2025-10-06 10:15:00 | 139.55 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-10-14 14:30:00 | 147.94 | 2025-10-16 11:15:00 | 147.18 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-10-15 09:15:00 | 149.07 | 2025-10-16 11:15:00 | 147.18 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-10-20 09:30:00 | 144.85 | 2025-10-21 13:15:00 | 149.60 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest1 | 2025-10-27 09:15:00 | 155.19 | 2025-10-28 13:15:00 | 153.15 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-10-27 14:15:00 | 153.50 | 2025-11-04 14:15:00 | 154.76 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2025-10-28 09:15:00 | 153.93 | 2025-11-04 15:15:00 | 155.31 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest2 | 2025-10-28 10:15:00 | 153.61 | 2025-11-04 15:15:00 | 155.31 | STOP_HIT | 1.00 | 1.11% |
| BUY | retest2 | 2025-10-28 10:45:00 | 153.63 | 2025-11-04 15:15:00 | 155.31 | STOP_HIT | 1.00 | 1.09% |
| BUY | retest2 | 2025-10-29 10:15:00 | 157.61 | 2025-11-04 15:15:00 | 155.31 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-10-29 11:15:00 | 156.73 | 2025-11-04 15:15:00 | 155.31 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-10-29 12:15:00 | 156.96 | 2025-11-04 15:15:00 | 155.31 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-10-31 11:45:00 | 156.88 | 2025-11-04 15:15:00 | 155.31 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-11-03 09:15:00 | 157.40 | 2025-11-04 15:15:00 | 155.31 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-11-11 09:30:00 | 150.91 | 2025-11-11 15:15:00 | 151.87 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-11-17 11:15:00 | 146.65 | 2025-11-19 13:15:00 | 139.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 11:15:00 | 146.65 | 2025-11-21 12:15:00 | 137.76 | STOP_HIT | 0.50 | 6.06% |
| BUY | retest2 | 2025-12-18 11:45:00 | 125.66 | 2025-12-18 13:15:00 | 124.83 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-12-24 09:15:00 | 130.75 | 2025-12-26 14:15:00 | 126.50 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2026-01-01 10:45:00 | 122.74 | 2026-01-02 09:15:00 | 125.13 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2026-01-01 12:00:00 | 122.80 | 2026-01-02 09:15:00 | 125.13 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2026-01-08 10:30:00 | 121.31 | 2026-01-09 09:15:00 | 115.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 10:30:00 | 121.31 | 2026-01-12 15:15:00 | 113.25 | STOP_HIT | 0.50 | 6.64% |
| SELL | retest1 | 2026-01-22 10:15:00 | 105.30 | 2026-01-22 15:15:00 | 107.80 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2026-01-23 10:15:00 | 106.37 | 2026-01-28 11:15:00 | 108.20 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2026-01-23 11:15:00 | 106.25 | 2026-01-28 11:15:00 | 108.20 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2026-01-29 13:30:00 | 107.43 | 2026-02-01 11:15:00 | 105.78 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2026-01-30 10:00:00 | 107.79 | 2026-02-01 11:15:00 | 105.78 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2026-02-01 09:45:00 | 107.03 | 2026-02-01 11:15:00 | 105.78 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2026-02-01 10:15:00 | 107.24 | 2026-02-01 11:15:00 | 105.78 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2026-02-11 14:45:00 | 110.22 | 2026-02-12 11:15:00 | 108.96 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-02-16 14:15:00 | 101.66 | 2026-02-19 14:15:00 | 96.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-16 14:15:00 | 101.66 | 2026-02-20 11:15:00 | 97.43 | STOP_HIT | 0.50 | 4.16% |
| SELL | retest2 | 2026-03-24 14:30:00 | 77.89 | 2026-03-25 09:15:00 | 81.17 | STOP_HIT | 1.00 | -4.21% |
| SELL | retest2 | 2026-03-24 15:00:00 | 77.97 | 2026-03-25 09:15:00 | 81.17 | STOP_HIT | 1.00 | -4.10% |
| BUY | retest2 | 2026-04-13 10:15:00 | 87.90 | 2026-04-16 11:15:00 | 96.36 | TARGET_HIT | 1.00 | 9.62% |
| BUY | retest2 | 2026-04-13 15:15:00 | 87.60 | 2026-04-17 09:15:00 | 96.69 | TARGET_HIT | 1.00 | 10.38% |
| BUY | retest2 | 2026-04-29 09:15:00 | 104.20 | 2026-04-29 14:15:00 | 102.10 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2026-04-29 10:45:00 | 104.23 | 2026-04-29 14:15:00 | 102.10 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2026-04-29 11:30:00 | 103.96 | 2026-04-29 14:15:00 | 102.10 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2026-05-07 12:30:00 | 106.13 | 2026-05-08 10:15:00 | 104.53 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2026-05-07 14:00:00 | 106.30 | 2026-05-08 10:15:00 | 104.53 | STOP_HIT | 1.00 | -1.67% |
