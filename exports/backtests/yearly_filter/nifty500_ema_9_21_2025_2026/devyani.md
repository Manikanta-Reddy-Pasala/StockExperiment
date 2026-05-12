# Devyani International Ltd. (DEVYANI)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 118.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 62 |
| ALERT1 | 45 |
| ALERT2 | 44 |
| ALERT2_SKIP | 23 |
| ALERT3 | 138 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 79 |
| PARTIAL | 20 |
| TARGET_HIT | 10 |
| STOP_HIT | 66 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 96 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 53 / 43
- **Target hits / Stop hits / Partials:** 10 / 66 / 20
- **Avg / median % per leg:** 2.32% / 2.86%
- **Sum % (uncompounded):** 222.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 32 | 13 | 40.6% | 7 | 25 | 0 | 2.04% | 65.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 32 | 13 | 40.6% | 7 | 25 | 0 | 2.04% | 65.3% |
| SELL (all) | 64 | 40 | 62.5% | 3 | 41 | 20 | 2.46% | 157.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 64 | 40 | 62.5% | 3 | 41 | 20 | 2.46% | 157.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 96 | 53 | 55.2% | 10 | 66 | 20 | 2.32% | 222.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 176.26 | 175.67 | 175.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 176.93 | 176.28 | 175.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 176.21 | 176.27 | 175.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 10:30:00 | 176.32 | 176.27 | 175.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 11:15:00 | 176.07 | 176.23 | 176.00 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 15:15:00 | 175.30 | 175.83 | 175.86 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 09:15:00 | 176.32 | 175.93 | 175.90 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 14:15:00 | 174.18 | 175.66 | 175.83 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 11:15:00 | 177.20 | 175.94 | 175.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 12:15:00 | 178.40 | 176.43 | 176.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 11:15:00 | 183.69 | 184.40 | 182.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 12:00:00 | 183.69 | 184.40 | 182.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 182.21 | 183.89 | 182.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 182.21 | 183.89 | 182.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 183.28 | 183.77 | 182.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 15:15:00 | 182.80 | 183.77 | 182.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 182.80 | 183.57 | 182.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 09:15:00 | 183.65 | 183.57 | 182.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 14:45:00 | 183.75 | 184.14 | 183.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 10:00:00 | 183.31 | 183.95 | 183.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 11:15:00 | 180.36 | 183.02 | 183.02 | SL hit (close<static) qty=1.00 sl=182.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 12:15:00 | 180.91 | 182.60 | 182.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 09:15:00 | 179.19 | 180.25 | 181.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 10:15:00 | 181.34 | 180.47 | 181.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 10:15:00 | 181.34 | 180.47 | 181.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 181.34 | 180.47 | 181.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 181.34 | 180.47 | 181.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 179.31 | 180.24 | 181.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:30:00 | 182.50 | 180.24 | 181.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 181.74 | 180.54 | 181.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:45:00 | 180.66 | 180.54 | 181.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 181.20 | 180.67 | 181.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 14:15:00 | 180.56 | 180.67 | 181.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 14:15:00 | 171.53 | 173.69 | 175.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-28 12:15:00 | 173.57 | 173.05 | 174.71 | SL hit (close>ema200) qty=0.50 sl=173.05 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 14:15:00 | 169.01 | 168.71 | 168.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 10:15:00 | 170.87 | 169.16 | 168.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 11:15:00 | 173.01 | 173.20 | 171.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 12:00:00 | 173.01 | 173.20 | 171.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 172.76 | 173.03 | 172.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:00:00 | 172.76 | 173.03 | 172.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 172.10 | 172.84 | 172.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 171.23 | 172.84 | 172.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 171.98 | 172.67 | 172.53 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 171.00 | 172.17 | 172.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 12:15:00 | 169.96 | 171.72 | 172.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 167.00 | 166.09 | 167.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 12:45:00 | 167.03 | 166.09 | 167.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 166.67 | 166.35 | 167.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 168.77 | 166.35 | 167.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 167.76 | 166.63 | 167.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:30:00 | 167.67 | 166.63 | 167.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 167.75 | 166.86 | 167.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:00:00 | 167.75 | 166.86 | 167.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 168.58 | 167.20 | 167.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 13:00:00 | 168.58 | 167.20 | 167.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-06-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 13:15:00 | 169.59 | 167.68 | 167.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 15:15:00 | 170.00 | 168.50 | 168.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 10:15:00 | 170.00 | 170.17 | 169.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 11:00:00 | 170.00 | 170.17 | 169.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 169.32 | 170.00 | 169.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:30:00 | 169.21 | 170.00 | 169.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 167.65 | 169.53 | 169.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 167.65 | 169.53 | 169.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 168.67 | 169.36 | 169.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 14:15:00 | 169.90 | 169.36 | 169.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 11:15:00 | 168.14 | 169.03 | 169.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 11:15:00 | 168.14 | 169.03 | 169.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 14:15:00 | 166.53 | 168.15 | 168.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 14:15:00 | 167.38 | 166.51 | 167.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 14:15:00 | 167.38 | 166.51 | 167.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 167.38 | 166.51 | 167.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 15:00:00 | 167.38 | 166.51 | 167.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 167.00 | 166.61 | 167.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:15:00 | 169.52 | 166.61 | 167.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 168.95 | 167.08 | 167.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 11:00:00 | 168.17 | 167.29 | 167.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 14:15:00 | 169.10 | 167.83 | 167.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 14:15:00 | 169.10 | 167.83 | 167.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 169.70 | 168.39 | 168.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 170.68 | 170.92 | 169.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 10:00:00 | 170.68 | 170.92 | 169.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 173.00 | 172.02 | 170.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:30:00 | 171.40 | 172.02 | 170.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 171.76 | 171.82 | 171.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 12:30:00 | 171.93 | 171.82 | 171.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 171.88 | 171.83 | 171.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 172.92 | 171.74 | 171.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 09:15:00 | 170.41 | 171.47 | 171.19 | SL hit (close<static) qty=1.00 sl=171.15 alert=retest2 |

### Cycle 12 — SELL (started 2025-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 11:15:00 | 169.23 | 170.73 | 170.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 12:15:00 | 167.92 | 170.17 | 170.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 14:15:00 | 168.68 | 168.01 | 168.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 14:15:00 | 168.68 | 168.01 | 168.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 168.68 | 168.01 | 168.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 15:00:00 | 168.68 | 168.01 | 168.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 168.84 | 168.17 | 168.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 168.27 | 168.17 | 168.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 167.27 | 167.99 | 168.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 12:00:00 | 166.63 | 167.56 | 168.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 15:15:00 | 166.60 | 165.97 | 166.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 09:15:00 | 172.91 | 167.46 | 167.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 09:15:00 | 172.91 | 167.46 | 167.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 10:15:00 | 174.35 | 172.22 | 170.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 13:15:00 | 172.24 | 172.67 | 171.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-07 14:00:00 | 172.24 | 172.67 | 171.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 173.64 | 172.77 | 171.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:45:00 | 173.39 | 172.77 | 171.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 172.00 | 172.52 | 171.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:30:00 | 171.60 | 172.52 | 171.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 171.56 | 172.33 | 171.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:45:00 | 171.00 | 172.33 | 171.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 171.49 | 172.16 | 171.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 171.49 | 172.16 | 171.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 171.50 | 172.03 | 171.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 174.49 | 172.03 | 171.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 11:00:00 | 172.05 | 172.03 | 171.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 14:15:00 | 171.06 | 171.50 | 171.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 14:15:00 | 171.06 | 171.50 | 171.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 10:15:00 | 170.25 | 171.13 | 171.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 11:15:00 | 171.15 | 171.14 | 171.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 11:15:00 | 171.15 | 171.14 | 171.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 171.15 | 171.14 | 171.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:30:00 | 171.32 | 171.14 | 171.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 171.63 | 171.24 | 171.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:45:00 | 171.50 | 171.24 | 171.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 171.70 | 171.34 | 171.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:30:00 | 171.66 | 171.34 | 171.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 169.84 | 171.06 | 171.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:30:00 | 169.34 | 170.65 | 171.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 172.67 | 169.67 | 169.77 | SL hit (close>static) qty=1.00 sl=171.50 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 172.33 | 170.20 | 170.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 12:15:00 | 174.10 | 171.38 | 170.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 12:15:00 | 173.88 | 173.98 | 172.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 13:00:00 | 173.88 | 173.98 | 172.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 173.34 | 173.81 | 172.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 173.83 | 173.81 | 172.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 11:15:00 | 174.58 | 175.02 | 175.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 11:15:00 | 174.58 | 175.02 | 175.05 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 13:15:00 | 175.30 | 175.07 | 175.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 14:15:00 | 177.45 | 175.55 | 175.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 15:15:00 | 175.50 | 175.54 | 175.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-23 09:15:00 | 175.36 | 175.54 | 175.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 177.44 | 175.92 | 175.50 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 15:15:00 | 175.00 | 175.30 | 175.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 173.13 | 174.49 | 174.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 163.03 | 162.62 | 163.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 163.03 | 162.62 | 163.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 163.03 | 162.62 | 163.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:00:00 | 163.03 | 162.62 | 163.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 163.49 | 162.84 | 163.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 163.49 | 162.84 | 163.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 163.87 | 163.05 | 163.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 163.87 | 163.05 | 163.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 164.01 | 163.24 | 163.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 164.01 | 163.24 | 163.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 163.90 | 163.37 | 163.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 163.00 | 163.37 | 163.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 161.50 | 163.00 | 163.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:15:00 | 159.66 | 161.85 | 162.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 14:45:00 | 160.38 | 159.98 | 161.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 14:15:00 | 160.02 | 159.94 | 160.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 15:15:00 | 152.36 | 154.09 | 155.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 09:15:00 | 151.68 | 153.51 | 154.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 09:15:00 | 152.02 | 153.51 | 154.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-13 13:15:00 | 156.15 | 153.36 | 154.18 | SL hit (close>ema200) qty=0.50 sl=153.36 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 15:15:00 | 159.03 | 155.40 | 155.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 10:15:00 | 165.74 | 158.21 | 156.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 09:15:00 | 172.80 | 173.25 | 170.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 09:30:00 | 172.67 | 173.25 | 170.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 170.82 | 171.90 | 170.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 170.61 | 171.90 | 170.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 170.00 | 171.52 | 170.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:30:00 | 170.51 | 171.52 | 170.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 170.81 | 171.38 | 170.66 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 15:15:00 | 169.49 | 170.34 | 170.38 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 09:15:00 | 172.65 | 170.80 | 170.58 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 12:15:00 | 169.47 | 170.41 | 170.45 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 13:15:00 | 172.05 | 170.74 | 170.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-26 14:15:00 | 174.60 | 171.51 | 170.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-29 09:15:00 | 173.67 | 174.44 | 173.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:00:00 | 173.67 | 174.44 | 173.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 174.89 | 174.53 | 173.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 09:15:00 | 175.13 | 174.37 | 173.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 13:00:00 | 176.30 | 174.72 | 174.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 09:30:00 | 175.66 | 174.91 | 174.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 10:45:00 | 175.14 | 174.92 | 174.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 11:15:00 | 178.12 | 175.56 | 174.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:30:00 | 179.66 | 176.28 | 175.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 10:15:00 | 183.49 | 185.61 | 185.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 10:15:00 | 183.49 | 185.61 | 185.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 12:15:00 | 181.90 | 183.44 | 184.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 14:15:00 | 177.70 | 176.49 | 177.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 14:15:00 | 177.70 | 176.49 | 177.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 177.70 | 176.49 | 177.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 177.70 | 176.49 | 177.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 179.10 | 177.02 | 178.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:15:00 | 180.87 | 177.02 | 178.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 178.66 | 177.34 | 178.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:30:00 | 182.45 | 177.34 | 178.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 177.13 | 177.30 | 178.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:30:00 | 176.58 | 177.03 | 177.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 09:15:00 | 180.10 | 176.55 | 176.76 | SL hit (close>static) qty=1.00 sl=178.80 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 10:15:00 | 178.89 | 177.02 | 176.96 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 09:15:00 | 175.10 | 176.67 | 176.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 10:15:00 | 174.89 | 176.31 | 176.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 12:15:00 | 170.63 | 169.10 | 171.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 13:00:00 | 170.63 | 169.10 | 171.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 170.33 | 169.35 | 170.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 169.31 | 169.78 | 170.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:00:00 | 169.33 | 169.69 | 170.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 168.30 | 168.82 | 169.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 13:45:00 | 169.73 | 169.40 | 169.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 171.33 | 169.79 | 169.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:00:00 | 171.33 | 169.79 | 169.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 170.90 | 170.01 | 169.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 170.90 | 170.01 | 169.91 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 09:15:00 | 168.87 | 169.78 | 169.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-03 10:15:00 | 168.06 | 169.44 | 169.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 09:15:00 | 166.44 | 166.27 | 167.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 09:15:00 | 166.44 | 166.27 | 167.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 166.44 | 166.27 | 167.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:30:00 | 167.01 | 166.27 | 167.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 163.39 | 164.54 | 165.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 11:30:00 | 162.50 | 163.85 | 164.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 12:00:00 | 162.80 | 163.86 | 164.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 14:00:00 | 162.70 | 163.49 | 164.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 14:30:00 | 162.71 | 163.22 | 163.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 163.35 | 163.13 | 163.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:00:00 | 162.30 | 162.97 | 163.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 10:30:00 | 162.26 | 161.66 | 162.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 15:15:00 | 164.00 | 162.87 | 162.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 15:15:00 | 164.00 | 162.87 | 162.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 12:15:00 | 165.90 | 163.95 | 163.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 11:15:00 | 166.50 | 167.18 | 166.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 12:00:00 | 166.50 | 167.18 | 166.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 165.75 | 166.89 | 166.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 12:30:00 | 165.76 | 166.89 | 166.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 165.82 | 166.68 | 166.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 166.30 | 166.26 | 166.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 09:15:00 | 165.50 | 166.38 | 166.34 | SL hit (close<static) qty=1.00 sl=165.51 alert=retest2 |

### Cycle 30 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 165.07 | 166.12 | 166.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 12:15:00 | 164.11 | 165.54 | 165.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 165.48 | 163.82 | 164.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 165.48 | 163.82 | 164.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 165.48 | 163.82 | 164.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 165.75 | 163.82 | 164.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 165.91 | 164.24 | 164.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:45:00 | 165.88 | 164.24 | 164.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 12:15:00 | 166.44 | 164.98 | 164.79 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 14:15:00 | 164.02 | 164.63 | 164.65 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 13:15:00 | 167.50 | 165.12 | 164.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 14:15:00 | 167.74 | 165.64 | 165.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 13:15:00 | 166.04 | 166.67 | 165.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 14:00:00 | 166.04 | 166.67 | 165.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 167.44 | 166.83 | 166.11 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 161.61 | 165.37 | 165.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 11:15:00 | 161.37 | 164.57 | 165.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 163.10 | 163.03 | 164.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 10:00:00 | 163.10 | 163.03 | 164.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 161.96 | 162.45 | 163.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 14:15:00 | 161.88 | 162.45 | 163.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 14:45:00 | 161.42 | 162.18 | 163.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:45:00 | 161.08 | 161.82 | 162.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 12:15:00 | 161.62 | 161.78 | 162.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 158.44 | 159.26 | 160.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:45:00 | 159.20 | 159.26 | 160.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 153.79 | 156.36 | 158.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 153.35 | 156.36 | 158.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 153.03 | 156.36 | 158.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 153.54 | 156.36 | 158.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-10 11:15:00 | 150.32 | 149.97 | 153.01 | SL hit (close>ema200) qty=0.50 sl=149.97 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 15:15:00 | 142.70 | 141.09 | 140.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 11:15:00 | 143.58 | 141.62 | 141.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 13:15:00 | 144.62 | 145.28 | 144.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-19 14:00:00 | 144.62 | 145.28 | 144.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 148.00 | 145.95 | 144.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:30:00 | 147.15 | 145.95 | 144.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 146.59 | 145.99 | 145.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 14:15:00 | 147.00 | 145.99 | 145.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 144.30 | 145.98 | 145.33 | SL hit (close<static) qty=1.00 sl=144.75 alert=retest2 |

### Cycle 36 — SELL (started 2025-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 15:15:00 | 144.14 | 144.92 | 145.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 09:15:00 | 142.00 | 144.34 | 144.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 14:15:00 | 139.80 | 139.70 | 141.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 15:00:00 | 139.80 | 139.70 | 141.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 141.15 | 139.99 | 140.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 141.15 | 139.99 | 140.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 140.85 | 140.16 | 140.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:45:00 | 142.17 | 140.16 | 140.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 140.14 | 140.16 | 140.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 13:45:00 | 139.63 | 140.02 | 140.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 15:00:00 | 139.15 | 139.85 | 140.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 09:30:00 | 138.98 | 139.55 | 140.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 09:15:00 | 132.65 | 134.15 | 135.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 09:15:00 | 132.19 | 134.15 | 135.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 09:15:00 | 132.03 | 134.15 | 135.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 135.00 | 133.84 | 134.67 | SL hit (close>ema200) qty=0.50 sl=133.84 alert=retest2 |

### Cycle 37 — BUY (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 14:15:00 | 136.83 | 135.25 | 135.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 14:15:00 | 138.49 | 135.89 | 135.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 09:15:00 | 134.58 | 135.85 | 135.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 134.58 | 135.85 | 135.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 134.58 | 135.85 | 135.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 134.58 | 135.85 | 135.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 136.70 | 136.02 | 135.66 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 134.50 | 135.65 | 135.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 133.89 | 135.06 | 135.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 134.21 | 134.17 | 134.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 134.21 | 134.17 | 134.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 134.78 | 134.29 | 134.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 134.78 | 134.29 | 134.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 135.24 | 134.48 | 134.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:45:00 | 135.09 | 134.48 | 134.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 134.98 | 134.58 | 134.79 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 135.20 | 134.92 | 134.91 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 14:15:00 | 134.70 | 134.90 | 134.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 10:15:00 | 134.53 | 134.77 | 134.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 134.68 | 134.61 | 134.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 134.68 | 134.61 | 134.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 134.68 | 134.61 | 134.72 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 134.89 | 134.78 | 134.78 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 14:15:00 | 134.50 | 134.74 | 134.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-12 15:15:00 | 134.00 | 134.60 | 134.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 126.56 | 126.43 | 127.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 126.56 | 126.43 | 127.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 126.56 | 126.43 | 127.68 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 131.00 | 128.19 | 128.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 132.31 | 129.32 | 128.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 12:15:00 | 143.97 | 144.20 | 141.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 13:00:00 | 143.97 | 144.20 | 141.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 142.50 | 144.10 | 142.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:15:00 | 144.91 | 143.38 | 142.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 14:15:00 | 142.11 | 142.96 | 142.80 | SL hit (close<static) qty=1.00 sl=142.21 alert=retest2 |

### Cycle 44 — SELL (started 2026-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 11:15:00 | 143.74 | 146.30 | 146.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 13:15:00 | 140.63 | 144.66 | 145.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 135.15 | 133.57 | 135.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 13:00:00 | 135.15 | 133.57 | 135.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 135.82 | 134.02 | 135.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 13:45:00 | 136.09 | 134.02 | 135.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 135.82 | 134.38 | 135.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 135.76 | 134.38 | 135.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 13:15:00 | 134.28 | 134.06 | 134.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 13:45:00 | 134.65 | 134.06 | 134.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 135.05 | 134.26 | 134.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:45:00 | 135.08 | 134.26 | 134.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 135.19 | 134.44 | 134.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 135.39 | 134.44 | 134.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 135.01 | 134.56 | 134.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:45:00 | 135.29 | 134.56 | 134.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 134.70 | 134.59 | 134.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 134.70 | 134.59 | 134.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 134.45 | 134.56 | 134.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:30:00 | 134.80 | 134.56 | 134.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 134.87 | 134.62 | 134.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 13:00:00 | 134.87 | 134.62 | 134.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 134.97 | 134.69 | 134.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 13:30:00 | 134.82 | 134.69 | 134.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 134.62 | 134.68 | 134.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:30:00 | 135.14 | 134.68 | 134.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 134.68 | 134.68 | 134.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:15:00 | 133.59 | 134.68 | 134.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 133.43 | 134.43 | 134.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:00:00 | 132.40 | 134.02 | 134.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:30:00 | 132.75 | 133.60 | 134.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:00:00 | 132.75 | 133.60 | 134.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 14:15:00 | 132.83 | 133.46 | 134.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 126.11 | 128.93 | 130.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 126.11 | 128.93 | 130.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 126.19 | 128.93 | 130.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 14:15:00 | 125.78 | 127.36 | 129.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 12:15:00 | 126.64 | 126.48 | 128.17 | SL hit (close>ema200) qty=0.50 sl=126.48 alert=retest2 |

### Cycle 45 — BUY (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 12:15:00 | 116.05 | 114.26 | 114.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 14:15:00 | 116.76 | 115.14 | 114.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 116.61 | 116.90 | 116.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 15:15:00 | 116.61 | 116.90 | 116.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 116.61 | 116.90 | 116.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 114.44 | 116.90 | 116.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 114.39 | 116.40 | 115.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:00:00 | 114.39 | 116.40 | 115.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 114.08 | 115.93 | 115.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:30:00 | 113.85 | 115.93 | 115.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2026-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 12:15:00 | 114.10 | 115.31 | 115.44 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 116.80 | 115.60 | 115.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 10:15:00 | 117.11 | 116.04 | 115.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-03 14:15:00 | 116.38 | 116.42 | 116.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 14:15:00 | 116.38 | 116.42 | 116.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 116.38 | 116.42 | 116.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-03 14:45:00 | 116.19 | 116.42 | 116.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 15:15:00 | 116.30 | 116.40 | 116.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 09:15:00 | 118.48 | 116.40 | 116.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-05 09:15:00 | 130.33 | 123.66 | 120.47 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 131.72 | 133.72 | 133.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 130.29 | 132.12 | 132.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 131.55 | 130.25 | 131.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 131.55 | 130.25 | 131.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 131.55 | 130.25 | 131.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 131.55 | 130.25 | 131.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 131.24 | 130.45 | 131.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 15:15:00 | 131.00 | 131.18 | 131.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 09:15:00 | 133.24 | 131.57 | 131.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 09:15:00 | 133.24 | 131.57 | 131.53 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 131.00 | 131.84 | 131.88 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 12:15:00 | 132.49 | 131.97 | 131.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 14:15:00 | 133.16 | 132.29 | 132.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 09:15:00 | 131.82 | 132.31 | 132.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 09:15:00 | 131.82 | 132.31 | 132.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 131.82 | 132.31 | 132.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:00:00 | 131.82 | 132.31 | 132.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 132.04 | 132.25 | 132.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:45:00 | 132.61 | 132.25 | 132.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 132.79 | 132.36 | 132.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:30:00 | 132.09 | 132.36 | 132.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 133.09 | 132.51 | 132.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:45:00 | 132.23 | 132.51 | 132.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 132.41 | 132.69 | 132.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:00:00 | 132.41 | 132.69 | 132.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 132.83 | 132.72 | 132.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:45:00 | 132.69 | 132.72 | 132.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 132.46 | 132.67 | 132.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:45:00 | 132.18 | 132.67 | 132.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 133.27 | 132.79 | 132.56 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 130.37 | 132.45 | 132.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 12:15:00 | 129.58 | 131.23 | 131.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 11:15:00 | 131.10 | 130.33 | 131.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 11:15:00 | 131.10 | 130.33 | 131.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 131.10 | 130.33 | 131.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 11:45:00 | 131.31 | 130.33 | 131.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 132.00 | 130.66 | 131.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:00:00 | 132.00 | 130.66 | 131.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 131.21 | 130.77 | 131.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 130.59 | 130.97 | 131.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 10:45:00 | 130.75 | 131.05 | 131.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 124.06 | 127.23 | 128.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 124.21 | 127.23 | 128.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-05 09:15:00 | 117.67 | 119.64 | 122.26 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 53 — BUY (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 10:15:00 | 116.70 | 112.84 | 112.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 11:15:00 | 117.60 | 113.79 | 113.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 112.24 | 115.20 | 114.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 112.24 | 115.20 | 114.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 112.24 | 115.20 | 114.27 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 109.51 | 113.02 | 113.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 105.51 | 110.44 | 111.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 12:15:00 | 110.37 | 107.77 | 109.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 12:15:00 | 110.37 | 107.77 | 109.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 110.37 | 107.77 | 109.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:45:00 | 110.58 | 107.77 | 109.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 108.79 | 107.98 | 109.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 14:15:00 | 107.89 | 107.98 | 109.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 106.55 | 108.93 | 109.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 102.50 | 103.88 | 105.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 101.22 | 103.88 | 105.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 15:15:00 | 101.60 | 101.25 | 103.05 | SL hit (close>ema200) qty=0.50 sl=101.25 alert=retest2 |

### Cycle 55 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 104.22 | 102.94 | 102.89 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 12:15:00 | 102.10 | 102.79 | 102.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 09:15:00 | 99.88 | 101.94 | 102.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 96.83 | 96.34 | 98.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 12:15:00 | 97.83 | 96.64 | 97.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 97.83 | 96.64 | 97.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 13:00:00 | 97.83 | 96.64 | 97.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 13:15:00 | 97.43 | 96.80 | 97.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 93.10 | 96.86 | 97.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 13:45:00 | 95.93 | 95.35 | 96.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 14:30:00 | 96.25 | 95.71 | 96.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 95.01 | 95.94 | 96.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 97.39 | 96.21 | 96.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:30:00 | 98.55 | 96.21 | 96.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-06 11:15:00 | 99.87 | 96.94 | 96.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 99.87 | 96.94 | 96.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 105.40 | 100.25 | 98.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 104.50 | 104.61 | 102.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 104.50 | 104.61 | 102.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 106.52 | 107.87 | 106.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:30:00 | 107.21 | 107.11 | 106.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 107.96 | 106.91 | 106.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 10:30:00 | 107.30 | 107.48 | 107.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 09:45:00 | 107.25 | 107.91 | 107.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 107.90 | 107.91 | 107.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 11:45:00 | 108.40 | 107.89 | 107.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 12:45:00 | 108.65 | 107.89 | 107.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 13:15:00 | 106.92 | 107.70 | 107.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-04-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 13:15:00 | 106.92 | 107.70 | 107.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 14:15:00 | 106.64 | 107.49 | 107.64 | Break + close below crossover candle low |

### Cycle 59 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 109.78 | 107.82 | 107.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 13:15:00 | 110.43 | 109.10 | 108.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 09:15:00 | 112.15 | 112.66 | 111.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 10:15:00 | 111.53 | 112.44 | 111.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 111.53 | 112.44 | 111.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:45:00 | 111.26 | 112.44 | 111.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 110.98 | 112.15 | 111.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:45:00 | 110.85 | 112.15 | 111.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 12:15:00 | 111.26 | 111.97 | 111.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 13:15:00 | 111.74 | 111.97 | 111.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 14:15:00 | 110.20 | 111.47 | 111.15 | SL hit (close<static) qty=1.00 sl=110.86 alert=retest2 |

### Cycle 60 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 109.06 | 110.80 | 110.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 107.40 | 110.12 | 110.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 112.41 | 108.99 | 109.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 112.41 | 108.99 | 109.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 112.41 | 108.99 | 109.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 113.05 | 108.99 | 109.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 112.16 | 109.62 | 109.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 112.16 | 109.62 | 109.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 113.05 | 110.31 | 110.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 113.35 | 110.92 | 110.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 09:15:00 | 110.88 | 111.31 | 110.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 09:15:00 | 110.88 | 111.31 | 110.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 110.88 | 111.31 | 110.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 10:15:00 | 112.18 | 111.31 | 110.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 10:45:00 | 112.31 | 111.47 | 110.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 11:45:00 | 112.16 | 111.58 | 111.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 13:15:00 | 112.03 | 111.65 | 111.12 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 112.05 | 111.73 | 111.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:45:00 | 111.50 | 111.73 | 111.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 112.40 | 112.02 | 111.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:45:00 | 111.52 | 112.02 | 111.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Target hit | 2026-04-29 13:15:00 | 123.40 | 116.94 | 114.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 13:15:00 | 118.46 | 120.05 | 120.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 15:15:00 | 117.85 | 119.34 | 119.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 14:15:00 | 121.08 | 118.97 | 119.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 14:15:00 | 121.08 | 118.97 | 119.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 121.08 | 118.97 | 119.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 15:00:00 | 121.08 | 118.97 | 119.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 120.51 | 119.28 | 119.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 120.87 | 119.28 | 119.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 118.93 | 119.41 | 119.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 09:30:00 | 118.22 | 118.92 | 119.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 12:00:00 | 118.35 | 118.85 | 119.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 12:30:00 | 118.18 | 118.73 | 119.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-20 09:15:00 | 183.65 | 2025-05-21 11:15:00 | 180.36 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-05-20 14:45:00 | 183.75 | 2025-05-21 11:15:00 | 180.36 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-05-21 10:00:00 | 183.31 | 2025-05-21 11:15:00 | 180.36 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-05-23 14:15:00 | 180.56 | 2025-05-27 14:15:00 | 171.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-23 14:15:00 | 180.56 | 2025-05-28 12:15:00 | 173.57 | STOP_HIT | 0.50 | 3.87% |
| BUY | retest2 | 2025-06-19 14:15:00 | 169.90 | 2025-06-20 11:15:00 | 168.14 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-06-24 11:00:00 | 168.17 | 2025-06-24 14:15:00 | 169.10 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-06-30 09:15:00 | 172.92 | 2025-06-30 09:15:00 | 170.41 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-07-02 12:00:00 | 166.63 | 2025-07-04 09:15:00 | 172.91 | STOP_HIT | 1.00 | -3.77% |
| SELL | retest2 | 2025-07-03 15:15:00 | 166.60 | 2025-07-04 09:15:00 | 172.91 | STOP_HIT | 1.00 | -3.79% |
| BUY | retest2 | 2025-07-09 09:15:00 | 174.49 | 2025-07-09 14:15:00 | 171.06 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-07-09 11:00:00 | 172.05 | 2025-07-09 14:15:00 | 171.06 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-07-11 10:30:00 | 169.34 | 2025-07-15 09:15:00 | 172.67 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-07-17 09:15:00 | 173.83 | 2025-07-22 11:15:00 | 174.58 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2025-08-06 09:15:00 | 159.66 | 2025-08-12 15:15:00 | 152.36 | PARTIAL | 0.50 | 4.57% |
| SELL | retest2 | 2025-08-06 14:45:00 | 160.38 | 2025-08-13 09:15:00 | 151.68 | PARTIAL | 0.50 | 5.43% |
| SELL | retest2 | 2025-08-07 14:15:00 | 160.02 | 2025-08-13 09:15:00 | 152.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-06 09:15:00 | 159.66 | 2025-08-13 13:15:00 | 156.15 | STOP_HIT | 0.50 | 2.20% |
| SELL | retest2 | 2025-08-06 14:45:00 | 160.38 | 2025-08-13 13:15:00 | 156.15 | STOP_HIT | 0.50 | 2.64% |
| SELL | retest2 | 2025-08-07 14:15:00 | 160.02 | 2025-08-13 13:15:00 | 156.15 | STOP_HIT | 0.50 | 2.42% |
| BUY | retest2 | 2025-09-01 09:15:00 | 175.13 | 2025-09-15 10:15:00 | 183.49 | STOP_HIT | 1.00 | 4.77% |
| BUY | retest2 | 2025-09-01 13:00:00 | 176.30 | 2025-09-15 10:15:00 | 183.49 | STOP_HIT | 1.00 | 4.08% |
| BUY | retest2 | 2025-09-02 09:30:00 | 175.66 | 2025-09-15 10:15:00 | 183.49 | STOP_HIT | 1.00 | 4.46% |
| BUY | retest2 | 2025-09-02 10:45:00 | 175.14 | 2025-09-15 10:15:00 | 183.49 | STOP_HIT | 1.00 | 4.77% |
| BUY | retest2 | 2025-09-04 09:30:00 | 179.66 | 2025-09-15 10:15:00 | 183.49 | STOP_HIT | 1.00 | 2.13% |
| SELL | retest2 | 2025-09-22 13:30:00 | 176.58 | 2025-09-24 09:15:00 | 180.10 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-09-30 09:15:00 | 169.31 | 2025-10-01 15:15:00 | 170.90 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-09-30 10:00:00 | 169.33 | 2025-10-01 15:15:00 | 170.90 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-10-01 09:15:00 | 168.30 | 2025-10-01 15:15:00 | 170.90 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-10-01 13:45:00 | 169.73 | 2025-10-01 15:15:00 | 170.90 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-10-09 11:30:00 | 162.50 | 2025-10-15 15:15:00 | 164.00 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-10-13 12:00:00 | 162.80 | 2025-10-15 15:15:00 | 164.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-10-13 14:00:00 | 162.70 | 2025-10-15 15:15:00 | 164.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-10-13 14:30:00 | 162.71 | 2025-10-15 15:15:00 | 164.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-10-14 11:00:00 | 162.30 | 2025-10-15 15:15:00 | 164.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-10-15 10:30:00 | 162.26 | 2025-10-15 15:15:00 | 164.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-10-21 13:45:00 | 166.30 | 2025-10-24 09:15:00 | 165.50 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-11-03 14:15:00 | 161.88 | 2025-11-07 09:15:00 | 153.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 14:45:00 | 161.42 | 2025-11-07 09:15:00 | 153.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 09:45:00 | 161.08 | 2025-11-07 09:15:00 | 153.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 12:15:00 | 161.62 | 2025-11-07 09:15:00 | 153.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 14:15:00 | 161.88 | 2025-11-10 11:15:00 | 150.32 | STOP_HIT | 0.50 | 7.14% |
| SELL | retest2 | 2025-11-03 14:45:00 | 161.42 | 2025-11-10 11:15:00 | 150.32 | STOP_HIT | 0.50 | 6.88% |
| SELL | retest2 | 2025-11-04 09:45:00 | 161.08 | 2025-11-10 11:15:00 | 150.32 | STOP_HIT | 0.50 | 6.68% |
| SELL | retest2 | 2025-11-04 12:15:00 | 161.62 | 2025-11-10 11:15:00 | 150.32 | STOP_HIT | 0.50 | 6.99% |
| SELL | retest2 | 2025-11-17 11:15:00 | 140.48 | 2025-11-17 15:15:00 | 142.70 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-11-17 12:00:00 | 140.72 | 2025-11-17 15:15:00 | 142.70 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-11-17 12:30:00 | 140.71 | 2025-11-17 15:15:00 | 142.70 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-11-17 13:30:00 | 140.42 | 2025-11-17 15:15:00 | 142.70 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-11-20 14:15:00 | 147.00 | 2025-11-21 09:15:00 | 144.30 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-11-26 13:45:00 | 139.63 | 2025-12-02 09:15:00 | 132.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 15:00:00 | 139.15 | 2025-12-02 09:15:00 | 132.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 09:30:00 | 138.98 | 2025-12-02 09:15:00 | 132.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 13:45:00 | 139.63 | 2025-12-03 09:15:00 | 135.00 | STOP_HIT | 0.50 | 3.32% |
| SELL | retest2 | 2025-11-26 15:00:00 | 139.15 | 2025-12-03 09:15:00 | 135.00 | STOP_HIT | 0.50 | 2.98% |
| SELL | retest2 | 2025-11-27 09:30:00 | 138.98 | 2025-12-03 09:15:00 | 135.00 | STOP_HIT | 0.50 | 2.86% |
| BUY | retest2 | 2025-12-30 10:15:00 | 144.91 | 2025-12-30 14:15:00 | 142.11 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-12-31 11:00:00 | 145.00 | 2026-01-02 09:15:00 | 159.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-31 13:15:00 | 148.45 | 2026-01-02 09:15:00 | 159.50 | TARGET_HIT | 1.00 | 7.44% |
| BUY | retest2 | 2026-01-01 13:45:00 | 145.00 | 2026-01-05 11:15:00 | 143.74 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2026-01-16 11:00:00 | 132.40 | 2026-01-20 11:15:00 | 126.11 | PARTIAL | 0.50 | 4.75% |
| SELL | retest2 | 2026-01-16 12:30:00 | 132.75 | 2026-01-20 11:15:00 | 126.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 13:00:00 | 132.75 | 2026-01-20 11:15:00 | 126.19 | PARTIAL | 0.50 | 4.94% |
| SELL | retest2 | 2026-01-16 14:15:00 | 132.83 | 2026-01-20 14:15:00 | 125.78 | PARTIAL | 0.50 | 5.31% |
| SELL | retest2 | 2026-01-16 11:00:00 | 132.40 | 2026-01-21 12:15:00 | 126.64 | STOP_HIT | 0.50 | 4.35% |
| SELL | retest2 | 2026-01-16 12:30:00 | 132.75 | 2026-01-21 12:15:00 | 126.64 | STOP_HIT | 0.50 | 4.60% |
| SELL | retest2 | 2026-01-16 13:00:00 | 132.75 | 2026-01-21 12:15:00 | 126.64 | STOP_HIT | 0.50 | 4.60% |
| SELL | retest2 | 2026-01-16 14:15:00 | 132.83 | 2026-01-21 12:15:00 | 126.64 | STOP_HIT | 0.50 | 4.66% |
| SELL | retest2 | 2026-01-22 10:45:00 | 124.38 | 2026-01-23 10:15:00 | 118.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 10:45:00 | 124.38 | 2026-01-27 09:15:00 | 111.94 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-04 09:15:00 | 118.48 | 2026-02-05 09:15:00 | 130.33 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-17 15:15:00 | 131.00 | 2026-02-18 09:15:00 | 133.24 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2026-02-26 09:15:00 | 130.59 | 2026-03-02 09:15:00 | 124.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 10:45:00 | 130.75 | 2026-03-02 09:15:00 | 124.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 09:15:00 | 130.59 | 2026-03-05 09:15:00 | 117.67 | TARGET_HIT | 0.50 | 9.89% |
| SELL | retest2 | 2026-02-26 10:45:00 | 130.75 | 2026-03-05 10:15:00 | 117.53 | TARGET_HIT | 0.50 | 10.11% |
| SELL | retest2 | 2026-03-17 14:15:00 | 107.89 | 2026-03-23 09:15:00 | 102.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-19 09:15:00 | 106.55 | 2026-03-23 09:15:00 | 101.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-17 14:15:00 | 107.89 | 2026-03-23 15:15:00 | 101.60 | STOP_HIT | 0.50 | 5.83% |
| SELL | retest2 | 2026-03-19 09:15:00 | 106.55 | 2026-03-23 15:15:00 | 101.60 | STOP_HIT | 0.50 | 4.65% |
| SELL | retest2 | 2026-04-02 09:15:00 | 93.10 | 2026-04-06 11:15:00 | 99.87 | STOP_HIT | 1.00 | -7.27% |
| SELL | retest2 | 2026-04-02 13:45:00 | 95.93 | 2026-04-06 11:15:00 | 99.87 | STOP_HIT | 1.00 | -4.11% |
| SELL | retest2 | 2026-04-02 14:30:00 | 96.25 | 2026-04-06 11:15:00 | 99.87 | STOP_HIT | 1.00 | -3.76% |
| SELL | retest2 | 2026-04-06 09:15:00 | 95.01 | 2026-04-06 11:15:00 | 99.87 | STOP_HIT | 1.00 | -5.12% |
| BUY | retest2 | 2026-04-13 13:30:00 | 107.21 | 2026-04-20 13:15:00 | 106.92 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2026-04-15 09:15:00 | 107.96 | 2026-04-20 13:15:00 | 106.92 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2026-04-16 10:30:00 | 107.30 | 2026-04-20 13:15:00 | 106.92 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2026-04-20 09:45:00 | 107.25 | 2026-04-20 13:15:00 | 106.92 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2026-04-20 11:45:00 | 108.40 | 2026-04-20 13:15:00 | 106.92 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-04-20 12:45:00 | 108.65 | 2026-04-20 13:15:00 | 106.92 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2026-04-23 13:15:00 | 111.74 | 2026-04-23 14:15:00 | 110.20 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-04-23 14:30:00 | 111.65 | 2026-04-23 15:15:00 | 110.30 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2026-04-28 10:15:00 | 112.18 | 2026-04-29 13:15:00 | 123.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-28 10:45:00 | 112.31 | 2026-04-29 13:15:00 | 123.38 | TARGET_HIT | 1.00 | 9.85% |
| BUY | retest2 | 2026-04-28 11:45:00 | 112.16 | 2026-04-29 13:15:00 | 123.23 | TARGET_HIT | 1.00 | 9.87% |
| BUY | retest2 | 2026-04-28 13:15:00 | 112.03 | 2026-04-29 14:15:00 | 123.54 | TARGET_HIT | 1.00 | 10.27% |
