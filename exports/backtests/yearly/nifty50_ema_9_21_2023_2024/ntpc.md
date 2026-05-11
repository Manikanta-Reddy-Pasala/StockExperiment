# NTPC (NTPC)

## Backtest Summary

- **Window:** 2023-03-13 09:15:00 → 2026-05-08 15:15:00 (5443 bars)
- **Last close:** 402.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 238 |
| ALERT1 | 166 |
| ALERT2 | 166 |
| ALERT2_SKIP | 79 |
| ALERT3 | 440 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 169 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 168 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 174 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 41 / 133
- **Target hits / Stop hits / Partials:** 2 / 168 / 4
- **Avg / median % per leg:** -0.10% / -0.70%
- **Sum % (uncompounded):** -17.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 90 | 22 | 24.4% | 2 | 88 | 0 | 0.19% | 17.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 90 | 22 | 24.4% | 2 | 88 | 0 | 0.19% | 17.3% |
| SELL (all) | 84 | 19 | 22.6% | 0 | 80 | 4 | -0.41% | -34.5% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.22% | 0.2% |
| SELL @ 3rd Alert (retest2) | 83 | 18 | 21.7% | 0 | 79 | 4 | -0.42% | -34.7% |
| retest1 (combined) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.22% | 0.2% |
| retest2 (combined) | 173 | 40 | 23.1% | 2 | 167 | 4 | -0.10% | -17.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-12 12:15:00 | 175.15 | 177.09 | 177.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-15 09:15:00 | 174.65 | 175.83 | 176.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-15 10:15:00 | 175.85 | 175.83 | 176.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-15 11:00:00 | 175.85 | 175.83 | 176.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-15 12:15:00 | 176.55 | 176.02 | 176.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-15 12:30:00 | 176.55 | 176.02 | 176.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-15 13:15:00 | 175.60 | 175.93 | 176.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-15 13:30:00 | 176.55 | 175.93 | 176.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 09:15:00 | 175.70 | 175.93 | 176.25 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 12:15:00 | 177.55 | 176.40 | 176.39 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 12:15:00 | 176.00 | 176.48 | 176.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-17 13:15:00 | 175.55 | 176.29 | 176.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-17 14:15:00 | 176.35 | 176.30 | 176.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-17 15:00:00 | 176.35 | 176.30 | 176.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 09:15:00 | 176.60 | 176.35 | 176.42 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2023-05-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-18 10:15:00 | 177.15 | 176.51 | 176.49 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2023-05-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 14:15:00 | 175.00 | 176.32 | 176.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-19 09:15:00 | 173.95 | 175.66 | 176.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-22 09:15:00 | 176.65 | 174.58 | 175.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 09:15:00 | 176.65 | 174.58 | 175.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 176.65 | 174.58 | 175.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 09:45:00 | 176.50 | 174.58 | 175.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 10:15:00 | 175.80 | 174.82 | 175.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-22 12:30:00 | 175.25 | 174.88 | 175.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-25 09:15:00 | 175.05 | 174.87 | 174.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2023-05-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 09:15:00 | 175.05 | 174.87 | 174.85 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-05-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 11:15:00 | 174.05 | 174.71 | 174.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-25 12:15:00 | 173.60 | 174.49 | 174.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-25 14:15:00 | 174.85 | 174.46 | 174.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 14:15:00 | 174.85 | 174.46 | 174.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 14:15:00 | 174.85 | 174.46 | 174.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-25 15:00:00 | 174.85 | 174.46 | 174.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 15:15:00 | 174.65 | 174.50 | 174.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-26 09:15:00 | 175.95 | 174.50 | 174.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 175.00 | 174.60 | 174.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-26 10:15:00 | 174.85 | 174.60 | 174.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-26 10:15:00 | 175.20 | 174.72 | 174.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2023-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 10:15:00 | 175.20 | 174.72 | 174.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 11:15:00 | 175.50 | 174.88 | 174.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-26 12:15:00 | 174.50 | 174.80 | 174.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-26 12:15:00 | 174.50 | 174.80 | 174.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 12:15:00 | 174.50 | 174.80 | 174.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 12:30:00 | 175.05 | 174.80 | 174.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 13:15:00 | 174.60 | 174.76 | 174.74 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-26 14:15:00 | 174.50 | 174.71 | 174.72 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-29 09:15:00 | 175.35 | 174.77 | 174.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-29 10:15:00 | 176.80 | 175.18 | 174.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 12:15:00 | 176.50 | 176.54 | 175.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-30 12:45:00 | 176.55 | 176.54 | 175.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 13:15:00 | 176.70 | 176.57 | 176.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 13:30:00 | 176.10 | 176.57 | 176.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 09:15:00 | 174.80 | 176.23 | 176.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 10:15:00 | 174.10 | 176.23 | 176.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2023-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 10:15:00 | 174.10 | 175.80 | 175.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-31 11:15:00 | 172.90 | 175.22 | 175.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-01 10:15:00 | 174.20 | 174.10 | 174.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-01 10:45:00 | 174.30 | 174.10 | 174.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 14:15:00 | 174.35 | 173.87 | 174.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 15:00:00 | 174.35 | 173.87 | 174.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 15:15:00 | 174.10 | 173.91 | 174.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-02 09:15:00 | 175.10 | 173.91 | 174.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 09:15:00 | 174.65 | 174.06 | 174.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-02 10:15:00 | 174.30 | 174.06 | 174.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-02 11:00:00 | 174.15 | 174.08 | 174.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-02 14:30:00 | 174.25 | 174.36 | 174.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-02 15:15:00 | 174.25 | 174.36 | 174.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 15:15:00 | 174.25 | 174.34 | 174.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-05 09:15:00 | 175.70 | 174.34 | 174.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-06-05 09:15:00 | 175.50 | 174.57 | 174.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2023-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-05 09:15:00 | 175.50 | 174.57 | 174.50 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-05 14:15:00 | 174.10 | 174.49 | 174.51 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-06 09:15:00 | 175.20 | 174.58 | 174.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-07 09:15:00 | 175.90 | 175.17 | 174.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-09 10:15:00 | 181.10 | 181.23 | 179.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-09 11:00:00 | 181.10 | 181.23 | 179.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 09:15:00 | 185.80 | 186.76 | 185.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 09:30:00 | 186.00 | 186.76 | 185.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 10:15:00 | 186.10 | 186.63 | 185.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 10:30:00 | 185.65 | 186.63 | 185.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 11:15:00 | 187.00 | 186.70 | 185.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 11:45:00 | 186.75 | 186.70 | 185.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 09:15:00 | 186.80 | 186.69 | 186.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-16 14:00:00 | 188.05 | 187.29 | 186.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-19 11:15:00 | 185.90 | 187.02 | 186.81 | SL hit (close<static) qty=1.00 sl=186.05 alert=retest2 |

### Cycle 15 — SELL (started 2023-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 13:15:00 | 186.05 | 186.67 | 186.68 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-20 14:15:00 | 187.50 | 186.47 | 186.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-20 15:15:00 | 188.00 | 186.78 | 186.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-21 10:15:00 | 187.00 | 187.01 | 186.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-21 11:00:00 | 187.00 | 187.01 | 186.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 11:15:00 | 187.20 | 187.05 | 186.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-21 13:45:00 | 187.90 | 187.23 | 186.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-22 09:15:00 | 185.80 | 186.98 | 186.89 | SL hit (close<static) qty=1.00 sl=186.65 alert=retest2 |

### Cycle 17 — SELL (started 2023-06-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 10:15:00 | 185.90 | 186.76 | 186.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 11:15:00 | 184.90 | 186.39 | 186.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-23 09:15:00 | 187.05 | 185.71 | 186.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-23 09:15:00 | 187.05 | 185.71 | 186.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 09:15:00 | 187.05 | 185.71 | 186.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-23 09:45:00 | 186.80 | 185.71 | 186.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 10:15:00 | 186.65 | 185.90 | 186.14 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2023-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-23 13:15:00 | 187.00 | 186.39 | 186.33 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-26 09:15:00 | 185.15 | 186.19 | 186.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-26 10:15:00 | 184.90 | 185.93 | 186.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-27 09:15:00 | 185.75 | 185.40 | 185.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-27 09:15:00 | 185.75 | 185.40 | 185.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 185.75 | 185.40 | 185.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:45:00 | 185.50 | 185.40 | 185.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 186.50 | 185.62 | 185.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 11:00:00 | 186.50 | 185.62 | 185.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 11:15:00 | 186.15 | 185.72 | 185.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 13:00:00 | 185.70 | 185.72 | 185.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-27 14:15:00 | 186.10 | 185.89 | 185.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2023-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 14:15:00 | 186.10 | 185.89 | 185.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 15:15:00 | 186.40 | 185.99 | 185.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 09:15:00 | 185.90 | 185.97 | 185.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-28 09:15:00 | 185.90 | 185.97 | 185.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 09:15:00 | 185.90 | 185.97 | 185.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-28 10:15:00 | 186.75 | 185.97 | 185.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-07 15:15:00 | 192.55 | 194.01 | 194.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2023-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 15:15:00 | 192.55 | 194.01 | 194.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 10:15:00 | 191.85 | 193.30 | 193.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 10:15:00 | 192.20 | 191.92 | 192.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-11 10:30:00 | 192.35 | 191.92 | 192.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 14:15:00 | 192.45 | 191.93 | 192.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 15:00:00 | 192.45 | 191.93 | 192.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 15:15:00 | 192.60 | 192.07 | 192.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-12 09:15:00 | 193.15 | 192.07 | 192.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 09:15:00 | 193.85 | 192.42 | 192.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-12 10:00:00 | 193.85 | 192.42 | 192.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2023-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 10:15:00 | 194.60 | 192.86 | 192.72 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-07-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 14:15:00 | 190.75 | 192.29 | 192.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-13 09:15:00 | 190.35 | 191.66 | 192.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 14:15:00 | 187.55 | 187.09 | 188.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-14 15:00:00 | 187.55 | 187.09 | 188.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 186.70 | 187.08 | 188.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-17 13:30:00 | 186.15 | 187.09 | 188.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-17 14:45:00 | 186.35 | 187.03 | 187.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-18 10:15:00 | 188.85 | 187.57 | 187.95 | SL hit (close>static) qty=1.00 sl=188.45 alert=retest2 |

### Cycle 24 — BUY (started 2023-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 09:15:00 | 195.40 | 189.00 | 188.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-25 09:15:00 | 199.25 | 196.04 | 194.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 09:15:00 | 221.80 | 222.79 | 217.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-02 09:45:00 | 223.15 | 222.79 | 217.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 11:15:00 | 217.90 | 221.33 | 218.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:45:00 | 217.85 | 221.33 | 218.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 12:15:00 | 218.30 | 220.73 | 218.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 13:15:00 | 217.75 | 220.73 | 218.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 13:15:00 | 217.60 | 220.10 | 218.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 13:45:00 | 217.75 | 220.10 | 218.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 14:15:00 | 218.90 | 219.86 | 218.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-03 09:15:00 | 221.70 | 219.73 | 218.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-03 14:45:00 | 220.10 | 220.28 | 219.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-04 15:15:00 | 218.00 | 219.38 | 219.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2023-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-04 15:15:00 | 218.00 | 219.38 | 219.41 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 09:15:00 | 220.40 | 219.59 | 219.50 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-07 12:15:00 | 218.15 | 219.46 | 219.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-07 14:15:00 | 217.95 | 218.99 | 219.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-08 14:15:00 | 217.50 | 216.97 | 217.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-08 15:00:00 | 217.50 | 216.97 | 217.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 15:15:00 | 218.15 | 217.21 | 217.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-09 09:15:00 | 218.20 | 217.21 | 217.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 09:15:00 | 216.95 | 217.15 | 217.82 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2023-08-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 15:15:00 | 218.80 | 217.97 | 217.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-10 09:15:00 | 220.60 | 218.50 | 218.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-10 10:15:00 | 218.00 | 218.40 | 218.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-10 10:15:00 | 218.00 | 218.40 | 218.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 10:15:00 | 218.00 | 218.40 | 218.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 11:00:00 | 218.00 | 218.40 | 218.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 11:15:00 | 217.70 | 218.26 | 218.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 12:00:00 | 217.70 | 218.26 | 218.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — SELL (started 2023-08-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 13:15:00 | 216.95 | 217.96 | 218.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 09:15:00 | 215.85 | 217.45 | 217.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 10:15:00 | 213.75 | 213.44 | 214.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-16 10:15:00 | 213.75 | 213.44 | 214.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 10:15:00 | 213.75 | 213.44 | 214.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 10:45:00 | 213.75 | 213.44 | 214.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 11:15:00 | 215.35 | 213.82 | 214.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 12:00:00 | 215.35 | 213.82 | 214.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 12:15:00 | 215.70 | 214.20 | 214.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 12:45:00 | 215.55 | 214.20 | 214.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2023-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 14:15:00 | 217.25 | 215.13 | 214.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-16 15:15:00 | 217.50 | 215.60 | 215.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-17 11:15:00 | 215.50 | 216.11 | 215.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 11:15:00 | 215.50 | 216.11 | 215.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 11:15:00 | 215.50 | 216.11 | 215.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-17 12:00:00 | 215.50 | 216.11 | 215.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 12:15:00 | 215.90 | 216.07 | 215.61 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2023-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 12:15:00 | 213.35 | 215.15 | 215.36 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 09:15:00 | 216.55 | 215.45 | 215.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-21 11:15:00 | 218.30 | 216.42 | 215.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 14:15:00 | 221.35 | 221.51 | 220.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-23 15:00:00 | 221.35 | 221.51 | 220.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 09:15:00 | 222.20 | 221.73 | 220.57 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2023-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 10:15:00 | 219.00 | 220.39 | 220.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 11:15:00 | 218.30 | 219.97 | 220.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 13:15:00 | 219.05 | 218.44 | 219.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 13:15:00 | 219.05 | 218.44 | 219.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 13:15:00 | 219.05 | 218.44 | 219.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 14:00:00 | 219.05 | 218.44 | 219.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 14:15:00 | 218.55 | 218.46 | 218.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 15:15:00 | 218.70 | 218.46 | 218.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 15:15:00 | 218.70 | 218.51 | 218.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 09:15:00 | 220.10 | 218.51 | 218.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 09:15:00 | 220.20 | 218.85 | 219.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 09:30:00 | 220.40 | 218.85 | 219.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 10:15:00 | 220.50 | 219.18 | 219.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 11:00:00 | 220.50 | 219.18 | 219.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2023-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 11:15:00 | 220.45 | 219.43 | 219.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 12:15:00 | 221.25 | 219.79 | 219.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 13:15:00 | 220.40 | 220.70 | 220.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-30 14:00:00 | 220.40 | 220.70 | 220.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 14:15:00 | 220.25 | 220.61 | 220.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-30 14:30:00 | 219.95 | 220.61 | 220.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 15:15:00 | 220.50 | 220.59 | 220.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-31 09:15:00 | 220.65 | 220.59 | 220.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-31 10:15:00 | 220.55 | 220.55 | 220.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-31 15:15:00 | 219.20 | 220.35 | 220.32 | SL hit (close<static) qty=1.00 sl=220.05 alert=retest2 |

### Cycle 35 — SELL (started 2023-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-01 09:15:00 | 219.75 | 220.23 | 220.27 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 10:15:00 | 223.00 | 220.78 | 220.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 11:15:00 | 228.05 | 222.23 | 221.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-05 11:15:00 | 233.25 | 233.30 | 230.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-05 11:30:00 | 233.00 | 233.30 | 230.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 10:15:00 | 230.25 | 232.78 | 231.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 11:00:00 | 230.25 | 232.78 | 231.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 11:15:00 | 230.75 | 232.37 | 231.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 11:45:00 | 229.60 | 232.37 | 231.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 09:15:00 | 229.95 | 231.34 | 231.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-07 10:00:00 | 229.95 | 231.34 | 231.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 10:15:00 | 230.60 | 231.19 | 231.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-07 10:30:00 | 230.45 | 231.19 | 231.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 11:15:00 | 231.00 | 231.15 | 231.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-07 12:45:00 | 234.65 | 232.16 | 231.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-12 14:15:00 | 234.65 | 238.34 | 238.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2023-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 14:15:00 | 234.65 | 238.34 | 238.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 15:15:00 | 234.35 | 237.54 | 238.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 12:15:00 | 237.55 | 236.87 | 237.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 12:15:00 | 237.55 | 236.87 | 237.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 12:15:00 | 237.55 | 236.87 | 237.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 12:30:00 | 237.70 | 236.87 | 237.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 13:15:00 | 236.85 | 236.86 | 237.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 11:30:00 | 236.40 | 237.29 | 237.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-14 14:15:00 | 237.70 | 237.33 | 237.46 | SL hit (close>static) qty=1.00 sl=237.65 alert=retest2 |

### Cycle 38 — BUY (started 2023-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-18 10:15:00 | 239.75 | 237.50 | 237.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-18 14:15:00 | 241.30 | 239.29 | 238.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-20 13:15:00 | 241.55 | 241.75 | 240.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-20 13:45:00 | 241.90 | 241.75 | 240.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 09:15:00 | 240.45 | 241.53 | 240.48 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-09-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 13:15:00 | 237.60 | 239.76 | 239.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-22 09:15:00 | 236.55 | 238.95 | 239.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 12:15:00 | 238.65 | 238.59 | 239.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-22 13:00:00 | 238.65 | 238.59 | 239.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 09:15:00 | 238.00 | 238.26 | 238.81 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2023-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 09:15:00 | 242.70 | 239.62 | 239.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-26 10:15:00 | 243.05 | 240.31 | 239.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-26 14:15:00 | 240.35 | 240.64 | 240.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-26 15:00:00 | 240.35 | 240.64 | 240.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 15:15:00 | 240.00 | 240.51 | 240.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-27 09:15:00 | 239.50 | 240.51 | 240.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 238.85 | 240.18 | 239.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-27 10:00:00 | 238.85 | 240.18 | 239.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 10:15:00 | 239.20 | 239.98 | 239.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-27 10:45:00 | 238.50 | 239.98 | 239.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — SELL (started 2023-09-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-27 11:15:00 | 238.45 | 239.68 | 239.71 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-09-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-28 09:15:00 | 240.80 | 239.68 | 239.65 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2023-09-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 12:15:00 | 238.80 | 239.60 | 239.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 13:15:00 | 238.00 | 239.28 | 239.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-28 15:15:00 | 239.40 | 238.90 | 239.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 15:15:00 | 239.40 | 238.90 | 239.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 15:15:00 | 239.40 | 238.90 | 239.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 09:15:00 | 246.80 | 238.90 | 239.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2023-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 09:15:00 | 247.50 | 240.62 | 240.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-29 12:15:00 | 247.95 | 243.70 | 241.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 10:15:00 | 242.80 | 244.42 | 242.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 10:15:00 | 242.80 | 244.42 | 242.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 10:15:00 | 242.80 | 244.42 | 242.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 11:00:00 | 242.80 | 244.42 | 242.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 11:15:00 | 242.40 | 244.02 | 242.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 11:45:00 | 241.95 | 244.02 | 242.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 12:15:00 | 242.20 | 243.66 | 242.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 12:45:00 | 241.90 | 243.66 | 242.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 13:15:00 | 241.80 | 243.28 | 242.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 14:00:00 | 241.80 | 243.28 | 242.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 14:15:00 | 241.15 | 242.86 | 242.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 14:45:00 | 240.90 | 242.86 | 242.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2023-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 09:15:00 | 236.60 | 241.39 | 241.96 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 14:15:00 | 236.10 | 235.34 | 235.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 09:15:00 | 238.05 | 236.05 | 235.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-16 09:15:00 | 241.65 | 241.82 | 240.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-16 10:00:00 | 241.65 | 241.82 | 240.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 10:15:00 | 244.00 | 245.09 | 244.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 11:00:00 | 244.00 | 245.09 | 244.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 11:15:00 | 243.50 | 244.78 | 243.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 12:00:00 | 243.50 | 244.78 | 243.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 12:15:00 | 243.50 | 244.52 | 243.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 12:45:00 | 243.00 | 244.52 | 243.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 13:15:00 | 243.65 | 244.35 | 243.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 14:15:00 | 243.25 | 244.35 | 243.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2023-10-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 15:15:00 | 242.35 | 243.55 | 243.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 09:15:00 | 240.35 | 242.91 | 243.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-20 13:15:00 | 240.30 | 239.83 | 240.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-20 14:00:00 | 240.30 | 239.83 | 240.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 14:15:00 | 241.35 | 240.13 | 240.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-20 14:45:00 | 241.40 | 240.13 | 240.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 15:15:00 | 241.15 | 240.34 | 240.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 09:15:00 | 240.35 | 240.34 | 240.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:15:00 | 228.33 | 232.08 | 234.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-26 14:15:00 | 231.45 | 231.20 | 233.30 | SL hit (close>ema200) qty=0.50 sl=231.20 alert=retest2 |

### Cycle 48 — BUY (started 2023-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 12:15:00 | 236.55 | 234.25 | 234.12 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2023-11-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 13:15:00 | 233.10 | 234.61 | 234.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 14:15:00 | 232.30 | 234.15 | 234.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 09:15:00 | 234.25 | 233.86 | 234.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 09:15:00 | 234.25 | 233.86 | 234.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 09:15:00 | 234.25 | 233.86 | 234.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 09:30:00 | 234.15 | 233.86 | 234.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 10:15:00 | 235.45 | 234.18 | 234.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 11:00:00 | 235.45 | 234.18 | 234.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 11:15:00 | 234.85 | 234.31 | 234.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 11:30:00 | 235.25 | 234.31 | 234.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 12:15:00 | 235.40 | 234.53 | 234.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 12:30:00 | 235.35 | 234.53 | 234.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2023-11-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 13:15:00 | 235.45 | 234.71 | 234.62 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2023-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 14:15:00 | 234.35 | 234.59 | 234.62 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2023-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-06 09:15:00 | 236.70 | 234.92 | 234.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 11:15:00 | 237.00 | 235.57 | 235.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-06 14:15:00 | 235.85 | 236.01 | 235.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-06 15:00:00 | 235.85 | 236.01 | 235.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 09:15:00 | 236.60 | 236.12 | 235.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-07 10:30:00 | 237.80 | 236.59 | 235.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 12:15:00 | 237.45 | 237.60 | 236.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-09 09:45:00 | 237.60 | 237.12 | 236.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-09 10:30:00 | 237.15 | 237.35 | 237.07 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 14:15:00 | 238.00 | 238.06 | 237.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 15:00:00 | 238.00 | 238.06 | 237.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 09:15:00 | 239.05 | 238.20 | 237.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-10 10:15:00 | 239.25 | 238.20 | 237.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-21 10:15:00 | 248.05 | 250.48 | 250.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2023-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-21 10:15:00 | 248.05 | 250.48 | 250.52 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2023-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-22 10:15:00 | 252.30 | 250.69 | 250.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-22 14:15:00 | 254.00 | 251.63 | 251.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-23 14:15:00 | 252.75 | 252.78 | 252.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-23 14:45:00 | 252.95 | 252.78 | 252.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 256.65 | 257.38 | 256.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 09:45:00 | 256.55 | 257.38 | 256.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 12:15:00 | 257.50 | 257.46 | 256.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-30 12:30:00 | 256.90 | 257.46 | 256.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 13:15:00 | 283.30 | 284.74 | 282.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 13:45:00 | 282.40 | 284.74 | 282.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 14:15:00 | 284.95 | 284.78 | 282.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-11 09:15:00 | 287.25 | 284.73 | 283.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-11 13:00:00 | 286.70 | 285.62 | 284.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-13 09:15:00 | 290.55 | 284.74 | 284.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-21 09:15:00 | 297.50 | 303.55 | 304.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2023-12-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-21 09:15:00 | 297.50 | 303.55 | 304.10 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2023-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 11:15:00 | 305.00 | 303.29 | 303.15 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2023-12-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-22 12:15:00 | 302.10 | 303.05 | 303.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-22 13:15:00 | 301.55 | 302.75 | 302.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 14:15:00 | 303.00 | 302.80 | 302.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 14:15:00 | 303.00 | 302.80 | 302.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 14:15:00 | 303.00 | 302.80 | 302.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 14:45:00 | 302.85 | 302.80 | 302.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 15:15:00 | 302.75 | 302.79 | 302.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-26 09:15:00 | 306.60 | 302.79 | 302.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2023-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 09:15:00 | 308.90 | 304.01 | 303.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-28 10:15:00 | 312.70 | 308.24 | 307.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 09:15:00 | 308.60 | 311.35 | 309.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 09:15:00 | 308.60 | 311.35 | 309.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 308.60 | 311.35 | 309.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-29 10:00:00 | 308.60 | 311.35 | 309.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 10:15:00 | 309.95 | 311.07 | 309.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 12:00:00 | 311.40 | 311.14 | 309.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 13:00:00 | 311.45 | 311.20 | 309.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 13:30:00 | 311.35 | 311.22 | 309.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 09:15:00 | 312.95 | 310.98 | 310.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 09:15:00 | 310.30 | 310.84 | 310.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 09:30:00 | 308.60 | 310.84 | 310.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 10:15:00 | 312.20 | 311.11 | 310.29 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-01-02 09:15:00 | 307.90 | 309.82 | 309.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2024-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 09:15:00 | 307.90 | 309.82 | 309.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-02 10:15:00 | 303.80 | 308.62 | 309.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-02 15:15:00 | 306.85 | 306.48 | 307.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-03 09:15:00 | 306.10 | 306.48 | 307.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 09:15:00 | 304.80 | 306.14 | 307.56 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2024-01-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-04 09:15:00 | 317.15 | 308.38 | 307.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 11:15:00 | 319.70 | 312.22 | 309.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-05 11:15:00 | 316.05 | 316.29 | 313.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-05 12:15:00 | 314.85 | 316.29 | 313.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 13:15:00 | 313.40 | 315.56 | 313.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-05 14:00:00 | 313.40 | 315.56 | 313.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 14:15:00 | 314.85 | 315.42 | 313.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-08 09:45:00 | 316.85 | 315.72 | 314.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-08 11:00:00 | 315.50 | 315.67 | 314.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-08 12:00:00 | 315.95 | 315.73 | 314.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-10 10:15:00 | 312.10 | 316.53 | 316.39 | SL hit (close<static) qty=1.00 sl=312.70 alert=retest2 |

### Cycle 61 — SELL (started 2024-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 11:15:00 | 310.20 | 315.27 | 315.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-10 12:15:00 | 309.00 | 314.01 | 315.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-10 14:15:00 | 313.35 | 313.28 | 314.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-10 14:45:00 | 313.30 | 313.28 | 314.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 316.95 | 314.05 | 314.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 09:30:00 | 316.60 | 314.05 | 314.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 10:15:00 | 314.75 | 314.19 | 314.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 11:15:00 | 314.10 | 314.19 | 314.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-15 12:30:00 | 314.05 | 313.18 | 313.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-15 13:15:00 | 315.70 | 313.68 | 313.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2024-01-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 13:15:00 | 315.70 | 313.68 | 313.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-15 14:15:00 | 317.20 | 314.39 | 313.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 10:15:00 | 314.25 | 314.82 | 314.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 10:15:00 | 314.25 | 314.82 | 314.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 10:15:00 | 314.25 | 314.82 | 314.22 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2024-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 12:15:00 | 311.00 | 313.59 | 313.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 12:15:00 | 309.90 | 312.07 | 312.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 304.95 | 302.72 | 306.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-19 10:00:00 | 304.95 | 302.72 | 306.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 11:15:00 | 306.35 | 303.78 | 306.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 12:00:00 | 306.35 | 303.78 | 306.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 12:15:00 | 308.85 | 304.79 | 306.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 12:45:00 | 307.90 | 304.79 | 306.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 13:15:00 | 308.00 | 305.43 | 306.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-19 14:30:00 | 307.35 | 306.10 | 306.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-20 09:15:00 | 312.85 | 307.89 | 307.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2024-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 09:15:00 | 312.85 | 307.89 | 307.39 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 14:15:00 | 303.05 | 307.74 | 308.33 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-01-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 09:15:00 | 315.80 | 309.31 | 308.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 09:15:00 | 320.75 | 314.73 | 311.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 10:15:00 | 321.30 | 321.68 | 317.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-30 10:45:00 | 320.20 | 321.68 | 317.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 13:15:00 | 317.20 | 320.13 | 318.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 14:00:00 | 317.20 | 320.13 | 318.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 14:15:00 | 315.50 | 319.20 | 317.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 14:30:00 | 315.65 | 319.20 | 317.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 10:15:00 | 314.85 | 317.47 | 317.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 10:30:00 | 314.00 | 317.47 | 317.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2024-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-31 11:15:00 | 315.95 | 317.17 | 317.21 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-01 09:15:00 | 322.25 | 317.80 | 317.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-02 09:15:00 | 326.60 | 321.35 | 319.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-05 15:15:00 | 336.00 | 336.06 | 331.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-06 09:15:00 | 330.85 | 336.06 | 331.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 331.90 | 335.23 | 331.36 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2024-02-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 10:15:00 | 328.70 | 331.56 | 331.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 10:15:00 | 322.05 | 329.22 | 330.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 09:15:00 | 320.90 | 319.08 | 322.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-13 10:15:00 | 321.80 | 319.62 | 322.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 10:15:00 | 321.80 | 319.62 | 322.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 11:00:00 | 321.80 | 319.62 | 322.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 12:15:00 | 322.95 | 320.18 | 322.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 13:00:00 | 322.95 | 320.18 | 322.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 13:15:00 | 321.50 | 320.45 | 322.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-13 14:30:00 | 320.65 | 320.76 | 322.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-13 15:15:00 | 320.90 | 320.76 | 322.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 10:00:00 | 320.35 | 320.70 | 321.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-14 13:15:00 | 323.90 | 321.50 | 321.86 | SL hit (close>static) qty=1.00 sl=323.00 alert=retest2 |

### Cycle 70 — BUY (started 2024-02-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 14:15:00 | 327.35 | 322.67 | 322.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 15:15:00 | 328.45 | 323.83 | 322.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 14:15:00 | 337.85 | 338.06 | 333.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-16 15:00:00 | 337.85 | 338.06 | 333.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 13:15:00 | 338.60 | 342.38 | 341.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 14:00:00 | 338.60 | 342.38 | 341.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 14:15:00 | 336.90 | 341.29 | 340.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 15:00:00 | 336.90 | 341.29 | 340.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2024-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 15:15:00 | 335.65 | 340.16 | 340.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 09:15:00 | 332.40 | 338.61 | 339.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 14:15:00 | 339.45 | 336.42 | 337.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 14:15:00 | 339.45 | 336.42 | 337.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 339.45 | 336.42 | 337.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 15:00:00 | 339.45 | 336.42 | 337.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 15:15:00 | 339.15 | 336.97 | 337.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-23 09:15:00 | 337.85 | 336.97 | 337.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-01 09:15:00 | 341.95 | 336.12 | 335.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — BUY (started 2024-03-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 09:15:00 | 341.95 | 336.12 | 335.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 13:15:00 | 342.55 | 339.28 | 337.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 09:15:00 | 348.25 | 354.89 | 351.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 09:15:00 | 348.25 | 354.89 | 351.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 348.25 | 354.89 | 351.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:00:00 | 348.25 | 354.89 | 351.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 10:15:00 | 348.50 | 353.61 | 351.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 11:00:00 | 348.50 | 353.61 | 351.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 352.95 | 351.45 | 350.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-07 09:30:00 | 351.45 | 351.45 | 350.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 10:15:00 | 351.10 | 351.38 | 350.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-07 10:45:00 | 350.70 | 351.38 | 350.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 11:15:00 | 352.40 | 351.59 | 351.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-07 11:30:00 | 350.05 | 351.59 | 351.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 14:15:00 | 352.75 | 351.98 | 351.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-07 14:45:00 | 351.80 | 351.98 | 351.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 09:15:00 | 349.65 | 351.55 | 351.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 10:00:00 | 349.65 | 351.55 | 351.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 10:15:00 | 349.95 | 351.23 | 351.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 10:30:00 | 351.00 | 351.23 | 351.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2024-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 11:15:00 | 347.35 | 350.45 | 350.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 09:15:00 | 346.25 | 348.70 | 349.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-15 15:15:00 | 315.65 | 315.50 | 321.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-18 09:15:00 | 319.15 | 315.50 | 321.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 09:15:00 | 317.50 | 315.90 | 320.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 09:15:00 | 314.10 | 317.15 | 319.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-21 09:15:00 | 323.20 | 315.29 | 315.41 | SL hit (close>static) qty=1.00 sl=322.90 alert=retest2 |

### Cycle 74 — BUY (started 2024-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 10:15:00 | 323.90 | 317.02 | 316.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 14:15:00 | 325.70 | 321.62 | 318.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 14:15:00 | 324.20 | 324.26 | 321.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-22 15:00:00 | 324.20 | 324.26 | 321.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 326.40 | 324.65 | 322.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-26 11:00:00 | 327.40 | 325.20 | 322.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-04 09:15:00 | 360.14 | 349.76 | 345.22 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2024-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 10:15:00 | 360.95 | 362.26 | 362.32 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2024-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-15 11:15:00 | 365.80 | 362.97 | 362.63 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-16 09:15:00 | 360.25 | 362.28 | 362.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-18 14:15:00 | 350.75 | 356.87 | 358.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-23 09:15:00 | 346.00 | 345.67 | 348.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-23 09:30:00 | 347.05 | 345.67 | 348.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 11:15:00 | 347.25 | 346.05 | 348.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 11:30:00 | 348.40 | 346.05 | 348.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 09:15:00 | 348.95 | 346.96 | 348.06 | EMA400 retest candle locked (from downside) |

### Cycle 78 — BUY (started 2024-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 12:15:00 | 351.40 | 348.74 | 348.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 13:15:00 | 352.30 | 349.46 | 349.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 13:15:00 | 356.05 | 356.30 | 354.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-26 13:30:00 | 356.20 | 356.30 | 354.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 12:15:00 | 366.45 | 368.52 | 366.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 13:00:00 | 366.45 | 368.52 | 366.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 13:15:00 | 362.00 | 367.22 | 365.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 14:00:00 | 362.00 | 367.22 | 365.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 14:15:00 | 366.20 | 367.02 | 365.76 | EMA400 retest candle locked (from upside) |

### Cycle 79 — SELL (started 2024-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 09:15:00 | 357.25 | 364.59 | 364.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 11:15:00 | 355.60 | 361.58 | 363.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 353.75 | 351.99 | 355.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 10:00:00 | 353.75 | 351.99 | 355.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 11:15:00 | 356.65 | 353.10 | 355.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 12:00:00 | 356.65 | 353.10 | 355.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 12:15:00 | 357.60 | 354.00 | 355.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 13:00:00 | 357.60 | 354.00 | 355.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 13:15:00 | 355.25 | 354.25 | 355.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 14:15:00 | 354.75 | 354.25 | 355.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 09:15:00 | 353.30 | 354.72 | 355.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 10:45:00 | 353.60 | 351.83 | 353.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 11:45:00 | 354.65 | 352.29 | 353.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 13:15:00 | 354.60 | 352.82 | 353.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 13:45:00 | 355.05 | 352.82 | 353.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 14:15:00 | 355.55 | 353.37 | 353.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 15:00:00 | 355.55 | 353.37 | 353.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 11:15:00 | 352.10 | 352.04 | 352.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 12:00:00 | 352.10 | 352.04 | 352.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 13:15:00 | 353.50 | 352.16 | 352.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 14:00:00 | 353.50 | 352.16 | 352.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 14:15:00 | 350.75 | 351.88 | 352.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 14:30:00 | 353.45 | 351.88 | 352.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 355.60 | 352.56 | 352.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:45:00 | 357.00 | 352.56 | 352.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-05-14 10:15:00 | 356.35 | 353.32 | 353.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 356.35 | 353.32 | 353.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 09:15:00 | 358.65 | 355.46 | 354.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 10:15:00 | 358.25 | 359.37 | 357.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 10:45:00 | 358.50 | 359.37 | 357.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 359.80 | 359.46 | 357.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 11:30:00 | 357.65 | 359.46 | 357.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 357.65 | 359.10 | 357.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:00:00 | 357.65 | 359.10 | 357.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 354.15 | 358.11 | 357.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:45:00 | 354.80 | 358.11 | 357.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 361.80 | 358.85 | 357.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 362.60 | 359.28 | 358.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:45:00 | 364.20 | 360.09 | 358.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 12:15:00 | 370.20 | 371.61 | 371.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 12:15:00 | 370.20 | 371.61 | 371.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 09:15:00 | 367.75 | 370.35 | 371.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 362.10 | 359.79 | 361.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 13:15:00 | 362.10 | 359.79 | 361.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 362.10 | 359.79 | 361.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:00:00 | 362.10 | 359.79 | 361.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 358.80 | 359.59 | 361.61 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 385.70 | 364.82 | 363.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 10:15:00 | 388.20 | 369.50 | 365.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 371.30 | 381.47 | 374.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 371.30 | 381.47 | 374.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 371.30 | 381.47 | 374.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 368.00 | 381.47 | 374.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 352.65 | 375.70 | 372.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 352.65 | 375.70 | 372.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 320.95 | 364.75 | 368.22 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-06-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 10:15:00 | 353.40 | 349.94 | 349.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 11:15:00 | 355.85 | 351.12 | 350.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 10:15:00 | 369.45 | 370.13 | 367.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-13 11:00:00 | 369.45 | 370.13 | 367.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 14:15:00 | 370.40 | 369.97 | 368.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 14:30:00 | 369.10 | 369.97 | 368.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 369.35 | 369.84 | 368.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 09:30:00 | 371.20 | 368.93 | 368.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 11:15:00 | 371.45 | 369.31 | 368.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 11:45:00 | 370.95 | 369.55 | 368.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 09:15:00 | 364.40 | 368.70 | 368.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2024-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 09:15:00 | 364.40 | 368.70 | 368.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 13:15:00 | 363.55 | 366.25 | 367.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 361.05 | 360.48 | 362.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-21 10:00:00 | 361.05 | 360.48 | 362.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 11:15:00 | 361.50 | 360.71 | 362.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 11:30:00 | 361.95 | 360.71 | 362.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 359.50 | 360.32 | 361.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 15:00:00 | 359.50 | 360.32 | 361.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 361.90 | 360.33 | 361.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:00:00 | 361.90 | 360.33 | 361.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 362.30 | 360.72 | 361.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:00:00 | 362.30 | 360.72 | 361.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 362.90 | 361.16 | 361.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 12:00:00 | 362.90 | 361.16 | 361.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 364.10 | 361.75 | 362.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 13:00:00 | 364.10 | 361.75 | 362.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2024-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 14:15:00 | 362.90 | 362.25 | 362.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 15:15:00 | 364.10 | 362.62 | 362.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 09:15:00 | 360.70 | 362.24 | 362.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 09:15:00 | 360.70 | 362.24 | 362.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 360.70 | 362.24 | 362.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 10:00:00 | 360.70 | 362.24 | 362.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2024-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 10:15:00 | 359.05 | 361.60 | 361.95 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 12:15:00 | 362.30 | 361.57 | 361.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 14:15:00 | 365.15 | 362.72 | 362.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 14:15:00 | 377.65 | 378.54 | 373.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-28 15:00:00 | 377.65 | 378.54 | 373.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 370.60 | 377.11 | 373.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:30:00 | 370.35 | 377.11 | 373.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 371.25 | 375.94 | 373.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 11:30:00 | 372.40 | 375.05 | 373.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 13:15:00 | 369.30 | 373.06 | 372.59 | SL hit (close<static) qty=1.00 sl=369.55 alert=retest2 |

### Cycle 89 — SELL (started 2024-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 15:15:00 | 370.40 | 372.04 | 372.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 09:15:00 | 367.95 | 371.22 | 371.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-02 14:15:00 | 370.60 | 369.02 | 370.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 14:15:00 | 370.60 | 369.02 | 370.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 14:15:00 | 370.60 | 369.02 | 370.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 15:00:00 | 370.60 | 369.02 | 370.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 15:15:00 | 370.60 | 369.33 | 370.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 09:15:00 | 370.20 | 369.33 | 370.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 371.00 | 369.67 | 370.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 09:30:00 | 371.50 | 369.67 | 370.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 372.85 | 370.30 | 370.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 11:00:00 | 372.85 | 370.30 | 370.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2024-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 12:15:00 | 372.35 | 371.02 | 370.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 377.35 | 372.94 | 371.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 13:15:00 | 373.70 | 373.77 | 372.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-04 13:45:00 | 373.55 | 373.77 | 372.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 373.05 | 373.62 | 372.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 14:45:00 | 372.05 | 373.62 | 372.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 373.60 | 373.62 | 372.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:15:00 | 374.40 | 373.62 | 372.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 375.45 | 373.99 | 373.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 14:30:00 | 378.55 | 376.16 | 374.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 12:00:00 | 377.70 | 377.18 | 376.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 09:15:00 | 379.00 | 377.29 | 376.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 12:00:00 | 377.40 | 377.00 | 376.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 12:15:00 | 377.70 | 377.14 | 376.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 14:30:00 | 379.45 | 377.97 | 377.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 09:15:00 | 375.90 | 377.58 | 377.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2024-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 09:15:00 | 375.90 | 377.58 | 377.59 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 10:15:00 | 379.00 | 377.86 | 377.72 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 11:15:00 | 374.75 | 377.24 | 377.45 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2024-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 09:15:00 | 381.35 | 378.10 | 377.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 10:15:00 | 384.25 | 379.33 | 378.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 12:15:00 | 383.60 | 384.52 | 382.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-16 12:30:00 | 383.50 | 384.52 | 382.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 382.35 | 384.09 | 382.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:00:00 | 382.35 | 384.09 | 382.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 380.35 | 383.34 | 382.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 15:00:00 | 380.35 | 383.34 | 382.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 380.45 | 382.76 | 382.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 378.15 | 382.76 | 382.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 375.10 | 380.64 | 381.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 373.10 | 377.74 | 379.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 374.60 | 371.23 | 374.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 374.60 | 371.23 | 374.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 374.60 | 371.23 | 374.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 374.60 | 371.23 | 374.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 374.00 | 371.78 | 374.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 14:30:00 | 371.85 | 373.48 | 374.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 15:15:00 | 372.75 | 373.48 | 374.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 11:15:00 | 373.35 | 374.04 | 374.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 11:15:00 | 380.05 | 375.25 | 375.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 11:15:00 | 380.05 | 375.25 | 375.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 09:15:00 | 390.75 | 381.27 | 378.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 11:15:00 | 391.80 | 392.03 | 389.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-26 12:00:00 | 391.80 | 392.03 | 389.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 13:15:00 | 395.05 | 395.38 | 393.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 14:15:00 | 394.95 | 395.38 | 393.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 14:15:00 | 394.10 | 395.12 | 393.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 09:15:00 | 411.90 | 394.96 | 393.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 11:15:00 | 410.75 | 415.41 | 415.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 11:15:00 | 410.75 | 415.41 | 415.48 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2024-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 11:15:00 | 419.25 | 415.81 | 415.37 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 09:15:00 | 412.95 | 415.90 | 415.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 10:15:00 | 411.70 | 415.06 | 415.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 09:15:00 | 413.50 | 411.60 | 413.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 09:15:00 | 413.50 | 411.60 | 413.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 413.50 | 411.60 | 413.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 09:30:00 | 414.90 | 411.60 | 413.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 10:15:00 | 409.65 | 411.21 | 412.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 11:30:00 | 409.10 | 410.74 | 412.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 13:00:00 | 409.50 | 410.49 | 412.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 14:15:00 | 408.90 | 410.36 | 412.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:15:00 | 402.55 | 410.62 | 411.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 395.20 | 397.06 | 399.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 10:15:00 | 393.80 | 397.06 | 399.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 11:00:00 | 394.25 | 396.50 | 399.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 09:15:00 | 405.35 | 399.11 | 399.33 | SL hit (close>static) qty=1.00 sl=400.95 alert=retest2 |

### Cycle 100 — BUY (started 2024-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 10:15:00 | 402.95 | 399.88 | 399.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 09:15:00 | 406.10 | 402.88 | 401.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 09:15:00 | 405.35 | 407.27 | 405.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-22 09:15:00 | 405.35 | 407.27 | 405.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 405.35 | 407.27 | 405.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 10:00:00 | 405.35 | 407.27 | 405.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 405.15 | 406.84 | 405.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 11:00:00 | 405.15 | 406.84 | 405.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 11:15:00 | 405.50 | 406.58 | 405.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 12:15:00 | 404.80 | 406.58 | 405.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 12:15:00 | 403.85 | 406.03 | 405.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 13:00:00 | 403.85 | 406.03 | 405.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 13:15:00 | 404.90 | 405.80 | 405.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 13:30:00 | 404.50 | 405.80 | 405.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 403.45 | 405.33 | 405.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 14:45:00 | 402.80 | 405.33 | 405.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2024-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 15:15:00 | 403.85 | 405.04 | 405.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 14:15:00 | 401.85 | 403.97 | 404.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 09:15:00 | 406.50 | 404.44 | 404.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 09:15:00 | 406.50 | 404.44 | 404.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 406.50 | 404.44 | 404.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 10:00:00 | 406.50 | 404.44 | 404.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — BUY (started 2024-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 10:15:00 | 406.75 | 404.90 | 404.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 11:15:00 | 411.40 | 406.20 | 405.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-27 10:15:00 | 411.25 | 411.62 | 409.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-27 11:00:00 | 411.25 | 411.62 | 409.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 12:15:00 | 410.20 | 411.09 | 409.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 10:15:00 | 412.00 | 410.72 | 409.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 12:30:00 | 412.50 | 410.85 | 409.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 13:15:00 | 412.30 | 410.85 | 409.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 14:15:00 | 409.00 | 410.85 | 410.13 | SL hit (close<static) qty=1.00 sl=409.10 alert=retest2 |

### Cycle 103 — SELL (started 2024-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 12:15:00 | 409.45 | 409.78 | 409.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 13:15:00 | 405.30 | 408.89 | 409.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 14:15:00 | 409.95 | 409.10 | 409.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 14:15:00 | 409.95 | 409.10 | 409.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 409.95 | 409.10 | 409.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 15:00:00 | 409.95 | 409.10 | 409.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 411.10 | 409.50 | 409.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 414.65 | 409.50 | 409.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 09:15:00 | 411.70 | 409.94 | 409.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 13:15:00 | 417.40 | 413.29 | 411.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 09:15:00 | 414.65 | 414.65 | 412.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 09:30:00 | 413.90 | 414.65 | 412.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 414.30 | 414.58 | 412.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 10:30:00 | 413.45 | 414.58 | 412.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 411.85 | 414.04 | 412.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:00:00 | 411.85 | 414.04 | 412.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 409.70 | 413.17 | 412.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 13:00:00 | 409.70 | 413.17 | 412.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 13:15:00 | 411.30 | 412.79 | 412.39 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2024-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 15:15:00 | 410.15 | 411.82 | 411.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 12:15:00 | 409.00 | 410.53 | 411.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 11:15:00 | 406.00 | 405.49 | 407.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-05 12:00:00 | 406.00 | 405.49 | 407.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 12:15:00 | 405.95 | 405.58 | 407.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 12:45:00 | 407.95 | 405.58 | 407.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 393.30 | 391.55 | 394.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 13:30:00 | 390.90 | 393.33 | 394.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 12:15:00 | 397.40 | 393.98 | 394.08 | SL hit (close>static) qty=1.00 sl=396.90 alert=retest2 |

### Cycle 106 — BUY (started 2024-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 13:15:00 | 400.25 | 395.24 | 394.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 14:15:00 | 405.00 | 397.19 | 395.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 13:15:00 | 401.65 | 401.68 | 398.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 14:00:00 | 401.65 | 401.68 | 398.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 417.85 | 421.96 | 418.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:45:00 | 420.35 | 421.96 | 418.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 424.15 | 422.40 | 419.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 09:15:00 | 428.65 | 423.02 | 420.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 12:15:00 | 436.60 | 439.66 | 439.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2024-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 12:15:00 | 436.60 | 439.66 | 439.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 09:15:00 | 430.80 | 436.15 | 437.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 11:15:00 | 438.00 | 435.99 | 437.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 11:15:00 | 438.00 | 435.99 | 437.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 438.00 | 435.99 | 437.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 12:00:00 | 438.00 | 435.99 | 437.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 431.20 | 435.03 | 436.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 12:30:00 | 433.20 | 435.03 | 436.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 424.80 | 420.35 | 425.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 424.80 | 420.35 | 425.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 421.35 | 420.55 | 424.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:30:00 | 424.40 | 420.55 | 424.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 420.40 | 420.85 | 423.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 15:00:00 | 418.20 | 420.66 | 422.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-10 09:15:00 | 427.80 | 421.95 | 422.72 | SL hit (close>static) qty=1.00 sl=424.45 alert=retest2 |

### Cycle 108 — BUY (started 2024-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 10:15:00 | 429.10 | 423.38 | 423.30 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 10:15:00 | 422.40 | 423.38 | 423.45 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 11:15:00 | 424.30 | 423.57 | 423.53 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 13:15:00 | 421.35 | 423.22 | 423.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 12:15:00 | 420.45 | 422.35 | 422.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 13:15:00 | 422.55 | 422.39 | 422.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 13:15:00 | 422.55 | 422.39 | 422.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 13:15:00 | 422.55 | 422.39 | 422.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 14:00:00 | 422.55 | 422.39 | 422.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 14:15:00 | 423.80 | 422.67 | 422.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 15:00:00 | 423.80 | 422.67 | 422.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2024-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 15:15:00 | 424.85 | 423.11 | 423.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 11:15:00 | 426.40 | 424.32 | 423.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 10:15:00 | 424.45 | 425.49 | 424.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 10:15:00 | 424.45 | 425.49 | 424.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 424.45 | 425.49 | 424.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:00:00 | 424.45 | 425.49 | 424.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 423.15 | 425.02 | 424.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:45:00 | 423.40 | 425.02 | 424.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 423.85 | 424.50 | 424.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 15:00:00 | 423.85 | 424.50 | 424.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 423.60 | 424.32 | 424.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:15:00 | 423.80 | 424.32 | 424.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 423.15 | 424.08 | 424.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 14:15:00 | 417.70 | 422.27 | 423.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 10:15:00 | 422.85 | 421.65 | 422.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 10:15:00 | 422.85 | 421.65 | 422.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 422.85 | 421.65 | 422.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 10:45:00 | 423.20 | 421.65 | 422.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 424.45 | 422.21 | 422.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:00:00 | 424.45 | 422.21 | 422.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 424.70 | 422.71 | 422.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:30:00 | 424.80 | 422.71 | 422.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — BUY (started 2024-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 14:15:00 | 424.95 | 423.44 | 423.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-21 13:15:00 | 425.75 | 423.89 | 423.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 15:15:00 | 423.15 | 423.97 | 423.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 15:15:00 | 423.15 | 423.97 | 423.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 15:15:00 | 423.15 | 423.97 | 423.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:15:00 | 424.05 | 423.97 | 423.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 422.10 | 423.59 | 423.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:00:00 | 422.10 | 423.59 | 423.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — SELL (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 10:15:00 | 420.70 | 423.02 | 423.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 14:15:00 | 415.50 | 420.01 | 421.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 09:15:00 | 411.65 | 410.66 | 414.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-24 10:00:00 | 411.65 | 410.66 | 414.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 413.10 | 411.15 | 414.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:30:00 | 414.35 | 411.15 | 414.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 400.95 | 400.94 | 405.38 | EMA400 retest candle locked (from downside) |

### Cycle 116 — BUY (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 11:15:00 | 406.85 | 405.93 | 405.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 13:15:00 | 409.20 | 406.70 | 406.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 12:15:00 | 409.00 | 409.81 | 408.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 13:00:00 | 409.00 | 409.81 | 408.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 13:15:00 | 405.85 | 409.02 | 408.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 14:00:00 | 405.85 | 409.02 | 408.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 408.45 | 408.90 | 408.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 11:15:00 | 409.80 | 408.75 | 408.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-31 13:15:00 | 406.00 | 407.69 | 407.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2024-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 13:15:00 | 406.00 | 407.69 | 407.87 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2024-11-01 17:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 17:15:00 | 412.70 | 408.87 | 408.38 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 399.40 | 407.45 | 407.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 10:15:00 | 395.55 | 405.07 | 406.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 401.50 | 400.86 | 402.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 14:00:00 | 401.50 | 400.86 | 402.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 402.85 | 401.26 | 402.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 15:00:00 | 402.85 | 401.26 | 402.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 403.05 | 401.62 | 402.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:15:00 | 407.55 | 401.62 | 402.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 406.00 | 402.49 | 403.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:30:00 | 405.90 | 402.49 | 403.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2024-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 12:15:00 | 409.20 | 404.56 | 403.94 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2024-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 15:15:00 | 403.95 | 404.67 | 404.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 09:15:00 | 401.20 | 403.97 | 404.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 11:15:00 | 399.25 | 398.41 | 400.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 11:15:00 | 399.25 | 398.41 | 400.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 399.25 | 398.41 | 400.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:45:00 | 399.60 | 398.41 | 400.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 370.00 | 368.68 | 373.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 15:00:00 | 365.75 | 368.55 | 371.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 373.55 | 364.80 | 364.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 373.55 | 364.80 | 364.65 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2024-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 11:15:00 | 362.05 | 366.50 | 366.59 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2024-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 11:15:00 | 369.45 | 366.00 | 365.85 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2024-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 12:15:00 | 364.35 | 366.42 | 366.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 13:15:00 | 363.00 | 365.74 | 366.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 15:15:00 | 365.25 | 365.08 | 365.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-29 09:15:00 | 363.55 | 365.08 | 365.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 360.60 | 364.19 | 365.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 10:30:00 | 359.90 | 362.36 | 363.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 10:45:00 | 360.10 | 359.83 | 361.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 12:15:00 | 365.95 | 361.47 | 361.88 | SL hit (close>static) qty=1.00 sl=365.40 alert=retest2 |

### Cycle 126 — BUY (started 2024-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 13:15:00 | 368.40 | 362.86 | 362.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 09:15:00 | 372.80 | 366.24 | 364.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 09:15:00 | 366.45 | 369.97 | 367.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 09:15:00 | 366.45 | 369.97 | 367.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 366.45 | 369.97 | 367.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:45:00 | 367.10 | 369.97 | 367.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 366.85 | 369.35 | 367.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:45:00 | 366.50 | 369.35 | 367.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 368.90 | 369.26 | 367.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:30:00 | 365.10 | 369.26 | 367.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 12:15:00 | 369.20 | 369.25 | 367.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 14:00:00 | 370.50 | 369.50 | 368.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 09:15:00 | 370.60 | 369.53 | 368.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 10:15:00 | 370.00 | 369.52 | 368.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 12:15:00 | 369.95 | 369.71 | 368.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 14:15:00 | 369.95 | 369.82 | 369.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 14:45:00 | 369.05 | 369.82 | 369.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 15:15:00 | 369.75 | 369.81 | 369.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 09:15:00 | 370.15 | 369.81 | 369.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 09:45:00 | 371.75 | 370.06 | 369.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 10:45:00 | 370.90 | 370.33 | 369.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 12:15:00 | 368.65 | 370.01 | 369.48 | SL hit (close<static) qty=1.00 sl=368.90 alert=retest2 |

### Cycle 127 — SELL (started 2024-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 13:15:00 | 367.15 | 369.08 | 369.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 14:15:00 | 365.45 | 367.64 | 368.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 13:15:00 | 357.10 | 356.36 | 359.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 14:00:00 | 357.10 | 356.36 | 359.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 340.00 | 339.23 | 342.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:30:00 | 343.80 | 339.23 | 342.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 339.80 | 339.35 | 342.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:00:00 | 339.10 | 339.30 | 342.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-27 10:15:00 | 338.90 | 336.19 | 335.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2024-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 10:15:00 | 338.90 | 336.19 | 335.93 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 10:15:00 | 334.15 | 335.81 | 336.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 12:15:00 | 333.70 | 335.29 | 335.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 13:15:00 | 333.60 | 331.99 | 333.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 13:15:00 | 333.60 | 331.99 | 333.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 333.60 | 331.99 | 333.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:00:00 | 333.60 | 331.99 | 333.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 333.90 | 332.37 | 333.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:45:00 | 333.85 | 332.37 | 333.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 15:15:00 | 332.70 | 332.44 | 333.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:15:00 | 333.25 | 332.44 | 333.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 330.65 | 332.08 | 333.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 11:45:00 | 329.80 | 331.55 | 332.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 12:15:00 | 335.55 | 333.04 | 332.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2025-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 12:15:00 | 335.55 | 333.04 | 332.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 13:15:00 | 337.70 | 333.97 | 333.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 338.30 | 339.42 | 337.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 338.30 | 339.42 | 337.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 338.30 | 339.42 | 337.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 337.15 | 339.42 | 337.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 332.20 | 337.97 | 337.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 332.20 | 337.97 | 337.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 332.40 | 336.86 | 336.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 330.65 | 336.86 | 336.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2025-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 12:15:00 | 331.10 | 335.71 | 336.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 328.95 | 334.35 | 335.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 308.90 | 305.01 | 310.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 09:30:00 | 308.10 | 305.01 | 310.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 310.55 | 306.12 | 310.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 310.55 | 306.12 | 310.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 310.75 | 307.04 | 310.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 12:00:00 | 310.75 | 307.04 | 310.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 12:15:00 | 310.10 | 307.65 | 310.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 12:30:00 | 311.30 | 307.65 | 310.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 310.05 | 308.53 | 310.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 15:00:00 | 310.05 | 308.53 | 310.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 312.10 | 309.25 | 310.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:15:00 | 316.15 | 309.25 | 310.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — BUY (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 09:15:00 | 322.05 | 311.81 | 311.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 15:15:00 | 322.85 | 319.03 | 315.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 15:15:00 | 326.00 | 326.18 | 323.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-20 09:15:00 | 328.55 | 326.18 | 323.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 331.40 | 332.65 | 329.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:30:00 | 330.90 | 332.65 | 329.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 329.65 | 331.89 | 329.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 12:30:00 | 328.60 | 331.89 | 329.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 328.15 | 331.14 | 329.61 | EMA400 retest candle locked (from upside) |

### Cycle 133 — SELL (started 2025-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 15:15:00 | 324.50 | 328.69 | 328.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 10:15:00 | 319.75 | 326.15 | 327.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 10:15:00 | 322.45 | 322.42 | 324.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 10:30:00 | 322.45 | 322.42 | 324.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 323.80 | 322.86 | 324.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 13:00:00 | 323.80 | 322.86 | 324.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 323.90 | 323.07 | 324.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 13:30:00 | 324.25 | 323.07 | 324.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 323.85 | 323.22 | 324.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 14:30:00 | 324.00 | 323.22 | 324.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 322.90 | 323.20 | 324.02 | EMA400 retest candle locked (from downside) |

### Cycle 134 — BUY (started 2025-01-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 11:15:00 | 328.05 | 324.71 | 324.60 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 14:15:00 | 323.20 | 324.45 | 324.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 321.70 | 323.83 | 324.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 12:15:00 | 322.70 | 322.58 | 323.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-27 13:15:00 | 322.40 | 322.54 | 323.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 13:15:00 | 322.40 | 322.54 | 323.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 13:45:00 | 323.25 | 322.54 | 323.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 309.95 | 319.91 | 321.95 | EMA400 retest candle locked (from downside) |

### Cycle 136 — BUY (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 15:15:00 | 322.90 | 320.00 | 319.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 325.10 | 321.02 | 320.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 09:15:00 | 320.00 | 322.06 | 321.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 09:15:00 | 320.00 | 322.06 | 321.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 320.00 | 322.06 | 321.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 10:45:00 | 323.90 | 322.47 | 321.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 11:45:00 | 322.80 | 322.70 | 321.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 14:30:00 | 323.20 | 323.06 | 322.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 15:15:00 | 318.00 | 322.22 | 322.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2025-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 15:15:00 | 318.00 | 322.22 | 322.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 306.20 | 319.01 | 321.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 15:15:00 | 312.00 | 311.85 | 315.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 09:15:00 | 313.75 | 311.85 | 315.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 11:15:00 | 314.50 | 312.83 | 315.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 11:45:00 | 314.10 | 312.83 | 315.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 12:15:00 | 317.60 | 313.78 | 315.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 13:00:00 | 317.60 | 313.78 | 315.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 13:15:00 | 318.55 | 314.74 | 315.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 13:45:00 | 319.45 | 314.74 | 315.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2025-02-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 15:15:00 | 319.70 | 316.49 | 316.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 321.75 | 317.54 | 316.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 316.35 | 318.51 | 317.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 09:15:00 | 316.35 | 318.51 | 317.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 316.35 | 318.51 | 317.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:00:00 | 316.35 | 318.51 | 317.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 316.65 | 318.14 | 317.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:15:00 | 316.30 | 318.14 | 317.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — SELL (started 2025-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 11:15:00 | 314.25 | 317.36 | 317.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 13:15:00 | 311.95 | 315.81 | 316.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 09:15:00 | 315.90 | 315.04 | 316.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 09:15:00 | 315.90 | 315.04 | 316.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 315.90 | 315.04 | 316.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:00:00 | 315.90 | 315.04 | 316.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 317.20 | 315.47 | 316.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:00:00 | 317.20 | 315.47 | 316.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 316.70 | 315.72 | 316.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:45:00 | 317.95 | 315.72 | 316.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 13:15:00 | 316.05 | 316.09 | 316.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 13:30:00 | 318.30 | 316.09 | 316.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 317.15 | 316.30 | 316.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 15:00:00 | 317.15 | 316.30 | 316.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 15:15:00 | 316.80 | 316.40 | 316.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 312.70 | 316.40 | 316.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 297.06 | 300.87 | 303.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 12:15:00 | 301.15 | 300.15 | 302.45 | SL hit (close>ema200) qty=0.50 sl=300.15 alert=retest2 |

### Cycle 140 — BUY (started 2025-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 13:15:00 | 306.85 | 303.04 | 302.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 14:15:00 | 311.00 | 304.63 | 303.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 15:15:00 | 325.10 | 325.76 | 321.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-24 09:15:00 | 323.35 | 325.76 | 321.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 324.50 | 325.51 | 321.88 | EMA400 retest candle locked (from upside) |

### Cycle 141 — SELL (started 2025-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 10:15:00 | 317.00 | 320.43 | 320.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 09:15:00 | 315.50 | 318.14 | 319.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 311.70 | 310.71 | 312.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 311.70 | 310.71 | 312.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 314.75 | 311.73 | 312.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 15:00:00 | 314.75 | 311.73 | 312.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 315.00 | 312.38 | 313.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 09:15:00 | 310.45 | 312.38 | 313.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 13:00:00 | 313.75 | 312.61 | 312.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 13:45:00 | 313.10 | 312.94 | 313.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 14:15:00 | 313.10 | 312.94 | 313.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 314.10 | 313.17 | 313.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 14:45:00 | 314.70 | 313.17 | 313.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 15:15:00 | 313.15 | 313.17 | 313.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:15:00 | 317.70 | 313.17 | 313.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 322.60 | 315.06 | 314.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 322.60 | 315.06 | 314.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 10:15:00 | 323.65 | 316.77 | 314.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 331.05 | 332.28 | 327.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 331.05 | 332.28 | 327.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 331.35 | 330.88 | 328.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 10:00:00 | 336.20 | 332.66 | 331.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-25 09:15:00 | 369.82 | 363.13 | 355.81 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 143 — SELL (started 2025-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 15:15:00 | 354.05 | 358.69 | 359.03 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 15:15:00 | 364.00 | 358.75 | 358.55 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-03-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 15:15:00 | 358.00 | 358.83 | 358.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 10:15:00 | 353.35 | 357.54 | 358.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 14:15:00 | 351.65 | 350.89 | 353.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 15:00:00 | 351.65 | 350.89 | 353.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 354.85 | 351.80 | 353.22 | EMA400 retest candle locked (from downside) |

### Cycle 146 — BUY (started 2025-04-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 12:15:00 | 360.00 | 355.21 | 354.58 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 351.50 | 355.05 | 355.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 14:15:00 | 350.25 | 353.53 | 354.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 13:15:00 | 348.35 | 347.21 | 350.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-07 13:45:00 | 346.80 | 347.21 | 350.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 14:15:00 | 350.45 | 347.86 | 350.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 15:00:00 | 350.45 | 347.86 | 350.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 349.75 | 348.24 | 350.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:15:00 | 357.10 | 348.24 | 350.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 353.00 | 349.19 | 350.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:30:00 | 355.65 | 349.19 | 350.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 11:15:00 | 355.00 | 350.99 | 351.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:00:00 | 355.00 | 350.99 | 351.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — BUY (started 2025-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 12:15:00 | 355.90 | 351.97 | 351.51 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 13:15:00 | 349.20 | 351.44 | 351.67 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 356.20 | 351.99 | 351.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 10:15:00 | 358.20 | 353.23 | 352.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-15 14:15:00 | 362.35 | 362.92 | 359.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-15 14:45:00 | 362.20 | 362.92 | 359.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 09:15:00 | 361.70 | 362.64 | 359.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 12:00:00 | 363.65 | 360.96 | 360.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 10:30:00 | 363.30 | 364.18 | 363.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-22 14:15:00 | 359.90 | 362.70 | 362.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — SELL (started 2025-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-22 14:15:00 | 359.90 | 362.70 | 362.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-23 12:15:00 | 359.50 | 360.94 | 361.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-23 13:15:00 | 362.65 | 361.28 | 361.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-23 13:15:00 | 362.65 | 361.28 | 361.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 13:15:00 | 362.65 | 361.28 | 361.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-23 14:00:00 | 362.65 | 361.28 | 361.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 14:15:00 | 363.85 | 361.79 | 362.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-23 15:00:00 | 363.85 | 361.79 | 362.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — BUY (started 2025-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 09:15:00 | 365.10 | 362.65 | 362.44 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 355.75 | 361.48 | 362.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 354.40 | 360.07 | 361.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 361.45 | 358.24 | 359.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 361.45 | 358.24 | 359.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 361.45 | 358.24 | 359.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:00:00 | 361.45 | 358.24 | 359.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 360.70 | 358.74 | 359.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:45:00 | 361.90 | 358.74 | 359.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — BUY (started 2025-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 14:15:00 | 362.25 | 360.47 | 360.30 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 11:15:00 | 357.85 | 360.26 | 360.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 13:15:00 | 355.55 | 358.94 | 359.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-30 09:15:00 | 360.10 | 358.82 | 359.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 09:15:00 | 360.10 | 358.82 | 359.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 360.10 | 358.82 | 359.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:30:00 | 360.55 | 358.82 | 359.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 359.60 | 358.98 | 359.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 11:00:00 | 359.60 | 358.98 | 359.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 11:15:00 | 357.70 | 358.72 | 359.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 13:00:00 | 355.50 | 358.08 | 358.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 10:15:00 | 356.15 | 356.51 | 357.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 337.72 | 342.66 | 346.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 338.34 | 342.66 | 346.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-07 15:15:00 | 342.60 | 342.18 | 344.38 | SL hit (close>ema200) qty=0.50 sl=342.18 alert=retest2 |

### Cycle 156 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 346.20 | 340.62 | 340.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 348.95 | 344.89 | 342.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 343.45 | 345.13 | 343.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 12:15:00 | 343.45 | 345.13 | 343.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 343.45 | 345.13 | 343.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:45:00 | 341.85 | 345.13 | 343.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 341.60 | 344.42 | 343.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:00:00 | 341.60 | 344.42 | 343.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 341.05 | 343.75 | 343.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:30:00 | 341.20 | 343.75 | 343.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 10:15:00 | 342.45 | 343.30 | 343.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 10:45:00 | 342.15 | 343.30 | 343.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 342.85 | 343.21 | 343.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:45:00 | 342.10 | 343.21 | 343.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — SELL (started 2025-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 12:15:00 | 341.15 | 342.80 | 342.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-14 13:15:00 | 339.20 | 342.08 | 342.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 13:15:00 | 343.25 | 338.73 | 340.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 13:15:00 | 343.25 | 338.73 | 340.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 343.25 | 338.73 | 340.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 14:00:00 | 343.25 | 338.73 | 340.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 341.95 | 339.38 | 340.25 | EMA400 retest candle locked (from downside) |

### Cycle 158 — BUY (started 2025-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 09:15:00 | 345.15 | 340.93 | 340.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 10:15:00 | 348.45 | 345.65 | 344.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 12:15:00 | 345.00 | 345.72 | 344.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 12:15:00 | 345.00 | 345.72 | 344.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 345.00 | 345.72 | 344.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 13:00:00 | 345.00 | 345.72 | 344.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 343.25 | 345.23 | 344.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 13:45:00 | 344.50 | 345.23 | 344.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 343.30 | 344.84 | 344.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:45:00 | 342.65 | 344.84 | 344.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 344.30 | 344.75 | 344.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:00:00 | 344.30 | 344.75 | 344.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 345.05 | 344.81 | 344.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 14:00:00 | 346.00 | 345.05 | 344.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 09:15:00 | 343.30 | 345.09 | 344.79 | SL hit (close<static) qty=1.00 sl=343.70 alert=retest2 |

### Cycle 159 — SELL (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 10:15:00 | 342.30 | 344.53 | 344.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 12:15:00 | 339.00 | 343.08 | 343.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 10:15:00 | 344.10 | 342.53 | 343.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 10:15:00 | 344.10 | 342.53 | 343.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 344.10 | 342.53 | 343.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:45:00 | 343.95 | 342.53 | 343.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 343.00 | 342.63 | 343.16 | EMA400 retest candle locked (from downside) |

### Cycle 160 — BUY (started 2025-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 15:15:00 | 344.50 | 343.57 | 343.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 346.70 | 344.20 | 343.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 11:15:00 | 343.50 | 344.29 | 343.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 11:15:00 | 343.50 | 344.29 | 343.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 343.50 | 344.29 | 343.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:00:00 | 343.50 | 344.29 | 343.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 344.20 | 344.27 | 343.93 | EMA400 retest candle locked (from upside) |

### Cycle 161 — SELL (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 09:15:00 | 338.85 | 342.93 | 343.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 09:15:00 | 336.60 | 339.12 | 339.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 13:15:00 | 327.95 | 327.93 | 330.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 13:30:00 | 328.00 | 327.93 | 330.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 328.95 | 328.13 | 329.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:00:00 | 328.95 | 328.13 | 329.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 329.15 | 328.47 | 329.80 | EMA400 retest candle locked (from downside) |

### Cycle 162 — BUY (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 11:15:00 | 331.20 | 329.85 | 329.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 14:15:00 | 332.85 | 330.86 | 330.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 339.25 | 340.17 | 337.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 14:00:00 | 339.25 | 340.17 | 337.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 338.15 | 339.44 | 338.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 340.30 | 339.44 | 338.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 340.30 | 339.61 | 338.22 | EMA400 retest candle locked (from upside) |

### Cycle 163 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 333.40 | 337.62 | 337.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 330.75 | 335.12 | 336.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 333.95 | 332.66 | 333.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 10:15:00 | 333.95 | 332.66 | 333.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 333.95 | 332.66 | 333.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 333.95 | 332.66 | 333.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 332.25 | 332.58 | 333.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:30:00 | 334.00 | 332.58 | 333.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 332.95 | 332.65 | 333.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:00:00 | 332.95 | 332.65 | 333.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 333.90 | 332.90 | 333.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 333.90 | 332.90 | 333.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 333.40 | 333.00 | 333.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:45:00 | 333.75 | 333.00 | 333.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 333.80 | 333.16 | 333.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 335.85 | 333.16 | 333.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 335.80 | 333.69 | 333.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 336.00 | 333.69 | 333.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 334.55 | 333.86 | 333.98 | EMA400 retest candle locked (from downside) |

### Cycle 164 — BUY (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 11:15:00 | 335.65 | 334.22 | 334.13 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 333.35 | 334.16 | 334.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 13:15:00 | 332.05 | 333.62 | 333.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 331.05 | 330.91 | 332.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 331.05 | 330.91 | 332.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 331.05 | 330.91 | 332.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:15:00 | 333.00 | 330.91 | 332.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 333.25 | 331.38 | 332.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:30:00 | 332.40 | 331.38 | 332.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 332.85 | 331.67 | 332.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:30:00 | 333.20 | 331.67 | 332.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 332.55 | 332.10 | 332.33 | EMA400 retest candle locked (from downside) |

### Cycle 166 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 335.35 | 332.75 | 332.60 | EMA200 above EMA400 |

### Cycle 167 — SELL (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 09:15:00 | 323.30 | 331.50 | 332.29 | EMA200 below EMA400 |

### Cycle 168 — BUY (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 09:15:00 | 334.85 | 331.66 | 331.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 13:15:00 | 335.75 | 333.51 | 332.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 15:15:00 | 335.75 | 337.37 | 335.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 15:15:00 | 335.75 | 337.37 | 335.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 335.75 | 337.37 | 335.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:30:00 | 334.15 | 336.79 | 335.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 335.00 | 336.43 | 335.58 | EMA400 retest candle locked (from upside) |

### Cycle 169 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 333.60 | 334.90 | 335.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 11:15:00 | 333.25 | 334.44 | 334.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 09:15:00 | 333.80 | 333.49 | 334.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 09:15:00 | 333.80 | 333.49 | 334.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 333.80 | 333.49 | 334.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:30:00 | 334.70 | 333.49 | 334.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 335.05 | 333.80 | 334.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:00:00 | 335.05 | 333.80 | 334.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 335.00 | 334.04 | 334.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:45:00 | 335.15 | 334.04 | 334.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 333.15 | 333.86 | 334.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 13:15:00 | 332.20 | 333.86 | 334.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 09:15:00 | 331.95 | 333.66 | 333.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 335.45 | 333.94 | 334.05 | SL hit (close>static) qty=1.00 sl=335.10 alert=retest2 |

### Cycle 170 — BUY (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 11:15:00 | 335.75 | 334.30 | 334.20 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 10:15:00 | 333.25 | 334.18 | 334.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 332.50 | 333.84 | 334.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 335.65 | 333.96 | 334.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 14:15:00 | 335.65 | 333.96 | 334.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 335.65 | 333.96 | 334.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 335.65 | 333.96 | 334.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 172 — BUY (started 2025-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 15:15:00 | 335.75 | 334.32 | 334.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 09:15:00 | 338.10 | 335.07 | 334.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 341.70 | 343.22 | 341.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 09:15:00 | 341.70 | 343.22 | 341.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 341.70 | 343.22 | 341.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:00:00 | 341.70 | 343.22 | 341.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 341.90 | 342.96 | 341.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 11:30:00 | 342.80 | 343.16 | 341.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 343.15 | 342.26 | 341.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 12:30:00 | 342.70 | 342.43 | 342.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 13:00:00 | 342.50 | 342.43 | 342.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 342.95 | 342.55 | 342.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:30:00 | 342.65 | 342.55 | 342.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 342.40 | 342.52 | 342.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:15:00 | 343.85 | 342.52 | 342.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 342.45 | 342.50 | 342.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:15:00 | 342.80 | 342.50 | 342.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 342.60 | 342.52 | 342.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 12:00:00 | 343.50 | 342.72 | 342.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 12:15:00 | 341.70 | 342.51 | 342.28 | SL hit (close<static) qty=1.00 sl=341.80 alert=retest2 |

### Cycle 173 — SELL (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 15:15:00 | 341.50 | 342.05 | 342.09 | EMA200 below EMA400 |

### Cycle 174 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 343.35 | 342.36 | 342.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 11:15:00 | 343.80 | 342.65 | 342.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 13:15:00 | 341.30 | 342.44 | 342.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 13:15:00 | 341.30 | 342.44 | 342.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 341.30 | 342.44 | 342.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:00:00 | 341.30 | 342.44 | 342.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 342.05 | 342.36 | 342.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:45:00 | 340.95 | 342.36 | 342.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — SELL (started 2025-07-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 15:15:00 | 341.80 | 342.25 | 342.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 13:15:00 | 341.75 | 342.14 | 342.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 14:15:00 | 342.25 | 342.16 | 342.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 14:15:00 | 342.25 | 342.16 | 342.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 342.25 | 342.16 | 342.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:45:00 | 342.65 | 342.16 | 342.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — BUY (started 2025-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 15:15:00 | 342.50 | 342.23 | 342.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 343.25 | 342.43 | 342.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 13:15:00 | 342.90 | 342.96 | 342.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 13:15:00 | 342.90 | 342.96 | 342.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 342.90 | 342.96 | 342.66 | EMA400 retest candle locked (from upside) |

### Cycle 177 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 341.25 | 342.37 | 342.47 | EMA200 below EMA400 |

### Cycle 178 — BUY (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 10:15:00 | 343.50 | 341.95 | 341.85 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 340.95 | 341.94 | 341.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 12:15:00 | 337.15 | 340.72 | 341.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 333.40 | 333.10 | 334.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:00:00 | 333.40 | 333.10 | 334.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 334.75 | 333.56 | 334.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 334.75 | 333.56 | 334.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 335.30 | 333.91 | 334.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 336.45 | 333.91 | 334.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 339.00 | 334.93 | 335.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:00:00 | 339.00 | 334.93 | 335.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — BUY (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 10:15:00 | 339.55 | 335.85 | 335.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 11:15:00 | 340.55 | 336.79 | 335.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 11:15:00 | 338.45 | 338.47 | 337.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 11:45:00 | 338.25 | 338.47 | 337.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 334.10 | 337.47 | 337.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 15:00:00 | 334.10 | 337.47 | 337.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — SELL (started 2025-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 15:15:00 | 333.70 | 336.71 | 336.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 332.35 | 335.84 | 336.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 331.80 | 331.41 | 333.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 12:00:00 | 331.80 | 331.41 | 333.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 333.00 | 331.73 | 333.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:45:00 | 333.00 | 331.73 | 333.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 331.75 | 331.86 | 332.86 | EMA400 retest candle locked (from downside) |

### Cycle 182 — BUY (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 10:15:00 | 334.75 | 333.31 | 333.14 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 15:15:00 | 332.55 | 333.17 | 333.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 330.60 | 332.65 | 332.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 10:15:00 | 333.25 | 331.29 | 331.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 10:15:00 | 333.25 | 331.29 | 331.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 333.25 | 331.29 | 331.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:00:00 | 333.25 | 331.29 | 331.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 333.75 | 331.78 | 331.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:15:00 | 334.40 | 331.78 | 331.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — BUY (started 2025-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 12:15:00 | 336.45 | 332.72 | 332.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 339.20 | 336.09 | 334.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 14:15:00 | 339.90 | 340.24 | 338.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 15:00:00 | 339.90 | 340.24 | 338.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 337.00 | 339.53 | 338.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 337.00 | 339.53 | 338.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 337.10 | 339.04 | 338.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:45:00 | 337.35 | 339.04 | 338.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 338.75 | 338.78 | 338.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:30:00 | 338.85 | 338.78 | 338.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 339.35 | 338.89 | 338.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 340.45 | 338.85 | 338.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 10:00:00 | 340.45 | 339.17 | 338.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 11:15:00 | 337.75 | 338.99 | 338.75 | SL hit (close<static) qty=1.00 sl=338.55 alert=retest2 |

### Cycle 185 — SELL (started 2025-08-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 14:15:00 | 335.80 | 338.09 | 338.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 10:15:00 | 335.25 | 337.04 | 337.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 09:15:00 | 339.50 | 336.50 | 337.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 09:15:00 | 339.50 | 336.50 | 337.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 339.50 | 336.50 | 337.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 339.80 | 336.50 | 337.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 339.15 | 337.03 | 337.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:15:00 | 340.10 | 337.03 | 337.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — BUY (started 2025-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 11:15:00 | 340.00 | 337.62 | 337.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 12:15:00 | 341.80 | 338.46 | 337.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 10:15:00 | 340.00 | 340.29 | 339.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 10:30:00 | 340.30 | 340.29 | 339.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 337.60 | 339.75 | 339.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 337.60 | 339.75 | 339.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 338.35 | 339.47 | 338.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 13:30:00 | 339.35 | 339.40 | 338.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 09:15:00 | 336.65 | 338.61 | 338.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — SELL (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 09:15:00 | 336.65 | 338.61 | 338.71 | EMA200 below EMA400 |

### Cycle 188 — BUY (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 12:15:00 | 339.20 | 338.34 | 338.30 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 14:15:00 | 337.50 | 338.17 | 338.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 335.20 | 337.48 | 337.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 330.70 | 329.14 | 330.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 330.70 | 329.14 | 330.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 330.70 | 329.14 | 330.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 331.25 | 329.14 | 330.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 330.90 | 329.49 | 330.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:45:00 | 331.00 | 329.49 | 330.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 329.50 | 329.50 | 330.83 | EMA400 retest candle locked (from downside) |

### Cycle 190 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 335.35 | 331.11 | 331.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 11:15:00 | 336.55 | 332.99 | 332.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 334.65 | 335.48 | 334.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 12:30:00 | 334.40 | 335.48 | 334.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 334.65 | 335.31 | 334.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:30:00 | 334.60 | 335.31 | 334.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 333.65 | 334.98 | 334.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 15:00:00 | 333.65 | 334.98 | 334.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 334.50 | 334.88 | 334.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 333.00 | 334.88 | 334.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 333.60 | 334.63 | 334.15 | EMA400 retest candle locked (from upside) |

### Cycle 191 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 330.75 | 333.35 | 333.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 329.90 | 332.66 | 333.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 329.45 | 329.12 | 330.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 329.45 | 329.12 | 330.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 329.45 | 329.12 | 330.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 10:30:00 | 328.25 | 328.84 | 330.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 10:15:00 | 330.30 | 327.44 | 327.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — BUY (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 10:15:00 | 330.30 | 327.44 | 327.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 11:15:00 | 332.10 | 328.37 | 327.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 11:15:00 | 328.60 | 330.29 | 329.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 11:15:00 | 328.60 | 330.29 | 329.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 328.60 | 330.29 | 329.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:00:00 | 328.60 | 330.29 | 329.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 329.40 | 330.11 | 329.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:30:00 | 328.80 | 330.11 | 329.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 330.90 | 330.27 | 329.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 14:15:00 | 331.15 | 330.27 | 329.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 14:45:00 | 331.30 | 330.59 | 329.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 14:30:00 | 331.35 | 331.38 | 330.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 15:15:00 | 331.80 | 331.38 | 330.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 334.85 | 335.84 | 334.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 334.85 | 335.84 | 334.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 335.05 | 335.68 | 334.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 14:45:00 | 336.20 | 335.95 | 335.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 09:15:00 | 338.40 | 342.74 | 343.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 193 — SELL (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 09:15:00 | 338.40 | 342.74 | 343.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 13:15:00 | 337.60 | 340.02 | 341.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 340.10 | 339.47 | 340.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 340.10 | 339.47 | 340.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 341.25 | 339.82 | 340.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:45:00 | 340.95 | 339.82 | 340.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 341.05 | 340.07 | 340.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 12:30:00 | 340.00 | 340.09 | 340.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 15:15:00 | 343.00 | 340.97 | 341.13 | SL hit (close>static) qty=1.00 sl=341.55 alert=retest2 |

### Cycle 194 — BUY (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 13:15:00 | 342.20 | 340.78 | 340.60 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 10:15:00 | 337.90 | 340.14 | 340.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 10:15:00 | 337.25 | 338.76 | 339.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 13:15:00 | 338.90 | 338.56 | 339.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-07 14:00:00 | 338.90 | 338.56 | 339.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 336.85 | 338.10 | 338.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 10:45:00 | 335.90 | 337.35 | 338.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 339.95 | 336.10 | 336.20 | SL hit (close>static) qty=1.00 sl=339.25 alert=retest2 |

### Cycle 196 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 10:15:00 | 340.75 | 337.03 | 336.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 15:15:00 | 342.00 | 340.16 | 338.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 10:15:00 | 338.85 | 340.04 | 339.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 10:15:00 | 338.85 | 340.04 | 339.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 338.85 | 340.04 | 339.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 338.85 | 340.04 | 339.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 337.35 | 339.50 | 338.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:45:00 | 338.25 | 339.50 | 338.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — SELL (started 2025-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 13:15:00 | 337.05 | 338.39 | 338.52 | EMA200 below EMA400 |

### Cycle 198 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 340.50 | 338.66 | 338.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 10:15:00 | 341.80 | 339.99 | 339.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 341.25 | 342.07 | 341.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 13:00:00 | 341.25 | 342.07 | 341.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 341.15 | 341.88 | 341.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 340.85 | 341.88 | 341.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 341.00 | 341.71 | 341.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:45:00 | 341.00 | 341.71 | 341.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 340.85 | 341.53 | 341.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:15:00 | 341.35 | 341.53 | 341.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 342.10 | 341.65 | 341.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 10:45:00 | 343.45 | 341.96 | 341.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 12:15:00 | 343.15 | 342.19 | 341.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 09:15:00 | 343.90 | 342.08 | 341.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 11:15:00 | 341.00 | 341.99 | 342.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 199 — SELL (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 11:15:00 | 341.00 | 341.99 | 342.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 12:15:00 | 339.80 | 341.55 | 341.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 342.30 | 341.02 | 341.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 342.30 | 341.02 | 341.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 342.30 | 341.02 | 341.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 342.30 | 341.02 | 341.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 342.05 | 341.23 | 341.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 14:00:00 | 341.40 | 341.58 | 341.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 10:00:00 | 341.35 | 341.56 | 341.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 343.60 | 340.38 | 340.70 | SL hit (close>static) qty=1.00 sl=342.60 alert=retest2 |

### Cycle 200 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 346.75 | 341.66 | 341.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 12:15:00 | 349.60 | 344.20 | 342.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 10:15:00 | 346.30 | 346.32 | 344.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 10:30:00 | 346.00 | 346.32 | 344.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 344.35 | 345.81 | 344.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:00:00 | 344.35 | 345.81 | 344.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 344.45 | 345.54 | 344.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:30:00 | 344.60 | 345.54 | 344.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 345.30 | 345.49 | 344.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:45:00 | 343.75 | 345.49 | 344.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 345.25 | 345.44 | 344.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:15:00 | 340.50 | 345.44 | 344.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — SELL (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 09:15:00 | 338.10 | 343.97 | 344.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 11:15:00 | 334.95 | 337.43 | 339.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 326.20 | 325.85 | 328.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:00:00 | 326.20 | 325.85 | 328.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 325.50 | 325.88 | 327.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 10:00:00 | 323.60 | 325.28 | 326.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 12:15:00 | 327.65 | 326.57 | 326.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — BUY (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 12:15:00 | 327.65 | 326.57 | 326.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 10:15:00 | 329.80 | 327.45 | 326.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 327.30 | 327.86 | 327.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 13:15:00 | 327.30 | 327.86 | 327.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 327.30 | 327.86 | 327.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 327.30 | 327.86 | 327.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 327.05 | 327.70 | 327.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 327.05 | 327.70 | 327.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 327.50 | 327.66 | 327.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 328.35 | 327.66 | 327.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 10:45:00 | 328.55 | 327.81 | 327.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 13:15:00 | 327.95 | 327.93 | 327.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 13:15:00 | 326.80 | 327.71 | 327.50 | SL hit (close<static) qty=1.00 sl=326.85 alert=retest2 |

### Cycle 203 — SELL (started 2025-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 14:15:00 | 328.10 | 328.56 | 328.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 327.10 | 328.25 | 328.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 327.95 | 326.95 | 327.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 327.95 | 326.95 | 327.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 327.95 | 326.95 | 327.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 14:45:00 | 326.45 | 327.32 | 327.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:45:00 | 326.75 | 326.98 | 327.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 13:15:00 | 326.50 | 326.83 | 327.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 14:30:00 | 326.65 | 326.71 | 327.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 326.65 | 326.65 | 326.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 11:45:00 | 324.70 | 326.17 | 326.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 12:30:00 | 325.20 | 325.89 | 326.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 09:30:00 | 325.25 | 324.67 | 325.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 10:15:00 | 325.40 | 324.67 | 325.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 327.15 | 325.17 | 325.38 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-26 10:15:00 | 327.15 | 325.17 | 325.38 | SL hit (close>static) qty=1.00 sl=327.10 alert=retest2 |

### Cycle 204 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 327.10 | 325.55 | 325.53 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 12:15:00 | 326.05 | 326.24 | 326.26 | EMA200 below EMA400 |

### Cycle 206 — BUY (started 2025-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 13:15:00 | 326.55 | 326.30 | 326.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 15:15:00 | 327.95 | 326.76 | 326.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 15:15:00 | 328.10 | 328.21 | 327.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-03 09:15:00 | 325.50 | 328.21 | 327.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 324.45 | 327.46 | 327.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 324.45 | 327.46 | 327.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 323.40 | 326.65 | 326.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 11:15:00 | 322.10 | 325.74 | 326.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 11:15:00 | 324.05 | 323.65 | 324.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 12:00:00 | 324.05 | 323.65 | 324.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 323.55 | 323.13 | 323.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:45:00 | 323.50 | 323.13 | 323.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 323.00 | 323.11 | 323.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 321.50 | 323.18 | 323.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 11:15:00 | 322.05 | 320.80 | 321.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 12:15:00 | 321.85 | 321.31 | 321.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — BUY (started 2025-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 12:15:00 | 321.85 | 321.31 | 321.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 15:15:00 | 322.90 | 322.27 | 321.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 323.25 | 324.10 | 323.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 323.25 | 324.10 | 323.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 323.25 | 324.10 | 323.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 12:15:00 | 324.70 | 324.16 | 323.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 13:15:00 | 324.55 | 324.18 | 323.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 11:15:00 | 320.40 | 322.84 | 323.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 320.40 | 322.84 | 323.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 319.75 | 322.22 | 322.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 10:15:00 | 321.45 | 321.39 | 322.10 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-17 11:45:00 | 320.55 | 321.22 | 321.95 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 321.40 | 321.20 | 321.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 318.60 | 321.23 | 321.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 319.85 | 318.73 | 319.36 | SL hit (close>ema400) qty=1.00 sl=319.36 alert=retest1 |

### Cycle 210 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 322.05 | 319.95 | 319.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 322.60 | 321.12 | 320.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 13:15:00 | 323.40 | 323.89 | 322.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 14:00:00 | 323.40 | 323.89 | 322.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 322.40 | 323.59 | 322.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:45:00 | 322.80 | 323.59 | 322.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 322.40 | 323.35 | 322.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 323.80 | 323.35 | 322.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 11:15:00 | 343.55 | 347.11 | 347.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 211 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 343.55 | 347.11 | 347.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 340.75 | 344.25 | 345.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 338.75 | 338.36 | 340.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:30:00 | 339.10 | 338.36 | 340.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 343.35 | 338.57 | 339.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:00:00 | 343.35 | 338.57 | 339.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 212 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 346.15 | 340.08 | 339.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 11:15:00 | 346.85 | 341.44 | 340.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 12:15:00 | 347.95 | 348.45 | 345.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 13:00:00 | 347.95 | 348.45 | 345.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 346.50 | 347.50 | 345.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 345.35 | 347.50 | 345.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 346.70 | 347.34 | 345.79 | EMA400 retest candle locked (from upside) |

### Cycle 213 — SELL (started 2026-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 14:15:00 | 343.65 | 345.10 | 345.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 15:15:00 | 343.35 | 344.75 | 345.00 | Break + close below crossover candle low |

### Cycle 214 — BUY (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-20 09:15:00 | 347.30 | 345.26 | 345.21 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 343.75 | 344.92 | 345.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 342.25 | 344.38 | 344.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 340.75 | 340.73 | 342.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 13:00:00 | 340.75 | 340.73 | 342.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 340.95 | 339.84 | 341.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 340.80 | 339.84 | 341.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 339.45 | 339.76 | 341.13 | EMA400 retest candle locked (from downside) |

### Cycle 216 — BUY (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 09:15:00 | 343.45 | 341.54 | 341.49 | EMA200 above EMA400 |

### Cycle 217 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 339.25 | 341.38 | 341.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 337.80 | 340.31 | 340.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 342.40 | 339.68 | 340.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 342.40 | 339.68 | 340.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 342.40 | 339.68 | 340.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:45:00 | 342.00 | 339.68 | 340.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 342.85 | 340.32 | 340.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:30:00 | 342.65 | 340.32 | 340.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 218 — BUY (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 11:15:00 | 343.35 | 340.92 | 340.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 14:15:00 | 344.25 | 341.81 | 341.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 10:15:00 | 355.00 | 355.52 | 351.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 12:15:00 | 352.70 | 354.56 | 352.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 352.70 | 354.56 | 352.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 13:45:00 | 354.20 | 354.51 | 352.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 14:30:00 | 354.50 | 354.87 | 352.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 359.55 | 354.71 | 352.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 11:15:00 | 346.80 | 353.68 | 352.84 | SL hit (close<static) qty=1.00 sl=351.10 alert=retest2 |

### Cycle 219 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 349.05 | 352.12 | 352.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 343.15 | 350.33 | 351.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 346.35 | 346.34 | 348.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 350.05 | 347.25 | 348.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 350.05 | 347.25 | 348.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 350.05 | 347.25 | 348.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 352.55 | 348.31 | 349.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 355.35 | 348.31 | 349.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 220 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 355.35 | 349.72 | 349.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 14:15:00 | 367.00 | 362.95 | 358.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 362.65 | 365.44 | 362.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 362.65 | 365.44 | 362.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 362.65 | 365.44 | 362.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 15:00:00 | 365.25 | 364.29 | 363.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 365.95 | 364.17 | 363.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 15:15:00 | 361.40 | 363.15 | 363.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 221 — SELL (started 2026-02-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-09 15:15:00 | 361.40 | 363.15 | 363.25 | EMA200 below EMA400 |

### Cycle 222 — BUY (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 09:15:00 | 365.30 | 363.58 | 363.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 10:15:00 | 366.95 | 364.25 | 363.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 12:15:00 | 367.85 | 368.19 | 367.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-12 13:00:00 | 367.85 | 368.19 | 367.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 368.25 | 368.14 | 367.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:45:00 | 367.95 | 368.14 | 367.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 366.10 | 367.71 | 367.20 | EMA400 retest candle locked (from upside) |

### Cycle 223 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 11:15:00 | 365.40 | 366.77 | 366.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 362.80 | 365.33 | 366.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 368.65 | 365.78 | 366.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 09:15:00 | 368.65 | 365.78 | 366.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 368.65 | 365.78 | 366.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:00:00 | 368.65 | 365.78 | 366.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 368.05 | 366.24 | 366.34 | EMA400 retest candle locked (from downside) |

### Cycle 224 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 11:15:00 | 368.70 | 366.73 | 366.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 369.95 | 368.27 | 367.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 11:15:00 | 368.30 | 368.52 | 367.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-18 12:00:00 | 368.30 | 368.52 | 367.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 12:15:00 | 368.00 | 368.41 | 367.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 13:00:00 | 368.00 | 368.41 | 367.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 367.90 | 368.31 | 367.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 13:45:00 | 367.50 | 368.31 | 367.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 368.20 | 368.29 | 367.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 369.35 | 368.23 | 367.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 365.40 | 367.66 | 367.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 225 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 365.40 | 367.66 | 367.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 365.00 | 366.86 | 367.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-19 12:15:00 | 366.95 | 366.88 | 367.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-19 13:00:00 | 366.95 | 366.88 | 367.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 363.20 | 366.14 | 366.94 | EMA400 retest candle locked (from downside) |

### Cycle 226 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 371.05 | 367.31 | 367.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 13:15:00 | 372.10 | 368.88 | 367.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 09:15:00 | 383.55 | 383.74 | 380.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 09:45:00 | 381.80 | 383.74 | 380.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 379.70 | 383.07 | 380.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:00:00 | 379.70 | 383.07 | 380.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 379.05 | 382.26 | 380.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 13:00:00 | 379.05 | 382.26 | 380.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 379.30 | 381.67 | 380.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 14:15:00 | 379.80 | 381.67 | 380.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 374.60 | 380.82 | 381.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 227 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 374.60 | 380.82 | 381.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 15:15:00 | 364.95 | 368.38 | 372.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 374.85 | 369.68 | 372.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 374.85 | 369.68 | 372.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 374.85 | 369.68 | 372.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:00:00 | 374.85 | 369.68 | 372.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 376.15 | 370.97 | 373.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:30:00 | 376.45 | 370.97 | 373.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 228 — BUY (started 2026-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 14:15:00 | 377.30 | 374.19 | 374.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 383.60 | 376.59 | 375.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 14:15:00 | 380.00 | 380.90 | 378.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-06 15:00:00 | 380.00 | 380.90 | 378.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 373.55 | 379.47 | 378.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:00:00 | 373.55 | 379.47 | 378.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 10:15:00 | 373.50 | 378.28 | 377.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 11:00:00 | 373.50 | 378.28 | 377.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 229 — SELL (started 2026-03-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 12:15:00 | 371.85 | 376.21 | 376.79 | EMA200 below EMA400 |

### Cycle 230 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 380.30 | 377.23 | 377.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 381.75 | 378.22 | 377.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 379.90 | 380.21 | 378.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 14:45:00 | 379.55 | 380.21 | 378.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 378.75 | 379.92 | 378.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 380.15 | 379.92 | 378.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 383.25 | 380.58 | 379.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:30:00 | 387.45 | 381.56 | 379.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 09:30:00 | 384.45 | 385.81 | 385.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 378.05 | 384.26 | 384.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 231 — SELL (started 2026-03-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 10:15:00 | 378.05 | 384.26 | 384.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 13:15:00 | 376.20 | 380.86 | 382.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 382.25 | 381.14 | 382.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 382.25 | 381.14 | 382.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 382.25 | 381.14 | 382.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 383.95 | 381.14 | 382.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 382.60 | 381.44 | 382.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:45:00 | 384.70 | 381.44 | 382.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 383.20 | 381.79 | 382.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 380.95 | 381.79 | 382.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 10:30:00 | 382.10 | 382.31 | 382.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 10:00:00 | 381.25 | 377.54 | 378.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 11:00:00 | 382.10 | 378.45 | 379.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 13:15:00 | 380.70 | 379.79 | 379.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 232 — BUY (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 13:15:00 | 380.70 | 379.79 | 379.69 | EMA200 above EMA400 |

### Cycle 233 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 371.65 | 378.58 | 379.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 11:15:00 | 370.10 | 375.83 | 377.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 374.70 | 373.62 | 375.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 11:15:00 | 374.70 | 373.62 | 375.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 374.70 | 373.62 | 375.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 11:45:00 | 375.95 | 373.62 | 375.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 375.90 | 374.08 | 375.42 | EMA400 retest candle locked (from downside) |

### Cycle 234 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 378.45 | 375.88 | 375.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 379.50 | 376.60 | 376.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 377.15 | 377.52 | 376.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 377.15 | 377.52 | 376.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 377.15 | 377.52 | 376.91 | EMA400 retest candle locked (from upside) |

### Cycle 235 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 375.75 | 376.79 | 376.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 374.20 | 375.85 | 376.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 14:15:00 | 360.15 | 360.13 | 364.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 15:00:00 | 360.15 | 360.13 | 364.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 13:15:00 | 363.45 | 360.02 | 362.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 14:00:00 | 363.45 | 360.02 | 362.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 14:15:00 | 366.50 | 361.32 | 362.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 15:00:00 | 366.50 | 361.32 | 362.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 236 — BUY (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 10:15:00 | 367.95 | 364.63 | 364.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 370.60 | 367.62 | 366.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 12:15:00 | 378.25 | 379.02 | 376.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 13:00:00 | 378.25 | 379.02 | 376.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 13:15:00 | 378.20 | 378.85 | 376.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 14:45:00 | 379.55 | 379.17 | 376.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 09:30:00 | 380.90 | 380.05 | 377.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 13:15:00 | 403.15 | 406.51 | 406.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 237 — SELL (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 13:15:00 | 403.15 | 406.51 | 406.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 14:15:00 | 402.50 | 405.71 | 406.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 15:15:00 | 400.00 | 399.84 | 402.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 09:15:00 | 403.40 | 399.84 | 402.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 401.55 | 400.18 | 402.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:30:00 | 399.70 | 400.42 | 401.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 398.00 | 400.19 | 401.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 10:00:00 | 398.95 | 399.94 | 401.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 12:15:00 | 400.65 | 398.77 | 398.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 238 — BUY (started 2026-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 12:15:00 | 400.65 | 398.77 | 398.71 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-22 12:30:00 | 175.25 | 2023-05-25 09:15:00 | 175.05 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2023-05-26 10:15:00 | 174.85 | 2023-05-26 10:15:00 | 175.20 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2023-06-02 10:15:00 | 174.30 | 2023-06-05 09:15:00 | 175.50 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2023-06-02 11:00:00 | 174.15 | 2023-06-05 09:15:00 | 175.50 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2023-06-02 14:30:00 | 174.25 | 2023-06-05 09:15:00 | 175.50 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2023-06-02 15:15:00 | 174.25 | 2023-06-05 09:15:00 | 175.50 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2023-06-16 14:00:00 | 188.05 | 2023-06-19 11:15:00 | 185.90 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2023-06-21 13:45:00 | 187.90 | 2023-06-22 09:15:00 | 185.80 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2023-06-27 13:00:00 | 185.70 | 2023-06-27 14:15:00 | 186.10 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2023-06-28 10:15:00 | 186.75 | 2023-07-07 15:15:00 | 192.55 | STOP_HIT | 1.00 | 3.11% |
| SELL | retest2 | 2023-07-17 13:30:00 | 186.15 | 2023-07-18 10:15:00 | 188.85 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2023-07-17 14:45:00 | 186.35 | 2023-07-18 10:15:00 | 188.85 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2023-08-03 09:15:00 | 221.70 | 2023-08-04 15:15:00 | 218.00 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2023-08-03 14:45:00 | 220.10 | 2023-08-04 15:15:00 | 218.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2023-08-31 09:15:00 | 220.65 | 2023-08-31 15:15:00 | 219.20 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2023-08-31 10:15:00 | 220.55 | 2023-08-31 15:15:00 | 219.20 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2023-09-07 12:45:00 | 234.65 | 2023-09-12 14:15:00 | 234.65 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2023-09-14 11:30:00 | 236.40 | 2023-09-14 14:15:00 | 237.70 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2023-09-15 11:30:00 | 236.40 | 2023-09-18 10:15:00 | 239.75 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2023-09-15 12:15:00 | 236.00 | 2023-09-18 10:15:00 | 239.75 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2023-09-15 15:00:00 | 235.95 | 2023-09-18 10:15:00 | 239.75 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2023-10-23 09:15:00 | 240.35 | 2023-10-26 09:15:00 | 228.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-23 09:15:00 | 240.35 | 2023-10-26 14:15:00 | 231.45 | STOP_HIT | 0.50 | 3.70% |
| BUY | retest2 | 2023-11-07 10:30:00 | 237.80 | 2023-11-21 10:15:00 | 248.05 | STOP_HIT | 1.00 | 4.31% |
| BUY | retest2 | 2023-11-08 12:15:00 | 237.45 | 2023-11-21 10:15:00 | 248.05 | STOP_HIT | 1.00 | 4.46% |
| BUY | retest2 | 2023-11-09 09:45:00 | 237.60 | 2023-11-21 10:15:00 | 248.05 | STOP_HIT | 1.00 | 4.40% |
| BUY | retest2 | 2023-11-09 10:30:00 | 237.15 | 2023-11-21 10:15:00 | 248.05 | STOP_HIT | 1.00 | 4.60% |
| BUY | retest2 | 2023-11-10 10:15:00 | 239.25 | 2023-11-21 10:15:00 | 248.05 | STOP_HIT | 1.00 | 3.68% |
| BUY | retest2 | 2023-12-11 09:15:00 | 287.25 | 2023-12-21 09:15:00 | 297.50 | STOP_HIT | 1.00 | 3.57% |
| BUY | retest2 | 2023-12-11 13:00:00 | 286.70 | 2023-12-21 09:15:00 | 297.50 | STOP_HIT | 1.00 | 3.77% |
| BUY | retest2 | 2023-12-13 09:15:00 | 290.55 | 2023-12-21 09:15:00 | 297.50 | STOP_HIT | 1.00 | 2.39% |
| BUY | retest2 | 2023-12-29 12:00:00 | 311.40 | 2024-01-02 09:15:00 | 307.90 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2023-12-29 13:00:00 | 311.45 | 2024-01-02 09:15:00 | 307.90 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2023-12-29 13:30:00 | 311.35 | 2024-01-02 09:15:00 | 307.90 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-01-01 09:15:00 | 312.95 | 2024-01-02 09:15:00 | 307.90 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-01-08 09:45:00 | 316.85 | 2024-01-10 10:15:00 | 312.10 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-01-08 11:00:00 | 315.50 | 2024-01-10 10:15:00 | 312.10 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-01-08 12:00:00 | 315.95 | 2024-01-10 10:15:00 | 312.10 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-01-11 11:15:00 | 314.10 | 2024-01-15 13:15:00 | 315.70 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-01-15 12:30:00 | 314.05 | 2024-01-15 13:15:00 | 315.70 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2024-01-19 14:30:00 | 307.35 | 2024-01-20 09:15:00 | 312.85 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2024-02-13 14:30:00 | 320.65 | 2024-02-14 13:15:00 | 323.90 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-02-13 15:15:00 | 320.90 | 2024-02-14 13:15:00 | 323.90 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2024-02-14 10:00:00 | 320.35 | 2024-02-14 13:15:00 | 323.90 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-02-23 09:15:00 | 337.85 | 2024-03-01 09:15:00 | 341.95 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-03-19 09:15:00 | 314.10 | 2024-03-21 09:15:00 | 323.20 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2024-03-26 11:00:00 | 327.40 | 2024-04-04 09:15:00 | 360.14 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-05-08 14:15:00 | 354.75 | 2024-05-14 10:15:00 | 356.35 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2024-05-09 09:15:00 | 353.30 | 2024-05-14 10:15:00 | 356.35 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-05-10 10:45:00 | 353.60 | 2024-05-14 10:15:00 | 356.35 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-05-10 11:45:00 | 354.65 | 2024-05-14 10:15:00 | 356.35 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2024-05-17 09:15:00 | 362.60 | 2024-05-27 12:15:00 | 370.20 | STOP_HIT | 1.00 | 2.10% |
| BUY | retest2 | 2024-05-17 09:45:00 | 364.20 | 2024-05-27 12:15:00 | 370.20 | STOP_HIT | 1.00 | 1.65% |
| BUY | retest2 | 2024-06-18 09:30:00 | 371.20 | 2024-06-19 09:15:00 | 364.40 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2024-06-18 11:15:00 | 371.45 | 2024-06-19 09:15:00 | 364.40 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2024-06-18 11:45:00 | 370.95 | 2024-06-19 09:15:00 | 364.40 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-07-01 11:30:00 | 372.40 | 2024-07-01 13:15:00 | 369.30 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-07-05 14:30:00 | 378.55 | 2024-07-12 09:15:00 | 375.90 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2024-07-09 12:00:00 | 377.70 | 2024-07-12 09:15:00 | 375.90 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2024-07-10 09:15:00 | 379.00 | 2024-07-12 09:15:00 | 375.90 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-07-10 12:00:00 | 377.40 | 2024-07-12 09:15:00 | 375.90 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2024-07-10 14:30:00 | 379.45 | 2024-07-12 09:15:00 | 375.90 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-07-22 14:30:00 | 371.85 | 2024-07-23 11:15:00 | 380.05 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2024-07-22 15:15:00 | 372.75 | 2024-07-23 11:15:00 | 380.05 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-07-23 11:15:00 | 373.35 | 2024-07-23 11:15:00 | 380.05 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-07-30 09:15:00 | 411.90 | 2024-08-05 11:15:00 | 410.75 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2024-08-09 11:30:00 | 409.10 | 2024-08-19 09:15:00 | 405.35 | STOP_HIT | 1.00 | 0.92% |
| SELL | retest2 | 2024-08-09 13:00:00 | 409.50 | 2024-08-19 09:15:00 | 405.35 | STOP_HIT | 1.00 | 1.01% |
| SELL | retest2 | 2024-08-09 14:15:00 | 408.90 | 2024-08-19 10:15:00 | 402.95 | STOP_HIT | 1.00 | 1.46% |
| SELL | retest2 | 2024-08-12 09:15:00 | 402.55 | 2024-08-19 10:15:00 | 402.95 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2024-08-16 10:15:00 | 393.80 | 2024-08-19 10:15:00 | 402.95 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2024-08-16 11:00:00 | 394.25 | 2024-08-19 10:15:00 | 402.95 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2024-08-28 10:15:00 | 412.00 | 2024-08-28 14:15:00 | 409.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-08-28 12:30:00 | 412.50 | 2024-08-28 14:15:00 | 409.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-08-28 13:15:00 | 412.30 | 2024-08-28 14:15:00 | 409.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-09-11 13:30:00 | 390.90 | 2024-09-12 12:15:00 | 397.40 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-09-23 09:15:00 | 428.65 | 2024-10-03 12:15:00 | 436.60 | STOP_HIT | 1.00 | 1.85% |
| SELL | retest2 | 2024-10-09 15:00:00 | 418.20 | 2024-10-10 09:15:00 | 427.80 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2024-10-31 11:15:00 | 409.80 | 2024-10-31 13:15:00 | 406.00 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2024-11-19 15:00:00 | 365.75 | 2024-11-25 09:15:00 | 373.55 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-12-02 10:30:00 | 359.90 | 2024-12-03 12:15:00 | 365.95 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2024-12-03 10:45:00 | 360.10 | 2024-12-03 12:15:00 | 365.95 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2024-12-05 14:00:00 | 370.50 | 2024-12-09 12:15:00 | 368.65 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-12-06 09:15:00 | 370.60 | 2024-12-09 12:15:00 | 368.65 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2024-12-06 10:15:00 | 370.00 | 2024-12-09 12:15:00 | 368.65 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2024-12-06 12:15:00 | 369.95 | 2024-12-10 12:15:00 | 367.80 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-12-09 09:15:00 | 370.15 | 2024-12-10 13:15:00 | 367.15 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-12-09 09:45:00 | 371.75 | 2024-12-10 13:15:00 | 367.15 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-12-09 10:45:00 | 370.90 | 2024-12-10 13:15:00 | 367.15 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-12-10 09:45:00 | 370.15 | 2024-12-10 13:15:00 | 367.15 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2024-12-20 12:00:00 | 339.10 | 2024-12-27 10:15:00 | 338.90 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-01-01 11:45:00 | 329.80 | 2025-01-02 12:15:00 | 335.55 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-01-31 10:45:00 | 323.90 | 2025-02-01 15:15:00 | 318.00 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-01-31 11:45:00 | 322.80 | 2025-02-01 15:15:00 | 318.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-01-31 14:30:00 | 323.20 | 2025-02-01 15:15:00 | 318.00 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-02-10 09:15:00 | 312.70 | 2025-02-17 09:15:00 | 297.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 312.70 | 2025-02-17 12:15:00 | 301.15 | STOP_HIT | 0.50 | 3.69% |
| SELL | retest2 | 2025-03-04 09:15:00 | 310.45 | 2025-03-05 09:15:00 | 322.60 | STOP_HIT | 1.00 | -3.91% |
| SELL | retest2 | 2025-03-04 13:00:00 | 313.75 | 2025-03-05 09:15:00 | 322.60 | STOP_HIT | 1.00 | -2.82% |
| SELL | retest2 | 2025-03-04 13:45:00 | 313.10 | 2025-03-05 09:15:00 | 322.60 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2025-03-04 14:15:00 | 313.10 | 2025-03-05 09:15:00 | 322.60 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2025-03-18 10:00:00 | 336.20 | 2025-03-25 09:15:00 | 369.82 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-17 12:00:00 | 363.65 | 2025-04-22 14:15:00 | 359.90 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-04-22 10:30:00 | 363.30 | 2025-04-22 14:15:00 | 359.90 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-04-30 13:00:00 | 355.50 | 2025-05-07 09:15:00 | 337.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-02 10:15:00 | 356.15 | 2025-05-07 09:15:00 | 338.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 13:00:00 | 355.50 | 2025-05-07 15:15:00 | 342.60 | STOP_HIT | 0.50 | 3.63% |
| SELL | retest2 | 2025-05-02 10:15:00 | 356.15 | 2025-05-07 15:15:00 | 342.60 | STOP_HIT | 0.50 | 3.80% |
| BUY | retest2 | 2025-05-21 14:00:00 | 346.00 | 2025-05-22 09:15:00 | 343.30 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-07-02 13:15:00 | 332.20 | 2025-07-03 10:15:00 | 335.45 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-07-03 09:15:00 | 331.95 | 2025-07-03 10:15:00 | 335.45 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-07-10 11:30:00 | 342.80 | 2025-07-14 12:15:00 | 341.70 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-07-11 09:15:00 | 343.15 | 2025-07-14 13:15:00 | 340.85 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-07-11 12:30:00 | 342.70 | 2025-07-14 13:15:00 | 340.85 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-07-11 13:00:00 | 342.50 | 2025-07-14 13:15:00 | 340.85 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-07-14 12:00:00 | 343.50 | 2025-07-14 13:15:00 | 340.85 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-08-18 09:15:00 | 340.45 | 2025-08-18 11:15:00 | 337.75 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-08-18 10:00:00 | 340.45 | 2025-08-18 11:15:00 | 337.75 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-08-21 13:30:00 | 339.35 | 2025-08-22 09:15:00 | 336.65 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-09-08 10:30:00 | 328.25 | 2025-09-11 10:15:00 | 330.30 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-09-12 14:15:00 | 331.15 | 2025-09-26 09:15:00 | 338.40 | STOP_HIT | 1.00 | 2.19% |
| BUY | retest2 | 2025-09-12 14:45:00 | 331.30 | 2025-09-26 09:15:00 | 338.40 | STOP_HIT | 1.00 | 2.14% |
| BUY | retest2 | 2025-09-15 14:30:00 | 331.35 | 2025-09-26 09:15:00 | 338.40 | STOP_HIT | 1.00 | 2.13% |
| BUY | retest2 | 2025-09-15 15:15:00 | 331.80 | 2025-09-26 09:15:00 | 338.40 | STOP_HIT | 1.00 | 1.99% |
| BUY | retest2 | 2025-09-18 14:45:00 | 336.20 | 2025-09-26 09:15:00 | 338.40 | STOP_HIT | 1.00 | 0.65% |
| SELL | retest2 | 2025-09-29 12:30:00 | 340.00 | 2025-09-29 15:15:00 | 343.00 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-09-30 10:15:00 | 340.15 | 2025-10-01 09:15:00 | 342.15 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-09-30 12:30:00 | 339.85 | 2025-10-01 09:15:00 | 342.15 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-09-30 14:00:00 | 340.15 | 2025-10-01 09:15:00 | 342.15 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-10-03 10:45:00 | 338.90 | 2025-10-03 13:15:00 | 342.20 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-10-08 10:45:00 | 335.90 | 2025-10-10 09:15:00 | 339.95 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-10-20 10:45:00 | 343.45 | 2025-10-24 11:15:00 | 341.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-10-20 12:15:00 | 343.15 | 2025-10-24 11:15:00 | 341.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-10-23 09:15:00 | 343.90 | 2025-10-24 11:15:00 | 341.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-10-27 14:00:00 | 341.40 | 2025-10-29 09:15:00 | 343.60 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-10-28 10:00:00 | 341.35 | 2025-10-29 09:15:00 | 343.60 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-11-11 10:00:00 | 323.60 | 2025-11-12 12:15:00 | 327.65 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-11-14 09:15:00 | 328.35 | 2025-11-14 13:15:00 | 326.80 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-11-14 10:45:00 | 328.55 | 2025-11-14 13:15:00 | 326.80 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-11-14 13:15:00 | 327.95 | 2025-11-14 13:15:00 | 326.80 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2025-11-14 15:00:00 | 328.95 | 2025-11-18 14:15:00 | 328.10 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-11-20 14:45:00 | 326.45 | 2025-11-26 10:15:00 | 327.15 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-11-21 09:45:00 | 326.75 | 2025-11-26 10:15:00 | 327.15 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2025-11-21 13:15:00 | 326.50 | 2025-11-26 10:15:00 | 327.15 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-11-21 14:30:00 | 326.65 | 2025-11-26 10:15:00 | 327.15 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-11-24 11:45:00 | 324.70 | 2025-11-26 11:15:00 | 327.10 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-11-24 12:30:00 | 325.20 | 2025-11-26 11:15:00 | 327.10 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-11-26 09:30:00 | 325.25 | 2025-11-26 11:15:00 | 327.10 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-11-26 10:15:00 | 325.40 | 2025-11-26 11:15:00 | 327.10 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-12-08 09:15:00 | 321.50 | 2025-12-10 12:15:00 | 321.85 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-12-10 11:15:00 | 322.05 | 2025-12-10 12:15:00 | 321.85 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2025-12-15 12:15:00 | 324.70 | 2025-12-16 11:15:00 | 320.40 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-12-15 13:15:00 | 324.55 | 2025-12-16 11:15:00 | 320.40 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest1 | 2025-12-17 11:45:00 | 320.55 | 2025-12-19 14:15:00 | 319.85 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2025-12-18 09:15:00 | 318.60 | 2025-12-22 10:15:00 | 322.05 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-12-26 09:15:00 | 323.80 | 2026-01-08 11:15:00 | 343.55 | STOP_HIT | 1.00 | 6.10% |
| BUY | retest2 | 2026-01-30 13:45:00 | 354.20 | 2026-02-01 11:15:00 | 346.80 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2026-01-30 14:30:00 | 354.50 | 2026-02-01 11:15:00 | 346.80 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2026-02-01 09:15:00 | 359.55 | 2026-02-01 11:15:00 | 346.80 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest2 | 2026-02-06 15:00:00 | 365.25 | 2026-02-09 15:15:00 | 361.40 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-02-09 09:15:00 | 365.95 | 2026-02-09 15:15:00 | 361.40 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2026-02-19 09:15:00 | 369.35 | 2026-02-19 09:15:00 | 365.40 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2026-02-26 14:15:00 | 379.80 | 2026-03-02 09:15:00 | 374.60 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-03-12 10:30:00 | 387.45 | 2026-03-16 10:15:00 | 378.05 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2026-03-16 09:30:00 | 384.45 | 2026-03-16 10:15:00 | 378.05 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2026-03-17 11:15:00 | 380.95 | 2026-03-20 13:15:00 | 380.70 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2026-03-18 10:30:00 | 382.10 | 2026-03-20 13:15:00 | 380.70 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2026-03-20 10:00:00 | 381.25 | 2026-03-20 13:15:00 | 380.70 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2026-03-20 11:00:00 | 382.10 | 2026-03-20 13:15:00 | 380.70 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2026-04-10 14:45:00 | 379.55 | 2026-04-29 13:15:00 | 403.15 | STOP_HIT | 1.00 | 6.22% |
| BUY | retest2 | 2026-04-13 09:30:00 | 380.90 | 2026-04-29 13:15:00 | 403.15 | STOP_HIT | 1.00 | 5.84% |
| SELL | retest2 | 2026-05-04 12:30:00 | 399.70 | 2026-05-07 12:15:00 | 400.65 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2026-05-05 09:15:00 | 398.00 | 2026-05-07 12:15:00 | 400.65 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2026-05-05 10:00:00 | 398.95 | 2026-05-07 12:15:00 | 400.65 | STOP_HIT | 1.00 | -0.43% |
