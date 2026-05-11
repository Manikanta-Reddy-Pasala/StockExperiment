# National Aluminium Co. Ltd. (NATIONALUM)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 401.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 142 |
| ALERT1 | 96 |
| ALERT2 | 95 |
| ALERT2_SKIP | 48 |
| ALERT3 | 259 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 104 |
| PARTIAL | 7 |
| TARGET_HIT | 6 |
| STOP_HIT | 99 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 112 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 33 / 79
- **Target hits / Stop hits / Partials:** 6 / 99 / 7
- **Avg / median % per leg:** 0.03% / -0.88%
- **Sum % (uncompounded):** 3.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 57 | 20 | 35.1% | 6 | 51 | 0 | 0.43% | 24.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 57 | 20 | 35.1% | 6 | 51 | 0 | 0.43% | 24.3% |
| SELL (all) | 55 | 13 | 23.6% | 0 | 48 | 7 | -0.37% | -20.6% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.84% | -3.7% |
| SELL @ 3rd Alert (retest2) | 53 | 13 | 24.5% | 0 | 46 | 7 | -0.32% | -16.9% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.84% | -3.7% |
| retest2 (combined) | 110 | 33 | 30.0% | 6 | 97 | 7 | 0.07% | 7.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 09:15:00 | 177.95 | 175.32 | 175.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 11:15:00 | 181.95 | 177.34 | 176.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 13:15:00 | 187.65 | 188.77 | 186.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 13:45:00 | 188.00 | 188.77 | 186.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 196.85 | 198.21 | 195.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 195.50 | 198.21 | 195.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 196.15 | 197.80 | 195.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 15:15:00 | 197.90 | 197.14 | 196.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 09:15:00 | 194.75 | 196.78 | 196.19 | SL hit (close<static) qty=1.00 sl=195.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 10:15:00 | 190.95 | 195.62 | 195.71 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 14:15:00 | 194.30 | 193.68 | 193.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-28 09:15:00 | 199.90 | 195.13 | 194.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 11:15:00 | 192.40 | 194.89 | 194.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 11:15:00 | 192.40 | 194.89 | 194.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 11:15:00 | 192.40 | 194.89 | 194.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:00:00 | 192.40 | 194.89 | 194.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 194.00 | 194.71 | 194.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:30:00 | 191.85 | 194.71 | 194.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 193.55 | 194.48 | 194.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 14:00:00 | 193.55 | 194.48 | 194.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 14:15:00 | 191.70 | 193.92 | 194.05 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 10:15:00 | 195.35 | 194.21 | 194.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 11:15:00 | 198.10 | 194.98 | 194.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 14:15:00 | 194.80 | 195.92 | 195.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 14:15:00 | 194.80 | 195.92 | 195.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 194.80 | 195.92 | 195.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 15:00:00 | 194.80 | 195.92 | 195.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 15:15:00 | 195.05 | 195.75 | 195.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:15:00 | 192.80 | 195.75 | 195.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 192.85 | 195.17 | 194.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 10:00:00 | 192.85 | 195.17 | 194.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 194.30 | 195.00 | 194.86 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2024-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 11:15:00 | 192.45 | 194.49 | 194.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 13:15:00 | 191.90 | 193.85 | 194.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 193.60 | 191.17 | 192.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 13:15:00 | 193.60 | 191.17 | 192.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 193.60 | 191.17 | 192.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:00:00 | 193.60 | 191.17 | 192.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 191.80 | 191.30 | 192.22 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 199.25 | 192.87 | 192.77 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 09:15:00 | 185.05 | 193.49 | 193.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 10:15:00 | 176.60 | 190.11 | 192.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 11:15:00 | 168.05 | 167.67 | 176.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 11:30:00 | 167.60 | 167.67 | 176.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 176.50 | 170.90 | 175.79 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2024-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 13:15:00 | 180.35 | 177.89 | 177.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 181.65 | 179.36 | 178.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 15:15:00 | 183.20 | 183.28 | 182.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 09:15:00 | 183.87 | 183.28 | 182.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 184.32 | 183.49 | 182.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 10:30:00 | 185.30 | 184.04 | 182.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 185.70 | 184.25 | 183.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 13:15:00 | 184.84 | 184.81 | 183.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 14:15:00 | 184.71 | 184.76 | 184.04 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 185.70 | 185.20 | 184.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 12:15:00 | 185.95 | 185.20 | 184.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 13:30:00 | 185.91 | 185.36 | 184.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 14:15:00 | 186.15 | 185.36 | 184.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 09:15:00 | 185.90 | 188.57 | 188.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 09:15:00 | 185.90 | 188.57 | 188.60 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 11:15:00 | 192.04 | 188.09 | 187.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 09:15:00 | 194.78 | 190.99 | 189.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-24 09:15:00 | 187.45 | 191.95 | 191.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 09:15:00 | 187.45 | 191.95 | 191.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 187.45 | 191.95 | 191.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:30:00 | 185.50 | 191.95 | 191.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 189.27 | 191.41 | 190.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 11:30:00 | 189.67 | 191.27 | 190.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 15:15:00 | 190.00 | 190.60 | 190.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 15:15:00 | 190.00 | 190.60 | 190.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 09:15:00 | 189.68 | 190.41 | 190.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 14:15:00 | 188.63 | 188.53 | 189.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-25 15:00:00 | 188.63 | 188.53 | 189.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 187.81 | 185.76 | 186.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:45:00 | 187.65 | 185.76 | 186.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 186.50 | 185.91 | 186.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:30:00 | 186.63 | 185.91 | 186.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 185.85 | 185.90 | 186.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 12:15:00 | 185.12 | 185.90 | 186.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 12:15:00 | 188.97 | 186.51 | 186.74 | SL hit (close>static) qty=1.00 sl=186.73 alert=retest2 |

### Cycle 13 — BUY (started 2024-06-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 15:15:00 | 187.21 | 186.91 | 186.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 09:15:00 | 190.68 | 187.66 | 187.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 11:15:00 | 190.90 | 191.37 | 189.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 12:00:00 | 190.90 | 191.37 | 189.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 190.40 | 191.17 | 189.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 190.40 | 191.17 | 189.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 196.86 | 202.06 | 201.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 196.86 | 202.06 | 201.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 195.67 | 200.78 | 200.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 194.41 | 200.78 | 200.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2024-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 11:15:00 | 197.65 | 200.16 | 200.28 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 11:15:00 | 200.50 | 199.33 | 199.25 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 12:15:00 | 198.60 | 199.18 | 199.20 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 10:15:00 | 200.02 | 199.25 | 199.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 11:15:00 | 201.65 | 199.73 | 199.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 13:15:00 | 200.00 | 200.06 | 199.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 13:15:00 | 200.00 | 200.06 | 199.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 200.00 | 200.06 | 199.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:00:00 | 200.00 | 200.06 | 199.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 199.91 | 200.03 | 199.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:30:00 | 200.00 | 200.03 | 199.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 199.42 | 199.91 | 199.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 194.49 | 199.91 | 199.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 09:15:00 | 194.60 | 198.85 | 199.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 185.49 | 192.36 | 195.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 190.50 | 188.42 | 191.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 10:00:00 | 190.50 | 188.42 | 191.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 190.92 | 188.92 | 191.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:30:00 | 192.09 | 188.92 | 191.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 192.60 | 189.66 | 191.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:45:00 | 192.38 | 189.66 | 191.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 192.31 | 190.19 | 191.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:15:00 | 192.59 | 190.19 | 191.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 191.38 | 190.43 | 191.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:30:00 | 192.51 | 190.43 | 191.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 190.57 | 190.46 | 191.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 15:15:00 | 191.35 | 190.46 | 191.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 191.35 | 190.63 | 191.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 09:15:00 | 187.10 | 190.63 | 191.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 177.74 | 186.47 | 188.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 188.00 | 186.08 | 187.90 | SL hit (close>ema200) qty=0.50 sl=186.08 alert=retest2 |

### Cycle 19 — BUY (started 2024-07-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 12:15:00 | 187.35 | 186.33 | 186.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 13:15:00 | 190.35 | 187.13 | 186.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 09:15:00 | 190.10 | 190.93 | 189.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 09:15:00 | 190.10 | 190.93 | 189.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 190.10 | 190.93 | 189.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 13:30:00 | 191.69 | 191.00 | 190.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 14:15:00 | 191.50 | 191.00 | 190.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 14:45:00 | 191.90 | 191.00 | 190.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:15:00 | 192.02 | 191.06 | 190.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 192.20 | 191.29 | 190.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 15:00:00 | 194.36 | 192.73 | 191.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 09:15:00 | 187.32 | 192.05 | 192.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 187.32 | 192.05 | 192.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 177.70 | 185.56 | 188.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 11:15:00 | 177.03 | 176.54 | 180.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 12:00:00 | 177.03 | 176.54 | 180.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 176.52 | 175.14 | 178.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 09:45:00 | 177.11 | 175.14 | 178.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 177.85 | 175.68 | 178.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:45:00 | 178.88 | 175.68 | 178.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 177.50 | 176.05 | 178.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:30:00 | 177.94 | 176.05 | 178.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 177.68 | 176.37 | 178.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:30:00 | 177.63 | 176.37 | 178.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 177.77 | 176.65 | 178.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 13:45:00 | 177.46 | 176.65 | 178.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 178.32 | 176.99 | 178.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:00:00 | 178.32 | 176.99 | 178.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 180.00 | 177.59 | 178.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:15:00 | 176.87 | 177.59 | 178.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 10:15:00 | 168.03 | 171.28 | 172.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-19 09:15:00 | 166.51 | 165.90 | 167.65 | SL hit (close>ema200) qty=0.50 sl=165.90 alert=retest2 |

### Cycle 21 — BUY (started 2024-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 15:15:00 | 172.32 | 168.00 | 167.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-23 13:15:00 | 172.67 | 171.44 | 170.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 09:15:00 | 183.74 | 183.91 | 180.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-28 10:00:00 | 183.74 | 183.91 | 180.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 182.04 | 183.84 | 182.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:15:00 | 181.02 | 183.84 | 182.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 181.02 | 183.27 | 182.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:30:00 | 180.20 | 183.27 | 182.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 11:15:00 | 180.60 | 182.74 | 181.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 11:30:00 | 181.05 | 182.74 | 181.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 182.30 | 182.75 | 182.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 15:00:00 | 182.30 | 182.75 | 182.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 15:15:00 | 183.00 | 182.80 | 182.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:15:00 | 180.45 | 182.80 | 182.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 180.19 | 182.28 | 182.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 10:00:00 | 180.19 | 182.28 | 182.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2024-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 10:15:00 | 180.60 | 181.94 | 182.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 11:15:00 | 179.66 | 181.49 | 181.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 09:15:00 | 176.68 | 175.78 | 177.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 09:15:00 | 176.68 | 175.78 | 177.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 176.68 | 175.78 | 177.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 09:15:00 | 174.90 | 176.48 | 177.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 12:45:00 | 175.69 | 175.46 | 176.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 13:30:00 | 175.42 | 175.33 | 176.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 13:15:00 | 176.90 | 173.81 | 173.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2024-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 13:15:00 | 176.90 | 173.81 | 173.78 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 12:15:00 | 173.10 | 173.91 | 173.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 13:15:00 | 171.91 | 173.51 | 173.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 09:15:00 | 173.30 | 173.04 | 173.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 09:15:00 | 173.30 | 173.04 | 173.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 173.30 | 173.04 | 173.43 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2024-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 12:15:00 | 176.15 | 174.08 | 173.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 13:15:00 | 177.62 | 174.79 | 174.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 11:15:00 | 186.78 | 186.89 | 184.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-17 12:00:00 | 186.78 | 186.89 | 184.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 185.02 | 186.74 | 185.13 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 09:15:00 | 181.59 | 184.34 | 184.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 10:15:00 | 177.57 | 182.99 | 183.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 181.97 | 181.73 | 182.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 14:30:00 | 181.62 | 181.73 | 182.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 183.50 | 182.08 | 182.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 184.19 | 182.08 | 182.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 183.95 | 182.46 | 183.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:30:00 | 184.20 | 182.46 | 183.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 184.47 | 182.99 | 183.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 12:00:00 | 184.47 | 182.99 | 183.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 12:15:00 | 184.19 | 183.23 | 183.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 13:15:00 | 184.11 | 183.23 | 183.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2024-09-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 14:15:00 | 184.00 | 183.35 | 183.31 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 09:15:00 | 181.84 | 183.07 | 183.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-23 10:15:00 | 181.27 | 182.71 | 183.02 | Break + close below crossover candle low |

### Cycle 29 — BUY (started 2024-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 09:15:00 | 189.69 | 182.96 | 182.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 13:15:00 | 191.39 | 187.21 | 185.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 09:15:00 | 193.64 | 194.01 | 191.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-26 09:30:00 | 192.79 | 194.01 | 191.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 219.66 | 220.94 | 217.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:30:00 | 217.79 | 220.94 | 217.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 221.45 | 221.04 | 218.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 14:00:00 | 222.13 | 221.17 | 218.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-07 09:15:00 | 217.63 | 220.20 | 219.02 | SL hit (close<static) qty=1.00 sl=218.10 alert=retest2 |

### Cycle 30 — SELL (started 2024-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 12:15:00 | 214.31 | 217.89 | 218.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-08 09:15:00 | 211.07 | 215.23 | 216.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 10:15:00 | 213.80 | 212.18 | 214.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 10:15:00 | 213.80 | 212.18 | 214.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 213.80 | 212.18 | 214.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:45:00 | 214.12 | 212.18 | 214.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 11:15:00 | 213.35 | 212.41 | 214.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 13:00:00 | 212.35 | 212.40 | 213.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-10 09:15:00 | 218.04 | 213.88 | 214.12 | SL hit (close>static) qty=1.00 sl=214.99 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 09:15:00 | 221.00 | 215.11 | 214.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 14:15:00 | 222.72 | 219.20 | 217.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 09:15:00 | 222.00 | 224.25 | 221.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 09:15:00 | 222.00 | 224.25 | 221.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 222.00 | 224.25 | 221.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 09:30:00 | 222.50 | 224.25 | 221.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 219.49 | 222.96 | 221.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 12:00:00 | 219.49 | 222.96 | 221.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 217.97 | 221.96 | 221.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 13:00:00 | 217.97 | 221.96 | 221.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2024-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 14:15:00 | 218.78 | 220.64 | 220.74 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-17 09:15:00 | 229.10 | 220.56 | 220.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 11:15:00 | 230.52 | 226.47 | 224.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 10:15:00 | 229.40 | 230.07 | 227.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-21 11:00:00 | 229.40 | 230.07 | 227.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 227.50 | 229.77 | 228.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:45:00 | 229.69 | 229.77 | 228.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 226.74 | 229.16 | 228.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:30:00 | 224.89 | 229.16 | 228.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-10-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 12:15:00 | 223.95 | 227.56 | 227.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 14:15:00 | 220.00 | 225.50 | 226.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 227.44 | 225.23 | 226.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 09:15:00 | 227.44 | 225.23 | 226.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 227.44 | 225.23 | 226.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:00:00 | 227.44 | 225.23 | 226.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 227.00 | 225.59 | 226.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:45:00 | 228.00 | 225.59 | 226.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 226.34 | 225.74 | 226.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:45:00 | 226.89 | 225.74 | 226.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 226.36 | 225.86 | 226.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:00:00 | 226.36 | 225.86 | 226.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 225.80 | 225.85 | 226.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:30:00 | 226.81 | 225.85 | 226.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 14:15:00 | 223.64 | 225.41 | 226.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 09:15:00 | 221.00 | 225.13 | 225.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-24 10:15:00 | 226.30 | 225.53 | 225.95 | SL hit (close>static) qty=1.00 sl=226.18 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 13:15:00 | 224.40 | 223.10 | 222.96 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 11:15:00 | 222.09 | 222.94 | 222.99 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 13:15:00 | 224.20 | 223.10 | 223.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 14:15:00 | 229.06 | 224.29 | 223.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 227.48 | 227.51 | 225.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 14:00:00 | 227.48 | 227.51 | 225.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 226.04 | 227.19 | 226.10 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2024-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 12:15:00 | 222.64 | 225.12 | 225.31 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 15:15:00 | 227.95 | 225.86 | 225.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 17:15:00 | 230.41 | 226.77 | 226.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 226.98 | 227.16 | 226.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 226.98 | 227.16 | 226.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 226.98 | 227.16 | 226.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 226.98 | 227.16 | 226.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 223.94 | 226.52 | 226.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 223.94 | 226.52 | 226.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 227.13 | 226.64 | 226.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 12:30:00 | 228.08 | 227.09 | 226.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 13:00:00 | 228.90 | 227.09 | 226.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 13:45:00 | 228.19 | 227.59 | 226.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 10:15:00 | 234.69 | 237.58 | 237.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 10:15:00 | 234.69 | 237.58 | 237.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 09:15:00 | 229.82 | 233.80 | 235.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 225.47 | 222.93 | 226.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 225.47 | 222.93 | 226.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 225.47 | 222.93 | 226.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 11:00:00 | 222.06 | 222.76 | 226.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 12:30:00 | 221.98 | 223.15 | 225.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 13:45:00 | 221.81 | 222.86 | 225.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 09:15:00 | 237.82 | 225.01 | 225.64 | SL hit (close>static) qty=1.00 sl=230.67 alert=retest2 |

### Cycle 41 — BUY (started 2024-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 10:15:00 | 238.67 | 227.74 | 226.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-18 11:15:00 | 243.64 | 230.92 | 228.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 11:15:00 | 238.00 | 238.13 | 234.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-19 12:00:00 | 238.00 | 238.13 | 234.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 11:15:00 | 247.53 | 252.09 | 249.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 12:00:00 | 247.53 | 252.09 | 249.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 12:15:00 | 250.10 | 251.69 | 249.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 13:15:00 | 251.58 | 251.69 | 249.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 15:00:00 | 251.99 | 251.35 | 249.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-27 09:15:00 | 246.59 | 250.90 | 250.75 | SL hit (close<static) qty=1.00 sl=246.61 alert=retest2 |

### Cycle 42 — SELL (started 2024-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 10:15:00 | 246.83 | 250.09 | 250.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 11:15:00 | 245.19 | 249.11 | 249.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 10:15:00 | 247.98 | 247.50 | 248.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 10:15:00 | 247.98 | 247.50 | 248.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 247.98 | 247.50 | 248.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 11:00:00 | 247.98 | 247.50 | 248.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 249.65 | 247.93 | 248.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 11:45:00 | 250.54 | 247.93 | 248.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 248.07 | 247.96 | 248.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 09:15:00 | 243.52 | 248.58 | 248.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 15:15:00 | 246.19 | 244.41 | 244.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2024-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 15:15:00 | 246.19 | 244.41 | 244.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 09:15:00 | 247.68 | 245.06 | 244.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 14:15:00 | 245.41 | 245.62 | 245.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 14:15:00 | 245.41 | 245.62 | 245.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 245.41 | 245.62 | 245.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 15:00:00 | 245.41 | 245.62 | 245.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 15:15:00 | 245.30 | 245.56 | 245.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 09:15:00 | 248.80 | 245.56 | 245.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 09:15:00 | 245.59 | 249.40 | 249.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 245.59 | 249.40 | 249.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 10:15:00 | 240.64 | 247.65 | 248.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-17 09:15:00 | 223.59 | 222.67 | 227.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 12:15:00 | 226.62 | 224.35 | 227.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 226.62 | 224.35 | 227.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:00:00 | 226.62 | 224.35 | 227.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 15:15:00 | 226.28 | 225.29 | 227.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 09:15:00 | 227.15 | 225.29 | 227.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 226.88 | 225.61 | 227.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 09:45:00 | 227.40 | 225.61 | 227.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 227.30 | 225.95 | 227.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 10:30:00 | 227.10 | 225.95 | 227.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 11:15:00 | 225.51 | 225.86 | 226.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 11:30:00 | 226.90 | 225.86 | 226.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 13:15:00 | 226.04 | 225.85 | 226.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 13:45:00 | 226.32 | 225.85 | 226.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 14:15:00 | 226.83 | 226.05 | 226.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 14:45:00 | 226.12 | 226.05 | 226.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 15:15:00 | 227.12 | 226.26 | 226.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 09:15:00 | 222.33 | 226.26 | 226.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 09:15:00 | 211.21 | 214.30 | 215.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-31 10:15:00 | 210.68 | 210.51 | 212.47 | SL hit (close>ema200) qty=0.50 sl=210.51 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 13:15:00 | 214.13 | 212.74 | 212.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 13:15:00 | 216.44 | 214.18 | 213.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 10:15:00 | 213.94 | 214.92 | 214.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 10:15:00 | 213.94 | 214.92 | 214.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 213.94 | 214.92 | 214.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 11:00:00 | 213.94 | 214.92 | 214.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 11:15:00 | 215.76 | 215.09 | 214.28 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 13:15:00 | 209.34 | 213.02 | 213.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 14:15:00 | 207.85 | 211.99 | 212.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 206.25 | 203.72 | 206.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 206.25 | 203.72 | 206.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 206.25 | 203.72 | 206.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 207.14 | 203.72 | 206.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 205.73 | 204.12 | 206.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:30:00 | 205.69 | 204.12 | 206.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 204.75 | 204.25 | 206.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:30:00 | 206.68 | 204.25 | 206.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 206.12 | 204.64 | 206.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:30:00 | 205.55 | 204.64 | 206.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 206.34 | 204.98 | 206.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:30:00 | 205.46 | 204.98 | 206.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 205.06 | 204.99 | 206.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 203.23 | 204.99 | 206.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 14:15:00 | 193.07 | 196.97 | 199.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 09:15:00 | 196.29 | 195.93 | 198.51 | SL hit (close>ema200) qty=0.50 sl=195.93 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 14:15:00 | 198.47 | 197.66 | 197.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 201.40 | 198.62 | 198.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 10:15:00 | 207.62 | 208.92 | 205.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 11:00:00 | 207.62 | 208.92 | 205.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 206.31 | 208.20 | 206.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 15:00:00 | 206.31 | 208.20 | 206.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 205.89 | 207.74 | 206.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 201.43 | 207.74 | 206.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 200.12 | 205.34 | 205.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 199.51 | 204.17 | 205.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 203.77 | 202.95 | 204.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 203.77 | 202.95 | 204.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 207.16 | 203.77 | 204.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 207.16 | 203.77 | 204.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 207.05 | 204.43 | 204.56 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2025-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 11:15:00 | 207.20 | 204.98 | 204.80 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 14:15:00 | 203.20 | 204.82 | 204.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 198.81 | 203.30 | 204.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 193.50 | 192.88 | 196.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 09:45:00 | 194.30 | 192.88 | 196.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 195.56 | 193.79 | 195.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:45:00 | 195.98 | 193.79 | 195.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 195.24 | 194.08 | 195.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 12:15:00 | 193.66 | 194.07 | 195.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 13:15:00 | 193.08 | 194.05 | 194.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-31 10:15:00 | 199.15 | 194.72 | 194.80 | SL hit (close>static) qty=1.00 sl=195.88 alert=retest2 |

### Cycle 51 — BUY (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 11:15:00 | 200.03 | 195.78 | 195.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 12:15:00 | 200.76 | 196.77 | 195.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 197.43 | 199.38 | 198.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 197.43 | 199.38 | 198.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 197.43 | 199.38 | 198.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 197.43 | 199.38 | 198.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 199.39 | 199.38 | 198.13 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 189.92 | 197.04 | 197.34 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 200.50 | 195.73 | 195.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 10:15:00 | 203.00 | 197.19 | 195.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 10:15:00 | 199.00 | 199.23 | 197.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 11:00:00 | 199.00 | 199.23 | 197.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 197.61 | 198.90 | 197.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:00:00 | 197.61 | 198.90 | 197.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 197.49 | 198.62 | 197.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 197.49 | 198.62 | 197.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 195.91 | 198.08 | 197.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:45:00 | 196.23 | 198.08 | 197.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 196.00 | 197.66 | 197.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 15:00:00 | 196.00 | 197.66 | 197.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 197.22 | 197.57 | 197.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 09:15:00 | 199.45 | 197.57 | 197.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:00:00 | 199.40 | 197.94 | 197.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:30:00 | 200.47 | 198.65 | 197.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 194.46 | 198.86 | 198.54 | SL hit (close<static) qty=1.00 sl=195.60 alert=retest2 |

### Cycle 54 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 195.19 | 198.13 | 198.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 193.51 | 197.21 | 197.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 187.90 | 186.63 | 190.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 10:45:00 | 187.25 | 186.63 | 190.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 191.80 | 188.00 | 190.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 191.80 | 188.00 | 190.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 188.90 | 188.18 | 189.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 09:15:00 | 187.93 | 188.69 | 189.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-13 09:15:00 | 192.90 | 189.53 | 190.17 | SL hit (close>static) qty=1.00 sl=191.88 alert=retest2 |

### Cycle 55 — BUY (started 2025-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 11:15:00 | 192.82 | 190.75 | 190.65 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 09:15:00 | 184.42 | 189.91 | 190.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 10:15:00 | 182.44 | 188.42 | 189.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 14:15:00 | 180.89 | 180.41 | 183.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 15:00:00 | 180.89 | 180.41 | 183.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 185.54 | 180.13 | 181.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 185.54 | 180.13 | 181.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 185.12 | 181.13 | 181.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:30:00 | 185.72 | 181.13 | 181.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 186.16 | 182.13 | 181.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 13:15:00 | 186.71 | 183.74 | 182.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 09:15:00 | 195.71 | 197.67 | 193.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 09:15:00 | 195.71 | 197.67 | 193.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 195.71 | 197.67 | 193.89 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2025-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 15:15:00 | 189.10 | 192.76 | 192.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 09:15:00 | 186.37 | 191.48 | 192.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 178.95 | 177.24 | 179.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 12:45:00 | 177.70 | 177.24 | 179.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 179.14 | 177.62 | 179.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 14:15:00 | 179.94 | 177.62 | 179.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 179.68 | 178.03 | 179.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 15:00:00 | 179.68 | 178.03 | 179.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 179.11 | 178.25 | 179.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 09:15:00 | 177.00 | 178.25 | 179.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-04 09:15:00 | 180.54 | 178.71 | 179.32 | SL hit (close>static) qty=1.00 sl=179.82 alert=retest2 |

### Cycle 59 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 181.78 | 179.77 | 179.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 10:15:00 | 183.75 | 180.57 | 180.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 193.06 | 193.61 | 191.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 10:00:00 | 193.06 | 193.61 | 191.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 191.20 | 192.89 | 191.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 191.20 | 192.89 | 191.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 191.24 | 192.56 | 191.19 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 188.79 | 190.20 | 190.37 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 14:15:00 | 191.79 | 189.87 | 189.71 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 12:15:00 | 188.64 | 189.68 | 189.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 13:15:00 | 188.16 | 189.38 | 189.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 189.32 | 186.92 | 187.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 09:15:00 | 189.32 | 186.92 | 187.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 189.32 | 186.92 | 187.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 189.32 | 186.92 | 187.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 187.90 | 187.11 | 187.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 12:30:00 | 186.92 | 187.41 | 187.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 13:00:00 | 187.40 | 187.41 | 187.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-19 09:15:00 | 186.97 | 187.87 | 187.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-20 09:30:00 | 187.21 | 186.45 | 186.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 187.90 | 186.74 | 187.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 11:00:00 | 187.90 | 186.74 | 187.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-20 11:15:00 | 191.29 | 187.65 | 187.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2025-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 11:15:00 | 191.29 | 187.65 | 187.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 12:15:00 | 191.75 | 188.47 | 187.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 09:15:00 | 186.04 | 188.72 | 188.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-21 09:15:00 | 186.04 | 188.72 | 188.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 186.04 | 188.72 | 188.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 10:00:00 | 186.04 | 188.72 | 188.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2025-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-21 10:15:00 | 184.76 | 187.93 | 187.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-21 12:15:00 | 183.17 | 186.48 | 187.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 14:15:00 | 185.88 | 185.83 | 186.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-21 15:00:00 | 185.88 | 185.83 | 186.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 187.74 | 186.15 | 186.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 09:15:00 | 184.58 | 186.85 | 186.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-28 15:15:00 | 175.35 | 177.25 | 178.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-02 15:15:00 | 174.84 | 174.59 | 175.76 | SL hit (close>ema200) qty=0.50 sl=174.59 alert=retest2 |

### Cycle 65 — BUY (started 2025-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 12:15:00 | 148.78 | 146.99 | 146.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 14:15:00 | 151.07 | 148.08 | 147.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 14:15:00 | 159.27 | 160.20 | 157.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 14:45:00 | 159.02 | 160.20 | 157.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 158.46 | 159.45 | 158.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 158.19 | 159.45 | 158.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 159.05 | 161.44 | 160.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 159.05 | 161.44 | 160.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 157.35 | 160.62 | 160.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 157.35 | 160.62 | 160.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 157.54 | 160.00 | 160.12 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 15:15:00 | 160.08 | 159.31 | 159.30 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 12:15:00 | 158.21 | 159.18 | 159.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 13:15:00 | 158.07 | 158.96 | 159.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-30 10:15:00 | 158.49 | 158.47 | 158.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 10:15:00 | 158.49 | 158.47 | 158.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 158.49 | 158.47 | 158.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 13:00:00 | 157.52 | 158.19 | 158.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-02 09:15:00 | 162.05 | 158.24 | 158.42 | SL hit (close>static) qty=1.00 sl=159.25 alert=retest2 |

### Cycle 69 — BUY (started 2025-05-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 10:15:00 | 160.32 | 158.66 | 158.60 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 11:15:00 | 157.26 | 159.30 | 159.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 155.07 | 157.67 | 158.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 157.75 | 156.98 | 157.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 10:30:00 | 157.29 | 156.98 | 157.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 157.92 | 157.17 | 157.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:30:00 | 158.17 | 157.17 | 157.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 158.19 | 157.37 | 158.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:45:00 | 158.30 | 157.37 | 158.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 158.60 | 157.62 | 158.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:00:00 | 158.60 | 157.62 | 158.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 158.24 | 157.74 | 158.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 09:15:00 | 157.46 | 157.84 | 158.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 09:15:00 | 158.85 | 158.04 | 158.15 | SL hit (close>static) qty=1.00 sl=158.80 alert=retest2 |

### Cycle 71 — BUY (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 11:15:00 | 158.53 | 158.26 | 158.24 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 12:15:00 | 156.98 | 158.00 | 158.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 155.11 | 157.42 | 157.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 11:15:00 | 155.98 | 155.50 | 156.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 12:00:00 | 155.98 | 155.50 | 156.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 12:15:00 | 156.60 | 155.72 | 156.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 12:45:00 | 156.68 | 155.72 | 156.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 13:15:00 | 156.56 | 155.88 | 156.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 14:15:00 | 155.55 | 155.88 | 156.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 15:15:00 | 157.10 | 156.26 | 156.62 | SL hit (close>static) qty=1.00 sl=156.90 alert=retest2 |

### Cycle 73 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 163.37 | 157.68 | 157.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 164.64 | 159.07 | 157.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 165.90 | 166.74 | 163.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 14:00:00 | 165.90 | 166.74 | 163.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 179.26 | 180.90 | 179.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 179.26 | 180.90 | 179.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 179.90 | 180.70 | 179.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 10:00:00 | 181.28 | 180.82 | 179.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 13:00:00 | 181.62 | 180.82 | 180.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 11:15:00 | 182.78 | 183.12 | 183.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 11:15:00 | 182.78 | 183.12 | 183.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 12:15:00 | 182.25 | 182.95 | 183.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 09:15:00 | 183.00 | 182.58 | 182.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 09:15:00 | 183.00 | 182.58 | 182.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 183.00 | 182.58 | 182.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:30:00 | 183.50 | 182.58 | 182.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 183.00 | 182.66 | 182.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:00:00 | 183.00 | 182.66 | 182.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 182.58 | 182.64 | 182.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 12:45:00 | 182.32 | 182.60 | 182.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 14:15:00 | 182.26 | 182.59 | 182.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 15:15:00 | 183.77 | 182.89 | 182.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2025-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 15:15:00 | 183.77 | 182.89 | 182.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 09:15:00 | 184.76 | 183.27 | 183.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 13:15:00 | 183.35 | 183.55 | 183.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 14:00:00 | 183.35 | 183.55 | 183.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 183.61 | 183.56 | 183.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:45:00 | 183.53 | 183.56 | 183.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 183.70 | 183.59 | 183.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 182.92 | 183.59 | 183.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 181.41 | 183.15 | 183.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 180.07 | 181.85 | 182.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 11:15:00 | 181.20 | 181.18 | 181.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 11:15:00 | 181.20 | 181.18 | 181.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 181.20 | 181.18 | 181.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:45:00 | 181.26 | 181.18 | 181.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 181.45 | 181.32 | 181.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:00:00 | 181.45 | 181.32 | 181.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 182.17 | 181.30 | 181.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:30:00 | 181.93 | 181.30 | 181.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 183.58 | 181.76 | 181.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:00:00 | 183.58 | 181.76 | 181.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 11:15:00 | 183.45 | 182.10 | 182.00 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 181.41 | 182.26 | 182.29 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 14:15:00 | 182.80 | 182.37 | 182.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 15:15:00 | 182.89 | 182.47 | 182.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 13:15:00 | 187.08 | 187.60 | 186.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 14:00:00 | 187.08 | 187.60 | 186.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 190.00 | 190.67 | 189.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:45:00 | 189.83 | 190.67 | 189.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 191.05 | 190.74 | 189.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 12:30:00 | 191.26 | 190.87 | 189.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 09:15:00 | 188.66 | 190.07 | 189.60 | SL hit (close<static) qty=1.00 sl=189.21 alert=retest2 |

### Cycle 80 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 186.92 | 188.92 | 189.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 10:15:00 | 186.75 | 187.96 | 188.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 186.74 | 186.19 | 187.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 12:00:00 | 186.74 | 186.19 | 187.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 187.20 | 186.39 | 187.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:45:00 | 186.91 | 186.39 | 187.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 187.82 | 186.68 | 187.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 187.82 | 186.68 | 187.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 188.26 | 186.99 | 187.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 15:15:00 | 187.60 | 186.99 | 187.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:00:00 | 187.77 | 187.25 | 187.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:30:00 | 187.63 | 187.32 | 187.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 12:15:00 | 187.89 | 185.05 | 184.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 187.89 | 185.05 | 184.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 13:15:00 | 188.26 | 185.70 | 185.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 189.12 | 189.16 | 187.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 14:00:00 | 189.12 | 189.16 | 187.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 188.69 | 189.44 | 188.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:00:00 | 188.69 | 189.44 | 188.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 188.25 | 189.20 | 188.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:45:00 | 187.88 | 189.20 | 188.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 187.85 | 188.93 | 188.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 188.98 | 188.93 | 188.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 10:30:00 | 188.87 | 188.94 | 188.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 10:15:00 | 188.82 | 191.01 | 191.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 188.82 | 191.01 | 191.06 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 10:15:00 | 192.30 | 191.06 | 190.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 11:15:00 | 194.30 | 191.71 | 191.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 12:15:00 | 192.70 | 192.76 | 192.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-03 13:00:00 | 192.70 | 192.76 | 192.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 192.36 | 192.64 | 192.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:30:00 | 192.63 | 192.64 | 192.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 192.18 | 192.54 | 192.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:15:00 | 192.78 | 192.54 | 192.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 193.29 | 192.69 | 192.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 10:30:00 | 193.62 | 192.79 | 192.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 13:15:00 | 191.16 | 192.21 | 192.20 | SL hit (close<static) qty=1.00 sl=191.80 alert=retest2 |

### Cycle 84 — SELL (started 2025-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 15:15:00 | 191.90 | 192.16 | 192.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 09:15:00 | 189.29 | 191.59 | 191.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 12:15:00 | 190.44 | 189.30 | 190.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 12:15:00 | 190.44 | 189.30 | 190.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 190.44 | 189.30 | 190.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:30:00 | 189.88 | 189.30 | 190.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 189.94 | 189.43 | 190.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:30:00 | 190.21 | 189.43 | 190.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 189.73 | 189.49 | 190.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 189.73 | 189.49 | 190.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 189.80 | 189.55 | 189.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 190.05 | 189.55 | 189.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 188.87 | 189.42 | 189.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 11:45:00 | 187.84 | 188.85 | 189.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 190.80 | 188.56 | 188.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 09:15:00 | 190.80 | 188.56 | 188.53 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 10:15:00 | 188.35 | 189.50 | 189.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 14:15:00 | 188.07 | 188.80 | 189.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 09:15:00 | 190.05 | 188.94 | 189.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 190.05 | 188.94 | 189.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 190.05 | 188.94 | 189.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:30:00 | 190.98 | 188.94 | 189.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 190.37 | 189.23 | 189.28 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 11:15:00 | 189.77 | 189.34 | 189.32 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-07-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 15:15:00 | 188.31 | 189.18 | 189.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 188.11 | 188.97 | 189.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 14:15:00 | 188.12 | 188.10 | 188.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-18 15:00:00 | 188.12 | 188.10 | 188.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 89 — BUY (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 09:15:00 | 193.86 | 189.25 | 189.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 10:15:00 | 195.55 | 190.51 | 189.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 10:15:00 | 197.28 | 197.43 | 196.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 11:00:00 | 197.28 | 197.43 | 196.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 196.59 | 197.89 | 196.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:00:00 | 196.59 | 197.89 | 196.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 195.33 | 197.38 | 196.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:00:00 | 195.33 | 197.38 | 196.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2025-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 12:15:00 | 194.27 | 196.31 | 196.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 15:15:00 | 193.50 | 195.08 | 195.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 11:15:00 | 189.37 | 189.17 | 191.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 12:00:00 | 189.37 | 189.17 | 191.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 187.42 | 188.91 | 190.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:45:00 | 187.04 | 188.67 | 190.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:30:00 | 186.99 | 187.48 | 188.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 14:15:00 | 187.20 | 185.06 | 185.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 187.20 | 185.06 | 185.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 10:15:00 | 187.65 | 186.21 | 185.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 13:15:00 | 186.17 | 186.33 | 185.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 13:15:00 | 186.17 | 186.33 | 185.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 186.17 | 186.33 | 185.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 15:00:00 | 187.39 | 186.54 | 185.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 12:00:00 | 186.95 | 187.58 | 187.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 12:15:00 | 185.27 | 187.11 | 187.00 | SL hit (close<static) qty=1.00 sl=185.72 alert=retest2 |

### Cycle 92 — SELL (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 09:15:00 | 186.65 | 187.41 | 187.43 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 10:15:00 | 187.62 | 187.45 | 187.44 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 11:15:00 | 186.30 | 187.22 | 187.34 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 12:15:00 | 188.23 | 187.46 | 187.36 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 14:15:00 | 186.54 | 187.22 | 187.26 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 190.80 | 187.83 | 187.53 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 187.12 | 188.18 | 188.19 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 189.09 | 188.08 | 188.05 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 11:15:00 | 187.71 | 188.01 | 188.02 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 12:15:00 | 188.56 | 188.12 | 188.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 13:15:00 | 188.68 | 188.23 | 188.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 15:15:00 | 188.25 | 188.26 | 188.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 15:15:00 | 188.25 | 188.26 | 188.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 188.25 | 188.26 | 188.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 09:15:00 | 188.51 | 188.26 | 188.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 12:15:00 | 189.67 | 190.67 | 190.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 12:15:00 | 189.67 | 190.67 | 190.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 188.75 | 190.13 | 190.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 191.70 | 190.24 | 190.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 191.70 | 190.24 | 190.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 191.70 | 190.24 | 190.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:30:00 | 192.20 | 190.24 | 190.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 191.40 | 190.47 | 190.57 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 11:15:00 | 191.36 | 190.65 | 190.64 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 189.40 | 190.42 | 190.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 185.87 | 188.44 | 189.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 188.00 | 186.14 | 187.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 188.00 | 186.14 | 187.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 188.00 | 186.14 | 187.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 188.00 | 186.14 | 187.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 187.72 | 186.46 | 187.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:45:00 | 188.12 | 186.46 | 187.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 187.04 | 186.66 | 187.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 13:30:00 | 187.29 | 186.66 | 187.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 186.17 | 186.56 | 187.12 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 189.05 | 187.37 | 187.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 190.62 | 188.02 | 187.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 14:15:00 | 206.29 | 206.63 | 203.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 15:00:00 | 206.29 | 206.63 | 203.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 208.11 | 210.91 | 209.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:30:00 | 208.10 | 210.91 | 209.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 207.40 | 210.21 | 209.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:45:00 | 207.66 | 210.21 | 209.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 14:15:00 | 206.48 | 208.40 | 208.58 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 210.00 | 208.92 | 208.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 09:15:00 | 213.40 | 210.04 | 209.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 13:15:00 | 216.45 | 217.87 | 216.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 14:00:00 | 216.45 | 217.87 | 216.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 216.55 | 217.60 | 216.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 217.89 | 217.50 | 216.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 10:15:00 | 215.00 | 216.80 | 216.07 | SL hit (close<static) qty=1.00 sl=216.07 alert=retest2 |

### Cycle 108 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 212.73 | 215.37 | 215.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 09:15:00 | 211.58 | 212.97 | 213.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 09:15:00 | 213.56 | 212.30 | 212.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 09:15:00 | 213.56 | 212.30 | 212.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 213.56 | 212.30 | 212.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:45:00 | 213.49 | 212.30 | 212.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 212.83 | 212.40 | 212.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 12:00:00 | 212.34 | 212.39 | 212.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 12:45:00 | 212.22 | 212.34 | 212.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 201.72 | 205.70 | 207.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 201.61 | 205.70 | 207.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 204.22 | 202.47 | 204.52 | SL hit (close>ema200) qty=0.50 sl=202.47 alert=retest2 |

### Cycle 109 — BUY (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 10:15:00 | 209.44 | 205.47 | 205.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 11:15:00 | 214.20 | 207.22 | 205.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 217.94 | 219.77 | 216.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 11:00:00 | 217.94 | 219.77 | 216.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 217.11 | 218.42 | 217.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 14:45:00 | 217.33 | 218.42 | 217.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 217.29 | 218.20 | 217.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:15:00 | 218.27 | 218.20 | 217.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 218.78 | 218.31 | 217.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 222.40 | 217.58 | 217.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 13:15:00 | 223.32 | 224.95 | 224.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2025-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 13:15:00 | 223.32 | 224.95 | 224.97 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 227.23 | 224.98 | 224.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 229.75 | 227.07 | 226.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 224.99 | 227.15 | 226.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 12:15:00 | 224.99 | 227.15 | 226.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 224.99 | 227.15 | 226.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 224.99 | 227.15 | 226.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 225.80 | 226.88 | 226.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:45:00 | 225.62 | 226.88 | 226.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 15:15:00 | 224.75 | 226.30 | 226.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 09:15:00 | 223.27 | 225.69 | 226.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 12:15:00 | 225.77 | 225.44 | 225.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 12:15:00 | 225.77 | 225.44 | 225.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 225.77 | 225.44 | 225.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:00:00 | 225.77 | 225.44 | 225.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 226.53 | 225.66 | 225.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:45:00 | 226.66 | 225.66 | 225.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 225.93 | 225.71 | 225.89 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 227.68 | 226.19 | 226.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 11:15:00 | 229.80 | 227.46 | 226.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 228.20 | 228.38 | 227.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 15:00:00 | 228.20 | 228.38 | 227.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 236.75 | 237.52 | 236.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:30:00 | 235.92 | 237.52 | 236.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 236.20 | 237.69 | 237.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:45:00 | 235.04 | 237.69 | 237.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 235.15 | 237.18 | 236.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:00:00 | 235.15 | 237.18 | 236.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 234.62 | 236.67 | 236.77 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 15:15:00 | 238.80 | 237.18 | 236.96 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 09:15:00 | 235.24 | 236.79 | 236.81 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 10:15:00 | 238.29 | 236.52 | 236.41 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 232.81 | 236.10 | 236.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 12:15:00 | 232.46 | 235.37 | 236.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 09:15:00 | 231.12 | 231.08 | 232.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 09:45:00 | 230.90 | 231.08 | 232.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 235.15 | 231.89 | 232.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:00:00 | 235.15 | 231.89 | 232.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 235.00 | 232.51 | 233.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:45:00 | 235.89 | 232.51 | 233.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2025-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 12:15:00 | 237.63 | 233.54 | 233.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 09:15:00 | 252.75 | 237.85 | 235.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 09:15:00 | 264.31 | 267.50 | 264.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 09:15:00 | 264.31 | 267.50 | 264.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 264.31 | 267.50 | 264.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:15:00 | 262.80 | 267.50 | 264.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 262.47 | 266.49 | 264.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:00:00 | 262.47 | 266.49 | 264.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 09:15:00 | 262.30 | 263.38 | 263.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 255.84 | 260.80 | 262.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 09:15:00 | 258.35 | 257.93 | 259.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 258.35 | 257.93 | 259.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 258.35 | 257.93 | 259.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:30:00 | 259.56 | 257.93 | 259.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 260.23 | 257.56 | 258.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:30:00 | 261.55 | 257.56 | 258.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 258.85 | 257.81 | 258.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:30:00 | 260.11 | 257.81 | 258.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 258.99 | 258.21 | 258.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 14:00:00 | 258.99 | 258.21 | 258.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 257.92 | 258.15 | 258.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 254.67 | 258.27 | 258.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 257.48 | 254.25 | 253.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 257.48 | 254.25 | 253.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 09:15:00 | 260.52 | 257.58 | 256.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 13:15:00 | 261.00 | 261.14 | 259.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 14:00:00 | 261.00 | 261.14 | 259.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 260.40 | 260.76 | 259.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 266.55 | 260.76 | 259.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 09:15:00 | 263.40 | 268.68 | 269.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 263.40 | 268.68 | 269.27 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 271.70 | 265.67 | 265.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 10:15:00 | 272.60 | 267.05 | 266.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 276.45 | 277.53 | 274.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 276.45 | 277.53 | 274.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 276.45 | 277.53 | 274.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 15:00:00 | 277.50 | 276.40 | 275.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:15:00 | 279.35 | 276.42 | 275.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 10:15:00 | 279.50 | 278.39 | 277.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 12:30:00 | 277.60 | 278.40 | 278.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 277.45 | 278.21 | 278.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 14:30:00 | 278.15 | 278.27 | 278.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 15:00:00 | 278.50 | 278.27 | 278.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-12-26 11:15:00 | 305.25 | 299.44 | 294.83 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 334.40 | 340.64 | 340.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 334.00 | 339.31 | 340.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 342.35 | 338.91 | 339.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 342.35 | 338.91 | 339.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 342.35 | 338.91 | 339.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:00:00 | 342.35 | 338.91 | 339.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 342.70 | 339.67 | 340.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:45:00 | 343.55 | 339.67 | 340.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 11:15:00 | 344.60 | 340.66 | 340.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-09 12:15:00 | 346.55 | 341.83 | 341.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-12 13:15:00 | 346.10 | 347.37 | 345.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:00:00 | 346.10 | 347.37 | 345.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 350.20 | 347.94 | 345.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:30:00 | 346.95 | 347.94 | 345.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 361.85 | 364.96 | 361.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:00:00 | 361.85 | 364.96 | 361.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 361.35 | 364.24 | 361.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:30:00 | 361.65 | 364.24 | 361.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 361.75 | 363.74 | 361.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 364.25 | 363.74 | 361.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 10:45:00 | 363.85 | 363.16 | 361.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 10:00:00 | 362.95 | 365.72 | 364.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 10:30:00 | 363.25 | 364.98 | 363.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 359.75 | 363.93 | 363.47 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-20 11:15:00 | 359.75 | 363.93 | 363.47 | SL hit (close<static) qty=1.00 sl=360.40 alert=retest2 |

### Cycle 126 — SELL (started 2026-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 13:15:00 | 359.60 | 362.54 | 362.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 14:15:00 | 358.50 | 361.73 | 362.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 361.50 | 360.49 | 361.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 12:15:00 | 361.50 | 360.49 | 361.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 361.50 | 360.49 | 361.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 12:45:00 | 361.70 | 360.49 | 361.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 356.50 | 359.69 | 360.98 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2026-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 12:15:00 | 363.10 | 361.69 | 361.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 13:15:00 | 364.75 | 362.30 | 361.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 14:15:00 | 370.30 | 371.83 | 368.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 14:45:00 | 370.75 | 371.83 | 368.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 396.00 | 415.53 | 405.67 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2026-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 14:15:00 | 384.75 | 398.25 | 399.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 15:15:00 | 383.00 | 395.20 | 398.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 367.85 | 363.38 | 375.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 15:15:00 | 371.50 | 365.22 | 371.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 371.50 | 365.22 | 371.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 370.75 | 365.22 | 371.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 366.75 | 365.52 | 370.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 10:15:00 | 364.30 | 365.52 | 370.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 10:15:00 | 374.00 | 371.55 | 371.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2026-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 10:15:00 | 374.00 | 371.55 | 371.38 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 360.25 | 370.93 | 371.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 351.80 | 362.26 | 366.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 363.40 | 357.03 | 360.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 363.40 | 357.03 | 360.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 363.40 | 357.03 | 360.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 363.40 | 357.03 | 360.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 367.50 | 359.13 | 361.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 367.50 | 359.13 | 361.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2026-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 14:15:00 | 365.00 | 362.27 | 362.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 366.20 | 363.58 | 362.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 12:15:00 | 366.25 | 367.14 | 365.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 12:45:00 | 367.40 | 367.14 | 365.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 369.00 | 367.51 | 366.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 09:30:00 | 370.30 | 368.58 | 366.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 352.20 | 366.09 | 366.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 352.20 | 366.09 | 366.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 349.00 | 356.53 | 361.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 15:15:00 | 348.90 | 348.65 | 353.48 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 09:15:00 | 341.75 | 348.65 | 353.48 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 342.70 | 342.91 | 344.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 10:15:00 | 342.05 | 342.91 | 344.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 14:15:00 | 342.50 | 342.78 | 344.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 09:15:00 | 342.30 | 341.21 | 342.08 | SL hit (close>ema400) qty=1.00 sl=342.08 alert=retest1 |

### Cycle 133 — BUY (started 2026-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 15:15:00 | 345.00 | 340.17 | 340.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 358.00 | 343.74 | 341.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 357.60 | 358.96 | 355.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 14:15:00 | 354.55 | 357.36 | 355.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 354.55 | 357.36 | 355.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 354.55 | 357.36 | 355.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 354.55 | 356.80 | 355.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 357.40 | 356.80 | 355.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 13:15:00 | 360.15 | 358.84 | 357.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 13:45:00 | 356.80 | 358.84 | 357.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 361.05 | 360.74 | 358.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:45:00 | 358.30 | 360.74 | 358.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 13:15:00 | 384.95 | 391.72 | 388.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 14:00:00 | 384.95 | 391.72 | 388.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 14:15:00 | 389.25 | 391.23 | 388.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 15:15:00 | 387.85 | 391.23 | 388.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 15:15:00 | 387.85 | 390.55 | 388.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:30:00 | 383.90 | 389.18 | 388.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 389.15 | 389.17 | 388.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 11:15:00 | 390.35 | 389.17 | 388.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 12:30:00 | 389.80 | 389.21 | 388.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 15:15:00 | 390.70 | 388.71 | 388.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 11:15:00 | 385.25 | 394.21 | 395.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 385.25 | 394.21 | 395.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 377.95 | 387.89 | 391.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 382.00 | 377.71 | 383.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 382.00 | 377.71 | 383.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 382.00 | 377.71 | 383.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 382.00 | 377.71 | 383.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 389.50 | 380.07 | 383.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 389.50 | 380.07 | 383.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 389.00 | 381.85 | 384.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:00:00 | 389.00 | 381.85 | 384.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 395.20 | 387.27 | 386.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 15:15:00 | 396.30 | 389.07 | 387.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 09:15:00 | 385.50 | 388.36 | 387.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 09:15:00 | 385.50 | 388.36 | 387.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 385.50 | 388.36 | 387.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:00:00 | 385.50 | 388.36 | 387.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 386.40 | 387.97 | 387.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:30:00 | 385.50 | 387.97 | 387.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 13:15:00 | 387.00 | 388.13 | 387.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 14:00:00 | 387.00 | 388.13 | 387.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 14:15:00 | 388.20 | 388.14 | 387.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 14:30:00 | 386.10 | 388.14 | 387.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 387.00 | 387.91 | 387.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:15:00 | 387.20 | 387.91 | 387.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 384.85 | 387.30 | 387.18 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 382.20 | 386.28 | 386.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 11:15:00 | 381.10 | 385.24 | 386.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 15:15:00 | 370.50 | 369.97 | 375.39 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 09:15:00 | 351.60 | 369.97 | 375.39 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 363.95 | 356.11 | 359.16 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 363.95 | 356.11 | 359.16 | SL hit (close>ema400) qty=1.00 sl=359.16 alert=retest1 |

### Cycle 137 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 369.05 | 362.42 | 361.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-27 12:15:00 | 371.65 | 366.54 | 364.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 387.95 | 396.31 | 389.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 387.95 | 396.31 | 389.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 387.95 | 396.31 | 389.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 387.95 | 396.31 | 389.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 387.90 | 394.62 | 389.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:45:00 | 387.00 | 394.62 | 389.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 400.85 | 409.07 | 405.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:45:00 | 397.10 | 409.07 | 405.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 403.10 | 407.88 | 405.71 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2026-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-08 14:15:00 | 399.75 | 403.86 | 404.30 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-09 09:15:00 | 411.45 | 404.79 | 404.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 15:15:00 | 413.05 | 410.05 | 407.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 09:15:00 | 407.65 | 409.57 | 407.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 09:15:00 | 407.65 | 409.57 | 407.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 407.65 | 409.57 | 407.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 10:00:00 | 407.65 | 409.57 | 407.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 409.25 | 409.51 | 407.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 13:00:00 | 409.80 | 409.34 | 408.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 13:30:00 | 409.95 | 410.30 | 408.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 15:15:00 | 426.00 | 428.73 | 428.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 426.00 | 428.73 | 428.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 13:15:00 | 424.30 | 426.47 | 427.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 09:15:00 | 431.90 | 426.70 | 427.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 09:15:00 | 431.90 | 426.70 | 427.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 431.90 | 426.70 | 427.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:45:00 | 431.85 | 426.70 | 427.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 431.30 | 427.62 | 427.71 | EMA400 retest candle locked (from downside) |

### Cycle 141 — BUY (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 11:15:00 | 432.80 | 428.66 | 428.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 13:15:00 | 435.50 | 430.71 | 429.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 09:15:00 | 431.30 | 436.97 | 434.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 09:15:00 | 431.30 | 436.97 | 434.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 431.30 | 436.97 | 434.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:45:00 | 430.70 | 436.97 | 434.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 429.10 | 435.40 | 434.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:00:00 | 429.10 | 435.40 | 434.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 437.35 | 435.85 | 434.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 12:45:00 | 438.30 | 435.85 | 434.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 437.00 | 436.08 | 434.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:00:00 | 437.00 | 436.08 | 434.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 443.00 | 440.96 | 438.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:30:00 | 437.60 | 440.96 | 438.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 438.85 | 440.70 | 439.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:45:00 | 439.50 | 440.70 | 439.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 439.05 | 440.37 | 439.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 14:45:00 | 442.00 | 440.47 | 439.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 09:15:00 | 434.35 | 439.65 | 439.18 | SL hit (close<static) qty=1.00 sl=438.55 alert=retest2 |

### Cycle 142 — SELL (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 11:15:00 | 436.20 | 438.40 | 438.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 14:15:00 | 433.70 | 437.24 | 438.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 15:15:00 | 409.60 | 408.45 | 415.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-05 09:15:00 | 409.00 | 408.45 | 415.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 414.05 | 409.57 | 415.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:00:00 | 414.05 | 409.57 | 415.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 417.60 | 411.18 | 415.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:00:00 | 417.60 | 411.18 | 415.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 413.70 | 411.68 | 415.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 14:30:00 | 407.60 | 412.93 | 414.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-22 15:15:00 | 197.90 | 2024-05-23 09:15:00 | 194.75 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-06-11 10:30:00 | 185.30 | 2024-06-19 09:15:00 | 185.90 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2024-06-12 09:15:00 | 185.70 | 2024-06-19 09:15:00 | 185.90 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2024-06-12 13:15:00 | 184.84 | 2024-06-19 09:15:00 | 185.90 | STOP_HIT | 1.00 | 0.57% |
| BUY | retest2 | 2024-06-12 14:15:00 | 184.71 | 2024-06-19 09:15:00 | 185.90 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2024-06-13 12:15:00 | 185.95 | 2024-06-19 09:15:00 | 185.90 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2024-06-13 13:30:00 | 185.91 | 2024-06-19 09:15:00 | 185.90 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2024-06-13 14:15:00 | 186.15 | 2024-06-19 09:15:00 | 185.90 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2024-06-24 11:30:00 | 189.67 | 2024-06-24 15:15:00 | 190.00 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2024-06-28 12:15:00 | 185.12 | 2024-06-28 12:15:00 | 188.97 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-07-23 09:15:00 | 187.10 | 2024-07-23 12:15:00 | 177.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-23 09:15:00 | 187.10 | 2024-07-24 09:15:00 | 188.00 | STOP_HIT | 0.50 | -0.48% |
| BUY | retest2 | 2024-07-30 13:30:00 | 191.69 | 2024-08-02 09:15:00 | 187.32 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2024-07-30 14:15:00 | 191.50 | 2024-08-02 09:15:00 | 187.32 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2024-07-30 14:45:00 | 191.90 | 2024-08-02 09:15:00 | 187.32 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2024-07-31 09:15:00 | 192.02 | 2024-08-02 09:15:00 | 187.32 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2024-07-31 15:00:00 | 194.36 | 2024-08-02 09:15:00 | 187.32 | STOP_HIT | 1.00 | -3.62% |
| SELL | retest2 | 2024-08-08 09:15:00 | 176.87 | 2024-08-14 10:15:00 | 168.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-08 09:15:00 | 176.87 | 2024-08-19 09:15:00 | 166.51 | STOP_HIT | 0.50 | 5.86% |
| SELL | retest2 | 2024-09-06 09:15:00 | 174.90 | 2024-09-10 13:15:00 | 176.90 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2024-09-06 12:45:00 | 175.69 | 2024-09-10 13:15:00 | 176.90 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-09-06 13:30:00 | 175.42 | 2024-09-10 13:15:00 | 176.90 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-10-04 14:00:00 | 222.13 | 2024-10-07 09:15:00 | 217.63 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2024-10-07 09:30:00 | 221.67 | 2024-10-07 10:15:00 | 215.79 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2024-10-09 13:00:00 | 212.35 | 2024-10-10 09:15:00 | 218.04 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2024-10-10 14:30:00 | 211.76 | 2024-10-11 09:15:00 | 221.00 | STOP_HIT | 1.00 | -4.36% |
| SELL | retest2 | 2024-10-24 09:15:00 | 221.00 | 2024-10-24 10:15:00 | 226.30 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2024-10-25 09:15:00 | 218.35 | 2024-10-28 13:15:00 | 224.40 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2024-11-04 12:30:00 | 228.08 | 2024-11-11 10:15:00 | 234.69 | STOP_HIT | 1.00 | 2.90% |
| BUY | retest2 | 2024-11-04 13:00:00 | 228.90 | 2024-11-11 10:15:00 | 234.69 | STOP_HIT | 1.00 | 2.53% |
| BUY | retest2 | 2024-11-04 13:45:00 | 228.19 | 2024-11-11 10:15:00 | 234.69 | STOP_HIT | 1.00 | 2.85% |
| SELL | retest2 | 2024-11-14 11:00:00 | 222.06 | 2024-11-18 09:15:00 | 237.82 | STOP_HIT | 1.00 | -7.10% |
| SELL | retest2 | 2024-11-14 12:30:00 | 221.98 | 2024-11-18 09:15:00 | 237.82 | STOP_HIT | 1.00 | -7.14% |
| SELL | retest2 | 2024-11-14 13:45:00 | 221.81 | 2024-11-18 09:15:00 | 237.82 | STOP_HIT | 1.00 | -7.22% |
| BUY | retest2 | 2024-11-25 13:15:00 | 251.58 | 2024-11-27 09:15:00 | 246.59 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-11-25 15:00:00 | 251.99 | 2024-11-27 09:15:00 | 246.59 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2024-11-29 09:15:00 | 243.52 | 2024-12-03 15:15:00 | 246.19 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-12-05 09:15:00 | 248.80 | 2024-12-12 09:15:00 | 245.59 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-12-19 09:15:00 | 222.33 | 2024-12-30 09:15:00 | 211.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-19 09:15:00 | 222.33 | 2024-12-31 10:15:00 | 210.68 | STOP_HIT | 0.50 | 5.24% |
| SELL | retest2 | 2025-01-08 09:15:00 | 203.23 | 2025-01-13 14:15:00 | 193.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 09:15:00 | 203.23 | 2025-01-14 09:15:00 | 196.29 | STOP_HIT | 0.50 | 3.41% |
| SELL | retest2 | 2025-01-30 12:15:00 | 193.66 | 2025-01-31 10:15:00 | 199.15 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-01-30 13:15:00 | 193.08 | 2025-01-31 10:15:00 | 199.15 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest2 | 2025-02-07 09:15:00 | 199.45 | 2025-02-10 09:15:00 | 194.46 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2025-02-07 10:00:00 | 199.40 | 2025-02-10 09:15:00 | 194.46 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2025-02-07 10:30:00 | 200.47 | 2025-02-10 09:15:00 | 194.46 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-02-13 09:15:00 | 187.93 | 2025-02-13 09:15:00 | 192.90 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2025-03-04 09:15:00 | 177.00 | 2025-03-04 09:15:00 | 180.54 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-03-04 13:30:00 | 178.75 | 2025-03-05 09:15:00 | 181.78 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-03-04 14:30:00 | 178.75 | 2025-03-05 09:15:00 | 181.78 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-03-04 15:15:00 | 178.75 | 2025-03-05 09:15:00 | 181.78 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-03-18 12:30:00 | 186.92 | 2025-03-20 11:15:00 | 191.29 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-03-18 13:00:00 | 187.40 | 2025-03-20 11:15:00 | 191.29 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-03-19 09:15:00 | 186.97 | 2025-03-20 11:15:00 | 191.29 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-03-20 09:30:00 | 187.21 | 2025-03-20 11:15:00 | 191.29 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-03-25 09:15:00 | 184.58 | 2025-03-28 15:15:00 | 175.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-25 09:15:00 | 184.58 | 2025-04-02 15:15:00 | 174.84 | STOP_HIT | 0.50 | 5.28% |
| SELL | retest2 | 2025-04-30 13:00:00 | 157.52 | 2025-05-02 09:15:00 | 162.05 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-05-08 09:15:00 | 157.46 | 2025-05-08 09:15:00 | 158.85 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-05-09 14:15:00 | 155.55 | 2025-05-09 15:15:00 | 157.10 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-05-21 10:00:00 | 181.28 | 2025-05-27 11:15:00 | 182.78 | STOP_HIT | 1.00 | 0.83% |
| BUY | retest2 | 2025-05-21 13:00:00 | 181.62 | 2025-05-27 11:15:00 | 182.78 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest2 | 2025-05-28 12:45:00 | 182.32 | 2025-05-28 15:15:00 | 183.77 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-05-28 14:15:00 | 182.26 | 2025-05-28 15:15:00 | 183.77 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-06-11 12:30:00 | 191.26 | 2025-06-12 09:15:00 | 188.66 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-06-16 15:15:00 | 187.60 | 2025-06-23 12:15:00 | 187.89 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-06-17 10:00:00 | 187.77 | 2025-06-23 12:15:00 | 187.89 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-06-17 10:30:00 | 187.63 | 2025-06-23 12:15:00 | 187.89 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2025-06-26 09:15:00 | 188.98 | 2025-07-01 10:15:00 | 188.82 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-06-26 10:30:00 | 188.87 | 2025-07-01 10:15:00 | 188.82 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-07-04 10:30:00 | 193.62 | 2025-07-04 13:15:00 | 191.16 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-07-09 11:45:00 | 187.84 | 2025-07-11 09:15:00 | 190.80 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-07-30 10:45:00 | 187.04 | 2025-08-04 14:15:00 | 187.20 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-07-31 09:30:00 | 186.99 | 2025-08-04 14:15:00 | 187.20 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-08-05 15:00:00 | 187.39 | 2025-08-07 12:15:00 | 185.27 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-08-07 12:00:00 | 186.95 | 2025-08-07 12:15:00 | 185.27 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-08-07 14:00:00 | 187.15 | 2025-08-11 09:15:00 | 186.65 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-08-08 12:00:00 | 187.00 | 2025-08-11 09:15:00 | 186.65 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-08-11 09:15:00 | 188.42 | 2025-08-11 09:15:00 | 186.65 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-08-19 09:15:00 | 188.51 | 2025-08-22 12:15:00 | 189.67 | STOP_HIT | 1.00 | 0.62% |
| BUY | retest2 | 2025-09-16 09:15:00 | 217.89 | 2025-09-16 10:15:00 | 215.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-09-22 12:00:00 | 212.34 | 2025-09-26 09:15:00 | 201.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 12:45:00 | 212.22 | 2025-09-26 09:15:00 | 201.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 12:00:00 | 212.34 | 2025-09-29 09:15:00 | 204.22 | STOP_HIT | 0.50 | 3.82% |
| SELL | retest2 | 2025-09-22 12:45:00 | 212.22 | 2025-09-29 09:15:00 | 204.22 | STOP_HIT | 0.50 | 3.77% |
| BUY | retest2 | 2025-10-08 09:15:00 | 222.40 | 2025-10-14 13:15:00 | 223.32 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2025-11-21 09:15:00 | 254.67 | 2025-11-26 09:15:00 | 257.48 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-12-01 09:15:00 | 266.55 | 2025-12-09 09:15:00 | 263.40 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-12-16 15:00:00 | 277.50 | 2025-12-26 11:15:00 | 305.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-17 09:15:00 | 279.35 | 2025-12-26 11:15:00 | 307.29 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-18 10:15:00 | 279.50 | 2025-12-26 11:15:00 | 307.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-19 12:30:00 | 277.60 | 2025-12-26 11:15:00 | 305.36 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-19 14:30:00 | 278.15 | 2025-12-26 11:15:00 | 305.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-19 15:00:00 | 278.50 | 2025-12-26 11:15:00 | 306.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-19 09:15:00 | 364.25 | 2026-01-20 11:15:00 | 359.75 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2026-01-19 10:45:00 | 363.85 | 2026-01-20 11:15:00 | 359.75 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2026-01-20 10:00:00 | 362.95 | 2026-01-20 11:15:00 | 359.75 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2026-01-20 10:30:00 | 363.25 | 2026-01-20 11:15:00 | 359.75 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2026-02-03 10:15:00 | 364.30 | 2026-02-04 10:15:00 | 374.00 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2026-02-12 09:30:00 | 370.30 | 2026-02-13 09:15:00 | 352.20 | STOP_HIT | 1.00 | -4.89% |
| SELL | retest1 | 2026-02-17 09:15:00 | 341.75 | 2026-02-23 09:15:00 | 342.30 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2026-02-19 10:15:00 | 342.05 | 2026-02-24 15:15:00 | 345.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2026-02-19 14:15:00 | 342.50 | 2026-02-24 15:15:00 | 345.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2026-02-23 10:00:00 | 342.30 | 2026-02-24 15:15:00 | 345.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2026-03-10 11:15:00 | 390.35 | 2026-03-13 11:15:00 | 385.25 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2026-03-10 12:30:00 | 389.80 | 2026-03-13 11:15:00 | 385.25 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2026-03-10 15:15:00 | 390.70 | 2026-03-13 11:15:00 | 385.25 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest1 | 2026-03-23 09:15:00 | 351.60 | 2026-03-25 09:15:00 | 363.95 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest2 | 2026-04-10 13:00:00 | 409.80 | 2026-04-20 15:15:00 | 426.00 | STOP_HIT | 1.00 | 3.95% |
| BUY | retest2 | 2026-04-10 13:30:00 | 409.95 | 2026-04-20 15:15:00 | 426.00 | STOP_HIT | 1.00 | 3.92% |
| BUY | retest2 | 2026-04-28 14:45:00 | 442.00 | 2026-04-29 09:15:00 | 434.35 | STOP_HIT | 1.00 | -1.73% |
