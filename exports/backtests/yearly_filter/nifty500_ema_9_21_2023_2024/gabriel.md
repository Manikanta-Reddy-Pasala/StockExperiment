# Gabriel India Ltd. (GABRIEL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1136.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 212 |
| ALERT1 | 146 |
| ALERT2 | 140 |
| ALERT2_SKIP | 77 |
| ALERT3 | 395 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 8 |
| ENTRY2 | 177 |
| PARTIAL | 21 |
| TARGET_HIT | 19 |
| STOP_HIT | 163 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 203 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 76 / 127
- **Target hits / Stop hits / Partials:** 19 / 163 / 21
- **Avg / median % per leg:** 0.45% / -0.97%
- **Sum % (uncompounded):** 91.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 100 | 30 | 30.0% | 16 | 84 | 0 | 0.46% | 46.0% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.20% | -12.8% |
| BUY @ 3rd Alert (retest2) | 96 | 30 | 31.2% | 16 | 80 | 0 | 0.61% | 58.8% |
| SELL (all) | 103 | 46 | 44.7% | 3 | 79 | 21 | 0.44% | 45.0% |
| SELL @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 4 | 1 | 0.91% | 4.6% |
| SELL @ 3rd Alert (retest2) | 98 | 44 | 44.9% | 3 | 75 | 20 | 0.41% | 40.4% |
| retest1 (combined) | 9 | 2 | 22.2% | 0 | 8 | 1 | -0.91% | -8.2% |
| retest2 (combined) | 194 | 74 | 38.1% | 19 | 155 | 20 | 0.51% | 99.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 12:15:00 | 169.90 | 171.97 | 172.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-17 11:15:00 | 169.20 | 170.78 | 171.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-17 13:15:00 | 170.70 | 170.64 | 171.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-17 13:15:00 | 170.70 | 170.64 | 171.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 13:15:00 | 170.70 | 170.64 | 171.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-17 13:45:00 | 171.05 | 170.64 | 171.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 14:15:00 | 169.80 | 170.47 | 171.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-17 14:45:00 | 169.35 | 170.47 | 171.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 09:15:00 | 170.05 | 170.34 | 170.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-18 09:30:00 | 171.35 | 170.34 | 170.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 10:15:00 | 171.10 | 170.50 | 170.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-18 10:45:00 | 171.35 | 170.50 | 170.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 11:15:00 | 171.00 | 170.60 | 170.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-18 11:30:00 | 171.45 | 170.60 | 170.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 09:15:00 | 167.40 | 169.21 | 170.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 09:30:00 | 168.70 | 169.21 | 170.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 11:15:00 | 169.60 | 169.28 | 169.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 11:45:00 | 170.25 | 169.28 | 169.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 12:15:00 | 170.65 | 169.56 | 170.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 13:00:00 | 170.65 | 169.56 | 170.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 13:15:00 | 170.15 | 169.68 | 170.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 13:30:00 | 170.55 | 169.68 | 170.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 14:15:00 | 170.30 | 169.80 | 170.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 15:15:00 | 170.80 | 169.80 | 170.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 15:15:00 | 170.80 | 170.00 | 170.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 09:15:00 | 171.50 | 170.00 | 170.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 171.00 | 170.20 | 170.21 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 10:15:00 | 170.45 | 170.25 | 170.23 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-22 11:15:00 | 169.05 | 170.01 | 170.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-22 12:15:00 | 168.45 | 169.70 | 169.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-23 11:15:00 | 170.00 | 168.92 | 169.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 11:15:00 | 170.00 | 168.92 | 169.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 11:15:00 | 170.00 | 168.92 | 169.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-23 11:30:00 | 169.95 | 168.92 | 169.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 12:15:00 | 171.00 | 169.34 | 169.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-23 14:15:00 | 169.80 | 169.49 | 169.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-24 10:00:00 | 169.70 | 169.09 | 169.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-24 11:15:00 | 171.00 | 169.70 | 169.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2023-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 11:15:00 | 171.00 | 169.70 | 169.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-24 13:15:00 | 173.35 | 170.48 | 169.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-25 09:15:00 | 171.10 | 171.35 | 170.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-25 10:00:00 | 171.10 | 171.35 | 170.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 13:15:00 | 171.40 | 172.10 | 171.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 14:00:00 | 171.40 | 172.10 | 171.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 14:15:00 | 172.10 | 172.10 | 171.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 15:00:00 | 172.10 | 172.10 | 171.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 10:15:00 | 171.90 | 172.35 | 171.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 10:30:00 | 171.95 | 172.35 | 171.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 11:15:00 | 171.85 | 172.25 | 171.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 11:45:00 | 171.95 | 172.25 | 171.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 14:15:00 | 172.55 | 172.23 | 171.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 14:30:00 | 171.95 | 172.23 | 171.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 10:15:00 | 173.95 | 174.70 | 173.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 10:45:00 | 173.70 | 174.70 | 173.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 11:15:00 | 172.95 | 174.35 | 173.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 11:45:00 | 172.60 | 174.35 | 173.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 12:15:00 | 172.95 | 174.07 | 173.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 13:00:00 | 172.95 | 174.07 | 173.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 13:15:00 | 173.75 | 174.01 | 173.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 13:30:00 | 173.30 | 174.01 | 173.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 14:15:00 | 173.60 | 173.92 | 173.60 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 10:15:00 | 172.85 | 173.41 | 173.43 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 11:15:00 | 175.50 | 173.83 | 173.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-31 12:15:00 | 176.85 | 174.43 | 173.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-02 13:15:00 | 178.45 | 179.48 | 178.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-02 14:00:00 | 178.45 | 179.48 | 178.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 14:15:00 | 179.55 | 179.50 | 178.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-05 09:15:00 | 181.60 | 179.25 | 178.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-05 10:45:00 | 179.80 | 179.64 | 178.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-08 14:15:00 | 181.20 | 183.08 | 183.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2023-06-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 14:15:00 | 181.20 | 183.08 | 183.11 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 09:15:00 | 185.85 | 182.94 | 182.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-12 15:15:00 | 189.00 | 186.05 | 184.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-13 09:15:00 | 185.85 | 186.01 | 184.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 09:15:00 | 185.85 | 186.01 | 184.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 09:15:00 | 185.85 | 186.01 | 184.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-13 10:00:00 | 185.85 | 186.01 | 184.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 10:15:00 | 187.00 | 186.21 | 184.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-14 09:15:00 | 190.05 | 186.44 | 185.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-14 11:45:00 | 188.55 | 187.69 | 186.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-14 15:00:00 | 188.10 | 187.73 | 186.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-15 09:45:00 | 188.55 | 187.63 | 186.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 11:15:00 | 187.50 | 187.60 | 187.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 11:45:00 | 187.45 | 187.60 | 187.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 12:15:00 | 188.00 | 187.68 | 187.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 12:30:00 | 187.20 | 187.68 | 187.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 13:15:00 | 186.60 | 187.46 | 187.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 14:00:00 | 186.60 | 187.46 | 187.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 14:15:00 | 187.40 | 187.45 | 187.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-15 14:45:00 | 186.90 | 187.45 | 187.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 15:15:00 | 186.00 | 187.16 | 186.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-16 09:15:00 | 192.40 | 187.16 | 186.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-21 12:15:00 | 189.85 | 190.25 | 190.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2023-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 12:15:00 | 189.85 | 190.25 | 190.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-21 13:15:00 | 189.00 | 190.00 | 190.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-22 10:15:00 | 191.40 | 189.40 | 189.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 10:15:00 | 191.40 | 189.40 | 189.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 10:15:00 | 191.40 | 189.40 | 189.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-22 11:00:00 | 191.40 | 189.40 | 189.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 11:15:00 | 191.65 | 189.85 | 189.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-22 12:15:00 | 190.30 | 189.85 | 189.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-22 14:45:00 | 190.70 | 189.80 | 189.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 09:45:00 | 190.45 | 188.12 | 188.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-28 09:15:00 | 190.25 | 188.43 | 188.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2023-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 09:15:00 | 190.25 | 188.43 | 188.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 10:15:00 | 195.10 | 191.60 | 190.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-03 09:15:00 | 194.25 | 194.47 | 192.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-03 10:00:00 | 194.25 | 194.47 | 192.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 12:15:00 | 192.95 | 193.79 | 192.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-03 13:15:00 | 193.90 | 193.79 | 192.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-04 10:30:00 | 194.60 | 193.87 | 193.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-14 12:15:00 | 203.70 | 204.51 | 204.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2023-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 12:15:00 | 203.70 | 204.51 | 204.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-14 13:15:00 | 203.00 | 204.21 | 204.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 15:15:00 | 203.90 | 203.86 | 204.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-14 15:15:00 | 203.90 | 203.86 | 204.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 15:15:00 | 203.90 | 203.86 | 204.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 09:15:00 | 203.85 | 203.86 | 204.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 204.00 | 203.89 | 204.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-17 13:30:00 | 202.15 | 203.13 | 203.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-18 09:30:00 | 202.35 | 202.40 | 203.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-20 09:15:00 | 207.25 | 202.25 | 202.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2023-07-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 09:15:00 | 207.25 | 202.25 | 202.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-20 10:15:00 | 209.15 | 203.63 | 202.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-26 11:15:00 | 216.65 | 217.04 | 215.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-26 12:00:00 | 216.65 | 217.04 | 215.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 09:15:00 | 220.00 | 221.23 | 219.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-28 10:00:00 | 220.00 | 221.23 | 219.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 10:15:00 | 220.65 | 221.11 | 219.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-31 10:00:00 | 222.45 | 219.86 | 219.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-02 13:15:00 | 218.70 | 224.31 | 224.31 | SL hit (close<static) qty=1.00 sl=219.55 alert=retest2 |

### Cycle 13 — SELL (started 2023-08-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 14:15:00 | 221.00 | 223.65 | 224.01 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-08-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 15:15:00 | 225.50 | 224.18 | 224.05 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-04 09:15:00 | 222.45 | 223.83 | 223.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-04 11:15:00 | 220.85 | 222.86 | 223.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-08 09:15:00 | 219.30 | 218.62 | 220.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-08 09:15:00 | 219.30 | 218.62 | 220.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 09:15:00 | 219.30 | 218.62 | 220.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-08 10:00:00 | 219.30 | 218.62 | 220.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 11:15:00 | 220.35 | 218.95 | 220.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-08 12:00:00 | 220.35 | 218.95 | 220.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 12:15:00 | 219.50 | 219.06 | 219.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-08 13:30:00 | 219.10 | 219.10 | 219.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-09 09:15:00 | 222.25 | 219.48 | 219.85 | SL hit (close>static) qty=1.00 sl=220.65 alert=retest2 |

### Cycle 16 — BUY (started 2023-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 10:15:00 | 224.00 | 220.38 | 220.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-09 11:15:00 | 227.80 | 221.87 | 220.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-10 10:15:00 | 224.70 | 226.01 | 223.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-10 11:00:00 | 224.70 | 226.01 | 223.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 13:15:00 | 224.35 | 225.89 | 224.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 14:00:00 | 224.35 | 225.89 | 224.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 14:15:00 | 224.90 | 225.69 | 224.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 14:30:00 | 223.95 | 225.69 | 224.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 15:15:00 | 227.85 | 226.12 | 224.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 10:00:00 | 224.50 | 225.80 | 224.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 10:15:00 | 224.35 | 225.51 | 224.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 11:00:00 | 224.35 | 225.51 | 224.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 11:15:00 | 227.45 | 225.90 | 224.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 11:30:00 | 224.10 | 225.90 | 224.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 09:15:00 | 226.20 | 226.86 | 225.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 09:30:00 | 222.05 | 226.86 | 225.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 10:15:00 | 223.85 | 226.26 | 225.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 11:00:00 | 223.85 | 226.26 | 225.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 11:15:00 | 224.05 | 225.82 | 225.56 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 12:15:00 | 222.00 | 225.05 | 225.23 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-14 13:15:00 | 230.40 | 226.12 | 225.70 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-08-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-17 10:15:00 | 224.25 | 226.15 | 226.39 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-18 09:15:00 | 235.70 | 227.38 | 226.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-18 10:15:00 | 239.90 | 229.89 | 227.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 11:15:00 | 245.60 | 246.57 | 242.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-22 11:45:00 | 245.80 | 246.57 | 242.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 11:15:00 | 295.15 | 257.60 | 249.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-25 09:30:00 | 300.90 | 283.01 | 272.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-09-05 09:15:00 | 330.99 | 322.90 | 320.00 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2023-09-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-06 13:15:00 | 315.65 | 321.77 | 322.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-07 09:15:00 | 305.10 | 316.63 | 319.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-08 09:15:00 | 314.80 | 312.91 | 315.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-08 09:15:00 | 314.80 | 312.91 | 315.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 09:15:00 | 314.80 | 312.91 | 315.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-08 12:00:00 | 311.40 | 312.83 | 315.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-11 10:30:00 | 311.35 | 312.57 | 314.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-11 11:15:00 | 310.50 | 312.57 | 314.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 09:15:00 | 295.83 | 309.52 | 311.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 09:15:00 | 295.78 | 309.52 | 311.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 09:15:00 | 294.97 | 309.52 | 311.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-12 09:30:00 | 305.15 | 309.52 | 311.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 12:15:00 | 289.89 | 303.27 | 308.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 09:15:00 | 294.00 | 297.90 | 303.81 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-09-13 10:15:00 | 300.90 | 298.50 | 303.54 | SL hit (close>ema200) qty=0.50 sl=298.50 alert=retest2 |

### Cycle 22 — BUY (started 2023-09-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 09:15:00 | 312.35 | 305.32 | 304.96 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-09-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 10:15:00 | 312.80 | 314.37 | 314.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 12:15:00 | 311.90 | 313.50 | 313.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-21 14:15:00 | 314.05 | 313.53 | 313.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 14:15:00 | 314.05 | 313.53 | 313.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 14:15:00 | 314.05 | 313.53 | 313.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 14:30:00 | 314.60 | 313.53 | 313.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 15:15:00 | 315.30 | 313.88 | 314.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 09:15:00 | 311.95 | 313.88 | 314.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 310.40 | 313.19 | 313.70 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 11:15:00 | 314.50 | 313.61 | 313.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-25 12:15:00 | 315.05 | 313.89 | 313.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-26 12:15:00 | 315.05 | 315.14 | 314.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 12:15:00 | 315.05 | 315.14 | 314.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 12:15:00 | 315.05 | 315.14 | 314.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 12:45:00 | 315.40 | 315.14 | 314.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 328.50 | 318.20 | 316.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-29 14:30:00 | 334.65 | 327.82 | 324.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-03 09:15:00 | 337.90 | 328.25 | 325.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-03 11:45:00 | 333.45 | 330.77 | 327.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-04 13:15:00 | 325.15 | 327.57 | 327.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2023-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 13:15:00 | 325.15 | 327.57 | 327.72 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-10-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 09:15:00 | 339.90 | 329.28 | 328.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 09:15:00 | 352.55 | 343.00 | 336.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-06 12:15:00 | 341.55 | 343.45 | 338.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-06 12:45:00 | 341.50 | 343.45 | 338.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 336.60 | 342.96 | 340.05 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 12:15:00 | 330.30 | 338.03 | 338.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 13:15:00 | 329.55 | 336.34 | 337.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 343.30 | 335.58 | 336.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 343.30 | 335.58 | 336.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 343.30 | 335.58 | 336.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 10:00:00 | 343.30 | 335.58 | 336.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 10:15:00 | 342.20 | 336.90 | 337.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 10:30:00 | 343.20 | 336.90 | 337.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2023-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 11:15:00 | 340.60 | 337.64 | 337.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 13:15:00 | 347.35 | 340.28 | 338.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 14:15:00 | 345.15 | 346.45 | 343.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-11 15:00:00 | 345.15 | 346.45 | 343.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 15:15:00 | 346.80 | 346.52 | 343.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-12 09:15:00 | 350.90 | 346.52 | 343.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 09:45:00 | 347.40 | 349.02 | 347.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 12:45:00 | 347.50 | 347.43 | 346.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-13 15:15:00 | 342.00 | 345.57 | 346.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2023-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 15:15:00 | 342.00 | 345.57 | 346.04 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-16 09:15:00 | 349.95 | 346.45 | 346.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-16 10:15:00 | 352.80 | 347.72 | 346.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-16 12:15:00 | 346.15 | 347.83 | 347.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-16 12:15:00 | 346.15 | 347.83 | 347.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 12:15:00 | 346.15 | 347.83 | 347.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-16 13:00:00 | 346.15 | 347.83 | 347.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 13:15:00 | 346.15 | 347.49 | 347.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-16 13:45:00 | 345.85 | 347.49 | 347.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 14:15:00 | 346.95 | 347.38 | 347.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-16 15:15:00 | 347.65 | 347.38 | 347.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-17 14:15:00 | 347.70 | 348.75 | 348.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-18 10:15:00 | 343.95 | 347.89 | 347.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2023-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 10:15:00 | 343.95 | 347.89 | 347.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 09:15:00 | 343.00 | 345.91 | 346.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 10:15:00 | 346.10 | 345.95 | 346.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-19 11:00:00 | 346.10 | 345.95 | 346.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 11:15:00 | 344.90 | 345.74 | 346.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 11:30:00 | 345.55 | 345.74 | 346.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 12:15:00 | 346.75 | 345.94 | 346.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 13:00:00 | 346.75 | 345.94 | 346.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 13:15:00 | 345.70 | 345.89 | 346.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-19 15:00:00 | 344.30 | 345.57 | 346.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-20 09:15:00 | 350.80 | 346.50 | 346.59 | SL hit (close>static) qty=1.00 sl=347.10 alert=retest2 |

### Cycle 32 — BUY (started 2023-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-20 10:15:00 | 349.15 | 347.03 | 346.82 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 12:15:00 | 341.00 | 345.66 | 346.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 09:15:00 | 325.10 | 340.19 | 343.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-26 12:15:00 | 317.45 | 315.50 | 321.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-26 12:45:00 | 317.35 | 315.50 | 321.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 14:15:00 | 323.00 | 317.08 | 321.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-26 15:00:00 | 323.00 | 317.08 | 321.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 15:15:00 | 323.00 | 318.26 | 321.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 09:15:00 | 331.85 | 318.26 | 321.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 333.30 | 321.27 | 322.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 10:00:00 | 333.30 | 321.27 | 322.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2023-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 10:15:00 | 334.40 | 323.90 | 323.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 09:15:00 | 344.95 | 334.98 | 331.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 11:15:00 | 335.60 | 335.80 | 332.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-31 11:45:00 | 336.25 | 335.80 | 332.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 15:15:00 | 336.00 | 336.00 | 333.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 09:15:00 | 333.60 | 336.00 | 333.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 09:15:00 | 332.20 | 335.24 | 333.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 09:45:00 | 331.65 | 335.24 | 333.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 10:15:00 | 329.90 | 334.17 | 333.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 11:00:00 | 329.90 | 334.17 | 333.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2023-11-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 13:15:00 | 325.00 | 330.91 | 331.70 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-11-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 13:15:00 | 333.45 | 331.13 | 331.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 09:15:00 | 336.45 | 332.90 | 331.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-06 09:15:00 | 335.75 | 337.52 | 335.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-06 09:15:00 | 335.75 | 337.52 | 335.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 09:15:00 | 335.75 | 337.52 | 335.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-06 09:45:00 | 335.00 | 337.52 | 335.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 10:15:00 | 333.60 | 336.73 | 335.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-06 11:00:00 | 333.60 | 336.73 | 335.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 11:15:00 | 334.10 | 336.21 | 335.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-06 12:00:00 | 334.10 | 336.21 | 335.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 13:15:00 | 335.50 | 336.04 | 335.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-06 13:30:00 | 335.50 | 336.04 | 335.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 14:15:00 | 336.00 | 336.03 | 335.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-06 14:30:00 | 335.40 | 336.03 | 335.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 15:15:00 | 336.90 | 336.21 | 335.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-07 09:15:00 | 339.40 | 336.21 | 335.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-07 12:00:00 | 339.25 | 337.22 | 336.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 09:15:00 | 341.50 | 337.35 | 336.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2023-11-13 09:15:00 | 373.34 | 365.11 | 359.20 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2023-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 10:15:00 | 403.20 | 408.22 | 408.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 11:15:00 | 397.20 | 406.02 | 407.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-24 09:15:00 | 407.80 | 399.82 | 401.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 09:15:00 | 407.80 | 399.82 | 401.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 407.80 | 399.82 | 401.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 09:30:00 | 409.85 | 399.82 | 401.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 10:15:00 | 406.65 | 401.18 | 402.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 11:15:00 | 408.75 | 401.18 | 402.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2023-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 12:15:00 | 407.60 | 403.44 | 403.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-24 14:15:00 | 409.50 | 405.54 | 404.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-29 09:15:00 | 409.50 | 410.91 | 408.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-29 09:15:00 | 409.50 | 410.91 | 408.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 09:15:00 | 409.50 | 410.91 | 408.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-29 15:15:00 | 415.00 | 410.17 | 408.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-30 10:00:00 | 415.75 | 412.06 | 410.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-30 15:15:00 | 407.40 | 418.08 | 414.66 | SL hit (close<static) qty=1.00 sl=408.25 alert=retest2 |

### Cycle 39 — SELL (started 2023-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 10:15:00 | 419.75 | 424.69 | 424.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 12:15:00 | 418.00 | 422.70 | 423.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 15:15:00 | 408.05 | 407.01 | 410.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-14 09:15:00 | 412.80 | 407.01 | 410.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 414.10 | 408.43 | 410.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-14 13:15:00 | 408.85 | 409.99 | 411.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-15 09:15:00 | 418.00 | 411.82 | 411.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2023-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 09:15:00 | 418.00 | 411.82 | 411.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-18 09:15:00 | 420.45 | 413.71 | 412.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 14:15:00 | 414.40 | 414.80 | 413.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-18 14:15:00 | 414.40 | 414.80 | 413.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 14:15:00 | 414.40 | 414.80 | 413.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 14:30:00 | 413.00 | 414.80 | 413.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 15:15:00 | 417.35 | 415.31 | 414.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 09:15:00 | 420.60 | 415.31 | 414.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 09:15:00 | 409.90 | 414.47 | 414.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2023-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 09:15:00 | 409.90 | 414.47 | 414.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 13:15:00 | 393.45 | 408.60 | 411.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 09:15:00 | 406.85 | 404.21 | 408.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-21 09:15:00 | 406.85 | 404.21 | 408.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 09:15:00 | 406.85 | 404.21 | 408.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 09:45:00 | 407.50 | 404.21 | 408.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 10:15:00 | 406.65 | 404.70 | 408.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 10:45:00 | 407.80 | 404.70 | 408.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 407.25 | 403.01 | 405.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 13:45:00 | 401.50 | 404.14 | 405.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-26 09:30:00 | 402.85 | 404.48 | 405.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-26 11:30:00 | 401.90 | 404.14 | 405.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-26 12:15:00 | 401.15 | 404.14 | 405.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 09:15:00 | 402.75 | 401.29 | 403.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-27 09:30:00 | 403.40 | 401.29 | 403.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 10:15:00 | 401.25 | 401.28 | 402.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-27 12:15:00 | 399.15 | 401.07 | 402.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-28 09:30:00 | 399.05 | 399.34 | 401.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-01 10:15:00 | 403.00 | 397.66 | 397.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2024-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-01 10:15:00 | 403.00 | 397.66 | 397.43 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-05 12:15:00 | 396.70 | 399.90 | 400.33 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2024-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-08 09:15:00 | 406.35 | 401.54 | 400.94 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2024-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 14:15:00 | 397.60 | 400.63 | 400.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-10 09:15:00 | 396.15 | 398.58 | 399.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-11 14:15:00 | 391.75 | 391.55 | 394.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-11 15:00:00 | 391.75 | 391.55 | 394.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 09:15:00 | 386.55 | 390.11 | 391.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-15 10:15:00 | 385.00 | 390.11 | 391.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-15 11:00:00 | 384.90 | 389.07 | 391.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-15 14:45:00 | 384.85 | 386.96 | 389.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-16 12:00:00 | 385.00 | 386.39 | 388.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 364.95 | 372.22 | 377.63 | EMA400 retest candle locked (from downside) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-18 09:15:00 | 365.75 | 372.22 | 377.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-18 09:15:00 | 365.65 | 372.22 | 377.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-18 09:15:00 | 365.61 | 372.22 | 377.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-18 09:15:00 | 365.75 | 372.22 | 377.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| ALERT3_SIDEWAYS | 2024-01-18 09:30:00 | 368.15 | 372.22 | 377.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-01-18 15:15:00 | 370.00 | 369.77 | 373.79 | SL hit (close>ema200) qty=0.50 sl=369.77 alert=retest2 |

### Cycle 46 — BUY (started 2024-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 11:15:00 | 388.90 | 377.47 | 376.65 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2024-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 13:15:00 | 375.10 | 382.35 | 382.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-24 11:15:00 | 374.45 | 378.17 | 380.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-25 14:15:00 | 376.85 | 375.84 | 377.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-25 15:00:00 | 376.85 | 375.84 | 377.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 09:15:00 | 377.35 | 376.33 | 377.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-29 10:15:00 | 375.00 | 376.33 | 377.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-31 11:15:00 | 376.90 | 373.75 | 373.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2024-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 11:15:00 | 376.90 | 373.75 | 373.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 12:15:00 | 377.60 | 374.52 | 374.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-02 15:15:00 | 383.20 | 385.33 | 382.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-02 15:15:00 | 383.20 | 385.33 | 382.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 15:15:00 | 383.20 | 385.33 | 382.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 09:15:00 | 379.00 | 385.33 | 382.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 09:15:00 | 381.75 | 384.61 | 382.78 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2024-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 13:15:00 | 377.40 | 381.40 | 381.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 14:15:00 | 375.00 | 380.12 | 381.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 09:15:00 | 385.90 | 380.70 | 381.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-06 09:15:00 | 385.90 | 380.70 | 381.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 385.90 | 380.70 | 381.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 09:45:00 | 381.85 | 380.70 | 381.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2024-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 10:15:00 | 386.50 | 381.86 | 381.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-07 12:15:00 | 398.00 | 388.18 | 385.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-08 09:15:00 | 392.60 | 393.56 | 389.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-08 09:30:00 | 395.75 | 393.56 | 389.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 15:15:00 | 393.00 | 392.73 | 390.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 09:15:00 | 378.50 | 392.73 | 390.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 375.95 | 389.37 | 389.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 09:45:00 | 367.25 | 389.37 | 389.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2024-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 10:15:00 | 371.00 | 385.70 | 387.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 09:15:00 | 361.75 | 373.51 | 379.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 15:15:00 | 351.00 | 350.12 | 358.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-14 09:15:00 | 351.70 | 350.12 | 358.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 09:15:00 | 364.40 | 352.98 | 359.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 10:00:00 | 364.40 | 352.98 | 359.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 10:15:00 | 367.55 | 355.89 | 359.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 10:45:00 | 369.05 | 355.89 | 359.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 13:15:00 | 361.00 | 360.11 | 361.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 14:45:00 | 358.00 | 359.58 | 360.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-15 14:00:00 | 359.70 | 359.75 | 360.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-16 10:30:00 | 359.15 | 359.62 | 360.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-16 11:45:00 | 358.50 | 359.53 | 359.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-16 12:15:00 | 365.10 | 360.64 | 360.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2024-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 12:15:00 | 365.10 | 360.64 | 360.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 13:15:00 | 370.00 | 362.51 | 361.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-20 10:15:00 | 366.20 | 368.95 | 366.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-20 10:15:00 | 366.20 | 368.95 | 366.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 10:15:00 | 366.20 | 368.95 | 366.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 11:00:00 | 366.20 | 368.95 | 366.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 11:15:00 | 364.95 | 368.15 | 366.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 11:45:00 | 365.80 | 368.15 | 366.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 12:15:00 | 365.20 | 367.56 | 366.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 12:30:00 | 365.00 | 367.56 | 366.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — SELL (started 2024-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 15:15:00 | 363.20 | 365.62 | 365.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 10:15:00 | 361.45 | 364.31 | 365.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 09:15:00 | 361.80 | 360.35 | 362.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 09:15:00 | 361.80 | 360.35 | 362.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 361.80 | 360.35 | 362.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 10:00:00 | 361.80 | 360.35 | 362.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 10:15:00 | 362.00 | 360.68 | 362.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 10:45:00 | 362.00 | 360.68 | 362.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 11:15:00 | 361.25 | 360.79 | 362.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 12:00:00 | 361.25 | 360.79 | 362.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 12:15:00 | 361.85 | 361.01 | 362.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 12:30:00 | 362.60 | 361.01 | 362.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 13:15:00 | 361.60 | 361.12 | 362.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-22 14:15:00 | 359.80 | 361.12 | 362.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-23 09:15:00 | 366.25 | 362.13 | 362.38 | SL hit (close>static) qty=1.00 sl=363.50 alert=retest2 |

### Cycle 54 — BUY (started 2024-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 10:15:00 | 365.75 | 362.86 | 362.68 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-23 14:15:00 | 360.35 | 362.48 | 362.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 11:15:00 | 358.05 | 360.97 | 361.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-28 15:15:00 | 349.70 | 347.08 | 350.51 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-29 10:15:00 | 344.00 | 346.58 | 349.97 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-29 12:30:00 | 343.90 | 345.33 | 348.47 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-29 13:45:00 | 344.10 | 345.01 | 348.04 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 348.10 | 345.23 | 347.36 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-03-01 09:15:00 | 348.10 | 345.23 | 347.36 | SL hit (close>ema400) qty=1.00 sl=347.36 alert=retest1 |

### Cycle 56 — BUY (started 2024-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-04 13:15:00 | 347.65 | 346.59 | 346.57 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 10:15:00 | 344.75 | 346.25 | 346.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 12:15:00 | 343.60 | 345.49 | 346.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-05 15:15:00 | 345.00 | 344.90 | 345.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-06 09:15:00 | 343.00 | 344.90 | 345.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 338.00 | 343.52 | 344.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-06 11:45:00 | 332.40 | 339.81 | 342.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 14:45:00 | 330.30 | 336.04 | 338.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 09:15:00 | 315.78 | 320.21 | 327.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 09:15:00 | 313.78 | 320.21 | 327.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-03-13 09:15:00 | 299.16 | 305.98 | 315.20 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 58 — BUY (started 2024-03-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 15:15:00 | 309.30 | 305.72 | 305.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-18 09:15:00 | 325.40 | 309.66 | 307.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 11:15:00 | 327.55 | 327.89 | 321.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-19 11:30:00 | 328.50 | 327.89 | 321.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 15:15:00 | 323.40 | 325.65 | 322.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 09:30:00 | 322.60 | 325.01 | 322.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 10:15:00 | 322.65 | 324.54 | 322.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 10:45:00 | 322.15 | 324.54 | 322.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 11:15:00 | 323.45 | 324.32 | 322.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 13:30:00 | 326.00 | 324.58 | 322.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 14:30:00 | 326.45 | 324.96 | 323.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-28 14:15:00 | 334.15 | 336.10 | 336.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2024-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 14:15:00 | 334.15 | 336.10 | 336.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-28 15:15:00 | 333.65 | 335.61 | 335.91 | Break + close below crossover candle low |

### Cycle 60 — BUY (started 2024-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 09:15:00 | 340.00 | 336.49 | 336.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 11:15:00 | 346.60 | 339.20 | 337.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-05 09:15:00 | 364.45 | 364.80 | 361.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 09:15:00 | 364.45 | 364.80 | 361.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 364.45 | 364.80 | 361.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-05 09:30:00 | 361.25 | 364.80 | 361.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 10:15:00 | 362.35 | 364.31 | 361.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-05 11:00:00 | 362.35 | 364.31 | 361.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 11:15:00 | 360.95 | 363.64 | 361.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-05 12:00:00 | 360.95 | 363.64 | 361.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 12:15:00 | 360.50 | 363.01 | 361.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-05 13:00:00 | 360.50 | 363.01 | 361.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 13:15:00 | 361.35 | 362.68 | 361.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-08 09:15:00 | 363.55 | 362.29 | 361.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-10 11:15:00 | 358.85 | 363.72 | 364.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2024-04-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-10 11:15:00 | 358.85 | 363.72 | 364.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 09:15:00 | 356.90 | 360.15 | 362.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-12 14:15:00 | 358.70 | 358.16 | 360.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-12 15:00:00 | 358.70 | 358.16 | 360.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 15:15:00 | 358.05 | 358.14 | 360.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-15 09:15:00 | 345.05 | 358.14 | 360.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 327.80 | 355.91 | 358.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-16 13:15:00 | 348.85 | 348.75 | 351.85 | SL hit (close>ema200) qty=0.50 sl=348.75 alert=retest2 |

### Cycle 62 — BUY (started 2024-04-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 13:15:00 | 354.50 | 352.75 | 352.58 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-04-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 09:15:00 | 346.00 | 351.31 | 351.96 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 11:15:00 | 352.70 | 349.30 | 348.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-25 09:15:00 | 357.50 | 352.52 | 351.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 11:15:00 | 352.35 | 352.49 | 351.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-25 11:30:00 | 352.50 | 352.49 | 351.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 12:15:00 | 352.70 | 352.53 | 351.51 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2024-04-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 11:15:00 | 349.15 | 351.28 | 351.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-26 12:15:00 | 348.70 | 350.76 | 351.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-26 15:15:00 | 350.75 | 350.47 | 350.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-29 09:15:00 | 352.70 | 350.47 | 350.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 66 — BUY (started 2024-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-29 09:15:00 | 363.20 | 353.02 | 351.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-29 11:15:00 | 367.80 | 357.25 | 354.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-02 10:15:00 | 385.90 | 389.01 | 379.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-02 11:00:00 | 385.90 | 389.01 | 379.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 10:15:00 | 378.95 | 384.90 | 381.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 11:00:00 | 378.95 | 384.90 | 381.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 11:15:00 | 377.10 | 383.34 | 381.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 11:30:00 | 376.75 | 383.34 | 381.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 15:15:00 | 382.75 | 382.04 | 381.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-06 09:15:00 | 377.20 | 382.04 | 381.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2024-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 09:15:00 | 374.00 | 380.43 | 380.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 10:15:00 | 366.75 | 374.06 | 376.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 369.05 | 367.49 | 371.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-08 09:15:00 | 369.05 | 367.49 | 371.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 369.05 | 367.49 | 371.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 12:15:00 | 356.10 | 361.63 | 365.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 09:30:00 | 356.55 | 362.63 | 364.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 10:30:00 | 353.95 | 361.52 | 363.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-14 11:15:00 | 365.75 | 363.83 | 363.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 365.75 | 363.83 | 363.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 13:15:00 | 371.00 | 365.70 | 364.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 10:15:00 | 371.35 | 374.24 | 371.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 10:15:00 | 371.35 | 374.24 | 371.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 10:15:00 | 371.35 | 374.24 | 371.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 11:00:00 | 371.35 | 374.24 | 371.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 374.00 | 374.19 | 371.84 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2024-05-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 12:15:00 | 370.00 | 370.93 | 371.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-17 15:15:00 | 367.05 | 369.77 | 370.46 | Break + close below crossover candle low |

### Cycle 70 — BUY (started 2024-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 09:15:00 | 376.30 | 371.08 | 370.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 12:15:00 | 378.20 | 374.92 | 373.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 14:15:00 | 388.85 | 389.37 | 385.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-24 14:30:00 | 386.40 | 389.37 | 385.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 383.05 | 387.75 | 385.24 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2024-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 09:15:00 | 377.90 | 383.20 | 383.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 12:15:00 | 374.30 | 380.11 | 382.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 365.35 | 360.73 | 364.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 365.35 | 360.73 | 364.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 365.35 | 360.73 | 364.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 10:45:00 | 361.50 | 360.33 | 364.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 343.43 | 358.64 | 361.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-04 12:15:00 | 361.30 | 354.55 | 358.44 | SL hit (close>ema200) qty=0.50 sl=354.55 alert=retest2 |

### Cycle 72 — BUY (started 2024-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 09:15:00 | 373.25 | 362.30 | 361.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 10:15:00 | 375.70 | 364.98 | 362.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 382.80 | 384.31 | 381.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-10 14:15:00 | 382.80 | 384.31 | 381.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 382.80 | 384.31 | 381.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:30:00 | 382.55 | 384.31 | 381.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 383.00 | 384.05 | 381.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:30:00 | 381.95 | 383.75 | 381.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 15:15:00 | 421.30 | 423.60 | 421.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 09:15:00 | 425.90 | 423.60 | 421.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-20 14:15:00 | 420.70 | 423.21 | 422.14 | SL hit (close<static) qty=1.00 sl=420.90 alert=retest2 |

### Cycle 73 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 472.95 | 478.33 | 479.02 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 12:15:00 | 482.05 | 479.42 | 479.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 497.10 | 483.99 | 481.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 15:15:00 | 489.75 | 489.96 | 486.20 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 09:15:00 | 495.50 | 489.96 | 486.20 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 489.65 | 495.98 | 493.18 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-08 10:15:00 | 489.65 | 495.98 | 493.18 | SL hit (close<ema400) qty=1.00 sl=493.18 alert=retest1 |

### Cycle 75 — SELL (started 2024-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 11:15:00 | 493.00 | 495.28 | 495.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 12:15:00 | 488.75 | 491.33 | 492.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 09:15:00 | 490.75 | 490.72 | 491.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 09:15:00 | 490.75 | 490.72 | 491.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 490.75 | 490.72 | 491.77 | EMA400 retest candle locked (from downside) |

### Cycle 76 — BUY (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 09:15:00 | 500.70 | 492.11 | 491.76 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 485.55 | 494.01 | 494.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 472.85 | 487.47 | 490.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 478.75 | 476.53 | 482.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 478.75 | 476.53 | 482.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 478.75 | 476.53 | 482.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:45:00 | 480.55 | 476.53 | 482.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 477.95 | 476.81 | 481.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 481.50 | 476.81 | 481.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 482.15 | 478.36 | 481.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:00:00 | 482.15 | 478.36 | 481.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 480.80 | 478.85 | 481.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:45:00 | 482.05 | 478.85 | 481.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 482.30 | 478.70 | 480.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 10:30:00 | 479.75 | 478.70 | 480.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 477.55 | 478.47 | 480.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 461.00 | 478.47 | 480.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 15:00:00 | 475.20 | 476.31 | 478.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 487.30 | 478.33 | 479.21 | SL hit (close>static) qty=1.00 sl=483.10 alert=retest2 |

### Cycle 78 — BUY (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 10:15:00 | 487.80 | 480.22 | 479.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 12:15:00 | 491.35 | 483.35 | 481.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 482.80 | 485.76 | 483.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 09:15:00 | 482.80 | 485.76 | 483.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 482.80 | 485.76 | 483.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 11:30:00 | 490.00 | 486.83 | 484.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 10:15:00 | 494.50 | 498.32 | 498.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 10:15:00 | 494.50 | 498.32 | 498.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 11:15:00 | 493.00 | 497.25 | 498.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 10:15:00 | 497.65 | 495.86 | 496.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 10:15:00 | 497.65 | 495.86 | 496.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 497.65 | 495.86 | 496.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 11:00:00 | 497.65 | 495.86 | 496.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 11:15:00 | 499.00 | 496.49 | 497.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 11:30:00 | 499.80 | 496.49 | 497.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 12:15:00 | 498.60 | 496.91 | 497.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 13:00:00 | 498.60 | 496.91 | 497.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 13:15:00 | 499.05 | 497.34 | 497.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 14:15:00 | 499.60 | 497.34 | 497.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 14:15:00 | 497.50 | 497.37 | 497.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 14:30:00 | 499.00 | 497.37 | 497.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — BUY (started 2024-08-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-02 15:15:00 | 499.35 | 497.77 | 497.60 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 483.55 | 494.92 | 496.32 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 13:15:00 | 490.85 | 485.05 | 484.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 10:15:00 | 492.45 | 488.96 | 486.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 12:15:00 | 489.40 | 489.53 | 487.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 13:00:00 | 489.40 | 489.53 | 487.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 491.50 | 489.76 | 488.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 501.60 | 489.76 | 488.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 15:15:00 | 493.00 | 499.34 | 499.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 15:15:00 | 493.00 | 498.07 | 498.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2024-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 15:15:00 | 493.00 | 498.07 | 498.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 09:15:00 | 479.70 | 494.40 | 496.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 12:15:00 | 496.00 | 488.84 | 493.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 12:15:00 | 496.00 | 488.84 | 493.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 12:15:00 | 496.00 | 488.84 | 493.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 13:00:00 | 496.00 | 488.84 | 493.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 13:15:00 | 512.05 | 493.48 | 494.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 14:00:00 | 512.05 | 493.48 | 494.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2024-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 14:15:00 | 511.15 | 497.02 | 496.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 09:15:00 | 521.05 | 504.53 | 500.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-16 12:15:00 | 504.25 | 508.12 | 503.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-16 13:00:00 | 504.25 | 508.12 | 503.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 13:15:00 | 499.05 | 506.30 | 502.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:00:00 | 499.05 | 506.30 | 502.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 486.90 | 502.42 | 501.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-16 15:00:00 | 486.90 | 502.42 | 501.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2024-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 15:15:00 | 486.50 | 499.24 | 500.00 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 12:15:00 | 509.15 | 500.90 | 500.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 09:15:00 | 529.00 | 508.84 | 504.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 09:15:00 | 531.05 | 531.86 | 524.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 09:45:00 | 530.95 | 531.86 | 524.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 528.60 | 536.40 | 533.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 10:00:00 | 528.60 | 536.40 | 533.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 529.50 | 535.02 | 533.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:00:00 | 529.50 | 535.02 | 533.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 536.00 | 534.84 | 533.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 12:45:00 | 535.75 | 534.84 | 533.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 13:15:00 | 535.75 | 535.02 | 533.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 14:15:00 | 534.90 | 535.02 | 533.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 14:15:00 | 537.90 | 535.60 | 534.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 14:45:00 | 537.00 | 535.60 | 534.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 537.00 | 536.74 | 534.87 | EMA400 retest candle locked (from upside) |

### Cycle 87 — SELL (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 10:15:00 | 523.35 | 533.09 | 534.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 13:15:00 | 521.55 | 527.98 | 531.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 529.50 | 518.31 | 521.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 529.50 | 518.31 | 521.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 529.50 | 518.31 | 521.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:00:00 | 529.50 | 518.31 | 521.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 531.10 | 520.87 | 522.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:30:00 | 531.25 | 520.87 | 522.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — BUY (started 2024-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 12:15:00 | 541.50 | 527.07 | 525.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 13:15:00 | 543.95 | 530.44 | 527.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 09:15:00 | 532.95 | 535.76 | 530.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 10:00:00 | 532.95 | 535.76 | 530.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 546.50 | 537.91 | 532.21 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2024-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 14:15:00 | 533.20 | 537.22 | 537.28 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 09:15:00 | 544.00 | 538.01 | 537.60 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 531.90 | 537.78 | 538.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 12:15:00 | 527.70 | 535.01 | 536.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 10:15:00 | 515.15 | 513.18 | 520.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 11:00:00 | 515.15 | 513.18 | 520.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 13:15:00 | 520.35 | 514.90 | 519.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 13:45:00 | 519.50 | 514.90 | 519.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 14:15:00 | 524.60 | 516.84 | 519.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 15:00:00 | 524.60 | 516.84 | 519.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 15:15:00 | 525.50 | 518.57 | 520.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 09:45:00 | 522.20 | 519.21 | 520.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 11:15:00 | 527.60 | 517.22 | 516.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2024-09-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 11:15:00 | 527.60 | 517.22 | 516.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 12:15:00 | 529.70 | 519.72 | 517.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 525.95 | 530.50 | 526.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 09:15:00 | 525.95 | 530.50 | 526.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 525.95 | 530.50 | 526.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:00:00 | 525.95 | 530.50 | 526.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 529.70 | 530.34 | 526.75 | EMA400 retest candle locked (from upside) |

### Cycle 93 — SELL (started 2024-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 10:15:00 | 519.20 | 524.55 | 525.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 12:15:00 | 514.85 | 521.58 | 523.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 12:15:00 | 518.30 | 516.96 | 519.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 12:15:00 | 518.30 | 516.96 | 519.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 518.30 | 516.96 | 519.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 13:00:00 | 518.30 | 516.96 | 519.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 13:15:00 | 518.90 | 517.35 | 519.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 14:15:00 | 519.75 | 517.35 | 519.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 522.00 | 518.28 | 519.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 14:30:00 | 523.15 | 518.28 | 519.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 524.00 | 519.42 | 520.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 523.30 | 519.42 | 520.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2024-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 11:15:00 | 527.25 | 522.08 | 521.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 12:15:00 | 531.50 | 523.96 | 522.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 10:15:00 | 526.15 | 527.03 | 524.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-23 10:30:00 | 526.55 | 527.03 | 524.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 11:15:00 | 523.60 | 526.34 | 524.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 12:00:00 | 523.60 | 526.34 | 524.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 12:15:00 | 524.35 | 525.94 | 524.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 12:30:00 | 524.55 | 525.94 | 524.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 13:15:00 | 526.45 | 526.04 | 524.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 13:30:00 | 523.90 | 526.04 | 524.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 524.25 | 525.75 | 524.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 10:00:00 | 524.25 | 525.75 | 524.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 522.95 | 525.19 | 524.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 11:00:00 | 522.95 | 525.19 | 524.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 525.65 | 525.28 | 524.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 11:45:00 | 525.00 | 525.28 | 524.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 12:15:00 | 527.05 | 525.64 | 525.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 12:30:00 | 525.55 | 525.64 | 525.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 528.00 | 527.31 | 526.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:45:00 | 525.30 | 526.79 | 525.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 522.15 | 525.86 | 525.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:30:00 | 521.75 | 525.86 | 525.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2024-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 11:15:00 | 518.25 | 524.34 | 524.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 09:15:00 | 513.40 | 519.00 | 520.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 15:15:00 | 519.80 | 517.60 | 518.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 15:15:00 | 519.80 | 517.60 | 518.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 519.80 | 517.60 | 518.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:30:00 | 521.25 | 518.66 | 519.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 523.00 | 519.53 | 519.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 11:15:00 | 516.25 | 519.53 | 519.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 11:15:00 | 516.15 | 518.85 | 519.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:30:00 | 513.45 | 516.81 | 518.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 487.78 | 505.27 | 510.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-07 10:15:00 | 462.11 | 485.00 | 496.18 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 96 — BUY (started 2024-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 09:15:00 | 442.45 | 432.54 | 431.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-24 14:15:00 | 450.15 | 441.27 | 436.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-25 09:15:00 | 436.55 | 442.03 | 438.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-25 09:15:00 | 436.55 | 442.03 | 438.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 436.55 | 442.03 | 438.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 09:45:00 | 439.15 | 442.03 | 438.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 434.40 | 440.50 | 437.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:45:00 | 433.60 | 440.50 | 437.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 11:15:00 | 436.95 | 439.79 | 437.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 12:15:00 | 433.70 | 439.79 | 437.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 12:15:00 | 432.35 | 438.30 | 437.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 13:00:00 | 432.35 | 438.30 | 437.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — SELL (started 2024-10-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 13:15:00 | 425.65 | 435.77 | 436.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 10:15:00 | 424.70 | 432.44 | 433.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 09:15:00 | 439.15 | 431.90 | 432.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 09:15:00 | 439.15 | 431.90 | 432.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 439.15 | 431.90 | 432.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:45:00 | 440.05 | 431.90 | 432.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 439.90 | 433.50 | 433.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 12:15:00 | 446.00 | 437.03 | 434.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 452.95 | 455.34 | 449.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 10:00:00 | 452.95 | 455.34 | 449.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 449.25 | 454.12 | 449.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 449.25 | 454.12 | 449.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 452.60 | 453.81 | 449.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 12:15:00 | 453.00 | 453.81 | 449.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 13:00:00 | 453.65 | 453.78 | 450.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 13:45:00 | 453.00 | 453.82 | 450.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:15:00 | 461.20 | 452.81 | 450.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 461.80 | 454.61 | 451.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 10:45:00 | 465.05 | 456.65 | 452.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 12:15:00 | 463.90 | 457.97 | 453.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 13:30:00 | 464.85 | 462.09 | 458.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 11:00:00 | 463.95 | 463.23 | 460.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 460.10 | 462.30 | 460.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:00:00 | 460.10 | 462.30 | 460.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 460.00 | 461.84 | 460.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:30:00 | 459.80 | 461.84 | 460.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 459.75 | 461.42 | 460.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 14:45:00 | 459.50 | 461.42 | 460.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 462.00 | 461.54 | 460.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 456.50 | 461.54 | 460.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 458.50 | 460.93 | 460.34 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-08 10:15:00 | 454.45 | 459.63 | 459.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 454.45 | 459.63 | 459.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 452.45 | 458.20 | 459.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 445.60 | 444.70 | 449.11 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 13:30:00 | 439.45 | 443.16 | 446.95 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 417.48 | 435.50 | 442.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-11-14 09:15:00 | 425.65 | 421.86 | 430.24 | SL hit (close>ema200) qty=0.50 sl=421.86 alert=retest1 |

### Cycle 100 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 429.65 | 427.10 | 426.79 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 416.10 | 424.83 | 425.99 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 427.30 | 424.76 | 424.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 15:15:00 | 429.70 | 426.13 | 425.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 12:15:00 | 443.10 | 444.32 | 439.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 13:00:00 | 443.10 | 444.32 | 439.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 441.15 | 443.53 | 440.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:30:00 | 441.80 | 443.53 | 440.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 440.50 | 442.92 | 440.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:00:00 | 440.50 | 442.92 | 440.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 439.45 | 442.23 | 440.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:45:00 | 438.95 | 442.23 | 440.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 439.45 | 441.67 | 440.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 12:30:00 | 439.65 | 441.67 | 440.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 441.05 | 442.02 | 441.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 12:45:00 | 447.55 | 444.22 | 442.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 13:30:00 | 448.20 | 444.95 | 442.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 09:15:00 | 452.45 | 445.86 | 443.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 10:00:00 | 448.25 | 450.91 | 448.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 447.95 | 450.32 | 448.09 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-05 09:15:00 | 441.90 | 446.37 | 446.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2024-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 09:15:00 | 441.90 | 446.37 | 446.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 10:15:00 | 439.75 | 445.05 | 446.26 | Break + close below crossover candle low |

### Cycle 104 — BUY (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 09:15:00 | 472.10 | 447.71 | 446.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 12:15:00 | 480.40 | 462.47 | 454.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-10 09:15:00 | 483.65 | 483.90 | 474.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-10 10:00:00 | 483.65 | 483.90 | 474.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 12:15:00 | 508.00 | 510.45 | 504.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 14:30:00 | 509.80 | 510.14 | 504.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 09:15:00 | 522.05 | 509.48 | 505.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-17 10:00:00 | 509.85 | 512.86 | 512.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-17 10:30:00 | 509.60 | 512.58 | 512.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 510.15 | 512.36 | 512.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:00:00 | 510.15 | 512.36 | 512.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 517.45 | 513.38 | 512.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:30:00 | 511.40 | 513.38 | 512.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 15:15:00 | 513.80 | 513.46 | 512.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 09:15:00 | 512.80 | 513.46 | 512.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-18 09:15:00 | 506.00 | 511.97 | 512.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2024-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 09:15:00 | 506.00 | 511.97 | 512.20 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 10:15:00 | 514.00 | 512.37 | 512.37 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 12:15:00 | 510.85 | 512.19 | 512.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 14:15:00 | 506.10 | 510.54 | 511.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 15:15:00 | 506.00 | 503.75 | 506.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 15:15:00 | 506.00 | 503.75 | 506.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 506.00 | 503.75 | 506.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:15:00 | 506.25 | 503.75 | 506.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 501.60 | 503.32 | 506.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:15:00 | 499.20 | 503.26 | 505.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 10:15:00 | 474.24 | 481.08 | 485.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-26 10:15:00 | 481.80 | 481.08 | 485.63 | SL hit (close>static) qty=0.50 sl=481.08 alert=retest2 |

### Cycle 108 — BUY (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 09:15:00 | 486.00 | 476.12 | 476.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 10:15:00 | 492.35 | 479.36 | 477.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 10:15:00 | 489.70 | 491.16 | 485.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-02 11:00:00 | 489.70 | 491.16 | 485.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 486.05 | 491.03 | 488.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 10:00:00 | 486.05 | 491.03 | 488.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 488.20 | 490.46 | 488.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 10:30:00 | 487.25 | 490.46 | 488.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 11:15:00 | 492.65 | 490.90 | 488.61 | EMA400 retest candle locked (from upside) |

### Cycle 109 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 476.70 | 486.19 | 487.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 468.50 | 482.65 | 485.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 10:15:00 | 469.15 | 469.08 | 475.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 11:00:00 | 469.15 | 469.08 | 475.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 466.25 | 468.64 | 472.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 12:45:00 | 460.70 | 465.90 | 469.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 437.66 | 447.99 | 455.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 09:15:00 | 434.80 | 434.44 | 443.44 | SL hit (close>ema200) qty=0.50 sl=434.44 alert=retest2 |

### Cycle 110 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 452.00 | 442.82 | 441.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 12:15:00 | 457.10 | 450.92 | 447.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 456.70 | 458.91 | 455.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 09:15:00 | 456.70 | 458.91 | 455.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 456.70 | 458.91 | 455.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 456.70 | 458.91 | 455.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 453.10 | 457.75 | 454.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 453.10 | 457.75 | 454.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 452.85 | 456.77 | 454.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 453.60 | 456.77 | 454.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 451.55 | 455.72 | 454.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 13:15:00 | 450.35 | 455.72 | 454.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2025-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 14:15:00 | 446.95 | 453.04 | 453.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 435.35 | 448.61 | 451.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 436.00 | 433.77 | 440.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 10:00:00 | 436.00 | 433.77 | 440.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 391.95 | 412.98 | 419.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 10:15:00 | 389.25 | 412.98 | 419.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 10:45:00 | 390.50 | 408.30 | 417.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 15:15:00 | 420.00 | 414.04 | 413.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 15:15:00 | 420.00 | 414.04 | 413.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 440.15 | 419.26 | 415.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 09:15:00 | 442.45 | 444.84 | 433.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 09:30:00 | 450.00 | 444.84 | 433.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 454.75 | 455.11 | 446.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 451.80 | 455.11 | 446.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 448.70 | 453.83 | 446.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 451.25 | 453.83 | 446.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 14:15:00 | 447.65 | 452.14 | 447.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 15:00:00 | 447.65 | 452.14 | 447.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 15:15:00 | 446.50 | 451.01 | 447.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 09:15:00 | 454.75 | 451.01 | 447.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 480.90 | 456.99 | 450.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 11:00:00 | 483.05 | 462.20 | 453.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 12:45:00 | 481.50 | 468.87 | 458.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 12:00:00 | 482.50 | 479.27 | 469.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 12:30:00 | 481.60 | 479.70 | 470.22 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 483.45 | 482.91 | 474.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 11:15:00 | 490.00 | 483.51 | 475.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 13:00:00 | 491.00 | 485.44 | 478.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 14:15:00 | 491.10 | 485.97 | 479.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 09:30:00 | 490.60 | 488.95 | 482.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 487.70 | 490.33 | 486.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:45:00 | 486.25 | 490.33 | 486.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 485.15 | 489.29 | 486.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:45:00 | 489.00 | 488.88 | 486.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 12:15:00 | 500.85 | 488.88 | 486.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 11:45:00 | 488.45 | 493.47 | 491.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 13:15:00 | 491.30 | 491.95 | 490.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 14:15:00 | 492.50 | 491.76 | 491.05 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-11 09:15:00 | 477.90 | 488.87 | 489.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 09:15:00 | 477.90 | 488.87 | 489.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 11:15:00 | 476.00 | 484.58 | 487.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 475.60 | 475.04 | 480.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:45:00 | 478.25 | 475.04 | 480.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 485.80 | 477.19 | 480.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 485.80 | 477.19 | 480.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 480.00 | 477.75 | 480.70 | EMA400 retest candle locked (from downside) |

### Cycle 114 — BUY (started 2025-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 15:15:00 | 490.00 | 483.48 | 482.99 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2025-02-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 12:15:00 | 475.80 | 485.74 | 487.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 13:15:00 | 470.80 | 482.75 | 485.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-19 09:15:00 | 457.60 | 453.38 | 460.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 09:15:00 | 457.60 | 453.38 | 460.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 457.60 | 453.38 | 460.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:30:00 | 458.40 | 453.38 | 460.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 11:15:00 | 457.60 | 454.41 | 459.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 12:00:00 | 457.60 | 454.41 | 459.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 14:15:00 | 459.50 | 455.77 | 459.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 15:00:00 | 459.50 | 455.77 | 459.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 15:15:00 | 465.10 | 457.64 | 459.65 | EMA400 retest candle locked (from downside) |

### Cycle 116 — BUY (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 10:15:00 | 472.85 | 462.51 | 461.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-21 11:15:00 | 476.80 | 469.19 | 466.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 09:15:00 | 474.45 | 475.31 | 470.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 09:15:00 | 474.45 | 475.31 | 470.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 474.45 | 475.31 | 470.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:45:00 | 471.90 | 475.31 | 470.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 10:15:00 | 473.85 | 475.02 | 471.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 10:45:00 | 472.40 | 475.02 | 471.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 483.40 | 482.54 | 477.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 10:45:00 | 481.35 | 482.54 | 477.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 11:15:00 | 482.00 | 487.33 | 483.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 12:00:00 | 482.00 | 487.33 | 483.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 12:15:00 | 481.25 | 486.12 | 483.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 13:00:00 | 481.25 | 486.12 | 483.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 15:15:00 | 487.10 | 484.18 | 483.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 09:15:00 | 476.00 | 484.18 | 483.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — SELL (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 09:15:00 | 466.10 | 480.57 | 481.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 10:15:00 | 462.70 | 476.99 | 479.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 14:15:00 | 453.85 | 453.83 | 462.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 14:45:00 | 456.90 | 453.83 | 462.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 457.90 | 453.87 | 460.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 457.90 | 453.87 | 460.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 11:15:00 | 448.80 | 453.38 | 459.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:30:00 | 452.70 | 453.38 | 459.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 491.45 | 456.39 | 457.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:45:00 | 496.35 | 456.39 | 457.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 491.35 | 463.39 | 460.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 13:15:00 | 499.10 | 479.56 | 469.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 11:15:00 | 514.45 | 516.52 | 507.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 12:00:00 | 514.45 | 516.52 | 507.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 502.70 | 512.49 | 507.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 15:00:00 | 502.70 | 512.49 | 507.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 501.70 | 510.33 | 507.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 496.00 | 510.33 | 507.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 502.25 | 506.75 | 506.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 12:00:00 | 502.25 | 506.75 | 506.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2025-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 12:15:00 | 501.20 | 505.64 | 505.71 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2025-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 14:15:00 | 508.50 | 506.01 | 505.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 15:15:00 | 510.05 | 506.82 | 506.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 11:15:00 | 508.40 | 508.61 | 507.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-12 11:30:00 | 509.25 | 508.61 | 507.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 522.25 | 511.34 | 508.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 13:30:00 | 527.60 | 514.47 | 510.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 10:15:00 | 524.05 | 520.99 | 514.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 12:15:00 | 510.40 | 513.61 | 513.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2025-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 12:15:00 | 510.40 | 513.61 | 513.78 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 546.45 | 519.61 | 516.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 10:15:00 | 549.55 | 525.60 | 519.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 569.50 | 574.40 | 559.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 569.50 | 574.40 | 559.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 574.00 | 592.93 | 588.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 574.00 | 592.93 | 588.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 573.10 | 588.96 | 586.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:00:00 | 573.10 | 588.96 | 586.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — SELL (started 2025-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 12:15:00 | 577.85 | 584.51 | 585.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 15:15:00 | 571.25 | 577.34 | 580.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 582.15 | 577.91 | 580.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 10:15:00 | 582.15 | 577.91 | 580.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 582.15 | 577.91 | 580.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:45:00 | 585.00 | 577.91 | 580.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 581.10 | 578.55 | 580.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:45:00 | 583.00 | 578.55 | 580.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2025-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 13:15:00 | 587.75 | 581.91 | 581.57 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-03-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 15:15:00 | 579.25 | 582.51 | 582.74 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 09:15:00 | 598.15 | 585.64 | 584.14 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2025-04-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 12:15:00 | 579.80 | 587.28 | 587.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 09:15:00 | 547.20 | 578.02 | 583.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 529.90 | 525.15 | 541.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 529.90 | 525.15 | 541.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 529.90 | 525.15 | 541.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 524.50 | 525.73 | 539.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 11:30:00 | 526.20 | 525.94 | 538.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 13:45:00 | 527.65 | 527.26 | 537.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 15:00:00 | 527.05 | 527.22 | 536.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 524.20 | 520.67 | 526.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 10:15:00 | 522.15 | 520.67 | 526.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 11:45:00 | 523.50 | 522.60 | 526.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 09:15:00 | 552.80 | 532.66 | 530.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 552.80 | 532.66 | 530.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 09:15:00 | 564.95 | 552.15 | 542.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 13:15:00 | 562.00 | 562.41 | 556.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 14:00:00 | 562.00 | 562.41 | 556.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 571.70 | 563.66 | 558.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 10:15:00 | 577.50 | 563.66 | 558.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 10:15:00 | 572.15 | 574.93 | 571.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 11:00:00 | 573.65 | 574.67 | 571.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 14:15:00 | 572.50 | 575.45 | 574.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-24 14:15:00 | 566.60 | 573.68 | 573.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2025-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 14:15:00 | 566.60 | 573.68 | 573.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 551.10 | 568.95 | 571.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 562.35 | 554.57 | 560.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 562.35 | 554.57 | 560.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 562.35 | 554.57 | 560.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 09:30:00 | 560.15 | 554.57 | 560.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 559.55 | 555.57 | 560.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 11:15:00 | 558.05 | 555.57 | 560.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 10:00:00 | 558.00 | 559.19 | 560.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-02 14:15:00 | 530.15 | 542.13 | 546.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-02 14:15:00 | 530.10 | 542.13 | 546.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-05 09:15:00 | 543.85 | 542.12 | 545.80 | SL hit (close>ema200) qty=0.50 sl=542.12 alert=retest2 |

### Cycle 130 — BUY (started 2025-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 12:15:00 | 555.70 | 547.92 | 546.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 13:15:00 | 558.90 | 550.11 | 548.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 550.05 | 551.41 | 549.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 09:15:00 | 550.05 | 551.41 | 549.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 550.05 | 551.41 | 549.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 10:15:00 | 560.00 | 551.41 | 549.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 10:00:00 | 555.85 | 570.37 | 567.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-12 11:15:00 | 611.44 | 585.68 | 576.64 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 131 — SELL (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 11:15:00 | 616.50 | 625.89 | 625.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-16 13:15:00 | 614.05 | 621.85 | 623.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-19 09:15:00 | 634.40 | 621.79 | 623.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-19 09:15:00 | 634.40 | 621.79 | 623.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 634.40 | 621.79 | 623.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:00:00 | 634.40 | 621.79 | 623.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 632.55 | 623.94 | 624.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:45:00 | 629.50 | 623.94 | 624.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — BUY (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 11:15:00 | 629.75 | 625.10 | 624.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 12:15:00 | 636.00 | 627.28 | 625.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 12:15:00 | 631.70 | 631.82 | 629.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 12:15:00 | 631.70 | 631.82 | 629.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 631.70 | 631.82 | 629.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:45:00 | 629.10 | 631.82 | 629.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 628.05 | 631.07 | 629.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 628.05 | 631.07 | 629.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 622.10 | 629.27 | 628.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 622.10 | 629.27 | 628.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 622.85 | 627.99 | 628.04 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 679.35 | 638.26 | 632.71 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 13:15:00 | 624.80 | 639.39 | 640.74 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2025-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 13:15:00 | 651.50 | 642.00 | 641.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 15:15:00 | 655.95 | 646.50 | 643.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 13:15:00 | 648.05 | 648.05 | 645.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 14:00:00 | 648.05 | 648.05 | 645.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 643.30 | 647.39 | 645.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:45:00 | 641.70 | 647.39 | 645.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 646.30 | 647.17 | 645.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 10:30:00 | 642.10 | 647.17 | 645.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 647.60 | 647.26 | 646.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:30:00 | 646.40 | 647.26 | 646.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 646.65 | 647.14 | 646.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:30:00 | 647.05 | 647.14 | 646.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 646.40 | 647.00 | 646.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:45:00 | 644.10 | 647.00 | 646.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 647.95 | 647.19 | 646.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 652.00 | 647.19 | 646.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 13:15:00 | 642.10 | 645.77 | 646.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 13:15:00 | 642.10 | 645.77 | 646.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 14:15:00 | 640.30 | 644.68 | 645.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 647.50 | 644.41 | 645.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 647.50 | 644.41 | 645.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 647.50 | 644.41 | 645.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 647.50 | 644.41 | 645.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 10:15:00 | 651.20 | 645.77 | 645.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 14:15:00 | 661.00 | 650.41 | 648.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 09:15:00 | 648.75 | 655.66 | 653.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 09:15:00 | 648.75 | 655.66 | 653.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 648.75 | 655.66 | 653.27 | EMA400 retest candle locked (from upside) |

### Cycle 139 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 650.20 | 651.94 | 652.10 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 09:15:00 | 657.55 | 652.78 | 652.44 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2025-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 15:15:00 | 650.10 | 652.08 | 652.32 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 654.95 | 652.65 | 652.56 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 645.75 | 651.54 | 652.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 14:15:00 | 638.65 | 648.97 | 650.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 09:15:00 | 636.35 | 634.49 | 638.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 09:15:00 | 636.35 | 634.49 | 638.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 636.35 | 634.49 | 638.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:00:00 | 636.35 | 634.49 | 638.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 643.30 | 636.25 | 638.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:45:00 | 643.00 | 636.25 | 638.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 644.00 | 637.80 | 639.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 12:30:00 | 639.10 | 638.21 | 639.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 14:15:00 | 644.45 | 640.01 | 639.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — BUY (started 2025-06-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 14:15:00 | 644.45 | 640.01 | 639.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 09:15:00 | 646.15 | 642.04 | 640.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 12:15:00 | 642.30 | 642.53 | 641.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 13:00:00 | 642.30 | 642.53 | 641.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 636.25 | 641.28 | 641.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 636.60 | 641.28 | 641.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 639.70 | 640.96 | 640.89 | EMA400 retest candle locked (from upside) |

### Cycle 145 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 639.00 | 640.57 | 640.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 635.95 | 639.65 | 640.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 626.80 | 607.06 | 610.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 626.80 | 607.06 | 610.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 626.80 | 607.06 | 610.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:00:00 | 626.80 | 607.06 | 610.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 611.50 | 607.94 | 610.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:15:00 | 608.55 | 608.34 | 610.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 09:45:00 | 604.95 | 607.48 | 609.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 12:15:00 | 603.85 | 597.85 | 597.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 603.85 | 597.85 | 597.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 621.45 | 604.63 | 600.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 09:15:00 | 929.90 | 954.41 | 880.95 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 09:45:00 | 953.60 | 942.59 | 908.12 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 15:15:00 | 955.00 | 942.05 | 920.66 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 917.20 | 936.66 | 923.62 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-07 10:15:00 | 917.20 | 936.66 | 923.62 | SL hit (close<ema400) qty=1.00 sl=923.62 alert=retest1 |

### Cycle 147 — SELL (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 10:15:00 | 1043.00 | 1076.75 | 1078.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 11:15:00 | 1031.00 | 1046.06 | 1058.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 09:15:00 | 1038.85 | 996.17 | 1008.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 1038.85 | 996.17 | 1008.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1038.85 | 996.17 | 1008.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:30:00 | 1033.00 | 996.17 | 1008.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 1030.90 | 1003.12 | 1010.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:45:00 | 1018.10 | 1010.35 | 1012.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 09:15:00 | 967.19 | 985.79 | 991.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 12:15:00 | 990.70 | 985.70 | 990.02 | SL hit (close>ema200) qty=0.50 sl=985.70 alert=retest2 |

### Cycle 148 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 1022.00 | 992.96 | 992.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 12:15:00 | 1039.05 | 1018.60 | 1007.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 1024.55 | 1030.35 | 1017.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 1024.55 | 1030.35 | 1017.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1024.55 | 1030.35 | 1017.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 1041.00 | 1030.72 | 1023.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 13:30:00 | 1043.00 | 1030.35 | 1025.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 15:15:00 | 999.80 | 1020.90 | 1021.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 999.80 | 1020.90 | 1021.94 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 10:15:00 | 1024.00 | 1022.80 | 1022.70 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 11:15:00 | 1011.10 | 1020.46 | 1021.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 13:15:00 | 1005.00 | 1016.11 | 1019.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 14:15:00 | 1021.30 | 1017.15 | 1019.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 14:15:00 | 1021.30 | 1017.15 | 1019.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1021.30 | 1017.15 | 1019.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:30:00 | 1022.40 | 1017.15 | 1019.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 1028.80 | 1019.48 | 1020.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 1033.10 | 1019.48 | 1020.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 1005.60 | 1006.33 | 1011.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 13:15:00 | 992.50 | 1003.84 | 1007.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:45:00 | 999.00 | 1006.61 | 1007.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 09:15:00 | 1020.40 | 1009.91 | 1009.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — BUY (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 09:15:00 | 1020.40 | 1009.91 | 1009.04 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 14:15:00 | 995.60 | 1006.49 | 1007.81 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 09:15:00 | 1049.70 | 1016.51 | 1012.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 10:15:00 | 1051.90 | 1023.59 | 1015.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 1082.00 | 1089.35 | 1068.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 10:00:00 | 1082.00 | 1089.35 | 1068.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 1073.00 | 1083.32 | 1073.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 15:00:00 | 1073.00 | 1083.32 | 1073.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 1075.00 | 1081.66 | 1073.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 1146.70 | 1081.66 | 1073.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 10:15:00 | 1141.60 | 1177.82 | 1178.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 10:15:00 | 1141.60 | 1177.82 | 1178.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 12:15:00 | 1135.60 | 1162.72 | 1171.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 1140.40 | 1128.91 | 1142.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 09:15:00 | 1140.40 | 1128.91 | 1142.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 1140.40 | 1128.91 | 1142.14 | EMA400 retest candle locked (from downside) |

### Cycle 156 — BUY (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 11:15:00 | 1154.90 | 1143.09 | 1142.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 09:15:00 | 1180.00 | 1154.16 | 1148.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 09:15:00 | 1220.20 | 1232.58 | 1210.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 09:45:00 | 1227.00 | 1232.58 | 1210.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 1223.00 | 1231.20 | 1219.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:15:00 | 1241.90 | 1231.20 | 1219.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 15:15:00 | 1226.90 | 1236.99 | 1229.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 10:15:00 | 1211.20 | 1230.38 | 1227.94 | SL hit (close<static) qty=1.00 sl=1219.00 alert=retest2 |

### Cycle 157 — SELL (started 2025-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 13:15:00 | 1217.00 | 1225.60 | 1226.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 14:15:00 | 1203.80 | 1221.24 | 1224.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1229.00 | 1219.39 | 1222.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 1229.00 | 1219.39 | 1222.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1229.00 | 1219.39 | 1222.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 1229.00 | 1219.39 | 1222.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1219.90 | 1219.49 | 1222.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 12:00:00 | 1215.00 | 1218.60 | 1221.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 09:15:00 | 1265.90 | 1230.48 | 1226.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 09:15:00 | 1265.90 | 1230.48 | 1226.46 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 11:15:00 | 1234.00 | 1243.70 | 1244.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 12:15:00 | 1229.00 | 1240.76 | 1243.47 | Break + close below crossover candle low |

### Cycle 160 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 1284.20 | 1245.43 | 1244.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 12:15:00 | 1296.90 | 1268.40 | 1256.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 09:15:00 | 1309.50 | 1311.12 | 1294.11 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 13:15:00 | 1321.80 | 1311.41 | 1298.41 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1271.10 | 1303.33 | 1298.85 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 1271.10 | 1303.33 | 1298.85 | SL hit (close<ema400) qty=1.00 sl=1298.85 alert=retest1 |

### Cycle 161 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 10:15:00 | 1256.60 | 1293.98 | 1295.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 11:15:00 | 1243.90 | 1283.97 | 1290.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 13:15:00 | 1256.20 | 1248.64 | 1263.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 13:15:00 | 1256.20 | 1248.64 | 1263.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 1256.20 | 1248.64 | 1263.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:45:00 | 1261.00 | 1248.64 | 1263.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 1277.40 | 1254.39 | 1264.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 1277.40 | 1254.39 | 1264.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 1300.10 | 1263.53 | 1267.95 | EMA400 retest candle locked (from downside) |

### Cycle 162 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 1300.60 | 1270.94 | 1270.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 10:15:00 | 1310.00 | 1278.76 | 1274.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 15:15:00 | 1275.00 | 1285.52 | 1280.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 15:15:00 | 1275.00 | 1285.52 | 1280.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 1275.00 | 1285.52 | 1280.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 09:15:00 | 1301.50 | 1285.52 | 1280.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 15:15:00 | 1290.00 | 1293.65 | 1288.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 09:30:00 | 1293.90 | 1293.24 | 1288.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 13:15:00 | 1278.60 | 1285.54 | 1286.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 13:15:00 | 1278.60 | 1285.54 | 1286.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 09:15:00 | 1269.70 | 1280.90 | 1283.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 09:15:00 | 1270.00 | 1258.92 | 1268.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 1270.00 | 1258.92 | 1268.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 1270.00 | 1258.92 | 1268.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:00:00 | 1270.00 | 1258.92 | 1268.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 1281.90 | 1263.52 | 1269.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:45:00 | 1286.20 | 1263.52 | 1269.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 1282.00 | 1267.22 | 1271.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:30:00 | 1278.20 | 1267.22 | 1271.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1235.20 | 1207.82 | 1223.00 | EMA400 retest candle locked (from downside) |

### Cycle 164 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 1244.60 | 1230.78 | 1229.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 10:15:00 | 1258.10 | 1236.24 | 1231.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 09:15:00 | 1270.70 | 1271.22 | 1260.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 09:30:00 | 1269.70 | 1271.22 | 1260.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 1298.00 | 1313.07 | 1306.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:00:00 | 1298.00 | 1313.07 | 1306.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 1290.60 | 1308.58 | 1305.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:30:00 | 1293.90 | 1308.58 | 1305.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — SELL (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 09:15:00 | 1284.70 | 1301.15 | 1302.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 10:15:00 | 1276.60 | 1296.24 | 1299.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 1240.70 | 1240.44 | 1253.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 11:00:00 | 1240.70 | 1240.44 | 1253.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 1259.80 | 1244.60 | 1252.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:00:00 | 1259.80 | 1244.60 | 1252.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 1296.90 | 1255.06 | 1256.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 1296.90 | 1255.06 | 1256.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — BUY (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 15:15:00 | 1288.00 | 1261.65 | 1259.10 | EMA200 above EMA400 |

### Cycle 167 — SELL (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 12:15:00 | 1233.20 | 1263.51 | 1263.63 | EMA200 below EMA400 |

### Cycle 168 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 1273.60 | 1261.03 | 1260.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 14:15:00 | 1296.00 | 1269.78 | 1264.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-21 14:15:00 | 1273.90 | 1275.91 | 1268.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 14:15:00 | 1273.90 | 1275.91 | 1268.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 1273.90 | 1275.91 | 1268.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 1273.90 | 1275.91 | 1268.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1271.00 | 1274.93 | 1269.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 1271.00 | 1274.93 | 1269.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 1266.90 | 1273.32 | 1268.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:00:00 | 1266.90 | 1273.32 | 1268.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 1266.30 | 1271.92 | 1268.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:45:00 | 1263.30 | 1271.92 | 1268.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 1263.60 | 1270.25 | 1268.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:30:00 | 1262.50 | 1270.25 | 1268.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 1245.50 | 1265.74 | 1266.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 09:15:00 | 1245.00 | 1258.40 | 1262.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 10:15:00 | 1245.60 | 1245.14 | 1251.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 10:30:00 | 1245.20 | 1245.14 | 1251.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 1251.00 | 1246.37 | 1251.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:30:00 | 1249.50 | 1246.37 | 1251.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 1260.00 | 1249.09 | 1252.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:45:00 | 1260.00 | 1249.09 | 1252.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 1240.00 | 1247.28 | 1250.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 15:15:00 | 1237.80 | 1247.28 | 1250.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 1263.00 | 1248.64 | 1250.43 | SL hit (close>static) qty=1.00 sl=1260.20 alert=retest2 |

### Cycle 170 — BUY (started 2025-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 12:15:00 | 1261.90 | 1252.93 | 1252.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 14:15:00 | 1284.30 | 1259.95 | 1255.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 1264.00 | 1280.09 | 1272.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 1264.00 | 1280.09 | 1272.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 1264.00 | 1280.09 | 1272.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 1264.00 | 1280.09 | 1272.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 1255.50 | 1275.18 | 1270.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:00:00 | 1255.50 | 1275.18 | 1270.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 171 — SELL (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 13:15:00 | 1257.80 | 1267.85 | 1268.10 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 1302.80 | 1273.02 | 1270.18 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 1267.00 | 1279.79 | 1281.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 1245.80 | 1267.25 | 1274.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 13:15:00 | 1235.90 | 1234.57 | 1246.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:45:00 | 1238.00 | 1234.57 | 1246.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 1247.00 | 1237.05 | 1246.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 1247.00 | 1237.05 | 1246.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 1236.00 | 1236.84 | 1245.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 1228.90 | 1236.84 | 1245.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1228.10 | 1235.09 | 1244.11 | EMA400 retest candle locked (from downside) |

### Cycle 174 — BUY (started 2025-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 15:15:00 | 1266.00 | 1248.79 | 1247.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 11:15:00 | 1272.20 | 1255.92 | 1251.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 15:15:00 | 1259.00 | 1259.84 | 1254.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 09:15:00 | 1271.00 | 1259.84 | 1254.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 1262.60 | 1260.39 | 1255.47 | EMA400 retest candle locked (from upside) |

### Cycle 175 — SELL (started 2025-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 12:15:00 | 1233.20 | 1253.97 | 1255.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 14:15:00 | 1230.00 | 1245.92 | 1251.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 09:15:00 | 1028.00 | 1000.28 | 1050.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 10:00:00 | 1028.00 | 1000.28 | 1050.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 1062.30 | 1012.68 | 1051.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:45:00 | 1061.90 | 1012.68 | 1051.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 1064.90 | 1023.13 | 1052.59 | EMA400 retest candle locked (from downside) |

### Cycle 176 — BUY (started 2025-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 15:15:00 | 1105.00 | 1065.58 | 1065.26 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 14:15:00 | 1052.40 | 1065.99 | 1066.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 1026.00 | 1055.91 | 1061.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 11:15:00 | 1066.70 | 1052.84 | 1058.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 11:15:00 | 1066.70 | 1052.84 | 1058.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 1066.70 | 1052.84 | 1058.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:00:00 | 1066.70 | 1052.84 | 1058.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 12:15:00 | 1073.30 | 1056.93 | 1060.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:30:00 | 1091.50 | 1056.93 | 1060.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 1046.10 | 1038.76 | 1046.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 12:30:00 | 1047.90 | 1038.76 | 1046.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 1045.40 | 1040.09 | 1046.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 13:30:00 | 1046.90 | 1040.09 | 1046.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 1056.20 | 1043.31 | 1047.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 1056.20 | 1043.31 | 1047.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 1044.00 | 1043.45 | 1047.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 1032.00 | 1043.45 | 1047.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 14:00:00 | 1031.90 | 1035.13 | 1041.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 11:15:00 | 1055.50 | 1042.93 | 1042.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 1055.50 | 1042.93 | 1042.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 13:15:00 | 1061.60 | 1048.61 | 1045.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 1047.70 | 1050.57 | 1047.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 1047.70 | 1050.57 | 1047.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 1047.70 | 1050.57 | 1047.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:45:00 | 1050.40 | 1050.57 | 1047.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 1050.50 | 1050.55 | 1047.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:15:00 | 1048.60 | 1050.55 | 1047.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 1054.60 | 1051.36 | 1048.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:30:00 | 1046.60 | 1051.36 | 1048.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 1040.50 | 1049.19 | 1047.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:00:00 | 1040.50 | 1049.19 | 1047.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1045.50 | 1048.45 | 1047.29 | EMA400 retest candle locked (from upside) |

### Cycle 179 — SELL (started 2025-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 15:15:00 | 1034.00 | 1044.08 | 1045.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 1025.30 | 1040.33 | 1043.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 1058.10 | 1036.92 | 1038.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 1058.10 | 1036.92 | 1038.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1058.10 | 1036.92 | 1038.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:30:00 | 1059.70 | 1036.92 | 1038.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 1051.70 | 1039.88 | 1040.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:30:00 | 1054.30 | 1039.88 | 1040.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — BUY (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 11:15:00 | 1050.20 | 1041.94 | 1041.02 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 1025.90 | 1039.90 | 1040.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 1002.70 | 1025.53 | 1032.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 09:15:00 | 988.30 | 984.02 | 998.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 10:15:00 | 995.00 | 986.21 | 997.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 995.00 | 986.21 | 997.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:45:00 | 996.70 | 986.21 | 997.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 942.90 | 935.21 | 948.38 | EMA400 retest candle locked (from downside) |

### Cycle 182 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 983.80 | 958.41 | 956.37 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 12:15:00 | 958.00 | 958.71 | 958.73 | EMA200 below EMA400 |

### Cycle 184 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 961.60 | 959.17 | 958.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 15:15:00 | 963.00 | 959.94 | 959.30 | Break + close above crossover candle high |

### Cycle 185 — SELL (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 09:15:00 | 954.50 | 958.85 | 958.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-12 10:15:00 | 948.50 | 956.78 | 957.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 14:15:00 | 974.10 | 954.67 | 955.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 14:15:00 | 974.10 | 954.67 | 955.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 974.10 | 954.67 | 955.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 15:00:00 | 974.10 | 954.67 | 955.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 15:15:00 | 956.00 | 954.93 | 955.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 09:15:00 | 948.90 | 954.93 | 955.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 10:15:00 | 963.10 | 956.77 | 956.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — BUY (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 10:15:00 | 963.10 | 956.77 | 956.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 10:15:00 | 976.90 | 964.15 | 960.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 09:15:00 | 959.90 | 968.76 | 965.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 959.90 | 968.76 | 965.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 959.90 | 968.76 | 965.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 959.90 | 968.76 | 965.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 956.60 | 966.33 | 964.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:00:00 | 956.60 | 966.33 | 964.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 953.00 | 961.80 | 962.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 945.10 | 956.80 | 959.84 | Break + close below crossover candle low |

### Cycle 188 — BUY (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 10:15:00 | 982.50 | 961.94 | 961.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 11:15:00 | 1022.50 | 974.05 | 967.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 11:15:00 | 1039.70 | 1040.31 | 1013.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-19 12:00:00 | 1039.70 | 1040.31 | 1013.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 1041.30 | 1048.98 | 1041.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:30:00 | 1039.20 | 1048.98 | 1041.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 1038.90 | 1046.97 | 1041.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 1038.90 | 1046.97 | 1041.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 1040.20 | 1045.61 | 1040.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 14:45:00 | 1051.40 | 1049.69 | 1043.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 13:45:00 | 1045.20 | 1048.24 | 1045.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:15:00 | 1042.40 | 1046.23 | 1045.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 10:15:00 | 1032.30 | 1043.45 | 1044.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 1032.30 | 1043.45 | 1044.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 14:15:00 | 1027.50 | 1037.40 | 1040.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 09:15:00 | 1039.20 | 1037.06 | 1040.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 1039.20 | 1037.06 | 1040.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1039.20 | 1037.06 | 1040.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:30:00 | 1046.60 | 1037.06 | 1040.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 1036.00 | 1036.85 | 1039.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:30:00 | 1041.90 | 1036.85 | 1039.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 1007.10 | 1016.48 | 1026.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 13:15:00 | 998.60 | 1009.17 | 1020.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 10:00:00 | 993.00 | 1002.64 | 1013.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:15:00 | 996.00 | 1002.35 | 1012.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 12:00:00 | 998.70 | 1001.62 | 1011.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 1008.50 | 1002.90 | 1010.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:00:00 | 1008.50 | 1002.90 | 1010.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 1008.90 | 1004.10 | 1009.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 1008.90 | 1004.10 | 1009.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 1006.70 | 1004.62 | 1009.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 1050.70 | 1004.62 | 1009.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1037.90 | 1011.27 | 1012.26 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-01 09:15:00 | 1037.90 | 1011.27 | 1012.26 | SL hit (close>static) qty=1.00 sl=1030.00 alert=retest2 |

### Cycle 190 — BUY (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 10:15:00 | 1038.90 | 1016.80 | 1014.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 1052.10 | 1035.60 | 1026.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 1058.00 | 1061.13 | 1049.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 1058.00 | 1061.13 | 1049.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1048.50 | 1057.67 | 1050.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:30:00 | 1047.70 | 1057.67 | 1050.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 1043.50 | 1054.84 | 1050.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:30:00 | 1043.50 | 1054.84 | 1050.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — SELL (started 2026-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 15:15:00 | 1040.00 | 1046.97 | 1047.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 09:15:00 | 1032.70 | 1044.12 | 1046.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 958.90 | 950.73 | 967.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 15:00:00 | 958.90 | 950.73 | 967.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 12:15:00 | 927.00 | 923.80 | 932.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 914.80 | 924.39 | 930.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 869.06 | 898.87 | 912.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 15:15:00 | 899.90 | 893.71 | 902.88 | SL hit (close>ema200) qty=0.50 sl=893.71 alert=retest2 |

### Cycle 192 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 920.50 | 908.47 | 907.08 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 900.00 | 906.08 | 906.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 11:15:00 | 890.70 | 903.01 | 904.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-23 14:15:00 | 900.90 | 900.50 | 903.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 14:15:00 | 900.90 | 900.50 | 903.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 900.90 | 900.50 | 903.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 900.90 | 900.50 | 903.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 893.60 | 888.49 | 893.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 897.30 | 888.49 | 893.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 893.30 | 889.45 | 893.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 13:30:00 | 883.00 | 890.21 | 892.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 14:15:00 | 896.90 | 891.54 | 893.21 | SL hit (close>static) qty=1.00 sl=894.80 alert=retest2 |

### Cycle 194 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 941.70 | 895.24 | 891.70 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 900.60 | 911.89 | 912.87 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 926.70 | 912.89 | 912.87 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2026-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-03 13:15:00 | 889.40 | 913.08 | 913.90 | EMA200 below EMA400 |

### Cycle 198 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 946.80 | 916.27 | 914.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 10:15:00 | 969.35 | 926.89 | 919.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 969.30 | 976.27 | 960.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 969.30 | 976.27 | 960.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 969.30 | 976.27 | 960.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 09:30:00 | 991.45 | 971.63 | 965.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 12:15:00 | 1003.05 | 1006.81 | 1007.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 199 — SELL (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 12:15:00 | 1003.05 | 1006.81 | 1007.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 13:15:00 | 996.05 | 1004.66 | 1006.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 11:15:00 | 972.80 | 972.57 | 982.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 11:45:00 | 974.05 | 972.57 | 982.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 979.00 | 970.89 | 974.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 979.00 | 970.89 | 974.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 994.95 | 975.70 | 976.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 994.95 | 975.70 | 976.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 975.65 | 975.69 | 976.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 12:30:00 | 969.00 | 974.66 | 975.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 13:15:00 | 970.00 | 964.90 | 964.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 200 — BUY (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 13:15:00 | 970.00 | 964.90 | 964.21 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 10:15:00 | 959.25 | 963.43 | 963.91 | EMA200 below EMA400 |

### Cycle 202 — BUY (started 2026-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 15:15:00 | 969.15 | 963.68 | 963.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 983.75 | 967.70 | 965.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 14:15:00 | 1008.75 | 1011.56 | 997.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 15:00:00 | 1008.75 | 1011.56 | 997.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1010.65 | 1010.82 | 999.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 11:30:00 | 1017.00 | 1011.32 | 1001.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 10:15:00 | 977.85 | 997.00 | 998.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 977.85 | 997.00 | 998.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 956.50 | 988.90 | 994.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 935.00 | 930.20 | 943.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 14:15:00 | 935.00 | 930.20 | 943.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 935.00 | 930.20 | 943.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 941.95 | 930.20 | 943.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 946.30 | 934.99 | 943.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:45:00 | 946.60 | 934.99 | 943.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 922.35 | 932.46 | 941.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:30:00 | 919.80 | 929.14 | 937.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 873.81 | 913.63 | 928.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 887.50 | 884.06 | 901.11 | SL hit (close>ema200) qty=0.50 sl=884.06 alert=retest2 |

### Cycle 204 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 867.60 | 854.09 | 853.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 876.60 | 862.51 | 857.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 852.95 | 862.93 | 859.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 852.95 | 862.93 | 859.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 852.95 | 862.93 | 859.38 | EMA400 retest candle locked (from upside) |

### Cycle 205 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 844.25 | 856.33 | 856.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 837.00 | 850.39 | 853.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 849.80 | 846.70 | 851.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 849.80 | 846.70 | 851.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 849.80 | 846.70 | 851.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 850.75 | 846.70 | 851.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 851.10 | 847.58 | 851.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 850.15 | 847.58 | 851.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 857.30 | 849.52 | 851.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:00:00 | 857.30 | 849.52 | 851.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 206 — BUY (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 12:15:00 | 868.00 | 853.22 | 853.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 13:15:00 | 870.00 | 856.58 | 854.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 838.65 | 854.86 | 854.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 09:15:00 | 838.65 | 854.86 | 854.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 838.65 | 854.86 | 854.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 838.65 | 854.86 | 854.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 837.15 | 851.32 | 852.92 | EMA200 below EMA400 |

### Cycle 208 — BUY (started 2026-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 13:15:00 | 862.00 | 848.05 | 847.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 908.75 | 863.59 | 855.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 858.00 | 875.47 | 867.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 858.00 | 875.47 | 867.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 858.00 | 875.47 | 867.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:45:00 | 850.55 | 875.47 | 867.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 834.00 | 867.18 | 864.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 834.00 | 867.18 | 864.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 209 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 828.65 | 859.47 | 861.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 13:15:00 | 825.45 | 847.45 | 855.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 14:15:00 | 825.60 | 819.91 | 833.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-30 15:00:00 | 825.60 | 819.91 | 833.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 863.40 | 829.90 | 835.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 863.30 | 829.90 | 835.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 865.80 | 840.78 | 839.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 869.75 | 846.57 | 842.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 842.95 | 854.36 | 848.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 842.95 | 854.36 | 848.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 842.95 | 854.36 | 848.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:30:00 | 858.95 | 856.51 | 850.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-09 09:15:00 | 944.85 | 916.85 | 902.46 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 211 — SELL (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 13:15:00 | 1004.95 | 1015.83 | 1016.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 996.80 | 1012.02 | 1014.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 987.40 | 981.61 | 993.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 987.40 | 981.61 | 993.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 987.40 | 981.61 | 993.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 991.45 | 981.61 | 993.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 987.50 | 982.79 | 992.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:30:00 | 990.60 | 982.79 | 992.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 1009.80 | 988.19 | 994.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:00:00 | 1009.80 | 988.19 | 994.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 1009.70 | 992.49 | 995.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:30:00 | 1008.80 | 992.49 | 995.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 212 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 1013.75 | 1000.17 | 998.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 10:15:00 | 1020.35 | 1007.71 | 1002.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 1018.00 | 1020.65 | 1014.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 13:15:00 | 1018.00 | 1020.65 | 1014.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 1018.00 | 1020.65 | 1014.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 1018.00 | 1020.65 | 1014.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 1011.25 | 1018.77 | 1014.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:00:00 | 1011.25 | 1018.77 | 1014.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 1012.05 | 1017.42 | 1014.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 1020.30 | 1017.42 | 1014.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 1028.20 | 1021.12 | 1017.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 13:30:00 | 1029.75 | 1022.90 | 1018.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 1053.40 | 1023.28 | 1019.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 12:15:00 | 1029.00 | 1026.49 | 1021.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 13:30:00 | 1031.50 | 1026.49 | 1022.73 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 1062.40 | 1033.71 | 1026.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 10:15:00 | 1075.20 | 1033.71 | 1026.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 13:45:00 | 1071.60 | 1067.27 | 1057.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-08 09:15:00 | 1132.73 | 1118.14 | 1095.03 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-23 14:15:00 | 169.80 | 2023-05-24 11:15:00 | 171.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2023-05-24 10:00:00 | 169.70 | 2023-05-24 11:15:00 | 171.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2023-06-05 09:15:00 | 181.60 | 2023-06-08 14:15:00 | 181.20 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2023-06-05 10:45:00 | 179.80 | 2023-06-08 14:15:00 | 181.20 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest2 | 2023-06-14 09:15:00 | 190.05 | 2023-06-21 12:15:00 | 189.85 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2023-06-14 11:45:00 | 188.55 | 2023-06-21 12:15:00 | 189.85 | STOP_HIT | 1.00 | 0.69% |
| BUY | retest2 | 2023-06-14 15:00:00 | 188.10 | 2023-06-21 12:15:00 | 189.85 | STOP_HIT | 1.00 | 0.93% |
| BUY | retest2 | 2023-06-15 09:45:00 | 188.55 | 2023-06-21 12:15:00 | 189.85 | STOP_HIT | 1.00 | 0.69% |
| BUY | retest2 | 2023-06-16 09:15:00 | 192.40 | 2023-06-21 12:15:00 | 189.85 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2023-06-22 12:15:00 | 190.30 | 2023-06-28 09:15:00 | 190.25 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2023-06-22 14:45:00 | 190.70 | 2023-06-28 09:15:00 | 190.25 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2023-06-27 09:45:00 | 190.45 | 2023-06-28 09:15:00 | 190.25 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2023-07-03 13:15:00 | 193.90 | 2023-07-14 12:15:00 | 203.70 | STOP_HIT | 1.00 | 5.05% |
| BUY | retest2 | 2023-07-04 10:30:00 | 194.60 | 2023-07-14 12:15:00 | 203.70 | STOP_HIT | 1.00 | 4.68% |
| SELL | retest2 | 2023-07-17 13:30:00 | 202.15 | 2023-07-20 09:15:00 | 207.25 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2023-07-18 09:30:00 | 202.35 | 2023-07-20 09:15:00 | 207.25 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2023-07-31 10:00:00 | 222.45 | 2023-08-02 13:15:00 | 218.70 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2023-08-08 13:30:00 | 219.10 | 2023-08-09 09:15:00 | 222.25 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2023-08-25 09:30:00 | 300.90 | 2023-09-05 09:15:00 | 330.99 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-09-08 12:00:00 | 311.40 | 2023-09-12 09:15:00 | 295.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-11 10:30:00 | 311.35 | 2023-09-12 09:15:00 | 295.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-11 11:15:00 | 310.50 | 2023-09-12 09:15:00 | 294.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-12 09:30:00 | 305.15 | 2023-09-12 12:15:00 | 289.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-08 12:00:00 | 311.40 | 2023-09-13 10:15:00 | 300.90 | STOP_HIT | 0.50 | 3.37% |
| SELL | retest2 | 2023-09-11 10:30:00 | 311.35 | 2023-09-13 10:15:00 | 300.90 | STOP_HIT | 0.50 | 3.36% |
| SELL | retest2 | 2023-09-11 11:15:00 | 310.50 | 2023-09-13 10:15:00 | 300.90 | STOP_HIT | 0.50 | 3.09% |
| SELL | retest2 | 2023-09-12 09:30:00 | 305.15 | 2023-09-13 10:15:00 | 300.90 | STOP_HIT | 0.50 | 1.39% |
| BUY | retest2 | 2023-09-29 14:30:00 | 334.65 | 2023-10-04 13:15:00 | 325.15 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2023-10-03 09:15:00 | 337.90 | 2023-10-04 13:15:00 | 325.15 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest2 | 2023-10-03 11:45:00 | 333.45 | 2023-10-04 13:15:00 | 325.15 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2023-10-12 09:15:00 | 350.90 | 2023-10-13 15:15:00 | 342.00 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2023-10-13 09:45:00 | 347.40 | 2023-10-13 15:15:00 | 342.00 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2023-10-13 12:45:00 | 347.50 | 2023-10-13 15:15:00 | 342.00 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2023-10-16 15:15:00 | 347.65 | 2023-10-18 10:15:00 | 343.95 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2023-10-17 14:15:00 | 347.70 | 2023-10-18 10:15:00 | 343.95 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2023-10-19 15:00:00 | 344.30 | 2023-10-20 09:15:00 | 350.80 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2023-11-07 09:15:00 | 339.40 | 2023-11-13 09:15:00 | 373.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-07 12:00:00 | 339.25 | 2023-11-13 09:15:00 | 373.18 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-08 09:15:00 | 341.50 | 2023-11-13 09:15:00 | 375.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-29 15:15:00 | 415.00 | 2023-11-30 15:15:00 | 407.40 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2023-11-30 10:00:00 | 415.75 | 2023-11-30 15:15:00 | 407.40 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2023-12-01 15:15:00 | 415.00 | 2023-12-08 10:15:00 | 419.75 | STOP_HIT | 1.00 | 1.14% |
| SELL | retest2 | 2023-12-14 13:15:00 | 408.85 | 2023-12-15 09:15:00 | 418.00 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2023-12-19 09:15:00 | 420.60 | 2023-12-20 09:15:00 | 409.90 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2023-12-22 13:45:00 | 401.50 | 2024-01-01 10:15:00 | 403.00 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2023-12-26 09:30:00 | 402.85 | 2024-01-01 10:15:00 | 403.00 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2023-12-26 11:30:00 | 401.90 | 2024-01-01 10:15:00 | 403.00 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2023-12-26 12:15:00 | 401.15 | 2024-01-01 10:15:00 | 403.00 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2023-12-27 12:15:00 | 399.15 | 2024-01-01 10:15:00 | 403.00 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2023-12-28 09:30:00 | 399.05 | 2024-01-01 10:15:00 | 403.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-01-15 10:15:00 | 385.00 | 2024-01-18 09:15:00 | 365.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-15 11:00:00 | 384.90 | 2024-01-18 09:15:00 | 365.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-15 14:45:00 | 384.85 | 2024-01-18 09:15:00 | 365.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-16 12:00:00 | 385.00 | 2024-01-18 09:15:00 | 365.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-15 10:15:00 | 385.00 | 2024-01-18 15:15:00 | 370.00 | STOP_HIT | 0.50 | 3.90% |
| SELL | retest2 | 2024-01-15 11:00:00 | 384.90 | 2024-01-18 15:15:00 | 370.00 | STOP_HIT | 0.50 | 3.87% |
| SELL | retest2 | 2024-01-15 14:45:00 | 384.85 | 2024-01-18 15:15:00 | 370.00 | STOP_HIT | 0.50 | 3.86% |
| SELL | retest2 | 2024-01-16 12:00:00 | 385.00 | 2024-01-18 15:15:00 | 370.00 | STOP_HIT | 0.50 | 3.90% |
| SELL | retest2 | 2024-01-29 10:15:00 | 375.00 | 2024-01-31 11:15:00 | 376.90 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-02-14 14:45:00 | 358.00 | 2024-02-16 12:15:00 | 365.10 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2024-02-15 14:00:00 | 359.70 | 2024-02-16 12:15:00 | 365.10 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2024-02-16 10:30:00 | 359.15 | 2024-02-16 12:15:00 | 365.10 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-02-16 11:45:00 | 358.50 | 2024-02-16 12:15:00 | 365.10 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-02-22 14:15:00 | 359.80 | 2024-02-23 09:15:00 | 366.25 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest1 | 2024-02-29 10:15:00 | 344.00 | 2024-03-01 09:15:00 | 348.10 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest1 | 2024-02-29 12:30:00 | 343.90 | 2024-03-01 09:15:00 | 348.10 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest1 | 2024-02-29 13:45:00 | 344.10 | 2024-03-01 09:15:00 | 348.10 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-03-01 10:45:00 | 344.90 | 2024-03-04 13:15:00 | 347.65 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-03-01 12:00:00 | 344.70 | 2024-03-04 13:15:00 | 347.65 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-03-01 12:45:00 | 344.85 | 2024-03-04 13:15:00 | 347.65 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2024-03-01 13:15:00 | 344.75 | 2024-03-04 13:15:00 | 347.65 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-03-06 11:45:00 | 332.40 | 2024-03-12 09:15:00 | 315.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-07 14:45:00 | 330.30 | 2024-03-12 09:15:00 | 313.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-06 11:45:00 | 332.40 | 2024-03-13 09:15:00 | 299.16 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-07 14:45:00 | 330.30 | 2024-03-13 09:15:00 | 297.27 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-03-20 13:30:00 | 326.00 | 2024-03-28 14:15:00 | 334.15 | STOP_HIT | 1.00 | 2.50% |
| BUY | retest2 | 2024-03-20 14:30:00 | 326.45 | 2024-03-28 14:15:00 | 334.15 | STOP_HIT | 1.00 | 2.36% |
| BUY | retest2 | 2024-04-08 09:15:00 | 363.55 | 2024-04-10 11:15:00 | 358.85 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-04-15 09:15:00 | 345.05 | 2024-04-15 09:15:00 | 327.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-15 09:15:00 | 345.05 | 2024-04-16 13:15:00 | 348.85 | STOP_HIT | 0.50 | -1.10% |
| SELL | retest2 | 2024-05-10 12:15:00 | 356.10 | 2024-05-14 11:15:00 | 365.75 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2024-05-13 09:30:00 | 356.55 | 2024-05-14 11:15:00 | 365.75 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2024-05-13 10:30:00 | 353.95 | 2024-05-14 11:15:00 | 365.75 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2024-06-03 10:45:00 | 361.50 | 2024-06-04 09:15:00 | 343.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 10:45:00 | 361.50 | 2024-06-04 12:15:00 | 361.30 | STOP_HIT | 0.50 | 0.06% |
| SELL | retest2 | 2024-06-04 14:45:00 | 362.65 | 2024-06-05 09:15:00 | 373.25 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2024-06-20 09:15:00 | 425.90 | 2024-06-20 14:15:00 | 420.70 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-06-21 09:15:00 | 426.50 | 2024-06-25 09:15:00 | 469.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-21 10:45:00 | 424.30 | 2024-06-25 09:15:00 | 466.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-21 11:15:00 | 424.65 | 2024-06-25 09:15:00 | 467.12 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-24 09:45:00 | 437.40 | 2024-06-26 09:15:00 | 481.14 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2024-07-05 09:15:00 | 495.50 | 2024-07-08 10:15:00 | 489.65 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-07-09 13:00:00 | 501.70 | 2024-07-10 09:15:00 | 489.40 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2024-07-09 14:45:00 | 501.30 | 2024-07-10 09:15:00 | 489.40 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2024-07-23 12:15:00 | 461.00 | 2024-07-24 09:15:00 | 487.30 | STOP_HIT | 1.00 | -5.70% |
| SELL | retest2 | 2024-07-23 15:00:00 | 475.20 | 2024-07-24 09:15:00 | 487.30 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2024-07-25 11:30:00 | 490.00 | 2024-08-01 10:15:00 | 494.50 | STOP_HIT | 1.00 | 0.92% |
| BUY | retest2 | 2024-08-09 09:15:00 | 501.60 | 2024-08-13 15:15:00 | 493.00 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2024-08-13 15:15:00 | 493.00 | 2024-08-13 15:15:00 | 493.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2024-09-11 09:45:00 | 522.20 | 2024-09-13 11:15:00 | 527.60 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-10-03 09:30:00 | 513.45 | 2024-10-04 09:15:00 | 487.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 09:30:00 | 513.45 | 2024-10-07 10:15:00 | 462.11 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-11-04 12:15:00 | 453.00 | 2024-11-08 10:15:00 | 454.45 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2024-11-04 13:00:00 | 453.65 | 2024-11-08 10:15:00 | 454.45 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2024-11-04 13:45:00 | 453.00 | 2024-11-08 10:15:00 | 454.45 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2024-11-05 09:15:00 | 461.20 | 2024-11-08 10:15:00 | 454.45 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-11-05 10:45:00 | 465.05 | 2024-11-08 10:15:00 | 454.45 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2024-11-05 12:15:00 | 463.90 | 2024-11-08 10:15:00 | 454.45 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-11-06 13:30:00 | 464.85 | 2024-11-08 10:15:00 | 454.45 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2024-11-07 11:00:00 | 463.95 | 2024-11-08 10:15:00 | 454.45 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest1 | 2024-11-12 13:30:00 | 439.45 | 2024-11-13 09:15:00 | 417.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-11-12 13:30:00 | 439.45 | 2024-11-14 09:15:00 | 425.65 | STOP_HIT | 0.50 | 3.14% |
| BUY | retest2 | 2024-12-02 12:45:00 | 447.55 | 2024-12-05 09:15:00 | 441.90 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-12-02 13:30:00 | 448.20 | 2024-12-05 09:15:00 | 441.90 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-12-03 09:15:00 | 452.45 | 2024-12-05 09:15:00 | 441.90 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2024-12-04 10:00:00 | 448.25 | 2024-12-05 09:15:00 | 441.90 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2024-12-12 14:30:00 | 509.80 | 2024-12-18 09:15:00 | 506.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-12-13 09:15:00 | 522.05 | 2024-12-18 09:15:00 | 506.00 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2024-12-17 10:00:00 | 509.85 | 2024-12-18 09:15:00 | 506.00 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-12-17 10:30:00 | 509.60 | 2024-12-18 09:15:00 | 506.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-12-20 12:15:00 | 499.20 | 2024-12-26 10:15:00 | 474.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 12:15:00 | 499.20 | 2024-12-26 10:15:00 | 481.80 | STOP_HIT | 0.50 | 3.49% |
| SELL | retest2 | 2025-01-09 12:45:00 | 460.70 | 2025-01-13 09:15:00 | 437.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 12:45:00 | 460.70 | 2025-01-14 09:15:00 | 434.80 | STOP_HIT | 0.50 | 5.62% |
| SELL | retest2 | 2025-01-28 10:15:00 | 389.25 | 2025-01-29 15:15:00 | 420.00 | STOP_HIT | 1.00 | -7.90% |
| SELL | retest2 | 2025-01-28 10:45:00 | 390.50 | 2025-01-29 15:15:00 | 420.00 | STOP_HIT | 1.00 | -7.55% |
| BUY | retest2 | 2025-02-03 11:00:00 | 483.05 | 2025-02-11 09:15:00 | 477.90 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-02-03 12:45:00 | 481.50 | 2025-02-11 09:15:00 | 477.90 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-02-04 12:00:00 | 482.50 | 2025-02-11 09:15:00 | 477.90 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-02-04 12:30:00 | 481.60 | 2025-02-11 09:15:00 | 477.90 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-02-05 11:15:00 | 490.00 | 2025-02-11 09:15:00 | 477.90 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-02-05 13:00:00 | 491.00 | 2025-02-11 09:15:00 | 477.90 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2025-02-05 14:15:00 | 491.10 | 2025-02-11 09:15:00 | 477.90 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2025-02-06 09:30:00 | 490.60 | 2025-02-11 09:15:00 | 477.90 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2025-02-07 11:45:00 | 489.00 | 2025-02-11 09:15:00 | 477.90 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2025-02-07 12:15:00 | 500.85 | 2025-02-11 09:15:00 | 477.90 | STOP_HIT | 1.00 | -4.58% |
| BUY | retest2 | 2025-02-10 11:45:00 | 488.45 | 2025-02-11 09:15:00 | 477.90 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-02-10 13:15:00 | 491.30 | 2025-02-11 09:15:00 | 477.90 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2025-03-12 13:30:00 | 527.60 | 2025-03-17 12:15:00 | 510.40 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2025-03-13 10:15:00 | 524.05 | 2025-03-17 12:15:00 | 510.40 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-04-08 10:30:00 | 524.50 | 2025-04-15 09:15:00 | 552.80 | STOP_HIT | 1.00 | -5.40% |
| SELL | retest2 | 2025-04-08 11:30:00 | 526.20 | 2025-04-15 09:15:00 | 552.80 | STOP_HIT | 1.00 | -5.06% |
| SELL | retest2 | 2025-04-08 13:45:00 | 527.65 | 2025-04-15 09:15:00 | 552.80 | STOP_HIT | 1.00 | -4.77% |
| SELL | retest2 | 2025-04-08 15:00:00 | 527.05 | 2025-04-15 09:15:00 | 552.80 | STOP_HIT | 1.00 | -4.89% |
| SELL | retest2 | 2025-04-11 10:15:00 | 522.15 | 2025-04-15 09:15:00 | 552.80 | STOP_HIT | 1.00 | -5.87% |
| SELL | retest2 | 2025-04-11 11:45:00 | 523.50 | 2025-04-15 09:15:00 | 552.80 | STOP_HIT | 1.00 | -5.60% |
| BUY | retest2 | 2025-04-21 10:15:00 | 577.50 | 2025-04-24 14:15:00 | 566.60 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-04-23 10:15:00 | 572.15 | 2025-04-24 14:15:00 | 566.60 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-04-23 11:00:00 | 573.65 | 2025-04-24 14:15:00 | 566.60 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-04-24 14:15:00 | 572.50 | 2025-04-24 14:15:00 | 566.60 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-04-28 11:15:00 | 558.05 | 2025-05-02 14:15:00 | 530.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-29 10:00:00 | 558.00 | 2025-05-02 14:15:00 | 530.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-28 11:15:00 | 558.05 | 2025-05-05 09:15:00 | 543.85 | STOP_HIT | 0.50 | 2.54% |
| SELL | retest2 | 2025-04-29 10:00:00 | 558.00 | 2025-05-05 09:15:00 | 543.85 | STOP_HIT | 0.50 | 2.54% |
| BUY | retest2 | 2025-05-07 10:15:00 | 560.00 | 2025-05-12 11:15:00 | 611.44 | TARGET_HIT | 1.00 | 9.18% |
| BUY | retest2 | 2025-05-09 10:00:00 | 555.85 | 2025-05-12 12:15:00 | 616.00 | TARGET_HIT | 1.00 | 10.82% |
| BUY | retest2 | 2025-05-29 09:15:00 | 652.00 | 2025-05-29 13:15:00 | 642.10 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-06-10 12:30:00 | 639.10 | 2025-06-10 14:15:00 | 644.45 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-06-18 12:15:00 | 608.55 | 2025-06-24 12:15:00 | 603.85 | STOP_HIT | 1.00 | 0.77% |
| SELL | retest2 | 2025-06-19 09:45:00 | 604.95 | 2025-06-24 12:15:00 | 603.85 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest1 | 2025-07-04 09:45:00 | 953.60 | 2025-07-07 10:15:00 | 917.20 | STOP_HIT | 1.00 | -3.82% |
| BUY | retest1 | 2025-07-04 15:15:00 | 955.00 | 2025-07-07 10:15:00 | 917.20 | STOP_HIT | 1.00 | -3.96% |
| BUY | retest2 | 2025-07-07 12:15:00 | 930.90 | 2025-07-09 09:15:00 | 1023.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-08 09:30:00 | 934.65 | 2025-07-09 09:15:00 | 1028.12 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-22 12:45:00 | 1018.10 | 2025-07-29 09:15:00 | 967.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 12:45:00 | 1018.10 | 2025-07-29 12:15:00 | 990.70 | STOP_HIT | 0.50 | 2.69% |
| BUY | retest2 | 2025-08-01 09:15:00 | 1041.00 | 2025-08-01 15:15:00 | 999.80 | STOP_HIT | 1.00 | -3.96% |
| BUY | retest2 | 2025-08-01 13:30:00 | 1043.00 | 2025-08-01 15:15:00 | 999.80 | STOP_HIT | 1.00 | -4.14% |
| SELL | retest2 | 2025-08-07 13:15:00 | 992.50 | 2025-08-11 09:15:00 | 1020.40 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-08-08 14:45:00 | 999.00 | 2025-08-11 09:15:00 | 1020.40 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-08-18 09:15:00 | 1146.70 | 2025-08-25 10:15:00 | 1141.60 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-09-04 09:15:00 | 1241.90 | 2025-09-05 10:15:00 | 1211.20 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-09-04 15:15:00 | 1226.90 | 2025-09-05 10:15:00 | 1211.20 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-09-05 12:30:00 | 1229.10 | 2025-09-05 13:15:00 | 1217.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-09-08 12:00:00 | 1215.00 | 2025-09-09 09:15:00 | 1265.90 | STOP_HIT | 1.00 | -4.19% |
| BUY | retest1 | 2025-09-17 13:15:00 | 1321.80 | 2025-09-18 09:15:00 | 1271.10 | STOP_HIT | 1.00 | -3.84% |
| BUY | retest2 | 2025-09-23 09:15:00 | 1301.50 | 2025-09-24 13:15:00 | 1278.60 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-09-23 15:15:00 | 1290.00 | 2025-09-24 13:15:00 | 1278.60 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-09-24 09:30:00 | 1293.90 | 2025-09-24 13:15:00 | 1278.60 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-10-27 15:15:00 | 1237.80 | 2025-10-28 10:15:00 | 1263.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-11-25 09:15:00 | 1032.00 | 2025-11-26 11:15:00 | 1055.50 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-11-25 14:00:00 | 1031.90 | 2025-11-26 11:15:00 | 1055.50 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2025-12-15 09:15:00 | 948.90 | 2025-12-15 10:15:00 | 963.10 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-12-23 14:45:00 | 1051.40 | 2025-12-26 10:15:00 | 1032.30 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-12-24 13:45:00 | 1045.20 | 2025-12-26 10:15:00 | 1032.30 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-12-26 10:15:00 | 1042.40 | 2025-12-26 10:15:00 | 1032.30 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-12-30 13:15:00 | 998.60 | 2026-01-01 09:15:00 | 1037.90 | STOP_HIT | 1.00 | -3.94% |
| SELL | retest2 | 2025-12-31 10:00:00 | 993.00 | 2026-01-01 09:15:00 | 1037.90 | STOP_HIT | 1.00 | -4.52% |
| SELL | retest2 | 2025-12-31 11:15:00 | 996.00 | 2026-01-01 09:15:00 | 1037.90 | STOP_HIT | 1.00 | -4.21% |
| SELL | retest2 | 2025-12-31 12:00:00 | 998.70 | 2026-01-01 09:15:00 | 1037.90 | STOP_HIT | 1.00 | -3.93% |
| SELL | retest2 | 2026-01-20 09:15:00 | 914.80 | 2026-01-21 09:15:00 | 869.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 09:15:00 | 914.80 | 2026-01-21 15:15:00 | 899.90 | STOP_HIT | 0.50 | 1.63% |
| SELL | retest2 | 2026-01-28 13:30:00 | 883.00 | 2026-01-28 14:15:00 | 896.90 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2026-01-29 09:30:00 | 882.10 | 2026-01-30 09:15:00 | 941.70 | STOP_HIT | 1.00 | -6.76% |
| BUY | retest2 | 2026-02-10 09:30:00 | 991.45 | 2026-02-13 12:15:00 | 1003.05 | STOP_HIT | 1.00 | 1.17% |
| SELL | retest2 | 2026-02-19 12:30:00 | 969.00 | 2026-02-23 13:15:00 | 970.00 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2026-02-27 11:30:00 | 1017.00 | 2026-03-02 10:15:00 | 977.85 | STOP_HIT | 1.00 | -3.85% |
| SELL | retest2 | 2026-03-06 14:30:00 | 919.80 | 2026-03-09 09:15:00 | 873.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 14:30:00 | 919.80 | 2026-03-10 10:15:00 | 887.50 | STOP_HIT | 0.50 | 3.51% |
| BUY | retest2 | 2026-04-02 11:30:00 | 858.95 | 2026-04-09 09:15:00 | 944.85 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-30 13:30:00 | 1029.75 | 2026-05-08 09:15:00 | 1132.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-05-04 09:15:00 | 1053.40 | 2026-05-08 09:15:00 | 1131.90 | TARGET_HIT | 1.00 | 7.45% |
| BUY | retest2 | 2026-05-04 12:15:00 | 1029.00 | 2026-05-08 09:15:00 | 1134.65 | TARGET_HIT | 1.00 | 10.27% |
