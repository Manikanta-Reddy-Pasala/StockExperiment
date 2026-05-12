# Manappuram Finance Ltd. (MANAPPURAM)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 315.05
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 141 |
| ALERT1 | 104 |
| ALERT2 | 103 |
| ALERT2_SKIP | 55 |
| ALERT3 | 294 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 105 |
| PARTIAL | 8 |
| TARGET_HIT | 12 |
| STOP_HIT | 99 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 119 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 45 / 74
- **Target hits / Stop hits / Partials:** 12 / 99 / 8
- **Avg / median % per leg:** 0.71% / -0.76%
- **Sum % (uncompounded):** 85.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 64 | 25 | 39.1% | 7 | 57 | 0 | 0.55% | 35.2% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 4 | 0 | -0.50% | -2.0% |
| BUY @ 3rd Alert (retest2) | 60 | 23 | 38.3% | 7 | 53 | 0 | 0.62% | 37.2% |
| SELL (all) | 55 | 20 | 36.4% | 5 | 42 | 8 | 0.91% | 49.9% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 1 | 0 | 5.23% | 10.5% |
| SELL @ 3rd Alert (retest2) | 53 | 18 | 34.0% | 4 | 41 | 8 | 0.74% | 39.4% |
| retest1 (combined) | 6 | 4 | 66.7% | 1 | 5 | 0 | 1.41% | 8.5% |
| retest2 (combined) | 113 | 41 | 36.3% | 11 | 94 | 8 | 0.68% | 76.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 14:15:00 | 175.90 | 174.96 | 174.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 09:15:00 | 179.45 | 176.01 | 175.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 14:15:00 | 183.65 | 183.99 | 182.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-17 14:30:00 | 182.65 | 183.99 | 182.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 09:15:00 | 183.50 | 183.89 | 182.58 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2024-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 12:15:00 | 181.15 | 182.17 | 182.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 13:15:00 | 180.35 | 181.81 | 182.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 14:15:00 | 178.60 | 178.40 | 179.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-23 15:00:00 | 178.60 | 178.40 | 179.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 15:15:00 | 180.00 | 178.72 | 179.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:15:00 | 180.25 | 178.72 | 179.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 179.65 | 178.91 | 179.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:30:00 | 180.35 | 178.91 | 179.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 177.90 | 178.71 | 179.26 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-05-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 15:15:00 | 181.25 | 179.65 | 179.52 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 09:15:00 | 175.90 | 178.90 | 179.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 10:15:00 | 174.50 | 178.02 | 178.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 13:15:00 | 177.30 | 177.02 | 178.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-27 14:00:00 | 177.30 | 177.02 | 178.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 173.90 | 170.22 | 170.72 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 11:15:00 | 173.55 | 171.39 | 171.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 12:15:00 | 174.75 | 172.06 | 171.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 167.75 | 171.81 | 171.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 167.75 | 171.81 | 171.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 167.75 | 171.81 | 171.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 166.20 | 171.81 | 171.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 161.00 | 169.65 | 170.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 157.25 | 167.17 | 169.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 09:15:00 | 164.05 | 163.73 | 166.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 10:00:00 | 164.05 | 163.73 | 166.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 168.45 | 164.67 | 166.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 168.45 | 164.67 | 166.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 168.90 | 165.52 | 166.95 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 173.90 | 168.53 | 167.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 10:15:00 | 175.65 | 169.95 | 168.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 177.85 | 178.24 | 176.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 14:45:00 | 177.67 | 178.24 | 176.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 181.83 | 178.83 | 177.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 10:30:00 | 182.57 | 179.63 | 177.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 14:15:00 | 182.77 | 180.99 | 178.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 182.43 | 181.25 | 179.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 10:15:00 | 182.49 | 181.34 | 179.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 185.30 | 185.59 | 184.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 09:45:00 | 185.00 | 185.59 | 184.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 189.41 | 186.36 | 185.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 13:15:00 | 190.76 | 187.70 | 186.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 12:00:00 | 191.31 | 189.73 | 188.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 09:15:00 | 192.26 | 190.45 | 188.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 10:00:00 | 191.10 | 191.66 | 191.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 191.45 | 192.59 | 192.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 10:00:00 | 191.45 | 192.59 | 192.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 191.10 | 192.29 | 191.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 11:00:00 | 191.10 | 192.29 | 191.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 11:15:00 | 190.41 | 191.92 | 191.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 12:00:00 | 190.41 | 191.92 | 191.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-25 12:15:00 | 191.30 | 191.79 | 191.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 12:15:00 | 191.30 | 191.79 | 191.80 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 13:15:00 | 192.90 | 192.01 | 191.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 09:15:00 | 196.25 | 193.12 | 192.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 14:15:00 | 207.85 | 208.82 | 205.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-28 14:30:00 | 207.10 | 208.82 | 205.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 208.35 | 210.74 | 208.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:45:00 | 208.13 | 210.74 | 208.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 208.27 | 210.25 | 208.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:30:00 | 207.08 | 210.25 | 208.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 14:15:00 | 206.11 | 207.81 | 207.85 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 10:15:00 | 209.16 | 207.93 | 207.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 12:15:00 | 210.70 | 209.46 | 208.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 15:15:00 | 209.40 | 209.66 | 209.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 15:15:00 | 209.40 | 209.66 | 209.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 15:15:00 | 209.40 | 209.66 | 209.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:15:00 | 208.58 | 209.66 | 209.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 207.50 | 209.23 | 209.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 10:00:00 | 207.50 | 209.23 | 209.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 207.67 | 208.92 | 208.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:00:00 | 207.67 | 208.92 | 208.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 11:15:00 | 207.13 | 208.56 | 208.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 14:15:00 | 206.74 | 207.80 | 208.31 | Break + close below crossover candle low |

### Cycle 13 — BUY (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 09:15:00 | 212.20 | 207.76 | 207.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 11:15:00 | 217.00 | 213.93 | 212.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 14:15:00 | 222.08 | 222.09 | 219.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 09:15:00 | 220.48 | 221.75 | 219.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 220.48 | 221.75 | 219.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 11:00:00 | 223.15 | 222.03 | 219.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 13:00:00 | 223.09 | 222.33 | 220.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 09:15:00 | 224.83 | 221.61 | 220.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 09:45:00 | 223.83 | 222.57 | 220.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 222.80 | 225.65 | 223.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 222.80 | 225.65 | 223.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 220.62 | 224.65 | 223.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:00:00 | 220.62 | 224.65 | 223.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 11:15:00 | 222.72 | 224.26 | 223.41 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-19 13:15:00 | 219.04 | 222.65 | 222.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 13:15:00 | 219.04 | 222.65 | 222.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 14:15:00 | 216.10 | 221.34 | 222.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 09:15:00 | 209.50 | 208.36 | 212.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-24 09:45:00 | 211.00 | 208.36 | 212.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 13:15:00 | 212.78 | 209.20 | 211.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 14:00:00 | 212.78 | 209.20 | 211.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 14:15:00 | 209.56 | 209.28 | 211.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 09:15:00 | 207.50 | 209.56 | 211.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 11:30:00 | 209.20 | 207.84 | 208.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 14:15:00 | 213.15 | 209.64 | 209.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 14:15:00 | 213.15 | 209.64 | 209.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 15:15:00 | 213.75 | 210.47 | 209.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 14:15:00 | 213.29 | 213.36 | 211.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 09:15:00 | 213.48 | 213.33 | 212.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 213.48 | 213.33 | 212.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 10:15:00 | 215.69 | 213.33 | 212.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 12:30:00 | 215.77 | 213.88 | 212.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 14:00:00 | 215.35 | 214.17 | 212.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:30:00 | 216.87 | 214.38 | 213.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 215.74 | 214.65 | 213.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:45:00 | 214.45 | 214.65 | 213.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 213.57 | 214.33 | 213.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:45:00 | 212.98 | 214.33 | 213.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 13:15:00 | 212.50 | 213.97 | 213.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 14:00:00 | 212.50 | 213.97 | 213.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 213.20 | 213.81 | 213.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 14:30:00 | 212.98 | 213.81 | 213.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-02 09:15:00 | 211.45 | 213.41 | 213.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 211.45 | 213.41 | 213.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 205.46 | 210.21 | 211.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 203.56 | 203.09 | 206.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 203.56 | 203.09 | 206.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 203.56 | 203.09 | 206.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 10:30:00 | 202.34 | 203.09 | 206.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 11:00:00 | 203.08 | 203.09 | 206.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 12:00:00 | 203.23 | 203.11 | 205.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 11:15:00 | 205.44 | 201.60 | 201.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 11:15:00 | 205.44 | 201.60 | 201.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 09:15:00 | 209.00 | 204.58 | 203.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 14:15:00 | 206.88 | 208.81 | 207.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 14:15:00 | 206.88 | 208.81 | 207.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 206.88 | 208.81 | 207.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 15:00:00 | 206.88 | 208.81 | 207.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 205.00 | 208.05 | 207.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:15:00 | 203.00 | 208.05 | 207.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 200.65 | 206.57 | 206.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 10:00:00 | 200.65 | 206.57 | 206.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 10:15:00 | 201.14 | 205.48 | 206.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 11:15:00 | 199.89 | 204.36 | 205.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 13:15:00 | 201.85 | 200.42 | 202.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 14:00:00 | 201.85 | 200.42 | 202.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 201.84 | 200.70 | 202.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 15:15:00 | 201.00 | 200.70 | 202.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 09:15:00 | 200.77 | 202.09 | 202.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-20 10:15:00 | 205.64 | 203.00 | 202.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 10:15:00 | 205.64 | 203.00 | 202.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 11:15:00 | 208.70 | 204.14 | 203.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 215.06 | 215.68 | 212.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 10:00:00 | 215.06 | 215.68 | 212.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 215.20 | 215.85 | 214.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:30:00 | 213.74 | 215.85 | 214.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 214.32 | 215.46 | 214.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:30:00 | 213.11 | 215.46 | 214.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 216.35 | 215.64 | 215.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:30:00 | 215.20 | 215.64 | 215.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 214.80 | 216.67 | 215.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:30:00 | 214.75 | 216.67 | 215.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 216.79 | 216.70 | 216.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 10:30:00 | 215.15 | 216.70 | 216.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 11:15:00 | 216.73 | 216.70 | 216.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 11:30:00 | 216.10 | 216.70 | 216.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 12:15:00 | 216.30 | 216.62 | 216.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 13:00:00 | 216.30 | 216.62 | 216.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 13:15:00 | 216.29 | 216.56 | 216.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 14:00:00 | 216.29 | 216.56 | 216.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 214.73 | 216.19 | 216.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 14:45:00 | 214.72 | 216.19 | 216.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 15:15:00 | 215.02 | 215.96 | 215.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:15:00 | 215.25 | 215.96 | 215.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 213.74 | 215.59 | 215.76 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 09:15:00 | 216.58 | 215.68 | 215.58 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 09:15:00 | 214.50 | 215.72 | 215.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 11:15:00 | 213.29 | 215.06 | 215.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-02 13:15:00 | 214.63 | 214.60 | 215.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 13:15:00 | 214.63 | 214.60 | 215.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 13:15:00 | 214.63 | 214.60 | 215.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 14:00:00 | 214.63 | 214.60 | 215.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 212.54 | 214.19 | 214.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 10:15:00 | 211.50 | 213.53 | 214.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 13:15:00 | 200.92 | 203.53 | 204.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-12 09:15:00 | 204.79 | 203.26 | 204.23 | SL hit (close>ema200) qty=0.50 sl=203.26 alert=retest2 |

### Cycle 23 — BUY (started 2024-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 14:15:00 | 205.84 | 204.54 | 204.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 214.41 | 206.69 | 205.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 11:15:00 | 210.65 | 211.12 | 209.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 11:45:00 | 211.17 | 211.12 | 209.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 210.42 | 211.92 | 210.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:30:00 | 210.45 | 211.92 | 210.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 210.55 | 211.65 | 210.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 11:00:00 | 210.55 | 211.65 | 210.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 11:15:00 | 211.47 | 211.61 | 210.52 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2024-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 14:15:00 | 210.33 | 210.61 | 210.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 207.87 | 209.96 | 210.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 206.92 | 206.87 | 208.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 15:00:00 | 206.92 | 206.87 | 208.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 206.58 | 206.34 | 207.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:45:00 | 207.28 | 206.34 | 207.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 207.03 | 206.47 | 207.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 11:00:00 | 207.03 | 206.47 | 207.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 11:15:00 | 207.13 | 206.61 | 207.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 12:00:00 | 207.13 | 206.61 | 207.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 12:15:00 | 208.26 | 206.94 | 207.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 12:30:00 | 208.33 | 206.94 | 207.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 13:15:00 | 208.31 | 207.21 | 207.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 13:30:00 | 208.20 | 207.21 | 207.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 14:15:00 | 208.59 | 207.49 | 207.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 09:15:00 | 210.50 | 208.27 | 207.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 11:15:00 | 207.52 | 208.19 | 207.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 11:15:00 | 207.52 | 208.19 | 207.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 207.52 | 208.19 | 207.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 12:00:00 | 207.52 | 208.19 | 207.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 12:15:00 | 207.58 | 208.07 | 207.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 13:15:00 | 207.26 | 208.07 | 207.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 13:15:00 | 206.82 | 207.82 | 207.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 13:45:00 | 206.70 | 207.82 | 207.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2024-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 14:15:00 | 204.95 | 207.24 | 207.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 15:15:00 | 204.50 | 206.70 | 207.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 13:15:00 | 202.27 | 201.64 | 203.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-26 14:00:00 | 202.27 | 201.64 | 203.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 203.70 | 202.05 | 203.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 14:30:00 | 204.03 | 202.05 | 203.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 203.21 | 202.29 | 203.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 203.50 | 202.29 | 203.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 205.20 | 202.87 | 203.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:00:00 | 205.20 | 202.87 | 203.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 204.90 | 203.27 | 203.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:45:00 | 205.15 | 203.27 | 203.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 12:15:00 | 202.90 | 203.22 | 203.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 15:00:00 | 202.52 | 203.08 | 203.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 09:15:00 | 202.53 | 203.08 | 203.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 12:15:00 | 202.44 | 202.71 | 203.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 13:00:00 | 202.35 | 202.64 | 203.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 201.50 | 202.41 | 202.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 14:00:00 | 201.50 | 202.41 | 202.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 09:15:00 | 192.39 | 197.05 | 199.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 09:15:00 | 192.40 | 197.05 | 199.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 09:15:00 | 192.32 | 197.05 | 199.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 09:15:00 | 192.23 | 197.05 | 199.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-04 09:15:00 | 193.48 | 193.26 | 195.83 | SL hit (close>ema200) qty=0.50 sl=193.26 alert=retest2 |

### Cycle 27 — BUY (started 2024-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 13:15:00 | 192.27 | 188.21 | 187.67 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 11:15:00 | 186.01 | 188.12 | 188.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 09:15:00 | 184.05 | 186.34 | 187.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 15:15:00 | 183.59 | 183.35 | 184.59 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 10:00:00 | 181.24 | 182.93 | 184.28 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 179.62 | 180.68 | 182.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 11:15:00 | 178.07 | 180.28 | 181.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 13:45:00 | 177.98 | 179.28 | 181.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-10-18 09:15:00 | 163.12 | 172.73 | 177.55 | Target hit (10%) qty=1.00 alert=retest1 |

### Cycle 29 — BUY (started 2024-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 13:15:00 | 148.09 | 147.32 | 147.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 14:15:00 | 149.69 | 147.79 | 147.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 09:15:00 | 156.46 | 158.47 | 155.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 09:15:00 | 156.46 | 158.47 | 155.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 156.46 | 158.47 | 155.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:00:00 | 159.90 | 157.44 | 156.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 10:15:00 | 153.19 | 156.42 | 156.04 | SL hit (close<static) qty=1.00 sl=153.25 alert=retest2 |

### Cycle 30 — SELL (started 2024-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 11:15:00 | 152.22 | 155.58 | 155.69 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-11-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 13:15:00 | 159.21 | 155.57 | 155.28 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 10:15:00 | 153.90 | 155.99 | 156.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 10:15:00 | 151.94 | 153.86 | 154.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 152.84 | 151.54 | 152.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 10:15:00 | 152.84 | 151.54 | 152.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 152.84 | 151.54 | 152.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:00:00 | 152.84 | 151.54 | 152.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 153.68 | 151.97 | 152.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:30:00 | 153.01 | 151.97 | 152.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 153.41 | 152.26 | 153.01 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2024-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 15:15:00 | 155.00 | 153.65 | 153.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-12 09:15:00 | 157.62 | 154.44 | 153.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 14:15:00 | 157.57 | 159.32 | 157.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-13 14:15:00 | 157.57 | 159.32 | 157.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 14:15:00 | 157.57 | 159.32 | 157.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 14:45:00 | 156.10 | 159.32 | 157.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 15:15:00 | 160.20 | 159.50 | 157.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-14 09:30:00 | 158.10 | 159.26 | 157.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 156.58 | 158.72 | 157.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:45:00 | 156.62 | 158.72 | 157.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 11:15:00 | 157.10 | 158.40 | 157.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-14 11:45:00 | 156.39 | 158.40 | 157.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 14:15:00 | 155.90 | 157.29 | 157.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 09:15:00 | 152.76 | 156.19 | 156.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 158.40 | 155.45 | 155.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 09:15:00 | 158.40 | 155.45 | 155.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 158.40 | 155.45 | 155.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 158.40 | 155.45 | 155.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 158.16 | 155.99 | 156.12 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 157.29 | 156.25 | 156.23 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 154.65 | 156.04 | 156.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 15:15:00 | 153.89 | 155.61 | 155.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 12:15:00 | 151.11 | 151.04 | 152.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 13:00:00 | 151.11 | 151.04 | 152.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 152.28 | 151.56 | 152.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:30:00 | 153.60 | 151.56 | 152.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 10:15:00 | 153.78 | 152.01 | 152.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 10:45:00 | 153.35 | 152.01 | 152.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 11:15:00 | 152.20 | 152.05 | 152.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 12:15:00 | 151.40 | 152.05 | 152.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-26 14:15:00 | 155.87 | 152.42 | 152.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 14:15:00 | 155.87 | 152.42 | 152.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 15:15:00 | 156.00 | 153.14 | 152.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 12:15:00 | 154.78 | 155.13 | 154.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 12:30:00 | 154.99 | 155.13 | 154.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 155.43 | 155.19 | 154.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 13:45:00 | 155.24 | 155.19 | 154.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 154.96 | 155.68 | 155.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:45:00 | 154.76 | 155.68 | 155.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 155.05 | 155.55 | 155.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 12:45:00 | 155.05 | 155.55 | 155.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 155.75 | 155.59 | 155.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 14:15:00 | 156.31 | 155.59 | 155.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 09:30:00 | 156.09 | 155.93 | 155.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-09 11:15:00 | 171.94 | 168.70 | 167.42 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 11:15:00 | 178.91 | 181.42 | 181.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 13:15:00 | 177.68 | 180.23 | 180.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 11:15:00 | 182.72 | 179.57 | 180.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 11:15:00 | 182.72 | 179.57 | 180.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 11:15:00 | 182.72 | 179.57 | 180.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 11:30:00 | 180.65 | 179.57 | 180.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 12:15:00 | 181.75 | 180.00 | 180.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 13:15:00 | 182.50 | 180.00 | 180.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2024-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 14:15:00 | 183.84 | 181.09 | 180.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-20 10:15:00 | 185.57 | 182.73 | 181.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 12:15:00 | 182.70 | 183.06 | 182.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 12:15:00 | 182.70 | 183.06 | 182.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 182.70 | 183.06 | 182.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 13:00:00 | 182.70 | 183.06 | 182.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 13:15:00 | 184.36 | 183.32 | 182.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 13:45:00 | 183.72 | 183.32 | 182.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 180.43 | 182.74 | 182.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 180.43 | 182.74 | 182.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 180.99 | 182.39 | 181.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 09:15:00 | 184.79 | 182.39 | 181.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-26 09:15:00 | 178.84 | 184.22 | 184.01 | SL hit (close<static) qty=1.00 sl=180.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 09:15:00 | 181.72 | 183.85 | 183.95 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 09:15:00 | 187.20 | 184.06 | 183.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 10:15:00 | 189.49 | 185.14 | 184.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 09:15:00 | 186.71 | 187.23 | 185.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 09:15:00 | 186.71 | 187.23 | 185.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 186.71 | 187.23 | 185.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 13:45:00 | 189.55 | 187.55 | 186.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 09:15:00 | 189.20 | 187.80 | 186.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 11:00:00 | 189.67 | 188.30 | 187.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 14:15:00 | 189.18 | 190.08 | 189.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-03 14:15:00 | 188.10 | 189.69 | 189.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-01-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 14:15:00 | 188.10 | 189.69 | 189.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 15:15:00 | 187.30 | 189.21 | 189.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 11:15:00 | 183.00 | 182.08 | 184.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 12:00:00 | 183.00 | 182.08 | 184.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 183.65 | 182.52 | 184.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:45:00 | 183.20 | 182.52 | 184.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 182.22 | 182.46 | 183.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 181.36 | 182.40 | 183.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 09:15:00 | 188.13 | 181.82 | 182.36 | SL hit (close>static) qty=1.00 sl=184.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 11:15:00 | 184.00 | 182.89 | 182.79 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 13:15:00 | 182.60 | 182.71 | 182.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 14:15:00 | 178.70 | 181.90 | 182.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 10:15:00 | 181.84 | 180.71 | 181.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 10:15:00 | 181.84 | 180.71 | 181.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 181.84 | 180.71 | 181.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:45:00 | 182.24 | 180.71 | 181.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 11:15:00 | 182.82 | 181.14 | 181.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 12:00:00 | 182.82 | 181.14 | 181.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 12:15:00 | 183.20 | 181.55 | 181.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 12:30:00 | 183.73 | 181.55 | 181.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 178.40 | 176.84 | 177.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 15:00:00 | 178.40 | 176.84 | 177.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 181.00 | 177.67 | 178.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:15:00 | 177.48 | 177.67 | 178.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 179.90 | 177.89 | 178.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 11:00:00 | 179.90 | 177.89 | 178.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 11:15:00 | 179.52 | 178.22 | 178.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 11:30:00 | 180.90 | 178.22 | 178.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-01-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 12:15:00 | 179.34 | 178.44 | 178.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 182.44 | 179.61 | 178.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 14:15:00 | 194.34 | 194.42 | 191.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 14:45:00 | 194.17 | 194.42 | 191.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 192.00 | 193.94 | 191.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 15:00:00 | 195.60 | 193.19 | 192.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-27 10:15:00 | 188.00 | 194.18 | 194.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 10:15:00 | 188.00 | 194.18 | 194.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 14:15:00 | 185.93 | 190.68 | 192.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 189.24 | 188.85 | 191.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:00:00 | 189.24 | 188.85 | 191.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 189.46 | 189.19 | 190.85 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 10:15:00 | 198.90 | 192.96 | 192.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 11:15:00 | 200.06 | 194.38 | 192.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 11:15:00 | 198.54 | 198.68 | 196.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 11:30:00 | 197.95 | 198.68 | 196.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 197.56 | 198.75 | 197.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 09:30:00 | 197.08 | 198.75 | 197.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 197.48 | 198.50 | 197.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 10:30:00 | 197.50 | 198.50 | 197.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 11:15:00 | 196.91 | 198.18 | 197.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 11:45:00 | 196.20 | 198.18 | 197.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 12:15:00 | 196.95 | 197.93 | 197.24 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 10:15:00 | 195.49 | 196.83 | 196.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-01 11:15:00 | 193.77 | 196.22 | 196.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 12:15:00 | 197.20 | 196.42 | 196.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 197.20 | 196.42 | 196.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 197.20 | 196.42 | 196.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:45:00 | 197.60 | 196.42 | 196.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 196.98 | 196.53 | 196.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:45:00 | 198.92 | 196.53 | 196.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 14:15:00 | 197.82 | 196.79 | 196.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 14:30:00 | 199.15 | 196.79 | 196.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 15:15:00 | 197.32 | 196.89 | 196.84 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 194.60 | 196.44 | 196.63 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 09:15:00 | 202.60 | 197.39 | 196.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 13:15:00 | 204.98 | 201.06 | 199.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 15:15:00 | 204.90 | 205.03 | 202.85 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-06 09:45:00 | 206.20 | 205.13 | 203.09 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-06 11:15:00 | 206.03 | 205.26 | 203.33 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 203.60 | 204.89 | 203.64 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-06 13:15:00 | 203.60 | 204.89 | 203.64 | SL hit (close<ema400) qty=1.00 sl=203.64 alert=retest1 |

### Cycle 52 — SELL (started 2025-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 15:15:00 | 203.80 | 204.62 | 204.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 199.29 | 203.55 | 204.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 197.88 | 193.99 | 196.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 197.88 | 193.99 | 196.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 197.88 | 193.99 | 196.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 197.88 | 193.99 | 196.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 197.89 | 194.77 | 196.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 14:00:00 | 196.51 | 196.31 | 196.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 09:15:00 | 186.68 | 192.83 | 195.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-14 10:15:00 | 176.86 | 189.19 | 193.29 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 53 — BUY (started 2025-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 15:15:00 | 193.22 | 189.51 | 189.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 11:15:00 | 196.90 | 191.70 | 190.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 11:15:00 | 197.22 | 198.13 | 195.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-20 11:15:00 | 197.22 | 198.13 | 195.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 11:15:00 | 197.22 | 198.13 | 195.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 11:45:00 | 196.01 | 198.13 | 195.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 201.12 | 202.10 | 199.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 09:15:00 | 206.00 | 202.10 | 199.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 09:15:00 | 205.51 | 204.20 | 202.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-28 09:15:00 | 197.45 | 204.94 | 204.79 | SL hit (close<static) qty=1.00 sl=198.99 alert=retest2 |

### Cycle 54 — SELL (started 2025-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 10:15:00 | 200.83 | 204.11 | 204.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-03 09:15:00 | 195.55 | 200.01 | 201.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 201.37 | 199.29 | 201.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 12:15:00 | 201.37 | 199.29 | 201.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 12:15:00 | 201.37 | 199.29 | 201.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:00:00 | 201.37 | 199.29 | 201.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 199.96 | 199.42 | 200.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 15:15:00 | 198.76 | 199.37 | 200.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 10:00:00 | 197.60 | 198.92 | 200.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 13:15:00 | 203.20 | 200.12 | 199.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 13:15:00 | 203.20 | 200.12 | 199.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 13:15:00 | 204.43 | 203.13 | 201.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 11:15:00 | 207.25 | 207.30 | 205.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 12:00:00 | 207.25 | 207.30 | 205.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 206.26 | 207.09 | 205.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 13:45:00 | 205.95 | 207.09 | 205.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 206.34 | 206.94 | 205.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:30:00 | 204.95 | 206.94 | 205.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 205.00 | 206.55 | 205.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 202.86 | 206.55 | 205.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 203.98 | 206.04 | 205.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 10:30:00 | 204.61 | 205.83 | 205.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 12:15:00 | 203.29 | 205.02 | 205.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 12:15:00 | 203.29 | 205.02 | 205.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 14:15:00 | 202.10 | 204.19 | 204.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 13:15:00 | 205.07 | 202.31 | 203.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 13:15:00 | 205.07 | 202.31 | 203.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 13:15:00 | 205.07 | 202.31 | 203.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 13:45:00 | 205.12 | 202.31 | 203.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 206.10 | 203.06 | 203.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 14:30:00 | 205.85 | 203.06 | 203.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 09:15:00 | 205.20 | 204.04 | 203.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 13:15:00 | 207.55 | 205.64 | 204.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-17 09:15:00 | 205.00 | 206.13 | 205.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 205.00 | 206.13 | 205.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 205.00 | 206.13 | 205.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:45:00 | 205.00 | 206.13 | 205.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 205.82 | 206.07 | 205.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:30:00 | 205.03 | 206.07 | 205.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 205.51 | 205.96 | 205.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 12:00:00 | 205.51 | 205.96 | 205.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 12:15:00 | 205.34 | 205.84 | 205.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 12:30:00 | 205.40 | 205.84 | 205.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 13:15:00 | 209.72 | 206.61 | 205.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 15:15:00 | 210.70 | 207.14 | 206.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 11:15:00 | 211.40 | 208.42 | 206.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-21 11:15:00 | 231.77 | 222.33 | 217.24 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 12:15:00 | 232.72 | 233.20 | 233.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 13:15:00 | 231.68 | 232.89 | 233.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 15:15:00 | 233.12 | 232.93 | 233.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 15:15:00 | 233.12 | 232.93 | 233.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 233.12 | 232.93 | 233.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 232.58 | 232.93 | 233.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 230.35 | 232.42 | 232.85 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2025-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 10:15:00 | 234.12 | 232.51 | 232.41 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 231.80 | 233.03 | 233.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 225.20 | 231.17 | 232.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 226.10 | 226.06 | 228.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 227.91 | 226.06 | 228.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 226.40 | 226.12 | 228.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 225.52 | 226.26 | 228.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 11:15:00 | 225.91 | 226.26 | 228.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:45:00 | 226.07 | 227.59 | 228.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 12:30:00 | 225.00 | 225.56 | 227.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 225.77 | 225.34 | 226.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-15 11:00:00 | 224.82 | 225.38 | 225.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-16 10:15:00 | 227.11 | 226.06 | 225.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2025-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 10:15:00 | 227.11 | 226.06 | 225.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 11:15:00 | 227.49 | 226.35 | 226.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 15:15:00 | 226.52 | 226.73 | 226.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 09:15:00 | 225.91 | 226.73 | 226.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 226.74 | 226.73 | 226.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 13:30:00 | 229.40 | 227.46 | 226.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:15:00 | 230.35 | 227.93 | 227.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-24 10:15:00 | 229.10 | 230.94 | 231.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 10:15:00 | 229.10 | 230.94 | 231.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 224.99 | 228.59 | 229.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 229.00 | 227.37 | 228.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 229.00 | 227.37 | 228.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 229.00 | 227.37 | 228.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:00:00 | 229.00 | 227.37 | 228.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 228.95 | 227.69 | 228.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:00:00 | 228.95 | 227.69 | 228.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2025-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 12:15:00 | 232.68 | 229.08 | 228.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 233.16 | 230.80 | 229.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 09:15:00 | 230.85 | 231.60 | 230.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 09:15:00 | 230.85 | 231.60 | 230.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 230.85 | 231.60 | 230.86 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2025-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-05 14:15:00 | 230.10 | 231.09 | 231.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 09:15:00 | 228.10 | 230.35 | 230.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 229.60 | 228.68 | 229.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 10:15:00 | 229.60 | 228.68 | 229.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 229.60 | 228.68 | 229.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:00:00 | 229.60 | 228.68 | 229.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 230.03 | 228.95 | 229.52 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2025-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 15:15:00 | 230.75 | 229.85 | 229.81 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 09:15:00 | 229.52 | 229.78 | 229.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 11:15:00 | 228.85 | 229.54 | 229.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 12:15:00 | 228.50 | 227.99 | 228.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 12:15:00 | 228.50 | 227.99 | 228.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 12:15:00 | 228.50 | 227.99 | 228.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 12:30:00 | 228.94 | 227.99 | 228.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 13:15:00 | 228.44 | 228.08 | 228.60 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 231.04 | 228.82 | 228.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 232.20 | 229.50 | 229.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 11:15:00 | 229.10 | 230.37 | 229.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 11:15:00 | 229.10 | 230.37 | 229.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 11:15:00 | 229.10 | 230.37 | 229.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:00:00 | 229.10 | 230.37 | 229.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 229.80 | 230.26 | 229.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 14:15:00 | 229.95 | 230.17 | 229.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-15 13:15:00 | 229.39 | 230.89 | 230.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 13:15:00 | 229.39 | 230.89 | 230.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-16 10:15:00 | 228.92 | 229.87 | 230.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 228.39 | 228.36 | 228.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 228.39 | 228.36 | 228.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 228.39 | 228.36 | 228.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 15:15:00 | 227.79 | 228.43 | 228.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 12:15:00 | 229.90 | 228.70 | 228.70 | SL hit (close>static) qty=1.00 sl=229.35 alert=retest2 |

### Cycle 69 — BUY (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 13:15:00 | 229.96 | 228.95 | 228.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 09:15:00 | 231.45 | 229.67 | 229.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 15:15:00 | 230.30 | 230.48 | 229.89 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 09:15:00 | 231.26 | 230.48 | 229.89 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 13:45:00 | 231.13 | 230.70 | 230.22 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 230.25 | 230.61 | 230.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 15:00:00 | 230.25 | 230.61 | 230.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 230.39 | 230.56 | 230.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:30:00 | 230.50 | 230.57 | 230.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 231.98 | 232.14 | 231.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:00:00 | 231.98 | 232.14 | 231.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 232.00 | 232.11 | 231.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:30:00 | 232.00 | 232.11 | 231.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 231.71 | 232.02 | 231.75 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-29 13:15:00 | 231.71 | 232.02 | 231.75 | SL hit (close<ema400) qty=1.00 sl=231.75 alert=retest1 |

### Cycle 70 — SELL (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 10:15:00 | 233.85 | 236.00 | 236.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 12:15:00 | 233.18 | 235.05 | 235.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 10:15:00 | 234.39 | 233.90 | 234.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 10:15:00 | 234.39 | 233.90 | 234.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 234.39 | 233.90 | 234.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:45:00 | 234.45 | 233.90 | 234.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 235.19 | 234.15 | 234.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:00:00 | 235.19 | 234.15 | 234.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 12:15:00 | 234.71 | 234.27 | 234.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:30:00 | 236.14 | 234.27 | 234.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 234.71 | 234.35 | 234.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 15:15:00 | 234.35 | 234.43 | 234.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 10:15:00 | 239.00 | 235.50 | 235.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 10:15:00 | 239.00 | 235.50 | 235.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 12:15:00 | 243.54 | 237.56 | 236.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 261.98 | 263.59 | 257.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 09:30:00 | 261.75 | 263.59 | 257.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 272.90 | 270.99 | 267.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 273.81 | 270.99 | 267.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 11:15:00 | 274.29 | 271.24 | 267.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 12:15:00 | 269.61 | 273.68 | 273.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 269.61 | 273.68 | 273.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 09:15:00 | 267.62 | 270.96 | 272.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 268.30 | 264.98 | 266.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 268.30 | 264.98 | 266.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 268.30 | 264.98 | 266.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:00:00 | 268.30 | 264.98 | 266.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 270.35 | 266.06 | 266.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 270.35 | 266.06 | 266.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2025-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 13:15:00 | 270.80 | 267.69 | 267.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 15:15:00 | 272.63 | 269.22 | 268.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 14:15:00 | 270.01 | 270.81 | 269.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-23 14:45:00 | 270.02 | 270.81 | 269.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 270.09 | 270.74 | 269.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:30:00 | 270.78 | 270.74 | 269.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 269.61 | 270.40 | 269.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 12:00:00 | 269.61 | 270.40 | 269.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 12:15:00 | 269.69 | 270.26 | 269.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 12:30:00 | 269.32 | 270.26 | 269.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 269.91 | 270.19 | 269.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:30:00 | 269.45 | 270.19 | 269.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 269.03 | 269.96 | 269.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 269.03 | 269.96 | 269.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 269.79 | 269.92 | 269.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 271.53 | 269.92 | 269.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:45:00 | 270.75 | 272.25 | 271.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 13:15:00 | 273.30 | 275.17 | 275.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 13:15:00 | 273.30 | 275.17 | 275.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 14:15:00 | 271.95 | 274.53 | 275.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 11:15:00 | 274.40 | 273.64 | 274.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 11:15:00 | 274.40 | 273.64 | 274.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 274.40 | 273.64 | 274.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:45:00 | 274.30 | 273.64 | 274.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 274.45 | 273.80 | 274.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:00:00 | 274.45 | 273.80 | 274.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 274.70 | 273.98 | 274.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:00:00 | 274.70 | 273.98 | 274.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 275.15 | 274.22 | 274.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 15:00:00 | 275.15 | 274.22 | 274.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 274.40 | 274.25 | 274.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:15:00 | 275.15 | 274.25 | 274.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 275.30 | 274.46 | 274.51 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 10:15:00 | 276.40 | 274.85 | 274.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 11:15:00 | 278.50 | 275.58 | 275.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 15:15:00 | 276.00 | 276.19 | 275.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-07 09:15:00 | 275.45 | 276.19 | 275.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 275.50 | 276.05 | 275.54 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 11:15:00 | 270.70 | 274.41 | 274.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 270.15 | 271.93 | 273.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 15:15:00 | 270.50 | 270.08 | 271.63 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-09 09:15:00 | 268.25 | 270.08 | 271.63 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 261.70 | 264.48 | 266.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:30:00 | 261.30 | 263.94 | 265.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 11:15:00 | 261.45 | 263.94 | 265.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 12:00:00 | 261.55 | 263.46 | 265.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 09:15:00 | 261.30 | 263.08 | 264.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 264.00 | 263.42 | 264.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:45:00 | 264.15 | 263.42 | 264.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 267.00 | 263.94 | 264.20 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 267.00 | 263.94 | 264.20 | SL hit (close>ema400) qty=1.00 sl=264.20 alert=retest1 |

### Cycle 77 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 267.30 | 264.61 | 264.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 13:15:00 | 269.40 | 266.37 | 265.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 15:15:00 | 270.20 | 270.57 | 269.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 09:15:00 | 273.40 | 270.57 | 269.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 274.50 | 271.36 | 269.97 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 269.75 | 272.26 | 272.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 266.70 | 271.15 | 271.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 252.65 | 250.71 | 252.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 10:00:00 | 252.65 | 250.71 | 252.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 255.20 | 251.60 | 253.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:45:00 | 254.90 | 251.60 | 253.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 255.95 | 252.47 | 253.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:45:00 | 256.80 | 252.47 | 253.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2025-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 12:15:00 | 260.35 | 254.05 | 254.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 13:15:00 | 263.30 | 255.90 | 254.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 09:15:00 | 258.45 | 258.67 | 256.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 09:15:00 | 258.45 | 258.67 | 256.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 258.45 | 258.67 | 256.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:00:00 | 258.45 | 258.67 | 256.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 256.85 | 258.21 | 256.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 11:30:00 | 256.60 | 258.21 | 256.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 257.05 | 257.98 | 256.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:30:00 | 258.40 | 257.98 | 256.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 257.75 | 257.93 | 256.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:30:00 | 257.10 | 257.93 | 256.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 257.00 | 257.99 | 257.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 257.00 | 257.99 | 257.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 257.80 | 257.95 | 257.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:15:00 | 256.10 | 257.95 | 257.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 256.15 | 257.59 | 257.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:30:00 | 255.20 | 257.59 | 257.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 256.00 | 257.28 | 257.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:30:00 | 255.95 | 257.28 | 257.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 256.05 | 256.96 | 256.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 15:00:00 | 256.05 | 256.96 | 256.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 15:15:00 | 256.00 | 256.77 | 256.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 255.40 | 256.40 | 256.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 11:15:00 | 256.85 | 256.49 | 256.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 11:15:00 | 256.85 | 256.49 | 256.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 256.85 | 256.49 | 256.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 12:00:00 | 256.85 | 256.49 | 256.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 254.50 | 256.09 | 256.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 13:15:00 | 254.10 | 256.09 | 256.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 259.55 | 256.79 | 256.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 259.55 | 256.79 | 256.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 15:15:00 | 261.90 | 257.81 | 257.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 14:15:00 | 258.10 | 260.40 | 259.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 14:15:00 | 258.10 | 260.40 | 259.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 258.10 | 260.40 | 259.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 258.10 | 260.40 | 259.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 257.35 | 259.79 | 258.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:15:00 | 259.40 | 259.79 | 258.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 256.40 | 259.11 | 258.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:30:00 | 255.30 | 259.11 | 258.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 264.15 | 260.12 | 259.19 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 15:15:00 | 255.50 | 258.61 | 258.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 14:15:00 | 253.90 | 256.33 | 257.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 256.05 | 255.93 | 257.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-13 10:00:00 | 256.05 | 255.93 | 257.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 256.05 | 255.33 | 256.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:30:00 | 256.55 | 255.33 | 256.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 256.25 | 255.51 | 256.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:30:00 | 256.50 | 255.51 | 256.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 09:15:00 | 265.35 | 257.33 | 257.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 268.20 | 263.85 | 260.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 265.70 | 266.18 | 263.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 09:45:00 | 265.00 | 266.18 | 263.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 265.70 | 267.87 | 266.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:00:00 | 265.70 | 267.87 | 266.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 264.95 | 267.29 | 266.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:30:00 | 265.30 | 267.29 | 266.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 265.40 | 266.91 | 265.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:30:00 | 264.80 | 266.91 | 265.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 265.55 | 266.42 | 265.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:30:00 | 265.60 | 266.42 | 265.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 265.90 | 266.32 | 265.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 11:00:00 | 267.15 | 266.55 | 266.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 11:30:00 | 267.20 | 266.49 | 266.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 15:15:00 | 264.85 | 265.97 | 265.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 264.85 | 265.97 | 265.98 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 11:15:00 | 267.35 | 266.16 | 266.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 09:15:00 | 272.00 | 267.78 | 266.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 13:15:00 | 269.75 | 269.94 | 268.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-25 14:00:00 | 269.75 | 269.94 | 268.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 269.00 | 269.58 | 268.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 267.80 | 269.58 | 268.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 267.70 | 269.20 | 268.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 267.05 | 269.20 | 268.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 267.95 | 268.95 | 268.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:30:00 | 267.35 | 268.95 | 268.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 269.10 | 268.98 | 268.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:45:00 | 268.65 | 268.98 | 268.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 270.95 | 269.38 | 268.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 13:15:00 | 272.50 | 269.38 | 268.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 14:15:00 | 266.95 | 268.89 | 268.58 | SL hit (close<static) qty=1.00 sl=268.50 alert=retest2 |

### Cycle 86 — SELL (started 2025-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 12:15:00 | 265.85 | 268.29 | 268.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 14:15:00 | 263.75 | 267.10 | 267.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 267.65 | 264.21 | 265.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 267.65 | 264.21 | 265.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 267.65 | 264.21 | 265.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 268.10 | 264.21 | 265.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 269.70 | 265.31 | 265.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 269.70 | 265.31 | 265.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 271.90 | 266.63 | 266.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 13:15:00 | 272.55 | 268.64 | 267.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 285.50 | 286.19 | 281.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 10:00:00 | 285.50 | 286.19 | 281.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 285.00 | 284.87 | 283.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:15:00 | 285.35 | 284.45 | 283.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 14:15:00 | 285.45 | 284.56 | 283.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 15:15:00 | 285.40 | 284.58 | 283.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 10:15:00 | 288.10 | 289.45 | 289.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 10:15:00 | 288.10 | 289.45 | 289.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 12:15:00 | 286.80 | 288.64 | 289.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 09:15:00 | 288.60 | 287.79 | 288.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 288.60 | 287.79 | 288.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 288.60 | 287.79 | 288.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:30:00 | 288.10 | 287.79 | 288.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 288.50 | 287.94 | 288.50 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 12:15:00 | 291.15 | 288.92 | 288.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 15:15:00 | 291.95 | 290.96 | 290.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 10:15:00 | 292.70 | 294.27 | 292.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 10:15:00 | 292.70 | 294.27 | 292.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 292.70 | 294.27 | 292.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:00:00 | 292.70 | 294.27 | 292.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 293.10 | 294.04 | 292.95 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 09:15:00 | 289.35 | 292.56 | 292.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 11:15:00 | 286.60 | 290.33 | 291.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 09:15:00 | 289.40 | 288.43 | 289.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 09:15:00 | 289.40 | 288.43 | 289.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 289.40 | 288.43 | 289.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:00:00 | 289.40 | 288.43 | 289.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 292.25 | 289.19 | 290.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:30:00 | 292.25 | 289.19 | 290.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 292.80 | 289.91 | 290.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:00:00 | 292.80 | 289.91 | 290.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2025-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 12:15:00 | 293.10 | 290.55 | 290.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 12:15:00 | 294.35 | 292.10 | 291.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 10:15:00 | 291.65 | 292.63 | 292.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 10:15:00 | 291.65 | 292.63 | 292.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 291.65 | 292.63 | 292.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 291.65 | 292.63 | 292.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 290.65 | 292.23 | 291.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 290.65 | 292.23 | 291.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 290.40 | 291.87 | 291.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:45:00 | 289.65 | 291.87 | 291.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 13:15:00 | 289.05 | 291.30 | 291.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 286.90 | 290.42 | 291.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 279.50 | 278.80 | 281.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 09:30:00 | 280.40 | 278.80 | 281.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 282.20 | 279.19 | 280.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:00:00 | 282.20 | 279.19 | 280.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 282.65 | 279.88 | 281.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 15:00:00 | 282.65 | 279.88 | 281.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 281.25 | 281.12 | 281.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 11:30:00 | 281.55 | 281.12 | 281.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 280.05 | 280.91 | 281.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:30:00 | 279.30 | 280.34 | 280.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 15:15:00 | 282.30 | 280.74 | 281.06 | SL hit (close>static) qty=1.00 sl=281.75 alert=retest2 |

### Cycle 93 — BUY (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 11:15:00 | 283.30 | 281.33 | 281.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 285.65 | 282.27 | 281.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 10:15:00 | 283.00 | 283.50 | 282.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 10:15:00 | 283.00 | 283.50 | 282.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 283.00 | 283.50 | 282.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:45:00 | 282.85 | 283.50 | 282.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 282.25 | 283.25 | 282.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 11:45:00 | 282.10 | 283.25 | 282.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 282.75 | 283.15 | 282.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 13:45:00 | 283.05 | 283.08 | 282.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 14:30:00 | 284.40 | 283.68 | 282.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 10:15:00 | 284.65 | 290.78 | 291.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 10:15:00 | 284.65 | 290.78 | 291.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 13:15:00 | 283.65 | 285.62 | 286.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 286.65 | 285.26 | 286.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 286.65 | 285.26 | 286.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 286.65 | 285.26 | 286.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 287.75 | 285.26 | 286.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 288.15 | 285.84 | 286.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:45:00 | 288.25 | 285.84 | 286.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 288.20 | 286.31 | 286.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:00:00 | 288.20 | 286.31 | 286.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 288.80 | 286.81 | 286.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 14:15:00 | 289.40 | 287.58 | 287.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 09:15:00 | 287.70 | 287.74 | 287.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 09:45:00 | 287.65 | 287.74 | 287.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 287.50 | 287.69 | 287.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:30:00 | 287.90 | 287.69 | 287.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 286.75 | 287.75 | 287.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 15:00:00 | 286.75 | 287.75 | 287.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 287.00 | 287.60 | 287.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 289.90 | 287.60 | 287.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 288.35 | 287.75 | 287.52 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 11:15:00 | 286.80 | 287.33 | 287.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 284.95 | 286.86 | 287.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 12:15:00 | 286.45 | 285.51 | 286.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 12:15:00 | 286.45 | 285.51 | 286.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 286.45 | 285.51 | 286.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:00:00 | 286.45 | 285.51 | 286.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 286.50 | 285.71 | 286.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:00:00 | 286.50 | 285.71 | 286.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 287.90 | 286.14 | 286.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 15:00:00 | 287.90 | 286.14 | 286.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2025-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 15:15:00 | 287.95 | 286.51 | 286.44 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 09:15:00 | 280.05 | 285.20 | 285.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 09:15:00 | 277.70 | 279.96 | 281.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 12:15:00 | 278.40 | 276.87 | 278.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 12:15:00 | 278.40 | 276.87 | 278.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 278.40 | 276.87 | 278.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:30:00 | 279.25 | 276.87 | 278.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 276.70 | 276.84 | 278.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:15:00 | 275.50 | 276.84 | 278.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 275.20 | 276.95 | 278.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 12:30:00 | 275.30 | 276.19 | 277.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 15:00:00 | 275.90 | 276.12 | 277.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 277.45 | 276.39 | 277.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 278.25 | 276.39 | 277.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 279.50 | 277.01 | 277.31 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-30 09:15:00 | 279.50 | 277.01 | 277.31 | SL hit (close>static) qty=1.00 sl=279.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 10:15:00 | 280.00 | 277.61 | 277.55 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 276.40 | 277.37 | 277.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 13:15:00 | 273.75 | 276.40 | 276.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 10:15:00 | 278.85 | 276.30 | 276.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 10:15:00 | 278.85 | 276.30 | 276.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 278.85 | 276.30 | 276.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 278.85 | 276.30 | 276.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 274.15 | 275.87 | 276.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 13:45:00 | 273.50 | 275.19 | 276.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 12:15:00 | 275.20 | 271.70 | 271.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 12:15:00 | 275.20 | 271.70 | 271.38 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 14:15:00 | 270.20 | 271.57 | 271.69 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 274.05 | 272.12 | 271.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 12:15:00 | 274.90 | 272.67 | 272.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 12:15:00 | 275.00 | 275.87 | 274.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 12:15:00 | 275.00 | 275.87 | 274.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 275.00 | 275.87 | 274.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:00:00 | 275.00 | 275.87 | 274.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 274.20 | 275.54 | 274.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:00:00 | 274.20 | 275.54 | 274.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 275.30 | 275.49 | 274.56 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 11:15:00 | 272.85 | 274.09 | 274.14 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 10:15:00 | 275.05 | 274.28 | 274.20 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 15:15:00 | 273.85 | 274.26 | 274.27 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 09:15:00 | 284.20 | 276.25 | 275.17 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 10:15:00 | 276.45 | 279.62 | 279.91 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 280.85 | 279.99 | 279.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 11:15:00 | 282.05 | 280.51 | 280.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 279.35 | 281.13 | 280.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 279.35 | 281.13 | 280.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 279.35 | 281.13 | 280.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:15:00 | 278.55 | 281.13 | 280.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 278.70 | 280.65 | 280.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:45:00 | 278.45 | 280.65 | 280.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 279.20 | 280.36 | 280.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 277.05 | 279.40 | 279.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 279.65 | 279.01 | 279.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 279.65 | 279.01 | 279.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 279.65 | 279.01 | 279.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:00:00 | 279.65 | 279.01 | 279.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 278.65 | 278.94 | 279.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:30:00 | 279.80 | 278.94 | 279.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 278.45 | 278.84 | 279.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 12:30:00 | 277.35 | 278.82 | 279.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 13:30:00 | 277.00 | 278.42 | 279.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 10:00:00 | 277.30 | 277.12 | 278.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 10:30:00 | 277.45 | 277.31 | 278.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 277.90 | 277.43 | 278.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 12:30:00 | 277.15 | 277.48 | 278.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 14:15:00 | 279.40 | 278.02 | 278.32 | SL hit (close>static) qty=1.00 sl=278.80 alert=retest2 |

### Cycle 111 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 282.40 | 279.05 | 278.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 11:15:00 | 284.50 | 280.77 | 279.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 11:15:00 | 284.60 | 285.22 | 282.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:45:00 | 284.70 | 285.22 | 282.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 283.00 | 284.92 | 284.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 12:45:00 | 283.15 | 284.92 | 284.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 285.00 | 284.94 | 284.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 15:15:00 | 285.55 | 284.86 | 284.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 10:15:00 | 285.60 | 284.80 | 284.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 13:15:00 | 282.40 | 283.91 | 284.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2025-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 13:15:00 | 282.40 | 283.91 | 284.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 09:15:00 | 278.90 | 282.48 | 283.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 15:15:00 | 279.50 | 278.85 | 280.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 09:15:00 | 277.20 | 278.85 | 280.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 276.10 | 274.96 | 276.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:30:00 | 276.40 | 274.96 | 276.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 277.30 | 275.42 | 276.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:45:00 | 277.40 | 275.42 | 276.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 277.95 | 275.93 | 276.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:00:00 | 277.95 | 275.93 | 276.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 14:15:00 | 279.60 | 276.66 | 276.64 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 273.85 | 276.15 | 276.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 272.10 | 275.34 | 276.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 274.55 | 273.22 | 274.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 10:15:00 | 274.55 | 273.22 | 274.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 274.55 | 273.22 | 274.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:00:00 | 274.55 | 273.22 | 274.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 275.10 | 273.60 | 274.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 275.10 | 273.60 | 274.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 276.20 | 274.12 | 274.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 276.20 | 274.12 | 274.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 274.25 | 274.71 | 274.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 282.30 | 274.71 | 274.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 281.80 | 276.13 | 275.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 285.85 | 282.30 | 280.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 12:15:00 | 285.80 | 286.07 | 284.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 12:45:00 | 285.75 | 286.07 | 284.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 282.75 | 285.31 | 284.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 283.30 | 285.31 | 284.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 283.90 | 285.03 | 284.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 283.90 | 285.03 | 284.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 13:15:00 | 282.70 | 283.71 | 283.83 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 09:15:00 | 285.00 | 284.00 | 283.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 09:15:00 | 288.60 | 285.97 | 285.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 14:15:00 | 287.35 | 287.84 | 286.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 14:15:00 | 287.35 | 287.84 | 286.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 287.35 | 287.84 | 286.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:45:00 | 285.80 | 287.84 | 286.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 287.00 | 287.68 | 286.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 286.70 | 287.68 | 286.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 284.80 | 287.10 | 286.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:45:00 | 285.35 | 287.10 | 286.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 285.85 | 286.85 | 286.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 11:15:00 | 287.30 | 286.85 | 286.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-12-24 15:15:00 | 316.03 | 309.71 | 302.75 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2025-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 12:15:00 | 306.35 | 309.38 | 309.51 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 15:15:00 | 311.00 | 309.54 | 309.53 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 11:15:00 | 306.60 | 309.21 | 309.41 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 10:15:00 | 313.65 | 309.23 | 309.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 15:15:00 | 314.50 | 312.25 | 310.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 12:15:00 | 313.55 | 313.70 | 312.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 13:00:00 | 313.55 | 313.70 | 312.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 310.15 | 312.99 | 311.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:00:00 | 310.15 | 312.99 | 311.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 311.10 | 312.61 | 311.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:45:00 | 310.15 | 312.61 | 311.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 09:15:00 | 307.00 | 311.23 | 311.33 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 11:15:00 | 312.80 | 309.15 | 309.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 14:15:00 | 319.90 | 312.57 | 310.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 313.00 | 313.92 | 311.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 09:45:00 | 314.75 | 313.92 | 311.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 307.95 | 312.73 | 311.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 307.95 | 312.73 | 311.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 310.75 | 312.33 | 311.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 12:45:00 | 311.55 | 312.43 | 311.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 14:45:00 | 311.35 | 311.39 | 311.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 15:15:00 | 308.50 | 310.81 | 310.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 15:15:00 | 308.50 | 310.81 | 310.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 11:15:00 | 307.40 | 309.52 | 310.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 10:15:00 | 303.50 | 297.15 | 300.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 10:15:00 | 303.50 | 297.15 | 300.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 303.50 | 297.15 | 300.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:00:00 | 303.50 | 297.15 | 300.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 304.65 | 298.65 | 300.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:00:00 | 304.65 | 298.65 | 300.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2026-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 14:15:00 | 308.20 | 302.62 | 302.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 14:15:00 | 309.40 | 306.53 | 304.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 11:15:00 | 312.75 | 312.82 | 310.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 11:45:00 | 312.15 | 312.82 | 310.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 307.30 | 311.96 | 310.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 306.40 | 311.96 | 310.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 305.55 | 310.67 | 310.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:30:00 | 307.75 | 310.67 | 310.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 305.20 | 309.58 | 309.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 303.25 | 308.31 | 309.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 300.90 | 300.29 | 303.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:15:00 | 302.75 | 300.29 | 303.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 295.10 | 299.25 | 302.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 300.60 | 299.25 | 302.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 299.95 | 298.45 | 300.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:30:00 | 300.35 | 298.45 | 300.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 299.45 | 298.65 | 300.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:15:00 | 302.40 | 298.65 | 300.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 299.90 | 298.90 | 300.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:45:00 | 298.25 | 298.78 | 300.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 09:15:00 | 283.34 | 293.92 | 294.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-01 09:15:00 | 268.43 | 286.33 | 289.87 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 127 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 289.40 | 280.13 | 279.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 14:15:00 | 292.75 | 285.32 | 282.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 287.80 | 291.26 | 288.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 287.80 | 291.26 | 288.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 287.80 | 291.26 | 288.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 14:30:00 | 293.25 | 291.71 | 289.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:30:00 | 292.45 | 291.58 | 290.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 298.35 | 304.62 | 305.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 298.35 | 304.62 | 305.11 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 306.65 | 303.29 | 303.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 13:15:00 | 311.10 | 307.71 | 305.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 307.20 | 308.21 | 306.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 09:45:00 | 307.60 | 308.21 | 306.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 308.05 | 308.18 | 306.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:30:00 | 306.55 | 308.18 | 306.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 306.55 | 307.83 | 306.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:45:00 | 305.90 | 307.83 | 306.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 306.95 | 307.66 | 306.86 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 305.25 | 306.39 | 306.41 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 307.15 | 306.48 | 306.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 308.50 | 306.93 | 306.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 11:15:00 | 306.10 | 306.85 | 306.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 11:15:00 | 306.10 | 306.85 | 306.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 306.10 | 306.85 | 306.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:45:00 | 305.95 | 306.85 | 306.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2026-02-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 12:15:00 | 305.00 | 306.48 | 306.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 13:15:00 | 304.35 | 306.05 | 306.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 13:15:00 | 304.50 | 304.36 | 305.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-24 14:00:00 | 304.50 | 304.36 | 305.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 304.50 | 304.39 | 305.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 15:00:00 | 304.50 | 304.39 | 305.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 305.70 | 304.65 | 305.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 09:30:00 | 303.40 | 304.50 | 305.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 09:15:00 | 288.23 | 298.56 | 301.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 15:15:00 | 293.75 | 293.16 | 296.85 | SL hit (close>ema200) qty=0.50 sl=293.16 alert=retest2 |

### Cycle 133 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 258.55 | 255.22 | 255.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 262.65 | 257.27 | 256.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 13:15:00 | 265.00 | 266.29 | 263.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 14:00:00 | 265.00 | 266.29 | 263.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 264.20 | 265.87 | 263.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 15:00:00 | 264.20 | 265.87 | 263.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 262.35 | 264.93 | 263.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 262.35 | 264.93 | 263.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 263.20 | 264.58 | 263.36 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 14:15:00 | 259.00 | 262.28 | 262.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 250.55 | 259.45 | 261.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 14:15:00 | 252.75 | 252.35 | 256.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-23 15:00:00 | 252.75 | 252.35 | 256.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 250.65 | 252.11 | 255.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 11:00:00 | 249.35 | 251.56 | 255.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 266.20 | 255.06 | 255.20 | SL hit (close>static) qty=1.00 sl=257.00 alert=retest2 |

### Cycle 135 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 269.15 | 257.88 | 256.47 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 252.75 | 257.28 | 257.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 251.65 | 254.31 | 255.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 257.15 | 253.53 | 254.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 257.15 | 253.53 | 254.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 257.15 | 253.53 | 254.55 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 259.70 | 256.03 | 255.54 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 251.80 | 254.98 | 255.24 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 09:15:00 | 257.75 | 255.50 | 255.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 10:15:00 | 258.90 | 256.18 | 255.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 255.85 | 258.06 | 257.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 255.85 | 258.06 | 257.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 255.85 | 258.06 | 257.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:45:00 | 256.15 | 258.06 | 257.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 10:15:00 | 255.00 | 257.45 | 256.89 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2026-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 12:15:00 | 254.75 | 256.53 | 256.54 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 264.55 | 257.80 | 257.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 269.50 | 260.14 | 258.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 265.25 | 266.00 | 262.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 09:45:00 | 264.20 | 266.00 | 262.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 262.60 | 264.94 | 263.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:45:00 | 263.45 | 264.94 | 263.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 263.45 | 264.64 | 263.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:30:00 | 263.80 | 264.64 | 263.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 263.55 | 264.42 | 263.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 266.10 | 264.42 | 263.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 09:45:00 | 265.10 | 265.92 | 264.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-22 11:15:00 | 292.71 | 284.29 | 278.46 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-11 10:30:00 | 182.57 | 2024-06-25 12:15:00 | 191.30 | STOP_HIT | 1.00 | 4.78% |
| BUY | retest2 | 2024-06-11 14:15:00 | 182.77 | 2024-06-25 12:15:00 | 191.30 | STOP_HIT | 1.00 | 4.67% |
| BUY | retest2 | 2024-06-12 09:15:00 | 182.43 | 2024-06-25 12:15:00 | 191.30 | STOP_HIT | 1.00 | 4.86% |
| BUY | retest2 | 2024-06-12 10:15:00 | 182.49 | 2024-06-25 12:15:00 | 191.30 | STOP_HIT | 1.00 | 4.83% |
| BUY | retest2 | 2024-06-18 13:15:00 | 190.76 | 2024-06-25 12:15:00 | 191.30 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2024-06-19 12:00:00 | 191.31 | 2024-06-25 12:15:00 | 191.30 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2024-06-20 09:15:00 | 192.26 | 2024-06-25 12:15:00 | 191.30 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-06-24 10:00:00 | 191.10 | 2024-06-25 12:15:00 | 191.30 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2024-07-16 11:00:00 | 223.15 | 2024-07-19 13:15:00 | 219.04 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2024-07-16 13:00:00 | 223.09 | 2024-07-19 13:15:00 | 219.04 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2024-07-18 09:15:00 | 224.83 | 2024-07-19 13:15:00 | 219.04 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2024-07-18 09:45:00 | 223.83 | 2024-07-19 13:15:00 | 219.04 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2024-07-25 09:15:00 | 207.50 | 2024-07-26 14:15:00 | 213.15 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2024-07-26 11:30:00 | 209.20 | 2024-07-26 14:15:00 | 213.15 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-07-30 10:15:00 | 215.69 | 2024-08-02 09:15:00 | 211.45 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-07-30 12:30:00 | 215.77 | 2024-08-02 09:15:00 | 211.45 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2024-07-30 14:00:00 | 215.35 | 2024-08-02 09:15:00 | 211.45 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-08-01 09:30:00 | 216.87 | 2024-08-02 09:15:00 | 211.45 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2024-08-06 10:30:00 | 202.34 | 2024-08-09 11:15:00 | 205.44 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-08-06 11:00:00 | 203.08 | 2024-08-09 11:15:00 | 205.44 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-08-06 12:00:00 | 203.23 | 2024-08-09 11:15:00 | 205.44 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-08-16 15:15:00 | 201.00 | 2024-08-20 10:15:00 | 205.64 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2024-08-20 09:15:00 | 200.77 | 2024-08-20 10:15:00 | 205.64 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2024-09-04 10:15:00 | 211.50 | 2024-09-11 13:15:00 | 200.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-04 10:15:00 | 211.50 | 2024-09-12 09:15:00 | 204.79 | STOP_HIT | 0.50 | 3.17% |
| SELL | retest2 | 2024-09-27 15:00:00 | 202.52 | 2024-10-03 09:15:00 | 192.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 09:15:00 | 202.53 | 2024-10-03 09:15:00 | 192.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 12:15:00 | 202.44 | 2024-10-03 09:15:00 | 192.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 13:00:00 | 202.35 | 2024-10-03 09:15:00 | 192.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 15:00:00 | 202.52 | 2024-10-04 09:15:00 | 193.48 | STOP_HIT | 0.50 | 4.46% |
| SELL | retest2 | 2024-09-30 09:15:00 | 202.53 | 2024-10-04 09:15:00 | 193.48 | STOP_HIT | 0.50 | 4.47% |
| SELL | retest2 | 2024-09-30 12:15:00 | 202.44 | 2024-10-04 09:15:00 | 193.48 | STOP_HIT | 0.50 | 4.43% |
| SELL | retest2 | 2024-09-30 13:00:00 | 202.35 | 2024-10-04 09:15:00 | 193.48 | STOP_HIT | 0.50 | 4.38% |
| SELL | retest1 | 2024-10-16 10:00:00 | 181.24 | 2024-10-18 09:15:00 | 163.12 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-17 11:15:00 | 178.07 | 2024-10-18 09:15:00 | 160.26 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-17 13:45:00 | 177.98 | 2024-10-18 09:15:00 | 160.18 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-01 18:00:00 | 159.90 | 2024-11-04 10:15:00 | 153.19 | STOP_HIT | 1.00 | -4.20% |
| SELL | retest2 | 2024-11-25 12:15:00 | 151.40 | 2024-11-26 14:15:00 | 155.87 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2024-11-29 14:15:00 | 156.31 | 2024-12-09 11:15:00 | 171.94 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-02 09:30:00 | 156.09 | 2024-12-09 11:15:00 | 171.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-23 09:15:00 | 184.79 | 2024-12-26 09:15:00 | 178.84 | STOP_HIT | 1.00 | -3.22% |
| BUY | retest2 | 2024-12-26 10:15:00 | 181.46 | 2024-12-27 09:15:00 | 181.72 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2024-12-26 10:45:00 | 183.83 | 2024-12-27 09:15:00 | 181.72 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-12-31 13:45:00 | 189.55 | 2025-01-03 14:15:00 | 188.10 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-01-01 09:15:00 | 189.20 | 2025-01-03 14:15:00 | 188.10 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-01-01 11:00:00 | 189.67 | 2025-01-03 14:15:00 | 188.10 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-01-03 14:15:00 | 189.18 | 2025-01-03 14:15:00 | 188.10 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-01-08 09:15:00 | 181.36 | 2025-01-09 09:15:00 | 188.13 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest2 | 2025-01-22 15:00:00 | 195.60 | 2025-01-27 10:15:00 | 188.00 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest1 | 2025-02-06 09:45:00 | 206.20 | 2025-02-06 13:15:00 | 203.60 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest1 | 2025-02-06 11:15:00 | 206.03 | 2025-02-06 13:15:00 | 203.60 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-02-07 11:30:00 | 207.61 | 2025-02-10 11:15:00 | 202.06 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2025-02-07 12:15:00 | 206.70 | 2025-02-10 11:15:00 | 202.06 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-02-07 15:00:00 | 207.50 | 2025-02-10 11:15:00 | 202.06 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-02-13 14:00:00 | 196.51 | 2025-02-14 09:15:00 | 186.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 14:00:00 | 196.51 | 2025-02-14 10:15:00 | 176.86 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-24 09:15:00 | 206.00 | 2025-02-28 09:15:00 | 197.45 | STOP_HIT | 1.00 | -4.15% |
| BUY | retest2 | 2025-02-25 09:15:00 | 205.51 | 2025-02-28 09:15:00 | 197.45 | STOP_HIT | 1.00 | -3.92% |
| SELL | retest2 | 2025-03-03 15:15:00 | 198.76 | 2025-03-05 13:15:00 | 203.20 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2025-03-04 10:00:00 | 197.60 | 2025-03-05 13:15:00 | 203.20 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2025-03-11 10:30:00 | 204.61 | 2025-03-11 12:15:00 | 203.29 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-03-17 15:15:00 | 210.70 | 2025-03-21 11:15:00 | 231.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-18 11:15:00 | 211.40 | 2025-03-21 12:15:00 | 232.54 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-08 10:30:00 | 225.52 | 2025-04-16 10:15:00 | 227.11 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-04-08 11:15:00 | 225.91 | 2025-04-16 10:15:00 | 227.11 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-04-09 09:45:00 | 226.07 | 2025-04-16 10:15:00 | 227.11 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-04-09 12:30:00 | 225.00 | 2025-04-16 10:15:00 | 227.11 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-04-15 11:00:00 | 224.82 | 2025-04-16 10:15:00 | 227.11 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-04-17 13:30:00 | 229.40 | 2025-04-24 10:15:00 | 229.10 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-04-21 09:15:00 | 230.35 | 2025-04-24 10:15:00 | 229.10 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-05-13 14:15:00 | 229.95 | 2025-05-15 13:15:00 | 229.39 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-05-21 15:15:00 | 227.79 | 2025-05-22 12:15:00 | 229.90 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest1 | 2025-05-26 09:15:00 | 231.26 | 2025-05-29 13:15:00 | 231.71 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest1 | 2025-05-26 13:45:00 | 231.13 | 2025-05-29 13:15:00 | 231.71 | STOP_HIT | 1.00 | 0.25% |
| BUY | retest2 | 2025-05-30 11:30:00 | 235.50 | 2025-06-04 10:15:00 | 233.85 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-06-05 15:15:00 | 234.35 | 2025-06-06 10:15:00 | 239.00 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-06-13 10:15:00 | 273.81 | 2025-06-17 12:15:00 | 269.61 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-06-13 11:15:00 | 274.29 | 2025-06-17 12:15:00 | 269.61 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-06-25 09:15:00 | 271.53 | 2025-07-02 13:15:00 | 273.30 | STOP_HIT | 1.00 | 0.65% |
| BUY | retest2 | 2025-06-27 09:45:00 | 270.75 | 2025-07-02 13:15:00 | 273.30 | STOP_HIT | 1.00 | 0.94% |
| SELL | retest1 | 2025-07-09 09:15:00 | 268.25 | 2025-07-15 09:15:00 | 267.00 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2025-07-11 10:30:00 | 261.30 | 2025-07-15 09:15:00 | 267.00 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-07-11 11:15:00 | 261.45 | 2025-07-15 09:15:00 | 267.00 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-07-11 12:00:00 | 261.55 | 2025-07-15 09:15:00 | 267.00 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-07-14 09:15:00 | 261.30 | 2025-07-15 09:15:00 | 267.00 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-08-07 13:15:00 | 254.10 | 2025-08-07 14:15:00 | 259.55 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-08-21 11:00:00 | 267.15 | 2025-08-21 15:15:00 | 264.85 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-08-21 11:30:00 | 267.20 | 2025-08-21 15:15:00 | 264.85 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-08-26 13:15:00 | 272.50 | 2025-08-26 14:15:00 | 266.95 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-09-05 13:15:00 | 285.35 | 2025-09-11 10:15:00 | 288.10 | STOP_HIT | 1.00 | 0.96% |
| BUY | retest2 | 2025-09-05 14:15:00 | 285.45 | 2025-09-11 10:15:00 | 288.10 | STOP_HIT | 1.00 | 0.93% |
| BUY | retest2 | 2025-09-05 15:15:00 | 285.40 | 2025-09-11 10:15:00 | 288.10 | STOP_HIT | 1.00 | 0.95% |
| SELL | retest2 | 2025-09-30 13:30:00 | 279.30 | 2025-09-30 15:15:00 | 282.30 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-10-03 13:45:00 | 283.05 | 2025-10-10 10:15:00 | 284.65 | STOP_HIT | 1.00 | 0.57% |
| BUY | retest2 | 2025-10-03 14:30:00 | 284.40 | 2025-10-10 10:15:00 | 284.65 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2025-10-28 14:15:00 | 275.50 | 2025-10-30 09:15:00 | 279.50 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-10-29 09:15:00 | 275.20 | 2025-10-30 09:15:00 | 279.50 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-10-29 12:30:00 | 275.30 | 2025-10-30 09:15:00 | 279.50 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-10-29 15:00:00 | 275.90 | 2025-10-30 09:15:00 | 279.50 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-10-31 13:45:00 | 273.50 | 2025-11-06 12:15:00 | 275.20 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-11-24 12:30:00 | 277.35 | 2025-11-25 14:15:00 | 279.40 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-11-24 13:30:00 | 277.00 | 2025-11-26 09:15:00 | 282.40 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-11-25 10:00:00 | 277.30 | 2025-11-26 09:15:00 | 282.40 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-11-25 10:30:00 | 277.45 | 2025-11-26 09:15:00 | 282.40 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-11-25 12:30:00 | 277.15 | 2025-11-26 09:15:00 | 282.40 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-11-28 15:15:00 | 285.55 | 2025-12-01 13:15:00 | 282.40 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-12-01 10:15:00 | 285.60 | 2025-12-01 13:15:00 | 282.40 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-12-19 11:15:00 | 287.30 | 2025-12-24 15:15:00 | 316.03 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-08 12:45:00 | 311.55 | 2026-01-08 15:15:00 | 308.50 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2026-01-08 14:45:00 | 311.35 | 2026-01-08 15:15:00 | 308.50 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-01-23 10:45:00 | 298.25 | 2026-01-30 09:15:00 | 283.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 10:45:00 | 298.25 | 2026-02-01 09:15:00 | 268.43 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-05 14:30:00 | 293.25 | 2026-02-13 10:15:00 | 298.35 | STOP_HIT | 1.00 | 1.74% |
| BUY | retest2 | 2026-02-06 11:30:00 | 292.45 | 2026-02-13 10:15:00 | 298.35 | STOP_HIT | 1.00 | 2.02% |
| SELL | retest2 | 2026-02-25 09:30:00 | 303.40 | 2026-02-26 09:15:00 | 288.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 09:30:00 | 303.40 | 2026-02-26 15:15:00 | 293.75 | STOP_HIT | 0.50 | 3.18% |
| SELL | retest2 | 2026-03-24 11:00:00 | 249.35 | 2026-03-25 09:15:00 | 266.20 | STOP_HIT | 1.00 | -6.76% |
| BUY | retest2 | 2026-04-10 09:15:00 | 266.10 | 2026-04-22 11:15:00 | 292.71 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 09:45:00 | 265.10 | 2026-04-22 11:15:00 | 291.61 | TARGET_HIT | 1.00 | 10.00% |
