# Anant Raj Ltd. (ANANTRAJ)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 561.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 189 |
| ALERT1 | 138 |
| ALERT2 | 133 |
| ALERT2_SKIP | 64 |
| ALERT3 | 336 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 9 |
| ENTRY2 | 160 |
| PARTIAL | 24 |
| TARGET_HIT | 11 |
| STOP_HIT | 158 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 193 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 78 / 115
- **Target hits / Stop hits / Partials:** 11 / 158 / 24
- **Avg / median % per leg:** 0.47% / -0.76%
- **Sum % (uncompounded):** 90.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 103 | 39 | 37.9% | 7 | 91 | 5 | 0.17% | 17.4% |
| BUY @ 2nd Alert (retest1) | 11 | 10 | 90.9% | 0 | 6 | 5 | 3.39% | 37.3% |
| BUY @ 3rd Alert (retest2) | 92 | 29 | 31.5% | 7 | 85 | 0 | -0.22% | -19.9% |
| SELL (all) | 90 | 39 | 43.3% | 4 | 67 | 19 | 0.81% | 72.8% |
| SELL @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 0 | 3 | 3 | 4.43% | 26.6% |
| SELL @ 3rd Alert (retest2) | 84 | 33 | 39.3% | 4 | 64 | 16 | 0.55% | 46.2% |
| retest1 (combined) | 17 | 16 | 94.1% | 0 | 9 | 8 | 3.76% | 63.9% |
| retest2 (combined) | 176 | 62 | 35.2% | 11 | 149 | 16 | 0.15% | 26.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 13:15:00 | 141.50 | 140.82 | 140.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-15 14:15:00 | 142.20 | 141.10 | 140.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-17 11:15:00 | 144.40 | 145.15 | 143.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-17 12:00:00 | 144.40 | 145.15 | 143.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 12:15:00 | 143.35 | 144.79 | 143.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-17 13:15:00 | 143.30 | 144.79 | 143.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 13:15:00 | 145.00 | 144.83 | 143.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-17 14:15:00 | 145.60 | 144.83 | 143.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-17 15:00:00 | 146.00 | 145.07 | 144.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-19 11:30:00 | 146.00 | 146.04 | 145.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-19 14:30:00 | 145.45 | 145.56 | 145.48 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 15:15:00 | 145.75 | 145.60 | 145.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-22 09:15:00 | 143.95 | 145.60 | 145.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-05-22 09:15:00 | 143.20 | 145.12 | 145.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2023-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-22 09:15:00 | 143.20 | 145.12 | 145.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-22 12:15:00 | 141.60 | 144.01 | 144.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-22 15:15:00 | 144.75 | 143.96 | 144.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 15:15:00 | 144.75 | 143.96 | 144.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 15:15:00 | 144.75 | 143.96 | 144.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-23 09:15:00 | 144.50 | 143.96 | 144.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 09:15:00 | 144.25 | 144.02 | 144.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-23 14:45:00 | 143.85 | 143.99 | 144.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-25 14:00:00 | 143.65 | 142.09 | 142.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-26 09:15:00 | 146.95 | 143.61 | 143.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2023-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 09:15:00 | 146.95 | 143.61 | 143.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-29 12:15:00 | 148.95 | 147.15 | 145.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-31 14:15:00 | 151.35 | 151.37 | 149.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-31 14:30:00 | 151.50 | 151.37 | 149.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 14:15:00 | 152.50 | 151.75 | 150.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-01 14:45:00 | 151.00 | 151.75 | 150.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 15:15:00 | 151.10 | 151.62 | 150.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 09:15:00 | 153.05 | 151.62 | 150.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-06-12 12:15:00 | 168.36 | 165.59 | 163.95 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2023-06-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 11:15:00 | 164.65 | 167.28 | 167.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-15 14:15:00 | 162.55 | 165.65 | 166.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-15 15:15:00 | 165.75 | 165.67 | 166.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-16 09:15:00 | 167.85 | 165.67 | 166.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 09:15:00 | 167.70 | 166.08 | 166.65 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2023-06-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 12:15:00 | 167.80 | 167.04 | 167.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-19 09:15:00 | 169.35 | 167.56 | 167.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-19 14:15:00 | 168.65 | 168.97 | 168.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-19 15:00:00 | 168.65 | 168.97 | 168.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 15:15:00 | 169.70 | 169.12 | 168.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-20 11:45:00 | 169.75 | 169.14 | 168.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-20 12:15:00 | 167.40 | 168.79 | 168.42 | SL hit (close<static) qty=1.00 sl=168.10 alert=retest2 |

### Cycle 6 — SELL (started 2023-06-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 09:15:00 | 167.35 | 168.51 | 168.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 10:15:00 | 166.00 | 168.01 | 168.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-22 13:15:00 | 168.10 | 167.21 | 167.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 13:15:00 | 168.10 | 167.21 | 167.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 13:15:00 | 168.10 | 167.21 | 167.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-22 13:45:00 | 167.35 | 167.21 | 167.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 14:15:00 | 167.30 | 167.23 | 167.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-22 15:15:00 | 164.30 | 167.23 | 167.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-23 13:45:00 | 165.90 | 165.26 | 166.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-26 15:15:00 | 169.90 | 166.20 | 166.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2023-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 15:15:00 | 169.90 | 166.20 | 166.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 09:15:00 | 171.90 | 167.34 | 166.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-04 09:15:00 | 180.65 | 181.37 | 178.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-04 10:00:00 | 180.65 | 181.37 | 178.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 11:15:00 | 180.25 | 181.12 | 178.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-05 10:00:00 | 181.95 | 180.02 | 179.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-05 11:15:00 | 182.00 | 180.20 | 179.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-05 12:00:00 | 181.80 | 180.52 | 179.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-06 10:00:00 | 181.65 | 180.67 | 179.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 13:15:00 | 181.35 | 181.16 | 180.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-06 13:45:00 | 181.40 | 181.16 | 180.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 09:15:00 | 181.30 | 181.21 | 180.63 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-07-07 13:15:00 | 179.85 | 180.35 | 180.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2023-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 13:15:00 | 179.85 | 180.35 | 180.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 15:15:00 | 177.95 | 179.73 | 180.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 09:15:00 | 183.00 | 178.61 | 178.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 09:15:00 | 183.00 | 178.61 | 178.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 183.00 | 178.61 | 178.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 10:00:00 | 183.00 | 178.61 | 178.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2023-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 10:15:00 | 185.10 | 179.91 | 179.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-13 09:15:00 | 186.15 | 184.27 | 183.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-13 11:15:00 | 184.00 | 184.25 | 183.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-13 11:45:00 | 184.45 | 184.25 | 183.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 13:15:00 | 180.35 | 183.37 | 183.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 14:00:00 | 180.35 | 183.37 | 183.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 14:15:00 | 182.55 | 183.21 | 182.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-14 10:30:00 | 182.85 | 183.02 | 182.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-14 12:15:00 | 182.25 | 182.78 | 182.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2023-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 12:15:00 | 182.25 | 182.78 | 182.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-14 13:15:00 | 180.95 | 182.42 | 182.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 14:15:00 | 183.65 | 182.66 | 182.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-14 14:15:00 | 183.65 | 182.66 | 182.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 14:15:00 | 183.65 | 182.66 | 182.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 15:00:00 | 183.65 | 182.66 | 182.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2023-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 15:15:00 | 184.90 | 183.11 | 182.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 11:15:00 | 188.50 | 184.60 | 183.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 11:15:00 | 185.20 | 186.27 | 185.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-18 11:15:00 | 185.20 | 186.27 | 185.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 11:15:00 | 185.20 | 186.27 | 185.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 12:00:00 | 185.20 | 186.27 | 185.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 12:15:00 | 186.70 | 186.36 | 185.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-18 13:15:00 | 187.45 | 186.36 | 185.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-24 10:15:00 | 190.35 | 194.09 | 194.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2023-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 10:15:00 | 190.35 | 194.09 | 194.34 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-07-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 12:15:00 | 194.95 | 193.02 | 192.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-27 09:15:00 | 196.00 | 194.33 | 193.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-31 10:15:00 | 199.25 | 199.84 | 198.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-31 11:00:00 | 199.25 | 199.84 | 198.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 11:15:00 | 198.10 | 199.49 | 198.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-31 11:45:00 | 198.00 | 199.49 | 198.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 12:15:00 | 200.50 | 199.69 | 198.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-31 13:30:00 | 200.95 | 199.94 | 198.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-01 11:15:00 | 201.15 | 200.88 | 199.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-02 11:15:00 | 200.90 | 202.08 | 201.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-02 12:00:00 | 201.15 | 201.90 | 201.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 12:15:00 | 199.55 | 201.43 | 200.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 13:00:00 | 199.55 | 201.43 | 200.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-08-02 13:15:00 | 193.70 | 199.88 | 200.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2023-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 13:15:00 | 193.70 | 199.88 | 200.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-03 14:15:00 | 193.20 | 196.69 | 198.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-04 09:15:00 | 196.85 | 196.23 | 197.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 09:15:00 | 196.85 | 196.23 | 197.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 196.85 | 196.23 | 197.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 15:00:00 | 194.40 | 195.57 | 196.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-07 15:15:00 | 197.85 | 197.04 | 197.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2023-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 15:15:00 | 197.85 | 197.04 | 197.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-08 10:15:00 | 198.25 | 197.40 | 197.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 14:15:00 | 196.15 | 197.49 | 197.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-08 14:15:00 | 196.15 | 197.49 | 197.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 14:15:00 | 196.15 | 197.49 | 197.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 15:00:00 | 196.15 | 197.49 | 197.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2023-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 15:15:00 | 194.95 | 196.98 | 197.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-09 09:15:00 | 191.85 | 195.95 | 196.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-11 09:15:00 | 193.90 | 191.90 | 193.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-11 09:15:00 | 193.90 | 191.90 | 193.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 09:15:00 | 193.90 | 191.90 | 193.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-11 09:45:00 | 193.80 | 191.90 | 193.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 10:15:00 | 192.00 | 191.92 | 193.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-11 12:00:00 | 191.00 | 191.73 | 192.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-11 15:15:00 | 191.00 | 191.70 | 192.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-14 12:00:00 | 190.85 | 190.88 | 191.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-16 10:15:00 | 198.70 | 192.57 | 192.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2023-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 10:15:00 | 198.70 | 192.57 | 192.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-23 09:15:00 | 205.25 | 201.93 | 200.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 15:15:00 | 204.00 | 204.13 | 202.68 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-24 09:45:00 | 204.80 | 204.24 | 202.87 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-24 10:15:00 | 205.50 | 204.24 | 202.87 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-25 09:15:00 | 215.04 | 207.64 | 205.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-25 09:15:00 | 215.78 | 207.64 | 205.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2023-08-28 14:15:00 | 213.40 | 213.60 | 211.08 | SL hit (close<ema200) qty=0.50 sl=213.60 alert=retest1 |

### Cycle 18 — SELL (started 2023-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-05 15:15:00 | 218.80 | 219.48 | 219.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-06 12:15:00 | 216.75 | 218.36 | 218.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-07 09:15:00 | 218.35 | 217.83 | 218.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-07 09:15:00 | 218.35 | 217.83 | 218.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 09:15:00 | 218.35 | 217.83 | 218.46 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2023-09-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 12:15:00 | 220.00 | 218.88 | 218.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-11 09:15:00 | 229.25 | 221.59 | 220.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-12 09:15:00 | 221.50 | 227.71 | 225.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-12 09:15:00 | 221.50 | 227.71 | 225.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 221.50 | 227.71 | 225.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:45:00 | 218.70 | 227.71 | 225.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 222.55 | 226.68 | 224.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 10:30:00 | 221.20 | 226.68 | 224.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2023-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 12:15:00 | 216.50 | 222.78 | 223.22 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-09-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 10:15:00 | 222.70 | 221.26 | 221.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-18 11:15:00 | 225.10 | 223.63 | 222.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 13:15:00 | 222.75 | 223.64 | 223.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-18 13:15:00 | 222.75 | 223.64 | 223.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 13:15:00 | 222.75 | 223.64 | 223.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 14:00:00 | 222.75 | 223.64 | 223.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 14:15:00 | 222.15 | 223.34 | 222.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 15:15:00 | 220.60 | 223.34 | 222.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 15:15:00 | 220.60 | 222.80 | 222.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 09:15:00 | 224.40 | 222.80 | 222.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2023-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 10:15:00 | 220.75 | 222.67 | 222.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 12:15:00 | 219.15 | 221.58 | 222.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 14:15:00 | 214.55 | 213.31 | 214.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 14:15:00 | 214.55 | 213.31 | 214.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 14:15:00 | 214.55 | 213.31 | 214.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 15:00:00 | 214.55 | 213.31 | 214.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 15:15:00 | 215.00 | 213.65 | 214.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 09:15:00 | 216.10 | 213.65 | 214.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 216.00 | 214.12 | 214.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 09:30:00 | 216.90 | 214.12 | 214.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 10:15:00 | 215.10 | 214.32 | 214.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-26 12:30:00 | 214.25 | 214.70 | 215.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-27 09:15:00 | 214.00 | 214.76 | 214.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-27 10:15:00 | 216.15 | 215.32 | 215.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2023-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 10:15:00 | 216.15 | 215.32 | 215.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 11:15:00 | 218.50 | 215.95 | 215.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 11:15:00 | 218.95 | 219.53 | 218.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-28 12:00:00 | 218.95 | 219.53 | 218.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 13:15:00 | 218.15 | 219.11 | 218.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-28 14:15:00 | 219.15 | 219.11 | 218.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-28 14:45:00 | 219.35 | 218.91 | 218.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-29 09:15:00 | 221.40 | 218.90 | 218.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-29 14:00:00 | 219.00 | 219.10 | 218.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 14:15:00 | 219.70 | 219.22 | 218.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-03 13:45:00 | 222.40 | 219.58 | 219.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-03 15:00:00 | 222.70 | 220.20 | 219.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-04 11:15:00 | 217.80 | 219.72 | 219.47 | SL hit (close<static) qty=1.00 sl=218.25 alert=retest2 |

### Cycle 24 — SELL (started 2023-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 12:15:00 | 216.35 | 219.05 | 219.19 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2023-10-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 09:15:00 | 231.45 | 221.18 | 220.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-05 10:15:00 | 233.10 | 223.57 | 221.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 09:15:00 | 228.30 | 232.25 | 229.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 09:15:00 | 228.30 | 232.25 | 229.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 228.30 | 232.25 | 229.79 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2023-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 13:15:00 | 224.75 | 228.44 | 228.59 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2023-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 10:15:00 | 234.60 | 229.55 | 228.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 11:15:00 | 235.40 | 230.72 | 229.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 10:15:00 | 230.30 | 232.07 | 230.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-11 10:15:00 | 230.30 | 232.07 | 230.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 10:15:00 | 230.30 | 232.07 | 230.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-11 11:00:00 | 230.30 | 232.07 | 230.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 11:15:00 | 229.55 | 231.57 | 230.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-11 11:45:00 | 230.65 | 231.57 | 230.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 12:15:00 | 228.85 | 231.02 | 230.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-11 12:45:00 | 228.00 | 231.02 | 230.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2023-10-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-11 14:15:00 | 228.20 | 229.98 | 230.20 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2023-10-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 10:15:00 | 231.15 | 230.31 | 230.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-13 11:15:00 | 232.90 | 231.83 | 231.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-17 10:15:00 | 237.10 | 237.87 | 236.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-17 10:45:00 | 237.50 | 237.87 | 236.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 11:15:00 | 237.55 | 237.80 | 236.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-17 11:45:00 | 236.45 | 237.80 | 236.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 12:15:00 | 238.10 | 237.86 | 236.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-17 12:30:00 | 236.05 | 237.86 | 236.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 13:15:00 | 236.35 | 237.56 | 236.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-17 14:00:00 | 236.35 | 237.56 | 236.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 14:15:00 | 239.30 | 237.91 | 236.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-19 09:45:00 | 241.30 | 238.34 | 237.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-19 10:45:00 | 240.55 | 238.67 | 237.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-19 14:15:00 | 235.65 | 237.45 | 237.39 | SL hit (close<static) qty=1.00 sl=236.05 alert=retest2 |

### Cycle 30 — SELL (started 2023-10-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 15:15:00 | 236.40 | 237.24 | 237.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 11:15:00 | 235.10 | 236.84 | 237.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-20 14:15:00 | 237.80 | 236.05 | 236.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 14:15:00 | 237.80 | 236.05 | 236.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 14:15:00 | 237.80 | 236.05 | 236.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-20 15:00:00 | 237.80 | 236.05 | 236.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 15:15:00 | 237.45 | 236.33 | 236.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-23 09:15:00 | 238.20 | 236.33 | 236.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 09:15:00 | 232.55 | 235.58 | 236.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 10:15:00 | 232.45 | 235.58 | 236.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 11:00:00 | 230.90 | 234.64 | 235.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 15:15:00 | 220.83 | 228.99 | 232.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-25 13:15:00 | 231.00 | 227.08 | 229.81 | SL hit (close>ema200) qty=0.50 sl=227.08 alert=retest2 |

### Cycle 31 — BUY (started 2023-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 09:15:00 | 235.50 | 229.08 | 229.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-27 10:15:00 | 240.30 | 231.32 | 230.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-30 15:15:00 | 239.20 | 239.42 | 236.71 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-31 09:15:00 | 246.15 | 239.42 | 236.71 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 12:15:00 | 239.50 | 242.02 | 240.72 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-11-01 12:15:00 | 239.50 | 242.02 | 240.72 | SL hit (close<ema400) qty=1.00 sl=240.72 alert=retest1 |

### Cycle 32 — SELL (started 2023-11-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-02 11:15:00 | 238.05 | 239.82 | 240.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-02 14:15:00 | 233.55 | 238.03 | 239.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-03 13:15:00 | 236.75 | 236.14 | 237.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-03 13:15:00 | 236.75 | 236.14 | 237.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 13:15:00 | 236.75 | 236.14 | 237.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-03 14:00:00 | 236.75 | 236.14 | 237.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 14:15:00 | 242.40 | 237.39 | 237.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-03 15:00:00 | 242.40 | 237.39 | 237.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2023-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 15:15:00 | 243.00 | 238.51 | 238.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 14:15:00 | 244.20 | 240.57 | 239.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 11:15:00 | 241.75 | 241.90 | 240.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-07 11:45:00 | 241.60 | 241.90 | 240.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 09:15:00 | 250.80 | 248.26 | 246.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-15 09:30:00 | 252.45 | 249.55 | 248.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-15 12:15:00 | 251.90 | 250.11 | 249.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-16 09:15:00 | 252.80 | 250.63 | 249.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-16 11:45:00 | 251.80 | 251.29 | 250.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 11:15:00 | 254.65 | 256.18 | 254.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 12:00:00 | 254.65 | 256.18 | 254.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 12:15:00 | 254.80 | 255.90 | 254.58 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-11-21 10:15:00 | 251.15 | 253.68 | 253.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2023-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-21 10:15:00 | 251.15 | 253.68 | 253.90 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-22 09:15:00 | 258.90 | 254.66 | 254.24 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-11-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-23 14:15:00 | 252.85 | 254.37 | 254.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-24 12:15:00 | 251.25 | 252.93 | 253.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 09:15:00 | 254.00 | 252.58 | 253.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 09:15:00 | 254.00 | 252.58 | 253.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 09:15:00 | 254.00 | 252.58 | 253.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 10:15:00 | 255.35 | 252.58 | 253.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2023-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 10:15:00 | 257.90 | 253.64 | 253.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 10:15:00 | 261.35 | 255.96 | 254.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 14:15:00 | 264.55 | 265.07 | 262.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-01 15:00:00 | 264.55 | 265.07 | 262.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 14:15:00 | 268.05 | 267.06 | 265.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-04 14:30:00 | 266.50 | 267.06 | 265.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 14:15:00 | 270.25 | 270.93 | 269.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-06 14:45:00 | 269.10 | 270.93 | 269.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 15:15:00 | 270.70 | 270.88 | 269.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-07 09:15:00 | 267.90 | 270.88 | 269.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 09:15:00 | 271.60 | 271.02 | 269.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-07 09:30:00 | 271.35 | 271.02 | 269.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 13:15:00 | 284.50 | 285.28 | 282.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-13 09:30:00 | 286.60 | 285.68 | 283.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 14:15:00 | 291.90 | 298.29 | 298.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 291.90 | 298.29 | 298.44 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-12-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-21 12:15:00 | 300.15 | 298.42 | 298.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 09:15:00 | 305.85 | 301.14 | 299.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-22 12:15:00 | 299.25 | 301.52 | 300.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 12:15:00 | 299.25 | 301.52 | 300.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 12:15:00 | 299.25 | 301.52 | 300.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-22 12:30:00 | 300.85 | 301.52 | 300.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 13:15:00 | 300.40 | 301.30 | 300.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-22 13:45:00 | 299.45 | 301.30 | 300.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 14:15:00 | 303.40 | 301.72 | 300.62 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2023-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-26 13:15:00 | 300.05 | 300.27 | 300.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-26 15:15:00 | 298.65 | 299.78 | 300.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-29 10:15:00 | 294.25 | 293.80 | 295.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 10:15:00 | 294.25 | 293.80 | 295.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 10:15:00 | 294.25 | 293.80 | 295.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-29 10:45:00 | 294.60 | 293.80 | 295.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 11:15:00 | 295.30 | 294.10 | 295.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-29 11:45:00 | 295.70 | 294.10 | 295.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 12:15:00 | 293.15 | 293.91 | 295.22 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2024-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-01 11:15:00 | 298.20 | 296.11 | 295.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-01 12:15:00 | 299.20 | 296.72 | 296.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-04 11:15:00 | 308.65 | 308.91 | 305.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-04 12:00:00 | 308.65 | 308.91 | 305.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 12:15:00 | 309.00 | 308.93 | 306.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-04 12:30:00 | 307.55 | 308.93 | 306.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 10:15:00 | 313.00 | 315.81 | 312.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 11:00:00 | 313.00 | 315.81 | 312.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 11:15:00 | 312.75 | 315.20 | 312.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 11:45:00 | 311.10 | 315.20 | 312.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 12:15:00 | 312.75 | 314.71 | 312.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-08 13:30:00 | 315.00 | 314.80 | 313.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-16 13:15:00 | 317.90 | 323.18 | 323.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 13:15:00 | 317.90 | 323.18 | 323.77 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 12:15:00 | 320.25 | 319.03 | 318.89 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 15:15:00 | 317.35 | 318.60 | 318.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 09:15:00 | 315.85 | 318.05 | 318.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 14:15:00 | 315.35 | 313.14 | 314.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 14:15:00 | 315.35 | 313.14 | 314.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 14:15:00 | 315.35 | 313.14 | 314.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 15:00:00 | 315.35 | 313.14 | 314.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 15:15:00 | 315.00 | 313.51 | 314.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:15:00 | 317.05 | 313.51 | 314.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 315.20 | 313.85 | 314.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 10:15:00 | 313.15 | 313.85 | 314.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 12:15:00 | 314.25 | 313.87 | 314.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-29 09:15:00 | 315.55 | 314.69 | 314.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2024-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 09:15:00 | 315.55 | 314.69 | 314.63 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-29 10:15:00 | 313.00 | 314.36 | 314.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-29 13:15:00 | 312.50 | 313.73 | 314.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-30 09:15:00 | 315.40 | 313.48 | 313.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 09:15:00 | 315.40 | 313.48 | 313.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 09:15:00 | 315.40 | 313.48 | 313.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-30 10:00:00 | 315.40 | 313.48 | 313.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 10:15:00 | 314.50 | 313.68 | 313.93 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2024-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 11:15:00 | 316.55 | 314.26 | 314.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 12:15:00 | 319.20 | 316.23 | 315.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 09:15:00 | 316.40 | 316.76 | 315.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-01 09:45:00 | 316.70 | 316.76 | 315.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 10:15:00 | 315.00 | 316.41 | 315.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 10:30:00 | 314.45 | 316.41 | 315.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 11:15:00 | 313.60 | 315.85 | 315.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 12:00:00 | 313.60 | 315.85 | 315.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2024-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 12:15:00 | 312.85 | 315.25 | 315.41 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 09:15:00 | 323.85 | 316.90 | 316.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-05 09:15:00 | 331.90 | 323.19 | 320.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-05 14:15:00 | 327.00 | 327.97 | 324.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-05 15:00:00 | 327.00 | 327.97 | 324.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 10:15:00 | 324.60 | 326.87 | 324.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-06 10:30:00 | 324.95 | 326.87 | 324.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 11:15:00 | 324.90 | 326.47 | 324.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-06 11:30:00 | 324.80 | 326.47 | 324.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 12:15:00 | 327.80 | 326.74 | 324.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-07 09:15:00 | 331.75 | 326.57 | 325.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-07 14:15:00 | 324.10 | 325.97 | 325.61 | SL hit (close<static) qty=1.00 sl=324.55 alert=retest2 |

### Cycle 50 — SELL (started 2024-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-13 10:15:00 | 329.95 | 337.86 | 338.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-13 11:15:00 | 325.10 | 335.30 | 337.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 14:15:00 | 333.95 | 333.50 | 336.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-13 14:15:00 | 333.95 | 333.50 | 336.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 14:15:00 | 333.95 | 333.50 | 336.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 15:00:00 | 333.95 | 333.50 | 336.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 15:15:00 | 336.00 | 334.00 | 336.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 09:30:00 | 339.15 | 335.31 | 336.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 10:15:00 | 342.00 | 336.65 | 336.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 11:00:00 | 342.00 | 336.65 | 336.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2024-02-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 11:15:00 | 345.00 | 338.32 | 337.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 12:15:00 | 347.85 | 340.22 | 338.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-14 15:15:00 | 339.20 | 341.14 | 339.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-14 15:15:00 | 339.20 | 341.14 | 339.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 15:15:00 | 339.20 | 341.14 | 339.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-15 09:15:00 | 338.65 | 341.14 | 339.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 09:15:00 | 350.20 | 342.95 | 340.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-16 14:30:00 | 355.25 | 347.34 | 344.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-21 11:15:00 | 344.70 | 345.89 | 346.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2024-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 11:15:00 | 344.70 | 345.89 | 346.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 13:15:00 | 339.25 | 344.13 | 345.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-23 09:15:00 | 352.30 | 342.28 | 342.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-23 09:15:00 | 352.30 | 342.28 | 342.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 09:15:00 | 352.30 | 342.28 | 342.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 10:00:00 | 352.30 | 342.28 | 342.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2024-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 10:15:00 | 348.85 | 343.59 | 343.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-26 15:15:00 | 355.00 | 351.74 | 349.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-27 09:15:00 | 351.00 | 351.59 | 349.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-27 10:00:00 | 351.00 | 351.59 | 349.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 10:15:00 | 349.60 | 351.20 | 349.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 10:30:00 | 348.50 | 351.20 | 349.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 11:15:00 | 349.15 | 350.79 | 349.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 11:45:00 | 349.75 | 350.79 | 349.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 12:15:00 | 349.65 | 350.56 | 349.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 12:45:00 | 349.85 | 350.56 | 349.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 13:15:00 | 347.75 | 350.00 | 349.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 14:00:00 | 347.75 | 350.00 | 349.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 14:15:00 | 348.85 | 349.77 | 349.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 14:45:00 | 349.55 | 349.77 | 349.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 15:15:00 | 348.10 | 349.43 | 349.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-28 09:15:00 | 349.60 | 349.43 | 349.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-28 09:15:00 | 346.00 | 348.75 | 348.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2024-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 09:15:00 | 346.00 | 348.75 | 348.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 10:15:00 | 341.25 | 347.25 | 348.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 10:15:00 | 341.30 | 341.16 | 343.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 11:00:00 | 341.30 | 341.16 | 343.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 14:15:00 | 349.45 | 342.87 | 343.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 15:00:00 | 349.45 | 342.87 | 343.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2024-02-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 15:15:00 | 357.30 | 345.76 | 345.08 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2024-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 09:15:00 | 345.00 | 352.48 | 352.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 10:15:00 | 342.05 | 346.65 | 348.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 10:15:00 | 304.20 | 300.83 | 313.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 11:00:00 | 304.20 | 300.83 | 313.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 09:15:00 | 295.30 | 296.24 | 300.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 11:00:00 | 294.85 | 295.96 | 300.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 14:15:00 | 294.85 | 295.81 | 299.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 09:45:00 | 294.10 | 295.67 | 298.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-21 10:15:00 | 302.60 | 293.91 | 293.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2024-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 10:15:00 | 302.60 | 293.91 | 293.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 12:15:00 | 304.60 | 297.02 | 295.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 14:15:00 | 303.25 | 304.39 | 301.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-22 15:00:00 | 303.25 | 304.39 | 301.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 299.75 | 303.05 | 301.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 09:30:00 | 298.90 | 303.05 | 301.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 10:15:00 | 296.00 | 301.64 | 300.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 11:00:00 | 296.00 | 301.64 | 300.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2024-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 12:15:00 | 296.70 | 299.90 | 300.02 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-26 15:15:00 | 303.45 | 300.69 | 300.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-27 09:15:00 | 310.75 | 302.70 | 301.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-28 15:15:00 | 310.20 | 311.79 | 309.04 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-01 09:15:00 | 319.90 | 311.79 | 309.04 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-02 09:15:00 | 335.89 | 322.91 | 317.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-04-03 13:15:00 | 330.85 | 331.27 | 326.76 | SL hit (close<ema200) qty=0.50 sl=331.27 alert=retest1 |

### Cycle 60 — SELL (started 2024-04-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 15:15:00 | 325.00 | 326.08 | 326.13 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-04-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 10:15:00 | 329.15 | 326.67 | 326.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 09:15:00 | 334.30 | 329.71 | 328.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 12:15:00 | 339.30 | 343.93 | 341.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 12:15:00 | 339.30 | 343.93 | 341.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 12:15:00 | 339.30 | 343.93 | 341.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 12:45:00 | 340.50 | 343.93 | 341.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 13:15:00 | 338.30 | 342.81 | 341.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 14:15:00 | 337.65 | 342.81 | 341.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 14:15:00 | 339.50 | 342.15 | 341.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 15:00:00 | 339.50 | 342.15 | 341.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2024-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 09:15:00 | 334.00 | 340.17 | 340.45 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2024-04-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 10:15:00 | 341.50 | 336.32 | 336.18 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-04-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 09:15:00 | 332.05 | 336.03 | 336.34 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2024-04-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-19 13:15:00 | 342.20 | 336.73 | 336.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 09:15:00 | 346.80 | 339.54 | 337.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 09:15:00 | 360.50 | 362.65 | 357.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-25 10:00:00 | 360.50 | 362.65 | 357.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 10:15:00 | 360.10 | 362.14 | 358.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 10:45:00 | 359.30 | 362.14 | 358.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 12:15:00 | 358.70 | 361.09 | 358.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 12:45:00 | 359.85 | 361.09 | 358.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 13:15:00 | 363.95 | 361.66 | 358.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 09:15:00 | 364.50 | 360.90 | 358.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 10:45:00 | 364.65 | 362.34 | 359.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 10:15:00 | 365.20 | 362.81 | 361.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 11:15:00 | 365.80 | 362.93 | 361.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 09:15:00 | 364.80 | 365.23 | 363.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 09:30:00 | 364.85 | 365.23 | 363.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 14:15:00 | 363.50 | 364.96 | 364.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 15:00:00 | 363.50 | 364.96 | 364.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 15:15:00 | 361.95 | 364.36 | 363.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 09:15:00 | 359.95 | 364.36 | 363.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-05-02 09:15:00 | 358.05 | 363.10 | 363.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2024-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 09:15:00 | 358.05 | 363.10 | 363.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 11:15:00 | 355.45 | 358.56 | 360.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-03 14:15:00 | 358.50 | 357.65 | 359.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-03 15:00:00 | 358.50 | 357.65 | 359.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 363.25 | 358.83 | 359.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 10:15:00 | 366.50 | 358.83 | 359.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2024-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-06 10:15:00 | 367.85 | 360.63 | 360.41 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2024-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 12:15:00 | 353.55 | 360.87 | 361.59 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2024-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-09 09:15:00 | 372.85 | 361.80 | 361.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 09:15:00 | 377.45 | 372.77 | 369.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 12:15:00 | 373.00 | 374.45 | 371.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-14 13:00:00 | 373.00 | 374.45 | 371.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 13:15:00 | 372.40 | 374.04 | 371.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 14:00:00 | 372.40 | 374.04 | 371.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 14:15:00 | 373.35 | 373.90 | 371.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-15 09:15:00 | 377.55 | 373.72 | 371.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 12:15:00 | 376.40 | 377.81 | 376.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 13:45:00 | 376.30 | 377.34 | 376.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 15:15:00 | 377.00 | 376.95 | 376.14 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 377.00 | 376.96 | 376.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:15:00 | 375.65 | 376.96 | 376.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 379.00 | 377.37 | 376.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 12:15:00 | 381.20 | 378.33 | 377.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 14:15:00 | 382.00 | 379.22 | 377.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-24 10:15:00 | 383.45 | 384.74 | 384.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2024-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 10:15:00 | 383.45 | 384.74 | 384.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 11:15:00 | 382.10 | 384.21 | 384.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 12:15:00 | 382.40 | 382.13 | 383.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-27 13:00:00 | 382.40 | 382.13 | 383.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 13:15:00 | 383.60 | 382.42 | 383.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 13:45:00 | 383.45 | 382.42 | 383.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 14:15:00 | 382.10 | 382.36 | 382.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 15:15:00 | 381.40 | 382.36 | 382.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-31 13:15:00 | 384.30 | 376.36 | 375.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2024-05-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 13:15:00 | 384.30 | 376.36 | 375.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 14:15:00 | 386.80 | 378.45 | 376.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 383.95 | 392.31 | 387.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 383.95 | 392.31 | 387.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 383.95 | 392.31 | 387.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 377.20 | 392.31 | 387.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 351.90 | 384.23 | 383.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 351.90 | 384.23 | 383.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 334.90 | 374.36 | 379.46 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2024-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 12:15:00 | 368.75 | 363.22 | 363.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 13:15:00 | 374.55 | 365.49 | 364.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 14:15:00 | 421.40 | 423.66 | 414.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 15:00:00 | 421.40 | 423.66 | 414.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 415.50 | 419.94 | 415.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:15:00 | 423.20 | 416.46 | 414.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 10:15:00 | 439.15 | 442.23 | 442.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 10:15:00 | 439.15 | 442.23 | 442.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 11:15:00 | 439.00 | 441.59 | 442.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 14:15:00 | 441.25 | 440.22 | 441.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 14:15:00 | 441.25 | 440.22 | 441.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 14:15:00 | 441.25 | 440.22 | 441.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 15:00:00 | 441.25 | 440.22 | 441.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 15:15:00 | 442.00 | 440.57 | 441.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:15:00 | 442.75 | 440.57 | 441.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 441.65 | 440.79 | 441.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 10:30:00 | 439.10 | 441.02 | 441.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 12:45:00 | 439.05 | 440.86 | 441.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 10:45:00 | 439.70 | 434.92 | 436.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 12:15:00 | 442.40 | 437.24 | 437.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2024-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 12:15:00 | 442.40 | 437.24 | 437.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 13:15:00 | 444.45 | 438.68 | 437.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 09:15:00 | 482.65 | 483.88 | 473.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-04 09:45:00 | 481.50 | 483.88 | 473.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 12:15:00 | 522.20 | 530.38 | 521.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 12:45:00 | 522.90 | 530.38 | 521.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 526.65 | 529.64 | 522.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 15:15:00 | 528.50 | 528.91 | 522.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 09:15:00 | 514.55 | 525.97 | 522.18 | SL hit (close<static) qty=1.00 sl=521.60 alert=retest2 |

### Cycle 76 — SELL (started 2024-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 12:15:00 | 512.00 | 519.68 | 519.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 13:15:00 | 505.10 | 516.77 | 518.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 09:15:00 | 491.40 | 491.17 | 496.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-16 09:30:00 | 490.15 | 491.17 | 496.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 472.10 | 478.93 | 484.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:45:00 | 478.50 | 478.93 | 484.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 12:15:00 | 487.40 | 478.47 | 482.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 12:45:00 | 483.90 | 478.47 | 482.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 13:15:00 | 489.85 | 480.74 | 483.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 13:45:00 | 494.65 | 480.74 | 483.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 15:15:00 | 483.00 | 481.08 | 483.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:15:00 | 513.80 | 481.08 | 483.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2024-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 09:15:00 | 530.55 | 490.97 | 487.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 12:15:00 | 541.50 | 524.74 | 519.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 12:15:00 | 551.25 | 551.32 | 542.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 12:30:00 | 552.05 | 551.32 | 542.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 552.25 | 553.78 | 550.39 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2024-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 12:15:00 | 539.90 | 547.00 | 547.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 14:15:00 | 532.55 | 542.91 | 545.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 515.00 | 509.90 | 519.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 515.00 | 509.90 | 519.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 515.00 | 509.90 | 519.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:45:00 | 517.75 | 509.90 | 519.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 506.75 | 505.40 | 512.15 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 09:15:00 | 529.65 | 512.69 | 512.61 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2024-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 13:15:00 | 516.45 | 519.59 | 519.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 14:15:00 | 514.00 | 518.47 | 519.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-13 11:15:00 | 518.00 | 516.48 | 517.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 11:15:00 | 518.00 | 516.48 | 517.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 11:15:00 | 518.00 | 516.48 | 517.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 12:30:00 | 510.00 | 515.16 | 517.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-14 14:15:00 | 519.75 | 516.11 | 515.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2024-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 14:15:00 | 519.75 | 516.11 | 515.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 09:15:00 | 535.45 | 520.28 | 517.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 11:15:00 | 592.10 | 592.56 | 574.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 12:00:00 | 592.10 | 592.56 | 574.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 593.40 | 594.98 | 589.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 11:30:00 | 595.05 | 594.49 | 589.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-22 12:15:00 | 586.15 | 592.82 | 589.52 | SL hit (close<static) qty=1.00 sl=589.00 alert=retest2 |

### Cycle 82 — SELL (started 2024-08-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 12:15:00 | 584.45 | 588.35 | 588.58 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 632.40 | 596.45 | 592.08 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 595.80 | 605.45 | 605.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 15:15:00 | 593.10 | 596.39 | 598.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 11:15:00 | 597.70 | 596.39 | 598.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-03 11:30:00 | 597.00 | 596.39 | 598.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 12:15:00 | 599.05 | 596.92 | 598.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 13:00:00 | 599.05 | 596.92 | 598.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 13:15:00 | 602.95 | 598.13 | 598.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 14:00:00 | 602.95 | 598.13 | 598.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2024-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 14:15:00 | 603.80 | 599.26 | 599.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 09:15:00 | 612.80 | 602.75 | 600.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 10:15:00 | 602.60 | 602.72 | 600.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-04 11:00:00 | 602.60 | 602.72 | 600.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 11:15:00 | 599.80 | 602.14 | 600.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 12:00:00 | 599.80 | 602.14 | 600.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 12:15:00 | 601.85 | 602.08 | 600.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 12:30:00 | 601.80 | 602.08 | 600.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 13:15:00 | 606.55 | 602.97 | 601.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 14:30:00 | 608.00 | 604.60 | 602.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 10:15:00 | 600.80 | 610.71 | 611.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2024-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 10:15:00 | 600.80 | 610.71 | 611.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 12:15:00 | 598.55 | 606.57 | 609.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 14:15:00 | 609.80 | 606.34 | 608.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 14:15:00 | 609.80 | 606.34 | 608.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 609.80 | 606.34 | 608.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 15:00:00 | 609.80 | 606.34 | 608.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 611.20 | 607.31 | 609.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:15:00 | 619.55 | 607.31 | 609.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 618.00 | 609.45 | 609.88 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 10:15:00 | 613.35 | 610.23 | 610.20 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-10 11:15:00 | 607.75 | 609.73 | 609.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 10:15:00 | 606.60 | 608.68 | 609.31 | Break + close below crossover candle low |

### Cycle 89 — BUY (started 2024-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 12:15:00 | 624.95 | 611.68 | 610.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 09:15:00 | 636.90 | 620.15 | 615.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 11:15:00 | 644.05 | 644.62 | 637.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 12:00:00 | 644.05 | 644.62 | 637.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 676.80 | 654.22 | 647.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 15:15:00 | 680.00 | 664.25 | 658.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-20 13:15:00 | 748.00 | 709.67 | 687.06 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 10:15:00 | 709.70 | 717.69 | 718.14 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 14:15:00 | 729.10 | 719.75 | 718.77 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-09-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 12:15:00 | 714.50 | 718.71 | 718.75 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 14:15:00 | 722.00 | 719.34 | 719.03 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 15:15:00 | 714.70 | 718.41 | 718.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 09:15:00 | 700.80 | 714.89 | 717.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 717.80 | 709.72 | 712.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 717.80 | 709.72 | 712.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 717.80 | 709.72 | 712.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:00:00 | 717.80 | 709.72 | 712.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 721.35 | 712.05 | 713.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:45:00 | 717.25 | 712.05 | 713.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 11:15:00 | 715.90 | 712.82 | 713.41 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2024-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 12:15:00 | 719.30 | 714.11 | 713.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 14:15:00 | 731.00 | 718.43 | 715.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 10:15:00 | 716.75 | 719.48 | 717.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 10:15:00 | 716.75 | 719.48 | 717.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 10:15:00 | 716.75 | 719.48 | 717.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 11:00:00 | 716.75 | 719.48 | 717.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 707.75 | 717.14 | 716.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 12:00:00 | 707.75 | 717.14 | 716.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2024-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 12:15:00 | 704.95 | 714.70 | 715.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 13:15:00 | 702.65 | 712.29 | 714.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 11:15:00 | 723.00 | 710.38 | 711.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 11:15:00 | 723.00 | 710.38 | 711.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 723.00 | 710.38 | 711.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 12:00:00 | 723.00 | 710.38 | 711.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 722.00 | 712.70 | 712.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 12:45:00 | 724.25 | 712.70 | 712.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2024-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 13:15:00 | 719.85 | 714.13 | 713.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-04 15:15:00 | 727.00 | 717.44 | 715.13 | Break + close above crossover candle high |

### Cycle 98 — SELL (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 10:15:00 | 690.85 | 712.78 | 713.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 13:15:00 | 671.85 | 695.49 | 704.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 685.85 | 684.83 | 696.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 09:45:00 | 686.65 | 684.83 | 696.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 702.00 | 688.26 | 697.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 702.00 | 688.26 | 697.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 707.55 | 692.12 | 698.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:30:00 | 715.00 | 692.12 | 698.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 14:15:00 | 724.00 | 705.41 | 703.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 15:15:00 | 735.00 | 711.32 | 706.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 10:15:00 | 745.40 | 748.10 | 738.02 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 13:15:00 | 751.75 | 747.63 | 739.52 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 13:45:00 | 754.20 | 748.39 | 740.61 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 750.50 | 749.53 | 743.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 10:30:00 | 756.85 | 751.08 | 744.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 09:15:00 | 789.34 | 771.25 | 762.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 09:15:00 | 791.91 | 771.25 | 762.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-16 11:15:00 | 766.25 | 770.57 | 763.71 | SL hit (close<ema200) qty=0.50 sl=770.57 alert=retest1 |

### Cycle 100 — SELL (started 2024-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 09:15:00 | 752.90 | 765.05 | 765.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 15:15:00 | 731.00 | 745.86 | 754.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 722.30 | 713.47 | 729.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 10:00:00 | 722.30 | 713.47 | 729.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 676.50 | 667.14 | 679.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 12:15:00 | 681.00 | 667.14 | 679.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 12:15:00 | 681.40 | 670.00 | 679.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 13:00:00 | 681.40 | 670.00 | 679.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 13:15:00 | 681.00 | 672.20 | 679.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 13:30:00 | 682.15 | 672.20 | 679.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 14:15:00 | 680.90 | 673.94 | 679.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 15:15:00 | 685.00 | 673.94 | 679.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 15:15:00 | 685.00 | 676.15 | 680.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 09:15:00 | 680.95 | 676.15 | 680.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 680.00 | 676.92 | 680.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 10:30:00 | 671.80 | 675.38 | 679.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 11:45:00 | 671.50 | 675.90 | 678.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 12:30:00 | 674.60 | 676.14 | 678.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 13:15:00 | 673.40 | 676.14 | 678.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 13:15:00 | 679.10 | 676.73 | 678.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 14:00:00 | 679.10 | 676.73 | 678.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 684.55 | 678.30 | 679.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 15:00:00 | 684.55 | 678.30 | 679.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 15:15:00 | 685.35 | 679.71 | 679.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:15:00 | 696.50 | 679.71 | 679.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-30 09:15:00 | 688.15 | 681.40 | 680.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 688.15 | 681.40 | 680.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 09:15:00 | 728.00 | 698.23 | 690.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 722.05 | 735.61 | 719.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 722.05 | 735.61 | 719.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 722.05 | 735.61 | 719.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 722.05 | 735.61 | 719.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 719.95 | 732.48 | 719.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 719.95 | 732.48 | 719.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 723.05 | 730.59 | 720.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 13:45:00 | 727.95 | 729.48 | 721.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 14:45:00 | 731.35 | 729.38 | 721.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:15:00 | 728.10 | 728.30 | 722.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:45:00 | 729.40 | 729.64 | 723.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 749.75 | 754.63 | 748.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 14:00:00 | 749.75 | 754.63 | 748.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 750.00 | 753.70 | 748.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 14:30:00 | 747.95 | 753.70 | 748.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 748.95 | 752.75 | 748.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 748.50 | 752.75 | 748.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 750.35 | 752.27 | 748.93 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-08 14:15:00 | 740.40 | 746.08 | 746.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2024-11-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 14:15:00 | 740.40 | 746.08 | 746.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 15:15:00 | 733.00 | 743.47 | 745.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 745.65 | 743.44 | 745.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 10:15:00 | 745.65 | 743.44 | 745.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 745.65 | 743.44 | 745.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:00:00 | 745.65 | 743.44 | 745.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 743.50 | 743.45 | 744.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 13:30:00 | 738.30 | 741.97 | 744.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 10:15:00 | 701.38 | 721.78 | 731.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-11-14 13:15:00 | 664.47 | 681.18 | 698.37 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 103 — BUY (started 2024-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 11:15:00 | 675.80 | 671.90 | 671.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 13:15:00 | 677.10 | 673.54 | 672.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 11:15:00 | 673.05 | 676.42 | 674.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 11:15:00 | 673.05 | 676.42 | 674.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 11:15:00 | 673.05 | 676.42 | 674.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 12:00:00 | 673.05 | 676.42 | 674.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 12:15:00 | 671.80 | 675.50 | 674.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 13:00:00 | 671.80 | 675.50 | 674.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 673.00 | 674.04 | 673.88 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2024-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-25 15:15:00 | 671.65 | 673.56 | 673.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 09:15:00 | 662.00 | 671.25 | 672.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 10:15:00 | 668.50 | 665.38 | 667.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 10:15:00 | 668.50 | 665.38 | 667.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 668.50 | 665.38 | 667.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:30:00 | 671.95 | 665.38 | 667.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 668.25 | 665.95 | 667.96 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2024-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 15:15:00 | 674.00 | 669.56 | 669.20 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 10:15:00 | 663.70 | 668.62 | 668.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 12:15:00 | 662.55 | 667.13 | 668.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 14:15:00 | 669.05 | 666.36 | 667.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 14:15:00 | 669.05 | 666.36 | 667.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 669.05 | 666.36 | 667.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:45:00 | 666.80 | 666.36 | 667.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 667.70 | 666.63 | 667.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 09:30:00 | 664.00 | 666.18 | 667.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 10:00:00 | 664.40 | 666.18 | 667.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 11:30:00 | 663.65 | 665.40 | 666.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 14:15:00 | 677.80 | 668.33 | 667.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2024-11-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 14:15:00 | 677.80 | 668.33 | 667.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 09:15:00 | 701.55 | 675.48 | 671.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 13:15:00 | 723.15 | 725.27 | 717.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 13:45:00 | 719.40 | 725.27 | 717.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 736.45 | 738.36 | 735.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:15:00 | 730.00 | 738.36 | 735.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 730.00 | 736.69 | 735.17 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2024-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 12:15:00 | 729.00 | 733.24 | 733.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 14:15:00 | 727.25 | 731.19 | 732.72 | Break + close below crossover candle low |

### Cycle 109 — BUY (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 09:15:00 | 746.00 | 733.58 | 733.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-12 11:15:00 | 751.70 | 739.08 | 736.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 09:15:00 | 741.55 | 746.46 | 741.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 09:15:00 | 741.55 | 746.46 | 741.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 741.55 | 746.46 | 741.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:45:00 | 739.30 | 746.46 | 741.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 739.70 | 745.11 | 741.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:30:00 | 736.65 | 745.11 | 741.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 748.00 | 745.69 | 742.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 13:45:00 | 751.00 | 746.56 | 743.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 14:30:00 | 751.50 | 749.19 | 744.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-17 09:15:00 | 826.10 | 790.50 | 771.79 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2024-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 12:15:00 | 821.25 | 828.55 | 829.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 10:15:00 | 814.65 | 822.14 | 825.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 14:15:00 | 818.70 | 815.99 | 821.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-26 15:00:00 | 818.70 | 815.99 | 821.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 823.00 | 817.39 | 821.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:30:00 | 830.40 | 818.34 | 821.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 820.45 | 818.76 | 821.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:30:00 | 822.70 | 818.76 | 821.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 829.50 | 820.85 | 821.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 14:00:00 | 829.50 | 820.85 | 821.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 823.45 | 821.37 | 821.75 | EMA400 retest candle locked (from downside) |

### Cycle 111 — BUY (started 2024-12-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 15:15:00 | 826.95 | 822.49 | 822.22 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 11:15:00 | 818.75 | 822.03 | 822.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 13:15:00 | 815.25 | 820.31 | 821.32 | Break + close below crossover candle low |

### Cycle 113 — BUY (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 14:15:00 | 841.50 | 824.55 | 823.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 15:15:00 | 850.00 | 829.64 | 825.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 09:15:00 | 829.55 | 829.62 | 825.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-31 10:00:00 | 829.55 | 829.62 | 825.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 14:15:00 | 871.80 | 864.96 | 856.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 14:30:00 | 858.00 | 864.96 | 856.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 865.95 | 872.52 | 866.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:15:00 | 857.70 | 872.52 | 866.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 854.85 | 868.98 | 865.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:45:00 | 855.35 | 868.98 | 865.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 860.20 | 867.23 | 865.03 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 13:15:00 | 854.10 | 862.97 | 863.37 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 09:15:00 | 875.00 | 863.28 | 863.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-07 12:15:00 | 881.95 | 869.61 | 866.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 10:15:00 | 914.10 | 916.78 | 900.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-09 10:45:00 | 913.10 | 916.78 | 900.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 12:15:00 | 904.35 | 912.69 | 901.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 12:45:00 | 907.85 | 912.69 | 901.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 13:15:00 | 900.80 | 910.32 | 901.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 14:00:00 | 900.80 | 910.32 | 901.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 14:15:00 | 907.85 | 909.82 | 902.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 09:15:00 | 910.70 | 908.46 | 902.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-10 09:15:00 | 882.75 | 903.32 | 900.38 | SL hit (close<static) qty=1.00 sl=896.50 alert=retest2 |

### Cycle 116 — SELL (started 2025-01-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 13:15:00 | 887.15 | 899.33 | 899.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 15:15:00 | 885.00 | 894.61 | 897.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 10:15:00 | 851.40 | 835.55 | 847.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-15 10:15:00 | 851.40 | 835.55 | 847.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 851.40 | 835.55 | 847.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:45:00 | 849.50 | 835.55 | 847.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 11:15:00 | 857.40 | 839.92 | 848.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 12:00:00 | 857.40 | 839.92 | 848.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-01-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 13:15:00 | 877.00 | 854.07 | 854.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 14:15:00 | 896.00 | 862.46 | 857.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 10:15:00 | 884.95 | 885.29 | 876.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 11:00:00 | 884.95 | 885.29 | 876.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 894.25 | 904.77 | 896.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 894.25 | 904.77 | 896.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 877.20 | 899.26 | 894.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 877.20 | 899.26 | 894.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 876.80 | 894.76 | 892.91 | EMA400 retest candle locked (from upside) |

### Cycle 118 — SELL (started 2025-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 12:15:00 | 868.95 | 889.60 | 890.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 842.60 | 871.71 | 881.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 868.45 | 847.74 | 860.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 868.45 | 847.74 | 860.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 868.45 | 847.74 | 860.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 868.45 | 847.74 | 860.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 866.50 | 851.49 | 861.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 11:30:00 | 859.10 | 853.47 | 861.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 14:15:00 | 816.14 | 833.40 | 845.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-27 09:15:00 | 773.19 | 811.29 | 832.88 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 119 — BUY (started 2025-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 10:15:00 | 616.75 | 600.62 | 600.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 13:15:00 | 627.35 | 609.26 | 604.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 09:15:00 | 601.10 | 615.59 | 609.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 09:15:00 | 601.10 | 615.59 | 609.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 601.10 | 615.59 | 609.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 10:00:00 | 601.10 | 615.59 | 609.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 583.25 | 609.12 | 606.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 11:00:00 | 583.25 | 609.12 | 606.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 11:15:00 | 581.15 | 603.52 | 604.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 13:15:00 | 573.40 | 585.63 | 593.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 590.80 | 582.62 | 589.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 09:15:00 | 590.80 | 582.62 | 589.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 590.80 | 582.62 | 589.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:45:00 | 594.05 | 582.62 | 589.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 587.65 | 583.62 | 589.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:45:00 | 587.50 | 583.62 | 589.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 14:15:00 | 585.75 | 584.23 | 587.87 | EMA400 retest candle locked (from downside) |

### Cycle 121 — BUY (started 2025-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 09:15:00 | 613.45 | 590.04 | 589.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 14:15:00 | 620.25 | 606.86 | 600.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-11 09:15:00 | 591.05 | 619.71 | 613.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-11 09:15:00 | 591.05 | 619.71 | 613.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 591.05 | 619.71 | 613.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 09:45:00 | 586.90 | 619.71 | 613.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 10:15:00 | 590.15 | 613.80 | 611.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 10:45:00 | 591.20 | 613.80 | 611.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 11:15:00 | 590.00 | 609.04 | 609.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-12 09:15:00 | 561.45 | 591.06 | 599.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 597.20 | 585.21 | 591.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 597.20 | 585.21 | 591.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 597.20 | 585.21 | 591.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 597.20 | 585.21 | 591.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 599.85 | 588.14 | 592.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:45:00 | 605.20 | 588.14 | 592.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2025-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 14:15:00 | 597.85 | 595.04 | 594.71 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 09:15:00 | 581.75 | 592.53 | 593.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 10:15:00 | 571.35 | 588.29 | 591.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-19 09:15:00 | 541.60 | 518.20 | 531.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 09:15:00 | 541.60 | 518.20 | 531.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 541.60 | 518.20 | 531.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 541.60 | 518.20 | 531.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 540.30 | 522.62 | 532.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:30:00 | 546.80 | 522.62 | 532.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 15:15:00 | 539.00 | 534.51 | 535.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 09:15:00 | 534.90 | 534.51 | 535.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 09:45:00 | 532.80 | 533.01 | 534.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 11:45:00 | 537.15 | 534.92 | 535.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 12:15:00 | 536.70 | 534.92 | 535.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-20 13:15:00 | 542.90 | 536.87 | 536.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 13:15:00 | 542.90 | 536.87 | 536.05 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 10:15:00 | 523.45 | 533.59 | 534.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 13:15:00 | 515.75 | 527.38 | 531.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 517.30 | 516.06 | 521.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 09:15:00 | 517.30 | 516.06 | 521.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 517.30 | 516.06 | 521.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:30:00 | 520.35 | 516.06 | 521.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 457.65 | 450.86 | 462.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:45:00 | 469.10 | 450.86 | 462.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 449.80 | 448.43 | 456.76 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2025-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 14:15:00 | 463.00 | 459.80 | 459.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 10:15:00 | 468.60 | 462.86 | 461.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 14:15:00 | 465.00 | 465.35 | 463.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 15:00:00 | 465.00 | 465.35 | 463.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 15:15:00 | 464.00 | 465.08 | 463.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:15:00 | 474.20 | 465.08 | 463.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 470.25 | 466.11 | 463.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 14:00:00 | 487.00 | 470.93 | 466.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 10:30:00 | 483.75 | 486.35 | 482.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 12:15:00 | 489.85 | 493.72 | 493.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2025-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 12:15:00 | 489.85 | 493.72 | 493.90 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 502.35 | 495.32 | 494.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 510.70 | 501.53 | 497.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 15:15:00 | 517.00 | 520.98 | 515.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 09:15:00 | 530.70 | 520.98 | 515.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 546.45 | 526.08 | 518.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 11:00:00 | 554.85 | 541.88 | 531.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 11:15:00 | 522.05 | 532.65 | 532.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 11:15:00 | 522.05 | 532.65 | 532.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 09:15:00 | 516.50 | 523.78 | 527.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 13:15:00 | 499.25 | 498.90 | 508.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 13:45:00 | 500.60 | 498.90 | 508.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 510.90 | 501.30 | 508.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 510.90 | 501.30 | 508.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 508.50 | 502.74 | 508.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 506.25 | 502.74 | 508.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 502.75 | 502.74 | 507.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 10:45:00 | 499.55 | 502.30 | 507.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 09:15:00 | 474.57 | 482.41 | 490.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-02 10:15:00 | 490.65 | 484.06 | 490.81 | SL hit (close>ema200) qty=0.50 sl=484.06 alert=retest2 |

### Cycle 131 — BUY (started 2025-04-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 14:15:00 | 439.15 | 432.62 | 431.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 15:15:00 | 440.65 | 434.23 | 432.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 466.00 | 468.15 | 460.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 09:30:00 | 466.00 | 468.15 | 460.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 469.40 | 470.26 | 465.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 12:15:00 | 497.35 | 475.37 | 468.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 10:30:00 | 490.40 | 494.52 | 493.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 13:15:00 | 490.60 | 492.43 | 492.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-24 13:15:00 | 488.10 | 491.56 | 491.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2025-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 13:15:00 | 488.10 | 491.56 | 491.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 14:15:00 | 487.75 | 490.80 | 491.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 09:15:00 | 463.70 | 460.41 | 467.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 463.70 | 460.41 | 467.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 463.70 | 460.41 | 467.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:30:00 | 463.95 | 460.41 | 467.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 465.55 | 462.36 | 465.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 15:00:00 | 465.55 | 462.36 | 465.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 466.45 | 463.18 | 465.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 09:15:00 | 458.65 | 463.18 | 465.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 15:15:00 | 462.40 | 458.98 | 458.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2025-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 15:15:00 | 462.40 | 458.98 | 458.75 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 455.35 | 458.25 | 458.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 11:15:00 | 453.75 | 456.94 | 457.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 451.80 | 448.11 | 452.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 11:00:00 | 451.80 | 448.11 | 452.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 449.00 | 448.28 | 451.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 12:45:00 | 446.70 | 447.94 | 451.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:00:00 | 447.50 | 449.59 | 450.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 424.36 | 439.86 | 445.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 425.12 | 439.86 | 445.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 15:15:00 | 431.50 | 428.21 | 435.66 | SL hit (close>ema200) qty=0.50 sl=428.21 alert=retest2 |

### Cycle 135 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 454.35 | 440.06 | 439.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 471.60 | 452.62 | 446.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 10:15:00 | 459.65 | 460.22 | 455.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 11:00:00 | 459.65 | 460.22 | 455.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 506.70 | 513.65 | 506.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 506.70 | 513.65 | 506.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 507.95 | 512.51 | 506.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 10:00:00 | 511.35 | 511.40 | 507.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 13:00:00 | 510.25 | 511.28 | 508.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 14:30:00 | 514.70 | 511.93 | 509.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 14:15:00 | 516.25 | 518.35 | 518.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 14:15:00 | 516.25 | 518.35 | 518.64 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 09:15:00 | 520.85 | 519.09 | 518.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 11:15:00 | 538.50 | 524.14 | 521.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 15:15:00 | 580.00 | 581.37 | 571.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-04 09:15:00 | 575.55 | 581.37 | 571.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 570.25 | 579.15 | 571.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 569.70 | 579.15 | 571.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 574.20 | 578.16 | 571.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:15:00 | 581.40 | 574.19 | 572.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 12:30:00 | 576.35 | 575.65 | 573.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 10:15:00 | 581.90 | 573.48 | 573.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:15:00 | 579.50 | 574.84 | 574.11 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 592.55 | 578.38 | 575.78 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-10 15:15:00 | 578.00 | 579.31 | 579.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2025-06-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 15:15:00 | 578.00 | 579.31 | 579.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 10:15:00 | 575.40 | 578.15 | 578.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 12:15:00 | 578.10 | 577.59 | 578.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 12:15:00 | 578.10 | 577.59 | 578.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 578.10 | 577.59 | 578.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:30:00 | 579.45 | 577.59 | 578.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 571.60 | 576.39 | 577.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 11:45:00 | 567.00 | 572.41 | 575.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-13 09:15:00 | 510.30 | 558.73 | 566.85 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 536.50 | 525.14 | 523.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 12:15:00 | 546.45 | 531.65 | 526.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 13:15:00 | 564.00 | 564.21 | 557.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 14:00:00 | 564.00 | 564.21 | 557.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 553.30 | 562.03 | 557.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 553.30 | 562.03 | 557.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 556.95 | 561.01 | 557.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 562.55 | 561.01 | 557.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 10:15:00 | 554.40 | 561.61 | 561.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 554.40 | 561.61 | 561.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 09:15:00 | 552.10 | 555.50 | 558.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 547.00 | 545.57 | 549.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 14:45:00 | 546.70 | 545.57 | 549.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 547.10 | 546.03 | 549.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 547.50 | 546.03 | 549.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 540.90 | 543.80 | 546.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 11:00:00 | 539.50 | 542.94 | 545.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 14:15:00 | 550.45 | 543.63 | 544.97 | SL hit (close>static) qty=1.00 sl=548.00 alert=retest2 |

### Cycle 141 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 549.75 | 546.03 | 545.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 09:15:00 | 560.15 | 550.29 | 548.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 11:15:00 | 605.00 | 607.11 | 597.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 12:00:00 | 605.00 | 607.11 | 597.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 602.50 | 604.80 | 599.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:15:00 | 596.60 | 604.80 | 599.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 592.55 | 602.35 | 598.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 592.55 | 602.35 | 598.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 594.50 | 600.78 | 598.54 | EMA400 retest candle locked (from upside) |

### Cycle 142 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 590.85 | 596.36 | 596.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 15:15:00 | 588.30 | 593.57 | 595.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 10:15:00 | 565.05 | 564.84 | 570.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-24 10:45:00 | 565.85 | 564.84 | 570.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 565.80 | 565.55 | 569.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:30:00 | 567.10 | 565.55 | 569.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 591.20 | 569.63 | 570.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:30:00 | 597.40 | 569.63 | 570.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-25 10:15:00 | 582.65 | 572.24 | 571.57 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 12:15:00 | 565.85 | 572.96 | 573.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 13:15:00 | 561.65 | 570.70 | 571.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 10:15:00 | 567.50 | 567.26 | 569.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 11:00:00 | 567.50 | 567.26 | 569.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 571.00 | 568.00 | 569.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:45:00 | 571.85 | 568.00 | 569.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 572.80 | 568.96 | 570.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:30:00 | 571.80 | 568.96 | 570.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 573.05 | 569.78 | 570.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:45:00 | 573.45 | 569.78 | 570.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 579.00 | 571.62 | 571.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 10:15:00 | 584.90 | 575.83 | 573.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 569.20 | 577.06 | 575.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 569.20 | 577.06 | 575.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 569.20 | 577.06 | 575.49 | EMA400 retest candle locked (from upside) |

### Cycle 146 — SELL (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 13:15:00 | 571.40 | 574.21 | 574.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 565.40 | 572.45 | 573.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 564.40 | 564.02 | 566.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 12:15:00 | 564.40 | 564.02 | 566.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 564.40 | 564.02 | 566.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:30:00 | 565.05 | 564.02 | 566.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 565.30 | 564.14 | 565.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:30:00 | 568.10 | 564.14 | 565.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 564.80 | 564.27 | 565.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:30:00 | 564.95 | 564.27 | 565.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 565.30 | 564.10 | 565.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:30:00 | 566.00 | 564.10 | 565.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 560.05 | 563.29 | 564.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 14:45:00 | 558.80 | 562.93 | 564.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 15:15:00 | 559.40 | 562.93 | 564.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:30:00 | 559.30 | 560.19 | 563.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 15:15:00 | 531.43 | 539.31 | 544.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:15:00 | 530.86 | 537.27 | 542.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:15:00 | 531.33 | 537.27 | 542.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 10:15:00 | 537.65 | 537.34 | 542.50 | SL hit (close>ema200) qty=0.50 sl=537.34 alert=retest2 |

### Cycle 147 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 547.90 | 539.21 | 538.86 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 14:15:00 | 538.60 | 541.17 | 541.42 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 09:15:00 | 553.30 | 543.29 | 542.33 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 11:15:00 | 544.00 | 546.63 | 546.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 13:15:00 | 543.35 | 545.48 | 546.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 522.70 | 519.51 | 524.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 522.70 | 519.51 | 524.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 522.70 | 519.51 | 524.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 526.00 | 519.51 | 524.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 522.70 | 520.13 | 522.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 525.55 | 520.13 | 522.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 529.70 | 522.04 | 523.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:45:00 | 531.35 | 522.04 | 523.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 534.30 | 524.49 | 524.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 11:15:00 | 536.00 | 526.79 | 525.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 539.25 | 540.62 | 536.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 10:15:00 | 539.00 | 540.62 | 536.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 536.40 | 539.77 | 536.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 536.40 | 539.77 | 536.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 536.70 | 539.16 | 536.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:45:00 | 541.50 | 537.12 | 536.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 533.30 | 536.25 | 535.81 | SL hit (close<static) qty=1.00 sl=535.10 alert=retest2 |

### Cycle 152 — SELL (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 12:15:00 | 529.20 | 534.84 | 535.21 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 536.85 | 535.43 | 535.33 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 11:15:00 | 532.65 | 534.81 | 535.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 12:15:00 | 531.75 | 534.19 | 534.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 533.45 | 524.35 | 527.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 533.45 | 524.35 | 527.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 533.45 | 524.35 | 527.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:00:00 | 533.45 | 524.35 | 527.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 533.00 | 526.08 | 527.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:30:00 | 535.35 | 526.08 | 527.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — BUY (started 2025-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 14:15:00 | 531.65 | 529.25 | 529.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 15:15:00 | 535.00 | 530.40 | 529.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 530.10 | 530.34 | 529.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 530.10 | 530.34 | 529.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 530.10 | 530.34 | 529.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:30:00 | 531.55 | 530.34 | 529.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 530.45 | 530.36 | 529.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:45:00 | 529.65 | 530.36 | 529.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 529.80 | 530.25 | 529.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:45:00 | 529.55 | 530.25 | 529.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 528.65 | 529.93 | 529.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:00:00 | 528.65 | 529.93 | 529.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 525.90 | 529.12 | 529.27 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 534.45 | 529.89 | 529.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 566.65 | 539.10 | 534.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 13:15:00 | 585.65 | 587.15 | 578.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 14:00:00 | 585.65 | 587.15 | 578.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 588.00 | 587.23 | 584.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 10:15:00 | 600.10 | 587.23 | 584.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-22 09:15:00 | 660.11 | 627.33 | 608.13 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 158 — SELL (started 2025-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 12:15:00 | 701.65 | 704.21 | 704.47 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 14:15:00 | 734.25 | 710.05 | 707.07 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 701.40 | 706.43 | 707.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 10:15:00 | 696.90 | 702.30 | 704.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 698.00 | 697.72 | 701.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 15:00:00 | 698.00 | 697.72 | 701.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 695.45 | 696.67 | 700.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 11:15:00 | 691.80 | 696.14 | 699.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 15:00:00 | 692.55 | 693.15 | 696.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 10:00:00 | 692.50 | 693.04 | 696.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 14:15:00 | 657.21 | 668.04 | 678.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 14:15:00 | 657.92 | 668.04 | 678.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 14:15:00 | 657.88 | 668.04 | 678.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 673.80 | 667.48 | 676.56 | SL hit (close>ema200) qty=0.50 sl=667.48 alert=retest2 |

### Cycle 161 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 636.90 | 628.05 | 627.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 09:15:00 | 646.70 | 636.03 | 632.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 14:15:00 | 642.70 | 644.93 | 641.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 15:00:00 | 642.70 | 644.93 | 641.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 637.45 | 643.43 | 641.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:30:00 | 637.75 | 643.43 | 641.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 641.70 | 643.08 | 641.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 12:15:00 | 645.85 | 642.39 | 641.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 627.80 | 643.10 | 643.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 627.80 | 643.10 | 643.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 11:15:00 | 623.70 | 636.61 | 640.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 14:15:00 | 622.30 | 616.83 | 624.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 15:00:00 | 622.30 | 616.83 | 624.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 629.80 | 620.25 | 625.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 636.35 | 620.25 | 625.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 625.95 | 623.44 | 625.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:30:00 | 626.30 | 623.44 | 625.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 626.75 | 624.10 | 625.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 14:00:00 | 626.75 | 624.10 | 625.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 627.25 | 624.73 | 625.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 15:00:00 | 627.25 | 624.73 | 625.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 626.00 | 624.98 | 625.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 618.70 | 624.98 | 625.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 634.75 | 620.35 | 619.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 634.75 | 620.35 | 619.13 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 15:15:00 | 621.55 | 623.70 | 623.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 617.95 | 622.55 | 623.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 625.65 | 619.93 | 620.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 625.65 | 619.93 | 620.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 625.65 | 619.93 | 620.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:45:00 | 627.55 | 619.93 | 620.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 624.85 | 620.91 | 621.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:30:00 | 626.00 | 620.91 | 621.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 603.00 | 611.67 | 615.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:30:00 | 605.20 | 611.67 | 615.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 610.45 | 594.90 | 597.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:45:00 | 610.60 | 594.90 | 597.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 610.00 | 597.92 | 598.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:30:00 | 611.90 | 597.92 | 598.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 607.75 | 600.74 | 600.06 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 11:15:00 | 594.85 | 599.63 | 600.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 12:15:00 | 589.00 | 597.51 | 599.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 582.50 | 580.80 | 586.54 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 10:15:00 | 578.60 | 580.80 | 586.54 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 11:30:00 | 578.95 | 580.22 | 585.28 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 12:30:00 | 578.00 | 579.93 | 584.69 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:15:00 | 549.67 | 557.28 | 566.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:15:00 | 550.00 | 557.28 | 566.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:15:00 | 549.10 | 557.28 | 566.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-12-03 15:15:00 | 556.20 | 554.69 | 561.92 | SL hit (close>ema200) qty=0.50 sl=554.69 alert=retest1 |

### Cycle 167 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 533.20 | 514.45 | 512.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 11:15:00 | 543.75 | 520.31 | 515.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 541.70 | 546.21 | 537.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 541.70 | 546.21 | 537.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 541.70 | 546.21 | 537.85 | EMA400 retest candle locked (from upside) |

### Cycle 168 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 528.30 | 535.88 | 536.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 525.20 | 533.75 | 535.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 532.20 | 529.78 | 532.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 12:00:00 | 532.20 | 529.78 | 532.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 530.55 | 529.94 | 532.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:30:00 | 532.00 | 529.94 | 532.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 531.35 | 530.22 | 532.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:45:00 | 533.40 | 530.22 | 532.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 532.60 | 530.70 | 532.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 532.60 | 530.70 | 532.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 530.60 | 530.68 | 532.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 535.15 | 530.68 | 532.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 540.00 | 532.54 | 532.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 541.80 | 532.54 | 532.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — BUY (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 10:15:00 | 544.60 | 534.95 | 533.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 551.75 | 541.54 | 537.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 557.30 | 557.97 | 553.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 15:00:00 | 557.30 | 557.97 | 553.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 561.00 | 563.57 | 559.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:00:00 | 561.00 | 563.57 | 559.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 559.45 | 562.23 | 559.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:00:00 | 559.45 | 562.23 | 559.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 559.35 | 561.65 | 559.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:45:00 | 557.05 | 561.65 | 559.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 553.20 | 559.96 | 559.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:00:00 | 553.20 | 559.96 | 559.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 552.65 | 558.50 | 558.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 545.60 | 555.20 | 557.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 537.25 | 533.59 | 541.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 14:00:00 | 537.25 | 533.59 | 541.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 545.15 | 535.90 | 541.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 545.15 | 535.90 | 541.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 548.00 | 538.32 | 542.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 09:30:00 | 541.70 | 539.23 | 542.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 10:15:00 | 554.90 | 542.36 | 543.24 | SL hit (close>static) qty=1.00 sl=548.40 alert=retest2 |

### Cycle 171 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 556.35 | 545.16 | 544.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 560.65 | 554.01 | 550.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 585.35 | 586.76 | 577.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 11:00:00 | 585.35 | 586.76 | 577.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 591.90 | 586.58 | 581.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 10:15:00 | 592.95 | 586.58 | 581.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 10:45:00 | 592.55 | 587.58 | 582.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 11:15:00 | 592.05 | 587.58 | 582.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 12:00:00 | 592.45 | 588.56 | 583.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 597.00 | 591.41 | 586.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 10:30:00 | 606.60 | 593.26 | 587.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 12:15:00 | 578.25 | 589.57 | 587.17 | SL hit (close<static) qty=1.00 sl=586.00 alert=retest2 |

### Cycle 172 — SELL (started 2026-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 14:15:00 | 572.80 | 584.58 | 585.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 15:15:00 | 569.65 | 581.59 | 583.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 13:15:00 | 558.25 | 552.74 | 561.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 13:15:00 | 558.25 | 552.74 | 561.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 558.25 | 552.74 | 561.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:00:00 | 558.25 | 552.74 | 561.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 566.15 | 555.42 | 562.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 15:00:00 | 566.15 | 555.42 | 562.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 571.70 | 558.67 | 563.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 566.45 | 558.67 | 563.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 565.65 | 561.34 | 563.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:15:00 | 563.80 | 561.34 | 563.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:00:00 | 563.70 | 563.25 | 563.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 11:00:00 | 560.60 | 562.72 | 563.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 535.61 | 541.31 | 547.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 535.51 | 541.31 | 547.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 532.57 | 541.31 | 547.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 09:15:00 | 527.05 | 526.98 | 535.93 | SL hit (close>ema200) qty=0.50 sl=526.98 alert=retest2 |

### Cycle 173 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 535.50 | 531.82 | 531.60 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 09:15:00 | 530.00 | 531.45 | 531.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 10:15:00 | 524.40 | 530.04 | 530.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 14:15:00 | 489.10 | 487.26 | 497.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 15:00:00 | 489.10 | 487.26 | 497.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 492.45 | 489.38 | 496.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:15:00 | 490.20 | 489.38 | 496.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 15:00:00 | 489.50 | 491.32 | 494.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 10:15:00 | 512.65 | 497.02 | 496.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 512.65 | 497.02 | 496.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 514.20 | 505.90 | 501.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-03 15:15:00 | 562.85 | 564.10 | 551.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 09:15:00 | 554.50 | 564.10 | 551.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 564.00 | 564.08 | 552.26 | EMA400 retest candle locked (from upside) |

### Cycle 176 — SELL (started 2026-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 13:15:00 | 545.85 | 551.53 | 551.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 539.20 | 547.91 | 550.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 15:15:00 | 554.00 | 546.17 | 547.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 15:15:00 | 554.00 | 546.17 | 547.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 554.00 | 546.17 | 547.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 561.05 | 546.17 | 547.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 565.85 | 550.11 | 549.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 574.75 | 563.45 | 557.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 13:15:00 | 565.50 | 568.25 | 562.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 14:00:00 | 565.50 | 568.25 | 562.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 565.00 | 567.60 | 562.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:30:00 | 564.85 | 567.60 | 562.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 555.80 | 565.09 | 562.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 555.80 | 565.09 | 562.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 558.85 | 563.84 | 562.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 11:45:00 | 561.45 | 563.77 | 562.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 551.50 | 560.17 | 561.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 551.50 | 560.17 | 561.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 14:15:00 | 548.65 | 554.29 | 557.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 13:15:00 | 542.45 | 539.23 | 543.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 13:15:00 | 542.45 | 539.23 | 543.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 542.45 | 539.23 | 543.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:00:00 | 542.45 | 539.23 | 543.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 543.50 | 540.09 | 543.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:30:00 | 544.70 | 540.09 | 543.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 543.05 | 540.68 | 543.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 541.00 | 540.68 | 543.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 544.20 | 541.38 | 543.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 543.90 | 541.38 | 543.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 538.35 | 540.78 | 543.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 11:45:00 | 535.20 | 539.75 | 542.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 10:15:00 | 554.50 | 544.04 | 543.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — BUY (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 10:15:00 | 554.50 | 544.04 | 543.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 11:15:00 | 560.15 | 547.26 | 545.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 13:15:00 | 557.20 | 558.35 | 553.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 13:15:00 | 557.20 | 558.35 | 553.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 557.20 | 558.35 | 553.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:00:00 | 557.20 | 558.35 | 553.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 546.40 | 555.96 | 553.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 546.40 | 555.96 | 553.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 544.20 | 553.61 | 552.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 545.00 | 553.61 | 552.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 551.70 | 552.09 | 551.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:00:00 | 551.70 | 552.09 | 551.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — SELL (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 12:15:00 | 549.15 | 551.50 | 551.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 11:15:00 | 543.75 | 548.15 | 549.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 14:15:00 | 548.20 | 546.82 | 548.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 14:15:00 | 548.20 | 546.82 | 548.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 548.20 | 546.82 | 548.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 15:00:00 | 548.20 | 546.82 | 548.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 548.85 | 547.22 | 548.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 542.45 | 547.22 | 548.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 515.33 | 528.33 | 533.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 13:15:00 | 488.21 | 499.67 | 511.62 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 181 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 464.85 | 452.00 | 451.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 11:15:00 | 466.35 | 454.87 | 452.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 456.70 | 462.49 | 458.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 456.70 | 462.49 | 458.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 456.70 | 462.49 | 458.09 | EMA400 retest candle locked (from upside) |

### Cycle 182 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 446.80 | 454.74 | 455.61 | EMA200 below EMA400 |

### Cycle 183 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 462.50 | 456.88 | 456.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 14:15:00 | 466.50 | 460.54 | 458.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 446.80 | 459.15 | 458.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 09:15:00 | 446.80 | 459.15 | 458.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 446.80 | 459.15 | 458.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 446.80 | 459.15 | 458.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 445.40 | 456.40 | 457.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 11:15:00 | 443.40 | 453.80 | 455.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 446.30 | 446.14 | 450.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:45:00 | 447.50 | 446.14 | 450.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 454.20 | 447.75 | 450.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 454.20 | 447.75 | 450.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 454.60 | 449.12 | 450.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 454.75 | 449.12 | 450.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 449.20 | 449.10 | 450.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 465.00 | 449.10 | 450.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 467.85 | 452.85 | 452.05 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 439.95 | 453.21 | 453.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 437.30 | 450.03 | 452.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 436.15 | 422.36 | 431.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 436.15 | 422.36 | 431.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 436.15 | 422.36 | 431.20 | EMA400 retest candle locked (from downside) |

### Cycle 187 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 447.25 | 436.69 | 435.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 454.60 | 442.43 | 439.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 10:15:00 | 443.60 | 444.01 | 441.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-06 11:00:00 | 443.60 | 444.01 | 441.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 454.65 | 451.81 | 446.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 13:45:00 | 461.40 | 455.06 | 450.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 475.05 | 456.42 | 451.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 10:15:00 | 507.54 | 491.01 | 485.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 188 — SELL (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 13:15:00 | 512.10 | 514.31 | 514.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 510.30 | 513.51 | 514.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 10:15:00 | 493.25 | 489.03 | 498.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 11:00:00 | 493.25 | 489.03 | 498.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 498.20 | 491.53 | 498.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:00:00 | 498.20 | 491.53 | 498.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 504.30 | 494.09 | 498.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:30:00 | 501.40 | 494.09 | 498.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 501.00 | 495.47 | 498.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 14:45:00 | 502.40 | 495.47 | 498.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 500.90 | 496.55 | 499.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:15:00 | 498.60 | 496.55 | 499.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 497.50 | 497.17 | 498.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:15:00 | 493.55 | 497.17 | 498.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 12:15:00 | 494.35 | 496.75 | 498.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 13:30:00 | 494.30 | 495.62 | 497.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 14:00:00 | 493.30 | 495.62 | 497.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 493.20 | 494.51 | 496.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:00:00 | 491.50 | 494.55 | 496.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 487.40 | 493.42 | 495.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 507.70 | 491.98 | 492.43 | SL hit (close>static) qty=1.00 sl=499.75 alert=retest2 |

### Cycle 189 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 508.80 | 495.35 | 493.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 515.30 | 506.80 | 501.05 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-17 14:15:00 | 145.60 | 2023-05-22 09:15:00 | 143.20 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2023-05-17 15:00:00 | 146.00 | 2023-05-22 09:15:00 | 143.20 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2023-05-19 11:30:00 | 146.00 | 2023-05-22 09:15:00 | 143.20 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2023-05-19 14:30:00 | 145.45 | 2023-05-22 09:15:00 | 143.20 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2023-05-23 14:45:00 | 143.85 | 2023-05-26 09:15:00 | 146.95 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2023-05-25 14:00:00 | 143.65 | 2023-05-26 09:15:00 | 146.95 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2023-06-02 09:15:00 | 153.05 | 2023-06-12 12:15:00 | 168.36 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-20 11:45:00 | 169.75 | 2023-06-20 12:15:00 | 167.40 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2023-06-20 15:15:00 | 169.95 | 2023-06-21 15:15:00 | 167.45 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2023-06-21 10:00:00 | 170.00 | 2023-06-21 15:15:00 | 167.45 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2023-06-21 11:00:00 | 170.00 | 2023-06-21 15:15:00 | 167.45 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2023-06-22 15:15:00 | 164.30 | 2023-06-26 15:15:00 | 169.90 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2023-06-23 13:45:00 | 165.90 | 2023-06-26 15:15:00 | 169.90 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2023-07-05 10:00:00 | 181.95 | 2023-07-07 13:15:00 | 179.85 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2023-07-05 11:15:00 | 182.00 | 2023-07-07 13:15:00 | 179.85 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2023-07-05 12:00:00 | 181.80 | 2023-07-07 13:15:00 | 179.85 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2023-07-06 10:00:00 | 181.65 | 2023-07-07 13:15:00 | 179.85 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2023-07-14 10:30:00 | 182.85 | 2023-07-14 12:15:00 | 182.25 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2023-07-18 13:15:00 | 187.45 | 2023-07-24 10:15:00 | 190.35 | STOP_HIT | 1.00 | 1.55% |
| BUY | retest2 | 2023-07-31 13:30:00 | 200.95 | 2023-08-02 13:15:00 | 193.70 | STOP_HIT | 1.00 | -3.61% |
| BUY | retest2 | 2023-08-01 11:15:00 | 201.15 | 2023-08-02 13:15:00 | 193.70 | STOP_HIT | 1.00 | -3.70% |
| BUY | retest2 | 2023-08-02 11:15:00 | 200.90 | 2023-08-02 13:15:00 | 193.70 | STOP_HIT | 1.00 | -3.58% |
| BUY | retest2 | 2023-08-02 12:00:00 | 201.15 | 2023-08-02 13:15:00 | 193.70 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2023-08-04 15:00:00 | 194.40 | 2023-08-07 15:15:00 | 197.85 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2023-08-11 12:00:00 | 191.00 | 2023-08-16 10:15:00 | 198.70 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2023-08-11 15:15:00 | 191.00 | 2023-08-16 10:15:00 | 198.70 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2023-08-14 12:00:00 | 190.85 | 2023-08-16 10:15:00 | 198.70 | STOP_HIT | 1.00 | -4.11% |
| BUY | retest1 | 2023-08-24 09:45:00 | 204.80 | 2023-08-25 09:15:00 | 215.04 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2023-08-24 10:15:00 | 205.50 | 2023-08-25 09:15:00 | 215.78 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2023-08-24 09:45:00 | 204.80 | 2023-08-28 14:15:00 | 213.40 | STOP_HIT | 0.50 | 4.20% |
| BUY | retest1 | 2023-08-24 10:15:00 | 205.50 | 2023-08-28 14:15:00 | 213.40 | STOP_HIT | 0.50 | 3.84% |
| BUY | retest2 | 2023-08-31 09:15:00 | 222.65 | 2023-09-05 15:15:00 | 218.80 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2023-09-01 12:00:00 | 218.30 | 2023-09-05 15:15:00 | 218.80 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2023-09-05 11:00:00 | 218.40 | 2023-09-05 15:15:00 | 218.80 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2023-09-26 12:30:00 | 214.25 | 2023-09-27 10:15:00 | 216.15 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2023-09-27 09:15:00 | 214.00 | 2023-09-27 10:15:00 | 216.15 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2023-09-28 14:15:00 | 219.15 | 2023-10-04 11:15:00 | 217.80 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2023-09-28 14:45:00 | 219.35 | 2023-10-04 11:15:00 | 217.80 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2023-09-29 09:15:00 | 221.40 | 2023-10-04 12:15:00 | 216.35 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2023-09-29 14:00:00 | 219.00 | 2023-10-04 12:15:00 | 216.35 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2023-10-03 13:45:00 | 222.40 | 2023-10-04 12:15:00 | 216.35 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2023-10-03 15:00:00 | 222.70 | 2023-10-04 12:15:00 | 216.35 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2023-10-19 09:45:00 | 241.30 | 2023-10-19 14:15:00 | 235.65 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2023-10-19 10:45:00 | 240.55 | 2023-10-19 14:15:00 | 235.65 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2023-10-23 10:15:00 | 232.45 | 2023-10-23 15:15:00 | 220.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-23 10:15:00 | 232.45 | 2023-10-25 13:15:00 | 231.00 | STOP_HIT | 0.50 | 0.62% |
| SELL | retest2 | 2023-10-23 11:00:00 | 230.90 | 2023-10-27 09:15:00 | 235.50 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2023-10-25 15:00:00 | 231.60 | 2023-10-27 09:15:00 | 235.50 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest1 | 2023-10-31 09:15:00 | 246.15 | 2023-11-01 12:15:00 | 239.50 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2023-11-02 09:15:00 | 242.30 | 2023-11-02 10:15:00 | 238.40 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2023-11-15 09:30:00 | 252.45 | 2023-11-21 10:15:00 | 251.15 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2023-11-15 12:15:00 | 251.90 | 2023-11-21 10:15:00 | 251.15 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2023-11-16 09:15:00 | 252.80 | 2023-11-21 10:15:00 | 251.15 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2023-11-16 11:45:00 | 251.80 | 2023-11-21 10:15:00 | 251.15 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2023-12-13 09:30:00 | 286.60 | 2023-12-20 14:15:00 | 291.90 | STOP_HIT | 1.00 | 1.85% |
| BUY | retest2 | 2024-01-08 13:30:00 | 315.00 | 2024-01-16 13:15:00 | 317.90 | STOP_HIT | 1.00 | 0.92% |
| SELL | retest2 | 2024-01-25 10:15:00 | 313.15 | 2024-01-29 09:15:00 | 315.55 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-01-25 12:15:00 | 314.25 | 2024-01-29 09:15:00 | 315.55 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2024-02-07 09:15:00 | 331.75 | 2024-02-07 14:15:00 | 324.10 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2024-02-08 09:15:00 | 338.40 | 2024-02-13 10:15:00 | 329.95 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2024-02-13 09:45:00 | 331.20 | 2024-02-13 10:15:00 | 329.95 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2024-02-16 14:30:00 | 355.25 | 2024-02-21 11:15:00 | 344.70 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2024-02-28 09:15:00 | 349.60 | 2024-02-28 09:15:00 | 346.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-03-18 11:00:00 | 294.85 | 2024-03-21 10:15:00 | 302.60 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2024-03-18 14:15:00 | 294.85 | 2024-03-21 10:15:00 | 302.60 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2024-03-19 09:45:00 | 294.10 | 2024-03-21 10:15:00 | 302.60 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest1 | 2024-04-01 09:15:00 | 319.90 | 2024-04-02 09:15:00 | 335.89 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-04-01 09:15:00 | 319.90 | 2024-04-03 13:15:00 | 330.85 | STOP_HIT | 0.50 | 3.42% |
| BUY | retest2 | 2024-04-26 09:15:00 | 364.50 | 2024-05-02 09:15:00 | 358.05 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-04-26 10:45:00 | 364.65 | 2024-05-02 09:15:00 | 358.05 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-04-29 10:15:00 | 365.20 | 2024-05-02 09:15:00 | 358.05 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2024-04-29 11:15:00 | 365.80 | 2024-05-02 09:15:00 | 358.05 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2024-05-15 09:15:00 | 377.55 | 2024-05-24 10:15:00 | 383.45 | STOP_HIT | 1.00 | 1.56% |
| BUY | retest2 | 2024-05-16 12:15:00 | 376.40 | 2024-05-24 10:15:00 | 383.45 | STOP_HIT | 1.00 | 1.87% |
| BUY | retest2 | 2024-05-16 13:45:00 | 376.30 | 2024-05-24 10:15:00 | 383.45 | STOP_HIT | 1.00 | 1.90% |
| BUY | retest2 | 2024-05-16 15:15:00 | 377.00 | 2024-05-24 10:15:00 | 383.45 | STOP_HIT | 1.00 | 1.71% |
| BUY | retest2 | 2024-05-17 12:15:00 | 381.20 | 2024-05-24 10:15:00 | 383.45 | STOP_HIT | 1.00 | 0.59% |
| BUY | retest2 | 2024-05-17 14:15:00 | 382.00 | 2024-05-24 10:15:00 | 383.45 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2024-05-27 15:15:00 | 381.40 | 2024-05-31 13:15:00 | 384.30 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-06-14 09:15:00 | 423.20 | 2024-06-26 10:15:00 | 439.15 | STOP_HIT | 1.00 | 3.77% |
| SELL | retest2 | 2024-06-27 10:30:00 | 439.10 | 2024-07-01 12:15:00 | 442.40 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-06-27 12:45:00 | 439.05 | 2024-07-01 12:15:00 | 442.40 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-07-01 10:45:00 | 439.70 | 2024-07-01 12:15:00 | 442.40 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-07-09 15:15:00 | 528.50 | 2024-07-10 09:15:00 | 514.55 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2024-08-13 12:30:00 | 510.00 | 2024-08-14 14:15:00 | 519.75 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2024-08-22 11:30:00 | 595.05 | 2024-08-22 12:15:00 | 586.15 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-08-23 09:15:00 | 595.60 | 2024-08-23 10:15:00 | 585.05 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-09-04 14:30:00 | 608.00 | 2024-09-09 10:15:00 | 600.80 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-09-19 15:15:00 | 680.00 | 2024-09-20 13:15:00 | 748.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2024-10-11 13:15:00 | 751.75 | 2024-10-16 09:15:00 | 789.34 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-10-11 13:45:00 | 754.20 | 2024-10-16 09:15:00 | 791.91 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-10-11 13:15:00 | 751.75 | 2024-10-16 11:15:00 | 766.25 | STOP_HIT | 0.50 | 1.93% |
| BUY | retest1 | 2024-10-11 13:45:00 | 754.20 | 2024-10-16 11:15:00 | 766.25 | STOP_HIT | 0.50 | 1.60% |
| BUY | retest2 | 2024-10-14 10:30:00 | 756.85 | 2024-10-18 09:15:00 | 752.90 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2024-10-29 10:30:00 | 671.80 | 2024-10-30 09:15:00 | 688.15 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2024-10-29 11:45:00 | 671.50 | 2024-10-30 09:15:00 | 688.15 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2024-10-29 12:30:00 | 674.60 | 2024-10-30 09:15:00 | 688.15 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2024-10-29 13:15:00 | 673.40 | 2024-10-30 09:15:00 | 688.15 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2024-11-04 13:45:00 | 727.95 | 2024-11-08 14:15:00 | 740.40 | STOP_HIT | 1.00 | 1.71% |
| BUY | retest2 | 2024-11-04 14:45:00 | 731.35 | 2024-11-08 14:15:00 | 740.40 | STOP_HIT | 1.00 | 1.24% |
| BUY | retest2 | 2024-11-05 09:15:00 | 728.10 | 2024-11-08 14:15:00 | 740.40 | STOP_HIT | 1.00 | 1.69% |
| BUY | retest2 | 2024-11-05 09:45:00 | 729.40 | 2024-11-08 14:15:00 | 740.40 | STOP_HIT | 1.00 | 1.51% |
| SELL | retest2 | 2024-11-11 13:30:00 | 738.30 | 2024-11-13 10:15:00 | 701.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 13:30:00 | 738.30 | 2024-11-14 13:15:00 | 664.47 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-29 09:30:00 | 664.00 | 2024-11-29 14:15:00 | 677.80 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-11-29 10:00:00 | 664.40 | 2024-11-29 14:15:00 | 677.80 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2024-11-29 11:30:00 | 663.65 | 2024-11-29 14:15:00 | 677.80 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2024-12-13 13:45:00 | 751.00 | 2024-12-17 09:15:00 | 826.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-13 14:30:00 | 751.50 | 2024-12-17 09:15:00 | 826.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-10 09:15:00 | 910.70 | 2025-01-10 09:15:00 | 882.75 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2025-01-10 11:00:00 | 912.60 | 2025-01-10 11:15:00 | 896.15 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-01-23 11:30:00 | 859.10 | 2025-01-24 14:15:00 | 816.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 11:30:00 | 859.10 | 2025-01-27 09:15:00 | 773.19 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-20 09:15:00 | 534.90 | 2025-02-20 13:15:00 | 542.90 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-02-20 09:45:00 | 532.80 | 2025-02-20 13:15:00 | 542.90 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-02-20 11:45:00 | 537.15 | 2025-02-20 13:15:00 | 542.90 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-02-20 12:15:00 | 536.70 | 2025-02-20 13:15:00 | 542.90 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-03-07 14:00:00 | 487.00 | 2025-03-17 12:15:00 | 489.85 | STOP_HIT | 1.00 | 0.59% |
| BUY | retest2 | 2025-03-11 10:30:00 | 483.75 | 2025-03-17 12:15:00 | 489.85 | STOP_HIT | 1.00 | 1.26% |
| BUY | retest2 | 2025-03-24 11:00:00 | 554.85 | 2025-03-25 11:15:00 | 522.05 | STOP_HIT | 1.00 | -5.91% |
| SELL | retest2 | 2025-03-28 10:45:00 | 499.55 | 2025-04-02 09:15:00 | 474.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-28 10:45:00 | 499.55 | 2025-04-02 10:15:00 | 490.65 | STOP_HIT | 0.50 | 1.78% |
| BUY | retest2 | 2025-04-21 12:15:00 | 497.35 | 2025-04-24 13:15:00 | 488.10 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-04-24 10:30:00 | 490.40 | 2025-04-24 13:15:00 | 488.10 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-04-24 13:15:00 | 490.60 | 2025-04-24 13:15:00 | 488.10 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-04-30 09:15:00 | 458.65 | 2025-05-05 15:15:00 | 462.40 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-05-07 12:45:00 | 446.70 | 2025-05-09 09:15:00 | 424.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 13:00:00 | 447.50 | 2025-05-09 09:15:00 | 425.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-07 12:45:00 | 446.70 | 2025-05-09 15:15:00 | 431.50 | STOP_HIT | 0.50 | 3.40% |
| SELL | retest2 | 2025-05-08 13:00:00 | 447.50 | 2025-05-09 15:15:00 | 431.50 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2025-05-12 09:45:00 | 447.50 | 2025-05-12 11:15:00 | 454.35 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-05-21 10:00:00 | 511.35 | 2025-05-28 14:15:00 | 516.25 | STOP_HIT | 1.00 | 0.96% |
| BUY | retest2 | 2025-05-21 13:00:00 | 510.25 | 2025-05-28 14:15:00 | 516.25 | STOP_HIT | 1.00 | 1.18% |
| BUY | retest2 | 2025-05-21 14:30:00 | 514.70 | 2025-05-28 14:15:00 | 516.25 | STOP_HIT | 1.00 | 0.30% |
| BUY | retest2 | 2025-06-05 09:15:00 | 581.40 | 2025-06-10 15:15:00 | 578.00 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-06-05 12:30:00 | 576.35 | 2025-06-10 15:15:00 | 578.00 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-06-06 10:15:00 | 581.90 | 2025-06-10 15:15:00 | 578.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-06-09 09:15:00 | 579.50 | 2025-06-10 15:15:00 | 578.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-06-12 11:45:00 | 567.00 | 2025-06-13 09:15:00 | 510.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-30 09:15:00 | 562.55 | 2025-07-02 10:15:00 | 554.40 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-07-08 11:00:00 | 539.50 | 2025-07-08 14:15:00 | 550.45 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-08-05 14:45:00 | 558.80 | 2025-08-08 15:15:00 | 531.43 | PARTIAL | 0.50 | 4.90% |
| SELL | retest2 | 2025-08-05 15:15:00 | 559.40 | 2025-08-11 09:15:00 | 530.86 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2025-08-06 09:30:00 | 559.30 | 2025-08-11 09:15:00 | 531.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 14:45:00 | 558.80 | 2025-08-11 10:15:00 | 537.65 | STOP_HIT | 0.50 | 3.78% |
| SELL | retest2 | 2025-08-05 15:15:00 | 559.40 | 2025-08-11 10:15:00 | 537.65 | STOP_HIT | 0.50 | 3.89% |
| SELL | retest2 | 2025-08-06 09:30:00 | 559.30 | 2025-08-11 10:15:00 | 537.65 | STOP_HIT | 0.50 | 3.87% |
| BUY | retest2 | 2025-09-05 09:45:00 | 541.50 | 2025-09-05 11:15:00 | 533.30 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-09-19 10:15:00 | 600.10 | 2025-09-22 09:15:00 | 660.11 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-10 11:15:00 | 691.80 | 2025-10-14 14:15:00 | 657.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-10 15:00:00 | 692.55 | 2025-10-14 14:15:00 | 657.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-13 10:00:00 | 692.50 | 2025-10-14 14:15:00 | 657.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-10 11:15:00 | 691.80 | 2025-10-15 09:15:00 | 673.80 | STOP_HIT | 0.50 | 2.60% |
| SELL | retest2 | 2025-10-10 15:00:00 | 692.55 | 2025-10-15 09:15:00 | 673.80 | STOP_HIT | 0.50 | 2.71% |
| SELL | retest2 | 2025-10-13 10:00:00 | 692.50 | 2025-10-15 09:15:00 | 673.80 | STOP_HIT | 0.50 | 2.70% |
| BUY | retest2 | 2025-11-03 12:15:00 | 645.85 | 2025-11-06 09:15:00 | 627.80 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2025-11-11 09:15:00 | 618.70 | 2025-11-17 09:15:00 | 634.75 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest1 | 2025-12-01 10:15:00 | 578.60 | 2025-12-03 11:15:00 | 549.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-12-01 11:30:00 | 578.95 | 2025-12-03 11:15:00 | 550.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-12-01 12:30:00 | 578.00 | 2025-12-03 11:15:00 | 549.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-12-01 10:15:00 | 578.60 | 2025-12-03 15:15:00 | 556.20 | STOP_HIT | 0.50 | 3.87% |
| SELL | retest1 | 2025-12-01 11:30:00 | 578.95 | 2025-12-03 15:15:00 | 556.20 | STOP_HIT | 0.50 | 3.93% |
| SELL | retest1 | 2025-12-01 12:30:00 | 578.00 | 2025-12-03 15:15:00 | 556.20 | STOP_HIT | 0.50 | 3.77% |
| SELL | retest2 | 2025-12-10 11:30:00 | 504.15 | 2025-12-12 09:15:00 | 525.15 | STOP_HIT | 1.00 | -4.17% |
| SELL | retest2 | 2025-12-11 10:45:00 | 505.50 | 2025-12-12 09:15:00 | 525.15 | STOP_HIT | 1.00 | -3.89% |
| SELL | retest2 | 2025-12-11 11:45:00 | 505.00 | 2025-12-12 09:15:00 | 525.15 | STOP_HIT | 1.00 | -3.99% |
| SELL | retest2 | 2025-12-11 12:45:00 | 505.65 | 2025-12-12 09:15:00 | 525.15 | STOP_HIT | 1.00 | -3.86% |
| SELL | retest2 | 2025-12-31 09:30:00 | 541.70 | 2025-12-31 10:15:00 | 554.90 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2026-01-07 10:15:00 | 592.95 | 2026-01-08 12:15:00 | 578.25 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2026-01-07 10:45:00 | 592.55 | 2026-01-08 14:15:00 | 572.80 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest2 | 2026-01-07 11:15:00 | 592.05 | 2026-01-08 14:15:00 | 572.80 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2026-01-07 12:00:00 | 592.45 | 2026-01-08 14:15:00 | 572.80 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2026-01-08 10:30:00 | 606.60 | 2026-01-08 14:15:00 | 572.80 | STOP_HIT | 1.00 | -5.57% |
| SELL | retest2 | 2026-01-13 11:15:00 | 563.80 | 2026-01-20 09:15:00 | 535.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 10:00:00 | 563.70 | 2026-01-20 09:15:00 | 535.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 11:00:00 | 560.60 | 2026-01-20 09:15:00 | 532.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 11:15:00 | 563.80 | 2026-01-21 09:15:00 | 527.05 | STOP_HIT | 0.50 | 6.52% |
| SELL | retest2 | 2026-01-14 10:00:00 | 563.70 | 2026-01-21 09:15:00 | 527.05 | STOP_HIT | 0.50 | 6.50% |
| SELL | retest2 | 2026-01-14 11:00:00 | 560.60 | 2026-01-21 09:15:00 | 527.05 | STOP_HIT | 0.50 | 5.98% |
| SELL | retest2 | 2026-01-29 10:15:00 | 490.20 | 2026-01-30 10:15:00 | 512.65 | STOP_HIT | 1.00 | -4.58% |
| SELL | retest2 | 2026-01-29 15:00:00 | 489.50 | 2026-01-30 10:15:00 | 512.65 | STOP_HIT | 1.00 | -4.73% |
| BUY | retest2 | 2026-02-11 11:45:00 | 561.45 | 2026-02-12 10:15:00 | 551.50 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2026-02-17 11:45:00 | 535.20 | 2026-02-18 10:15:00 | 554.50 | STOP_HIT | 1.00 | -3.61% |
| SELL | retest2 | 2026-02-24 09:15:00 | 542.45 | 2026-03-02 09:15:00 | 515.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 542.45 | 2026-03-04 13:15:00 | 488.21 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-04-07 13:45:00 | 461.40 | 2026-04-15 10:15:00 | 507.54 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-08 09:15:00 | 475.05 | 2026-04-21 10:15:00 | 522.56 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-28 11:15:00 | 493.55 | 2026-05-04 09:15:00 | 507.70 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2026-04-28 12:15:00 | 494.35 | 2026-05-04 09:15:00 | 507.70 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2026-04-28 13:30:00 | 494.30 | 2026-05-04 09:15:00 | 507.70 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2026-04-28 14:00:00 | 493.30 | 2026-05-04 09:15:00 | 507.70 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2026-04-29 14:00:00 | 491.50 | 2026-05-04 09:15:00 | 507.70 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2026-04-30 09:15:00 | 487.40 | 2026-05-04 09:15:00 | 507.70 | STOP_HIT | 1.00 | -4.16% |
