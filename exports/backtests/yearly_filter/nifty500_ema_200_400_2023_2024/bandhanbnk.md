# Bandhan Bank Ltd. (BANDHANBNK)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 206.25
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 11 |
| ALERT2 | 10 |
| ALERT2_SKIP | 3 |
| ALERT3 | 59 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 47 |
| PARTIAL | 4 |
| TARGET_HIT | 7 |
| STOP_HIT | 44 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 55 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 42
- **Target hits / Stop hits / Partials:** 7 / 44 / 4
- **Avg / median % per leg:** -0.21% / -2.06%
- **Sum % (uncompounded):** -11.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 5 | 20.8% | 5 | 19 | 0 | -0.14% | -3.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 24 | 5 | 20.8% | 5 | 19 | 0 | -0.14% | -3.3% |
| SELL (all) | 31 | 8 | 25.8% | 2 | 25 | 4 | -0.26% | -8.1% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.80% | -11.2% |
| SELL @ 3rd Alert (retest2) | 27 | 8 | 29.6% | 2 | 21 | 4 | 0.12% | 3.1% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.80% | -11.2% |
| retest2 (combined) | 51 | 13 | 25.5% | 7 | 40 | 4 | -0.00% | -0.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 09:15:00 | 222.80 | 239.58 | 239.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-12 11:15:00 | 221.05 | 239.22 | 239.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-01 09:15:00 | 227.45 | 226.13 | 231.22 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-01 10:30:00 | 226.30 | 226.13 | 231.19 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-02 09:15:00 | 224.70 | 226.20 | 231.10 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-03 12:00:00 | 225.95 | 226.01 | 230.76 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-04 09:30:00 | 226.40 | 226.00 | 230.65 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 13:15:00 | 229.75 | 226.21 | 230.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-07 13:45:00 | 229.80 | 226.21 | 230.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 09:15:00 | 232.15 | 226.33 | 230.50 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-08-08 09:15:00 | 232.15 | 226.33 | 230.50 | SL hit (close>ema400) qty=1.00 sl=230.50 alert=retest1 |

### Cycle 2 — BUY (started 2023-09-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 15:15:00 | 238.50 | 231.75 | 231.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-08 09:15:00 | 242.45 | 231.86 | 231.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-12 15:15:00 | 232.00 | 233.31 | 232.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-12 15:15:00 | 232.00 | 233.31 | 232.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 15:15:00 | 232.00 | 233.31 | 232.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-13 09:15:00 | 234.60 | 233.31 | 232.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 09:15:00 | 231.00 | 233.29 | 232.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-13 09:30:00 | 229.90 | 233.29 | 232.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 10:15:00 | 232.50 | 233.28 | 232.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-13 11:15:00 | 233.10 | 233.28 | 232.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-13 13:45:00 | 234.40 | 233.28 | 232.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2023-09-27 15:15:00 | 256.41 | 240.45 | 236.84 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2023-10-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 15:15:00 | 212.10 | 238.10 | 238.18 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 10:15:00 | 252.95 | 230.02 | 229.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-03 11:15:00 | 257.05 | 237.11 | 234.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 10:15:00 | 239.00 | 240.26 | 236.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-08 11:00:00 | 239.00 | 240.26 | 236.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 11:15:00 | 236.80 | 240.22 | 236.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 11:45:00 | 236.60 | 240.22 | 236.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 12:15:00 | 235.90 | 240.18 | 236.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 12:45:00 | 236.00 | 240.18 | 236.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 13:15:00 | 235.05 | 240.13 | 236.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 14:00:00 | 235.05 | 240.13 | 236.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 15:15:00 | 232.15 | 239.98 | 236.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 09:15:00 | 237.80 | 239.98 | 236.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 10:00:00 | 234.75 | 239.93 | 236.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-10 09:15:00 | 230.40 | 239.61 | 236.15 | SL hit (close<static) qty=1.00 sl=230.95 alert=retest2 |

### Cycle 5 — SELL (started 2024-01-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-24 13:15:00 | 222.50 | 233.96 | 233.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 11:15:00 | 221.20 | 233.46 | 233.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-02 10:15:00 | 231.90 | 231.15 | 232.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-02 11:00:00 | 231.90 | 231.15 | 232.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 11:15:00 | 231.30 | 231.15 | 232.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-02 11:30:00 | 231.90 | 231.15 | 232.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 200.00 | 192.11 | 201.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-05 11:15:00 | 198.45 | 192.18 | 201.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-05 12:00:00 | 198.25 | 192.24 | 201.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-08 09:15:00 | 188.53 | 192.38 | 201.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-08 09:15:00 | 188.34 | 192.38 | 201.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-04-15 09:15:00 | 178.60 | 190.27 | 199.21 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 6 — BUY (started 2024-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 10:15:00 | 205.04 | 191.53 | 191.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 11:15:00 | 206.22 | 191.68 | 191.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 195.59 | 199.92 | 196.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 09:15:00 | 195.59 | 199.92 | 196.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 195.59 | 199.92 | 196.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 195.59 | 199.92 | 196.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 193.09 | 199.86 | 196.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 191.99 | 199.86 | 196.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 196.88 | 199.27 | 196.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:15:00 | 196.19 | 199.27 | 196.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 196.00 | 199.24 | 196.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 09:15:00 | 201.46 | 198.61 | 196.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 12:15:00 | 195.00 | 198.54 | 196.48 | SL hit (close<static) qty=1.00 sl=195.08 alert=retest2 |

### Cycle 7 — SELL (started 2024-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 15:15:00 | 184.49 | 200.12 | 200.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 182.04 | 196.31 | 197.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 09:15:00 | 177.08 | 175.97 | 182.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-04 10:00:00 | 177.08 | 175.97 | 182.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 10:15:00 | 150.82 | 144.28 | 150.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-07 11:00:00 | 150.82 | 144.28 | 150.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 149.12 | 144.33 | 150.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-10 14:15:00 | 148.92 | 144.84 | 150.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-11 09:15:00 | 141.47 | 144.85 | 150.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-21 10:15:00 | 144.55 | 143.25 | 148.31 | SL hit (close>ema200) qty=0.50 sl=143.25 alert=retest2 |

### Cycle 8 — BUY (started 2025-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 09:15:00 | 168.90 | 150.05 | 150.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 12:15:00 | 170.61 | 150.64 | 150.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 14:15:00 | 157.31 | 157.70 | 154.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 14:30:00 | 157.32 | 157.70 | 154.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 157.00 | 157.87 | 154.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 12:00:00 | 157.27 | 157.84 | 154.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 12:45:00 | 157.46 | 157.83 | 154.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 15:15:00 | 157.19 | 157.80 | 154.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-19 12:15:00 | 173.00 | 160.72 | 157.04 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 166.72 | 172.71 | 172.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 10:15:00 | 166.19 | 172.59 | 172.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 12:15:00 | 172.91 | 171.35 | 171.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 12:15:00 | 172.91 | 171.35 | 171.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 172.91 | 171.35 | 171.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:30:00 | 172.95 | 171.35 | 171.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 173.32 | 171.37 | 172.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:45:00 | 173.49 | 171.37 | 172.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 172.19 | 172.07 | 172.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 15:00:00 | 171.47 | 172.08 | 172.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 14:15:00 | 162.90 | 171.23 | 171.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 167.49 | 166.69 | 168.76 | SL hit (close>ema200) qty=0.50 sl=166.69 alert=retest2 |

### Cycle 10 — BUY (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 12:15:00 | 167.15 | 151.07 | 151.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 10:15:00 | 168.54 | 151.85 | 151.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 09:15:00 | 169.12 | 171.80 | 164.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-16 10:00:00 | 169.12 | 171.80 | 164.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 164.80 | 171.73 | 164.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:45:00 | 164.61 | 171.73 | 164.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 163.04 | 171.65 | 164.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 12:00:00 | 163.04 | 171.65 | 164.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 12:15:00 | 159.00 | 171.52 | 164.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 12:45:00 | 159.36 | 171.52 | 164.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 14:15:00 | 163.78 | 169.98 | 164.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 14:30:00 | 163.18 | 169.98 | 164.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 12:15:00 | 142.15 | 160.77 | 160.79 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 170.67 | 160.63 | 160.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 10:15:00 | 171.55 | 160.74 | 160.66 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-08-01 10:30:00 | 226.30 | 2023-08-08 09:15:00 | 232.15 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest1 | 2023-08-02 09:15:00 | 224.70 | 2023-08-08 09:15:00 | 232.15 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest1 | 2023-08-03 12:00:00 | 225.95 | 2023-08-08 09:15:00 | 232.15 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest1 | 2023-08-04 09:30:00 | 226.40 | 2023-08-08 09:15:00 | 232.15 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2023-08-10 13:45:00 | 227.75 | 2023-08-18 10:15:00 | 233.00 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2023-08-11 09:15:00 | 228.20 | 2023-08-18 10:15:00 | 233.00 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2023-08-11 10:15:00 | 228.55 | 2023-08-18 10:15:00 | 233.00 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2023-08-11 11:00:00 | 228.30 | 2023-08-18 10:15:00 | 233.00 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2023-08-21 10:00:00 | 228.10 | 2023-08-23 09:15:00 | 233.75 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2023-09-13 11:15:00 | 233.10 | 2023-09-27 15:15:00 | 256.41 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-13 13:45:00 | 234.40 | 2023-09-28 09:15:00 | 257.84 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-19 09:45:00 | 233.05 | 2023-10-19 10:15:00 | 228.40 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2024-01-09 09:15:00 | 237.80 | 2024-01-10 09:15:00 | 230.40 | STOP_HIT | 1.00 | -3.11% |
| BUY | retest2 | 2024-01-09 10:00:00 | 234.75 | 2024-01-10 09:15:00 | 230.40 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2024-01-11 09:15:00 | 234.45 | 2024-01-12 14:15:00 | 230.00 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2024-01-11 14:15:00 | 233.70 | 2024-01-12 14:15:00 | 230.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2024-04-05 11:15:00 | 198.45 | 2024-04-08 09:15:00 | 188.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-05 12:00:00 | 198.25 | 2024-04-08 09:15:00 | 188.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-05 11:15:00 | 198.45 | 2024-04-15 09:15:00 | 178.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-04-05 12:00:00 | 198.25 | 2024-04-15 09:15:00 | 178.43 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-10 14:15:00 | 198.20 | 2024-06-20 09:15:00 | 204.00 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2024-06-10 15:00:00 | 198.45 | 2024-06-20 09:15:00 | 204.00 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2024-07-16 09:15:00 | 201.46 | 2024-07-16 12:15:00 | 195.00 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2024-07-16 13:30:00 | 199.20 | 2024-07-19 09:15:00 | 192.10 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest2 | 2024-07-18 09:45:00 | 199.86 | 2024-07-19 09:15:00 | 192.10 | STOP_HIT | 1.00 | -3.88% |
| BUY | retest2 | 2024-07-22 14:15:00 | 198.65 | 2024-07-23 12:15:00 | 194.17 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2024-07-23 11:15:00 | 197.80 | 2024-07-23 12:15:00 | 194.17 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2024-07-29 09:15:00 | 210.02 | 2024-08-13 09:15:00 | 192.01 | STOP_HIT | 1.00 | -8.58% |
| BUY | retest2 | 2024-08-12 10:15:00 | 197.53 | 2024-08-13 09:15:00 | 192.01 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2024-08-21 09:15:00 | 200.07 | 2024-08-29 09:15:00 | 194.79 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2024-08-30 15:00:00 | 200.54 | 2024-09-06 14:15:00 | 196.39 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2024-09-02 10:00:00 | 201.25 | 2024-09-06 14:15:00 | 196.39 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2024-09-05 09:15:00 | 203.95 | 2024-09-06 14:15:00 | 196.39 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2024-09-10 12:45:00 | 200.29 | 2024-09-11 11:15:00 | 196.89 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-09-11 13:15:00 | 199.91 | 2024-09-11 14:15:00 | 195.80 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2024-09-13 09:15:00 | 200.06 | 2024-10-01 09:15:00 | 195.75 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-03-10 14:15:00 | 148.92 | 2025-03-11 09:15:00 | 141.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-10 14:15:00 | 148.92 | 2025-03-21 10:15:00 | 144.55 | STOP_HIT | 0.50 | 2.93% |
| SELL | retest2 | 2025-03-25 10:00:00 | 148.45 | 2025-03-26 11:15:00 | 151.18 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-03-25 10:30:00 | 148.68 | 2025-03-26 11:15:00 | 151.18 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-03-26 15:00:00 | 148.78 | 2025-04-02 11:15:00 | 151.34 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-03-28 12:15:00 | 147.16 | 2025-04-02 14:15:00 | 151.51 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2025-04-01 11:30:00 | 148.10 | 2025-04-02 14:15:00 | 151.51 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-04-02 09:15:00 | 147.75 | 2025-04-02 14:15:00 | 151.51 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2025-04-07 09:15:00 | 146.01 | 2025-04-15 09:15:00 | 152.18 | STOP_HIT | 1.00 | -4.23% |
| SELL | retest2 | 2025-04-09 09:15:00 | 147.96 | 2025-04-15 09:15:00 | 152.18 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2025-04-11 10:15:00 | 148.37 | 2025-04-15 09:15:00 | 152.18 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2025-05-09 12:00:00 | 157.27 | 2025-05-19 12:15:00 | 173.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-09 12:45:00 | 157.46 | 2025-05-19 12:15:00 | 172.91 | TARGET_HIT | 1.00 | 9.81% |
| BUY | retest2 | 2025-05-09 15:15:00 | 157.19 | 2025-06-02 09:15:00 | 173.21 | TARGET_HIT | 1.00 | 10.19% |
| SELL | retest2 | 2025-08-25 15:00:00 | 171.47 | 2025-08-28 14:15:00 | 162.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 15:00:00 | 171.47 | 2025-09-18 09:15:00 | 167.49 | STOP_HIT | 0.50 | 2.32% |
| SELL | retest2 | 2025-10-23 13:15:00 | 171.83 | 2025-10-27 11:15:00 | 173.17 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-10-23 13:45:00 | 171.87 | 2025-10-27 11:15:00 | 173.17 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-10-23 15:00:00 | 170.97 | 2025-10-27 11:15:00 | 173.17 | STOP_HIT | 1.00 | -1.29% |
