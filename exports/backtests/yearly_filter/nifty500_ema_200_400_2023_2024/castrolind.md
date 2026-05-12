# Castrol India Ltd. (CASTROLIND)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 185.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT2_SKIP | 4 |
| ALERT3 | 31 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 35 |
| PARTIAL | 7 |
| TARGET_HIT | 8 |
| STOP_HIT | 34 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 46 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 30
- **Target hits / Stop hits / Partials:** 8 / 31 / 7
- **Avg / median % per leg:** 0.72% / -1.07%
- **Sum % (uncompounded):** 32.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 2 | 8.3% | 1 | 23 | 0 | -2.41% | -57.8% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.33% | -17.3% |
| BUY @ 3rd Alert (retest2) | 20 | 2 | 10.0% | 1 | 19 | 0 | -2.02% | -40.5% |
| SELL (all) | 22 | 14 | 63.6% | 7 | 8 | 7 | 4.12% | 90.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 22 | 14 | 63.6% | 7 | 8 | 7 | 4.12% | 90.7% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.33% | -17.3% |
| retest2 (combined) | 42 | 16 | 38.1% | 8 | 27 | 7 | 1.20% | 50.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-19 09:15:00 | 121.55 | 115.75 | 115.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-20 09:15:00 | 122.70 | 116.15 | 115.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 13:15:00 | 142.30 | 142.48 | 135.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-24 13:30:00 | 142.00 | 142.48 | 135.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 12:15:00 | 141.15 | 146.59 | 141.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 13:00:00 | 141.15 | 146.59 | 141.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 13:15:00 | 140.80 | 146.53 | 141.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 14:00:00 | 140.80 | 146.53 | 141.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 10:15:00 | 141.90 | 146.32 | 141.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-22 11:45:00 | 142.15 | 146.28 | 141.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-22 14:15:00 | 139.85 | 146.11 | 141.58 | SL hit (close<static) qty=1.00 sl=140.20 alert=retest2 |

### Cycle 2 — SELL (started 2023-11-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-02 14:15:00 | 134.95 | 140.21 | 140.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-03 10:15:00 | 134.15 | 140.04 | 140.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 09:15:00 | 138.25 | 136.98 | 138.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-23 09:15:00 | 138.25 | 136.98 | 138.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 138.25 | 136.98 | 138.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 10:00:00 | 138.25 | 136.98 | 138.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 10:15:00 | 138.20 | 136.99 | 138.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 10:30:00 | 138.10 | 136.99 | 138.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 11:15:00 | 137.95 | 137.00 | 138.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 11:30:00 | 138.75 | 137.00 | 138.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 12:15:00 | 137.50 | 137.00 | 138.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 14:30:00 | 136.95 | 137.00 | 138.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-24 09:30:00 | 136.85 | 137.00 | 138.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-24 11:15:00 | 140.55 | 137.04 | 138.20 | SL hit (close>static) qty=1.00 sl=138.25 alert=retest2 |

### Cycle 3 — BUY (started 2023-12-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 12:15:00 | 149.65 | 139.07 | 139.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-19 09:15:00 | 151.40 | 140.57 | 139.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 10:15:00 | 168.80 | 169.28 | 159.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-17 11:00:00 | 168.80 | 169.28 | 159.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 13:15:00 | 190.45 | 201.16 | 190.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 14:00:00 | 190.45 | 201.16 | 190.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 14:15:00 | 190.95 | 201.06 | 190.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 14:30:00 | 191.05 | 201.06 | 190.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 15:15:00 | 191.10 | 200.96 | 190.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-14 09:15:00 | 193.95 | 200.96 | 190.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 09:15:00 | 196.00 | 200.91 | 190.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-14 11:30:00 | 200.25 | 200.85 | 190.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-18 09:15:00 | 200.45 | 200.46 | 190.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-28 14:15:00 | 185.60 | 197.09 | 191.20 | SL hit (close<static) qty=1.00 sl=186.05 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 14:15:00 | 193.15 | 198.11 | 198.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 09:15:00 | 191.00 | 197.99 | 198.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 196.90 | 195.45 | 196.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 196.90 | 195.45 | 196.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 196.90 | 195.45 | 196.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 12:45:00 | 193.65 | 195.43 | 196.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 10:15:00 | 183.97 | 195.16 | 196.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-06-04 11:15:00 | 174.28 | 194.94 | 196.35 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 15:15:00 | 202.90 | 196.99 | 196.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 09:15:00 | 205.51 | 197.07 | 197.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 12:15:00 | 201.15 | 201.43 | 199.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-28 12:45:00 | 200.98 | 201.43 | 199.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 251.70 | 259.70 | 251.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 15:00:00 | 251.70 | 259.70 | 251.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 250.20 | 259.54 | 251.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:00:00 | 250.20 | 259.54 | 251.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 245.90 | 259.40 | 251.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:00:00 | 245.90 | 259.40 | 251.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 251.50 | 258.56 | 251.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 11:15:00 | 251.80 | 258.56 | 251.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 11:45:00 | 252.05 | 258.50 | 251.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 13:15:00 | 249.35 | 258.35 | 251.34 | SL hit (close<static) qty=1.00 sl=249.45 alert=retest2 |

### Cycle 6 — SELL (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 09:15:00 | 230.82 | 247.56 | 247.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 09:15:00 | 226.39 | 241.96 | 244.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 10:15:00 | 208.60 | 205.81 | 217.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-02 11:00:00 | 208.60 | 205.81 | 217.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 215.88 | 206.29 | 217.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:00:00 | 215.88 | 206.29 | 217.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 216.22 | 206.85 | 216.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:30:00 | 218.36 | 206.85 | 216.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 216.30 | 206.94 | 216.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 12:00:00 | 215.33 | 210.20 | 217.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 09:15:00 | 204.56 | 210.13 | 216.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-12-30 09:15:00 | 193.80 | 205.85 | 212.37 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 7 — BUY (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 09:15:00 | 217.14 | 199.81 | 199.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-27 09:15:00 | 219.31 | 201.00 | 200.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 15:15:00 | 218.50 | 218.52 | 211.36 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 09:15:00 | 222.80 | 218.52 | 211.36 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 11:45:00 | 219.44 | 218.55 | 211.48 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 12:15:00 | 219.85 | 218.55 | 211.48 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 12:45:00 | 219.67 | 218.55 | 211.52 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 212.25 | 218.40 | 212.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:45:00 | 212.69 | 218.40 | 212.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 210.89 | 218.32 | 212.07 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-25 11:15:00 | 210.89 | 218.32 | 212.07 | SL hit (close<ema400) qty=1.00 sl=212.07 alert=retest1 |

### Cycle 8 — SELL (started 2025-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-15 11:15:00 | 201.85 | 208.38 | 208.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 10:15:00 | 201.27 | 207.07 | 207.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 15:15:00 | 204.70 | 204.43 | 205.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-13 09:15:00 | 206.00 | 204.43 | 205.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 205.58 | 204.44 | 205.98 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 12:15:00 | 218.70 | 207.03 | 206.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 09:15:00 | 222.25 | 208.66 | 207.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 212.50 | 213.14 | 210.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 212.50 | 213.14 | 210.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 212.50 | 213.14 | 210.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 213.45 | 213.14 | 210.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 14:30:00 | 213.52 | 212.90 | 210.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 09:45:00 | 213.43 | 212.89 | 210.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 15:15:00 | 208.00 | 212.52 | 210.59 | SL hit (close<static) qty=1.00 sl=208.08 alert=retest2 |

### Cycle 10 — SELL (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 11:15:00 | 206.55 | 215.56 | 215.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 205.51 | 213.36 | 214.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 09:15:00 | 206.12 | 204.13 | 208.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 09:15:00 | 206.12 | 204.13 | 208.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 206.12 | 204.13 | 208.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 09:15:00 | 203.70 | 204.21 | 208.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 11:00:00 | 203.70 | 204.20 | 208.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 12:15:00 | 203.75 | 204.06 | 207.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 12:45:00 | 203.57 | 204.06 | 207.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 204.55 | 201.57 | 204.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 09:30:00 | 203.79 | 201.59 | 204.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:15:00 | 193.51 | 199.98 | 202.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:15:00 | 193.51 | 199.98 | 202.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:15:00 | 193.56 | 199.98 | 202.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:15:00 | 193.39 | 199.98 | 202.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:15:00 | 193.60 | 199.98 | 202.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-15 09:15:00 | 183.33 | 190.91 | 194.69 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-06-05 14:30:00 | 114.65 | 2023-06-12 10:15:00 | 116.25 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2023-06-06 09:45:00 | 114.75 | 2023-06-12 10:15:00 | 116.25 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2023-09-22 11:45:00 | 142.15 | 2023-09-22 14:15:00 | 139.85 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2023-10-12 15:00:00 | 143.05 | 2023-10-23 09:15:00 | 139.70 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2023-10-18 13:00:00 | 142.30 | 2023-10-23 09:15:00 | 139.70 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2023-11-23 14:30:00 | 136.95 | 2023-11-24 11:15:00 | 140.55 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2023-11-24 09:30:00 | 136.85 | 2023-11-24 11:15:00 | 140.55 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2024-03-14 11:30:00 | 200.25 | 2024-03-28 14:15:00 | 185.60 | STOP_HIT | 1.00 | -7.32% |
| BUY | retest2 | 2024-03-18 09:15:00 | 200.45 | 2024-03-28 14:15:00 | 185.60 | STOP_HIT | 1.00 | -7.41% |
| BUY | retest2 | 2024-04-01 15:00:00 | 201.70 | 2024-04-09 10:15:00 | 221.87 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-06 14:45:00 | 198.40 | 2024-05-24 14:15:00 | 193.15 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2024-06-03 12:45:00 | 193.65 | 2024-06-04 10:15:00 | 183.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 12:45:00 | 193.65 | 2024-06-04 11:15:00 | 174.28 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-07 15:15:00 | 194.35 | 2024-06-10 10:15:00 | 201.08 | STOP_HIT | 1.00 | -3.46% |
| BUY | retest2 | 2024-09-20 11:15:00 | 251.80 | 2024-09-20 13:15:00 | 249.35 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-09-20 11:45:00 | 252.05 | 2024-09-20 13:15:00 | 249.35 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2024-09-23 10:30:00 | 252.55 | 2024-09-24 14:15:00 | 249.90 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-09-23 13:15:00 | 251.90 | 2024-09-24 14:15:00 | 249.90 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-09-24 09:15:00 | 253.25 | 2024-09-25 09:15:00 | 248.20 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-09-24 13:00:00 | 252.95 | 2024-09-25 09:15:00 | 248.20 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-12-11 12:00:00 | 215.33 | 2024-12-18 09:15:00 | 204.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-11 12:00:00 | 215.33 | 2024-12-30 09:15:00 | 193.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-21 10:00:00 | 215.12 | 2025-02-25 09:15:00 | 217.14 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-02-21 10:30:00 | 215.31 | 2025-02-25 09:15:00 | 217.14 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-02-21 15:15:00 | 214.88 | 2025-02-25 09:15:00 | 217.14 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest1 | 2025-03-20 09:15:00 | 222.80 | 2025-03-25 11:15:00 | 210.89 | STOP_HIT | 1.00 | -5.35% |
| BUY | retest1 | 2025-03-20 11:45:00 | 219.44 | 2025-03-25 11:15:00 | 210.89 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest1 | 2025-03-20 12:15:00 | 219.85 | 2025-03-25 11:15:00 | 210.89 | STOP_HIT | 1.00 | -4.08% |
| BUY | retest1 | 2025-03-20 12:45:00 | 219.67 | 2025-03-25 11:15:00 | 210.89 | STOP_HIT | 1.00 | -4.00% |
| BUY | retest2 | 2025-06-13 10:15:00 | 213.45 | 2025-06-18 15:15:00 | 208.00 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-06-16 14:30:00 | 213.52 | 2025-06-18 15:15:00 | 208.00 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2025-06-17 09:45:00 | 213.43 | 2025-06-18 15:15:00 | 208.00 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-06-26 10:00:00 | 213.91 | 2025-08-06 11:15:00 | 215.50 | STOP_HIT | 1.00 | 0.74% |
| BUY | retest2 | 2025-08-01 09:15:00 | 220.56 | 2025-08-06 11:15:00 | 215.50 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-08-01 09:45:00 | 221.24 | 2025-08-06 11:15:00 | 215.50 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2025-08-05 09:15:00 | 223.15 | 2025-08-14 09:15:00 | 205.95 | STOP_HIT | 1.00 | -7.71% |
| SELL | retest2 | 2025-09-18 09:15:00 | 203.70 | 2025-11-03 09:15:00 | 193.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 11:00:00 | 203.70 | 2025-11-03 09:15:00 | 193.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 12:15:00 | 203.75 | 2025-11-03 09:15:00 | 193.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 12:45:00 | 203.57 | 2025-11-03 09:15:00 | 193.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-16 09:30:00 | 203.79 | 2025-11-03 09:15:00 | 193.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 09:15:00 | 203.70 | 2025-12-15 09:15:00 | 183.33 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-18 11:00:00 | 203.70 | 2025-12-15 09:15:00 | 183.33 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-22 12:15:00 | 203.75 | 2025-12-15 09:15:00 | 183.38 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-22 12:45:00 | 203.57 | 2025-12-15 09:15:00 | 183.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-16 09:30:00 | 203.79 | 2025-12-15 09:15:00 | 183.41 | TARGET_HIT | 0.50 | 10.00% |
