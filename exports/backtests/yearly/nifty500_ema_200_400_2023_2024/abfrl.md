# Aditya Birla Fashion and Retail Ltd. (ABFRL)

## Backtest Summary

- **Window:** 2022-04-07 13:15:00 → 2026-05-08 15:15:00 (7050 bars)
- **Last close:** 66.15
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 6 |
| ALERT3 | 69 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 55 |
| PARTIAL | 8 |
| TARGET_HIT | 11 |
| STOP_HIT | 44 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 63 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 42
- **Target hits / Stop hits / Partials:** 11 / 44 / 8
- **Avg / median % per leg:** 1.18% / -1.50%
- **Sum % (uncompounded):** 74.18%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 5 | 16.1% | 5 | 26 | 0 | 0.17% | 5.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 31 | 5 | 16.1% | 5 | 26 | 0 | 0.17% | 5.1% |
| SELL (all) | 32 | 16 | 50.0% | 6 | 18 | 8 | 2.16% | 69.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 32 | 16 | 50.0% | 6 | 18 | 8 | 2.16% | 69.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 63 | 21 | 33.3% | 11 | 44 | 8 | 1.18% | 74.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 09:15:00 | 223.60 | 213.66 | 213.64 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 12:15:00 | 200.90 | 213.82 | 213.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-08 13:15:00 | 200.30 | 213.69 | 213.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-09 13:15:00 | 214.35 | 213.41 | 213.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 13:15:00 | 214.35 | 213.41 | 213.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 13:15:00 | 214.35 | 213.41 | 213.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-09 14:00:00 | 214.35 | 213.41 | 213.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 14:15:00 | 215.90 | 213.44 | 213.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-09 15:00:00 | 215.90 | 213.44 | 213.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 14:15:00 | 213.20 | 213.55 | 213.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-10 14:30:00 | 214.45 | 213.55 | 213.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 15:15:00 | 213.60 | 213.55 | 213.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-11 09:15:00 | 213.35 | 213.55 | 213.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 09:15:00 | 212.80 | 213.54 | 213.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-11 10:30:00 | 212.55 | 213.53 | 213.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-11 11:00:00 | 212.05 | 213.53 | 213.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-14 15:00:00 | 212.15 | 213.30 | 213.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 10:00:00 | 212.25 | 213.13 | 213.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 11:15:00 | 214.20 | 213.13 | 213.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-18 12:00:00 | 214.20 | 213.13 | 213.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 12:15:00 | 214.25 | 213.14 | 213.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-18 12:30:00 | 214.15 | 213.14 | 213.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-08-18 13:15:00 | 215.70 | 213.17 | 213.45 | SL hit (close>static) qty=1.00 sl=214.45 alert=retest2 |

### Cycle 3 — BUY (started 2023-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 14:15:00 | 218.90 | 213.71 | 213.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-23 09:15:00 | 220.95 | 213.84 | 213.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-25 11:15:00 | 214.45 | 214.57 | 214.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 11:15:00 | 214.45 | 214.57 | 214.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 11:15:00 | 214.45 | 214.57 | 214.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 12:00:00 | 214.45 | 214.57 | 214.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 12:15:00 | 214.20 | 214.57 | 214.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 12:30:00 | 213.75 | 214.57 | 214.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 13:15:00 | 214.25 | 214.57 | 214.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 14:15:00 | 213.90 | 214.57 | 214.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 14:15:00 | 213.95 | 214.56 | 214.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 15:15:00 | 213.00 | 214.56 | 214.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 15:15:00 | 213.00 | 214.55 | 214.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-28 09:15:00 | 213.25 | 214.55 | 214.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 10:15:00 | 214.25 | 214.54 | 214.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-28 10:45:00 | 213.85 | 214.54 | 214.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 11:15:00 | 213.95 | 214.54 | 214.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-28 12:00:00 | 213.95 | 214.54 | 214.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 12:15:00 | 215.15 | 214.54 | 214.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-28 13:30:00 | 215.50 | 214.55 | 214.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-30 09:15:00 | 217.65 | 214.68 | 214.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-22 10:30:00 | 216.15 | 221.44 | 218.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-22 14:15:00 | 216.10 | 221.29 | 218.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 10:15:00 | 215.45 | 221.05 | 218.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-25 11:00:00 | 215.45 | 221.05 | 218.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-09-28 12:15:00 | 212.55 | 219.96 | 218.35 | SL hit (close<static) qty=1.00 sl=213.75 alert=retest2 |

### Cycle 4 — SELL (started 2023-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 14:15:00 | 216.75 | 218.79 | 218.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-10 15:15:00 | 216.00 | 218.76 | 218.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-13 09:15:00 | 219.50 | 218.76 | 218.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-13 09:15:00 | 219.50 | 218.76 | 218.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 09:15:00 | 219.50 | 218.76 | 218.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-13 10:00:00 | 219.50 | 218.76 | 218.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 10:15:00 | 216.15 | 218.74 | 218.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-15 14:45:00 | 215.05 | 218.50 | 218.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-16 15:15:00 | 215.05 | 218.32 | 218.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-17 11:00:00 | 215.00 | 218.22 | 218.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-17 11:30:00 | 214.90 | 218.19 | 218.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 220.30 | 217.51 | 218.08 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-11-23 09:15:00 | 220.30 | 217.51 | 218.08 | SL hit (close>static) qty=1.00 sl=220.05 alert=retest2 |

### Cycle 5 — BUY (started 2023-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 15:15:00 | 227.70 | 218.68 | 218.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 09:15:00 | 231.10 | 218.81 | 218.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-13 11:15:00 | 226.70 | 226.89 | 223.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-13 12:00:00 | 226.70 | 226.89 | 223.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 220.75 | 228.09 | 224.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 14:00:00 | 220.75 | 228.09 | 224.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 14:15:00 | 216.70 | 227.97 | 224.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 15:00:00 | 216.70 | 227.97 | 224.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 10:15:00 | 223.75 | 225.66 | 224.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 12:30:00 | 225.15 | 225.51 | 224.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-01-04 11:15:00 | 247.67 | 227.88 | 225.37 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 12:15:00 | 221.85 | 232.37 | 232.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 14:15:00 | 220.30 | 232.15 | 232.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-02 09:15:00 | 245.15 | 217.21 | 223.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-02 09:15:00 | 245.15 | 217.21 | 223.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 09:15:00 | 245.15 | 217.21 | 223.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-02 10:00:00 | 245.15 | 217.21 | 223.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 10:15:00 | 230.30 | 225.96 | 226.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 10:45:00 | 231.40 | 225.96 | 226.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 231.00 | 226.26 | 226.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 11:00:00 | 231.00 | 226.26 | 226.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 10:15:00 | 227.85 | 227.09 | 227.12 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-04-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-19 12:15:00 | 231.30 | 227.16 | 227.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-19 13:15:00 | 232.15 | 227.21 | 227.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 268.75 | 269.70 | 256.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-04 11:00:00 | 268.75 | 269.70 | 256.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 265.00 | 269.60 | 256.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:30:00 | 261.40 | 269.60 | 256.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 311.35 | 316.51 | 301.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 306.00 | 316.51 | 301.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 311.40 | 323.19 | 311.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 310.05 | 323.08 | 311.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 310.25 | 322.96 | 311.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:00:00 | 310.25 | 322.96 | 311.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 11:15:00 | 312.25 | 322.85 | 311.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:45:00 | 310.90 | 322.85 | 311.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 12:15:00 | 311.60 | 322.74 | 311.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 12:30:00 | 311.15 | 322.74 | 311.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 13:15:00 | 310.00 | 322.61 | 311.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 14:00:00 | 310.00 | 322.61 | 311.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 15:15:00 | 313.00 | 322.40 | 311.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 09:30:00 | 315.20 | 322.33 | 311.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 13:00:00 | 314.00 | 320.87 | 312.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 12:00:00 | 313.80 | 320.70 | 313.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 13:00:00 | 316.05 | 320.66 | 313.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 314.75 | 320.56 | 313.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 15:00:00 | 314.75 | 320.56 | 313.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 311.50 | 320.36 | 313.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:45:00 | 311.95 | 320.36 | 313.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 11:15:00 | 311.05 | 320.27 | 313.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 14:45:00 | 312.50 | 320.03 | 313.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-02 09:15:00 | 309.10 | 319.52 | 313.66 | SL hit (close<static) qty=1.00 sl=310.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 10:15:00 | 304.40 | 323.92 | 323.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 10:15:00 | 301.20 | 322.52 | 323.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 09:15:00 | 306.00 | 305.36 | 312.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 13:15:00 | 309.60 | 305.64 | 312.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 13:15:00 | 309.60 | 305.64 | 312.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 13:30:00 | 309.75 | 305.64 | 312.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 316.60 | 305.81 | 312.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:00:00 | 316.60 | 305.81 | 312.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 316.00 | 305.91 | 312.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:30:00 | 317.50 | 305.91 | 312.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 312.30 | 306.72 | 312.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 09:30:00 | 310.80 | 308.88 | 312.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 13:15:00 | 311.00 | 308.67 | 312.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 10:00:00 | 310.50 | 308.68 | 312.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 10:30:00 | 311.15 | 308.75 | 312.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 14:15:00 | 295.26 | 306.88 | 310.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 14:15:00 | 295.45 | 306.88 | 310.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 14:15:00 | 294.97 | 306.88 | 310.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 14:15:00 | 295.59 | 306.88 | 310.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-12-23 09:15:00 | 279.72 | 303.26 | 308.43 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 9 — BUY (started 2025-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 12:15:00 | 278.25 | 262.39 | 262.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 281.15 | 263.98 | 263.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 09:15:00 | 89.95 | 265.67 | 264.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 89.95 | 265.67 | 264.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 89.95 | 265.67 | 264.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:00:00 | 89.95 | 265.67 | 264.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 11:15:00 | 90.45 | 262.21 | 262.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 12:15:00 | 90.00 | 260.49 | 261.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 09:15:00 | 79.94 | 77.73 | 95.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-22 10:00:00 | 79.94 | 77.73 | 95.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 89.38 | 80.20 | 91.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:30:00 | 91.27 | 80.20 | 91.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 90.55 | 80.57 | 91.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 10:30:00 | 89.55 | 81.31 | 91.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 11:00:00 | 89.65 | 81.31 | 91.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 09:45:00 | 88.90 | 83.23 | 90.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 11:15:00 | 89.34 | 83.30 | 90.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 90.71 | 83.37 | 90.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:45:00 | 90.54 | 83.37 | 90.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 90.74 | 83.44 | 90.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 13:15:00 | 90.32 | 83.44 | 90.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 15:15:00 | 89.99 | 83.58 | 90.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 10:15:00 | 91.82 | 84.21 | 90.79 | SL hit (close>static) qty=1.00 sl=91.60 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-08-11 10:30:00 | 212.55 | 2023-08-18 13:15:00 | 215.70 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2023-08-11 11:00:00 | 212.05 | 2023-08-18 13:15:00 | 215.70 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2023-08-14 15:00:00 | 212.15 | 2023-08-18 13:15:00 | 215.70 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2023-08-18 10:00:00 | 212.25 | 2023-08-18 13:15:00 | 215.70 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2023-08-28 13:30:00 | 215.50 | 2023-09-28 12:15:00 | 212.55 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2023-08-30 09:15:00 | 217.65 | 2023-09-28 12:15:00 | 212.55 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2023-09-22 10:30:00 | 216.15 | 2023-09-28 12:15:00 | 212.55 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2023-09-22 14:15:00 | 216.10 | 2023-09-28 12:15:00 | 212.55 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2023-10-10 09:30:00 | 217.80 | 2023-10-20 09:15:00 | 239.58 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-23 13:45:00 | 217.40 | 2023-10-23 15:15:00 | 214.00 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2023-10-23 14:15:00 | 217.15 | 2023-10-23 15:15:00 | 214.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2023-10-25 09:30:00 | 219.60 | 2023-10-25 12:15:00 | 212.15 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2023-11-07 13:45:00 | 218.95 | 2023-11-09 09:15:00 | 214.90 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2023-11-15 14:45:00 | 215.05 | 2023-11-23 09:15:00 | 220.30 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2023-11-16 15:15:00 | 215.05 | 2023-11-23 09:15:00 | 220.30 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2023-11-17 11:00:00 | 215.00 | 2023-11-23 09:15:00 | 220.30 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2023-11-17 11:30:00 | 214.90 | 2023-11-23 09:15:00 | 220.30 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2024-01-01 12:30:00 | 225.15 | 2024-01-04 11:15:00 | 247.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-10 13:00:00 | 225.60 | 2024-01-18 09:15:00 | 221.95 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2024-01-17 12:15:00 | 225.25 | 2024-01-18 09:15:00 | 221.95 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-01-17 13:45:00 | 225.10 | 2024-01-18 09:15:00 | 221.95 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2024-01-19 12:30:00 | 227.45 | 2024-01-19 14:15:00 | 223.90 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-01-23 11:00:00 | 227.35 | 2024-01-23 12:15:00 | 225.30 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-01-24 09:15:00 | 231.30 | 2024-02-02 09:15:00 | 254.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-21 15:00:00 | 227.60 | 2024-02-26 14:15:00 | 225.45 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-03-04 09:15:00 | 233.60 | 2024-03-05 14:15:00 | 228.85 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2024-03-04 11:00:00 | 233.70 | 2024-03-05 14:15:00 | 228.85 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2024-08-16 09:30:00 | 315.20 | 2024-09-02 09:15:00 | 309.10 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2024-08-23 13:00:00 | 314.00 | 2024-09-02 09:15:00 | 309.10 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-08-28 12:00:00 | 313.80 | 2024-09-02 09:15:00 | 309.10 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-08-28 13:00:00 | 316.05 | 2024-09-02 09:15:00 | 309.10 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2024-08-29 14:45:00 | 312.50 | 2024-09-02 09:15:00 | 309.10 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-09-02 10:30:00 | 315.90 | 2024-09-04 13:15:00 | 310.30 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-09-02 13:30:00 | 312.75 | 2024-09-04 13:15:00 | 310.30 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-09-05 09:15:00 | 312.50 | 2024-09-06 10:15:00 | 309.95 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-09-06 11:45:00 | 311.20 | 2024-09-09 09:15:00 | 305.85 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-09-09 12:45:00 | 311.35 | 2024-09-23 12:15:00 | 342.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-09 13:45:00 | 311.20 | 2024-09-23 12:15:00 | 342.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-23 09:15:00 | 312.70 | 2024-10-25 09:15:00 | 299.45 | STOP_HIT | 1.00 | -4.24% |
| SELL | retest2 | 2024-12-05 09:30:00 | 310.80 | 2024-12-17 14:15:00 | 295.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-09 13:15:00 | 311.00 | 2024-12-17 14:15:00 | 295.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-10 10:00:00 | 310.50 | 2024-12-17 14:15:00 | 294.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-11 10:30:00 | 311.15 | 2024-12-17 14:15:00 | 295.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-05 09:30:00 | 310.80 | 2024-12-23 09:15:00 | 279.72 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-09 13:15:00 | 311.00 | 2024-12-23 09:15:00 | 279.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-10 10:00:00 | 310.50 | 2024-12-23 09:15:00 | 279.45 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-11 10:30:00 | 311.15 | 2024-12-23 09:15:00 | 280.03 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-01 15:15:00 | 282.55 | 2025-02-03 10:15:00 | 290.25 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-02-04 10:30:00 | 282.35 | 2025-02-10 11:15:00 | 268.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 09:15:00 | 282.85 | 2025-02-10 11:15:00 | 268.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-04 10:30:00 | 282.35 | 2025-02-11 13:15:00 | 254.12 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-06 09:15:00 | 282.85 | 2025-02-11 13:15:00 | 254.57 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-10 10:30:00 | 89.55 | 2025-09-19 10:15:00 | 91.82 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2025-09-10 11:00:00 | 89.65 | 2025-09-19 10:15:00 | 91.82 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-09-17 09:45:00 | 88.90 | 2025-09-19 14:15:00 | 91.74 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2025-09-17 11:15:00 | 89.34 | 2025-09-22 09:15:00 | 92.52 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2025-09-17 13:15:00 | 90.32 | 2025-09-22 09:15:00 | 92.52 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-09-17 15:15:00 | 89.99 | 2025-09-22 09:15:00 | 92.52 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-09-19 13:00:00 | 90.44 | 2025-09-22 09:15:00 | 92.52 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-09-25 09:15:00 | 90.05 | 2025-09-26 11:15:00 | 85.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 10:30:00 | 89.11 | 2025-09-26 13:15:00 | 84.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 09:15:00 | 90.05 | 2025-10-01 14:15:00 | 86.26 | STOP_HIT | 0.50 | 4.21% |
| SELL | retest2 | 2025-09-25 10:30:00 | 89.11 | 2025-10-01 14:15:00 | 86.26 | STOP_HIT | 0.50 | 3.20% |
