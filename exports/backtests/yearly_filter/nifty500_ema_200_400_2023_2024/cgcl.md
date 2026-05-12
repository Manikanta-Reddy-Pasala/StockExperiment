# Capri Global Capital Ltd. (CGCL)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 197.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT2_SKIP | 5 |
| ALERT3 | 67 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 71 |
| PARTIAL | 10 |
| TARGET_HIT | 8 |
| STOP_HIT | 66 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 81 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 60
- **Target hits / Stop hits / Partials:** 8 / 63 / 10
- **Avg / median % per leg:** -0.69% / -1.95%
- **Sum % (uncompounded):** -55.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 2 | 14.3% | 2 | 12 | 0 | -1.82% | -25.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 2 | 14.3% | 2 | 12 | 0 | -1.82% | -25.5% |
| SELL (all) | 67 | 19 | 28.4% | 6 | 51 | 10 | -0.45% | -30.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 67 | 19 | 28.4% | 6 | 51 | 10 | -0.45% | -30.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 81 | 21 | 25.9% | 8 | 63 | 10 | -0.69% | -55.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-17 12:15:00 | 168.83 | 163.51 | 163.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-17 13:15:00 | 169.26 | 163.57 | 163.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-28 10:15:00 | 194.00 | 194.26 | 187.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-28 10:30:00 | 194.95 | 194.26 | 187.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 15:15:00 | 189.98 | 194.98 | 190.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-16 09:15:00 | 190.48 | 194.98 | 190.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 09:15:00 | 191.14 | 194.95 | 190.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-16 10:15:00 | 191.74 | 194.95 | 190.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-16 13:15:00 | 192.05 | 194.82 | 190.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-18 11:15:00 | 188.30 | 194.55 | 190.24 | SL hit (close<static) qty=1.00 sl=188.86 alert=retest2 |

### Cycle 2 — SELL (started 2023-10-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 11:15:00 | 187.51 | 193.91 | 193.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 12:15:00 | 186.95 | 193.84 | 193.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 10:15:00 | 192.85 | 192.17 | 192.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 10:15:00 | 192.85 | 192.17 | 192.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 192.85 | 192.17 | 192.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 10:30:00 | 193.06 | 192.17 | 192.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 11:15:00 | 193.10 | 192.18 | 192.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-27 13:15:00 | 192.31 | 192.19 | 192.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-02 10:00:00 | 192.20 | 191.89 | 192.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-02 11:00:00 | 190.96 | 191.88 | 192.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-02 14:00:00 | 191.43 | 191.85 | 192.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 09:15:00 | 192.48 | 191.85 | 192.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-03 10:00:00 | 192.48 | 191.85 | 192.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 10:15:00 | 192.60 | 191.86 | 192.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-03 10:45:00 | 192.64 | 191.86 | 192.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 11:15:00 | 192.50 | 191.86 | 192.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-03 13:45:00 | 192.19 | 191.87 | 192.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-07 12:00:00 | 191.66 | 191.56 | 192.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 10:15:00 | 191.78 | 191.27 | 192.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-16 14:45:00 | 192.11 | 191.02 | 191.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 15:15:00 | 193.25 | 191.05 | 191.99 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-11-16 15:15:00 | 193.25 | 191.05 | 191.99 | SL hit (close>static) qty=1.00 sl=192.68 alert=retest2 |

### Cycle 3 — BUY (started 2023-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-29 13:15:00 | 192.85 | 191.08 | 191.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-01 12:15:00 | 193.26 | 191.18 | 191.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-12 14:15:00 | 224.74 | 226.41 | 215.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-12 15:00:00 | 224.74 | 226.41 | 215.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 13:15:00 | 228.80 | 241.15 | 230.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 14:00:00 | 228.80 | 241.15 | 230.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 14:15:00 | 229.80 | 241.04 | 230.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 15:15:00 | 228.70 | 241.04 | 230.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 15:15:00 | 228.70 | 240.92 | 230.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 09:15:00 | 225.30 | 240.92 | 230.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 10:15:00 | 224.75 | 240.63 | 230.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 11:00:00 | 224.75 | 240.63 | 230.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 12:15:00 | 201.10 | 222.92 | 222.96 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 15:15:00 | 239.50 | 222.16 | 222.10 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 09:15:00 | 214.00 | 223.35 | 223.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 09:15:00 | 209.00 | 222.62 | 223.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 10:15:00 | 222.55 | 221.19 | 222.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 10:15:00 | 222.55 | 221.19 | 222.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 10:15:00 | 222.55 | 221.19 | 222.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 10:45:00 | 225.55 | 221.19 | 222.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 11:15:00 | 224.30 | 221.22 | 222.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 12:00:00 | 224.30 | 221.22 | 222.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 12:15:00 | 223.45 | 221.24 | 222.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 13:15:00 | 221.65 | 221.24 | 222.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 14:15:00 | 221.75 | 221.26 | 222.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 212.75 | 221.31 | 222.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 210.57 | 221.25 | 222.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 210.66 | 221.25 | 222.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-06-04 12:15:00 | 199.49 | 220.83 | 222.01 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 7 — BUY (started 2024-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 15:15:00 | 206.60 | 203.01 | 203.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 14:15:00 | 207.20 | 203.19 | 203.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 13:15:00 | 202.85 | 203.26 | 203.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 13:15:00 | 202.85 | 203.26 | 203.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 13:15:00 | 202.85 | 203.26 | 203.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 14:00:00 | 202.85 | 203.26 | 203.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 14:15:00 | 202.90 | 203.26 | 203.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 15:00:00 | 202.90 | 203.26 | 203.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 15:15:00 | 203.20 | 203.26 | 203.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:15:00 | 201.01 | 203.26 | 203.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 200.72 | 203.23 | 203.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:30:00 | 201.48 | 203.23 | 203.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 201.31 | 203.21 | 203.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:30:00 | 199.60 | 203.21 | 203.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 202.30 | 203.15 | 203.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 202.97 | 203.15 | 203.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 202.38 | 203.14 | 203.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:15:00 | 201.98 | 203.14 | 203.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 201.20 | 203.13 | 203.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 11:00:00 | 201.20 | 203.13 | 203.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 15:15:00 | 200.50 | 203.00 | 203.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 09:15:00 | 197.99 | 202.95 | 202.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-24 09:15:00 | 198.66 | 186.27 | 191.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 198.66 | 186.27 | 191.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 198.66 | 186.27 | 191.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 09:15:00 | 186.77 | 187.16 | 191.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 177.43 | 186.95 | 191.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-29 09:15:00 | 168.09 | 186.26 | 191.09 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 174.90 | 166.50 | 166.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 12:15:00 | 177.95 | 166.61 | 166.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 14:15:00 | 167.90 | 168.17 | 167.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 15:00:00 | 167.90 | 168.17 | 167.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 167.99 | 168.17 | 167.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 167.12 | 168.17 | 167.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 166.05 | 168.15 | 167.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:30:00 | 166.18 | 168.15 | 167.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 166.24 | 168.13 | 167.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:30:00 | 165.63 | 168.13 | 167.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 166.77 | 168.11 | 167.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:45:00 | 166.95 | 168.11 | 167.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 167.77 | 168.10 | 167.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 11:30:00 | 168.64 | 167.73 | 167.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 09:15:00 | 169.28 | 167.88 | 167.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-24 09:15:00 | 185.50 | 174.02 | 171.05 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 13:15:00 | 180.01 | 190.91 | 190.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 12:15:00 | 179.29 | 188.36 | 189.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 185.50 | 183.80 | 186.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 185.50 | 183.80 | 186.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 185.50 | 183.80 | 186.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:45:00 | 185.98 | 183.80 | 186.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 185.77 | 183.82 | 186.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 185.77 | 183.82 | 186.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 186.57 | 183.85 | 186.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:00:00 | 186.57 | 183.85 | 186.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 185.97 | 183.87 | 186.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 13:15:00 | 185.40 | 183.92 | 186.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 10:00:00 | 185.37 | 183.97 | 186.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 11:15:00 | 188.51 | 184.04 | 186.44 | SL hit (close>static) qty=1.00 sl=187.20 alert=retest2 |

### Cycle 11 — BUY (started 2026-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 15:15:00 | 182.00 | 173.11 | 173.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 09:15:00 | 185.50 | 173.23 | 173.17 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-08-16 10:15:00 | 191.74 | 2023-08-18 11:15:00 | 188.30 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2023-08-16 13:15:00 | 192.05 | 2023-08-18 11:15:00 | 188.30 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2023-08-21 09:15:00 | 206.26 | 2023-09-25 11:15:00 | 191.63 | STOP_HIT | 1.00 | -7.09% |
| BUY | retest2 | 2023-08-22 09:15:00 | 201.39 | 2023-09-25 11:15:00 | 191.63 | STOP_HIT | 1.00 | -4.85% |
| BUY | retest2 | 2023-09-21 11:00:00 | 201.23 | 2023-09-26 10:15:00 | 192.00 | STOP_HIT | 1.00 | -4.59% |
| BUY | retest2 | 2023-09-25 09:15:00 | 199.60 | 2023-09-26 10:15:00 | 192.00 | STOP_HIT | 1.00 | -3.81% |
| BUY | retest2 | 2023-09-26 09:45:00 | 199.15 | 2023-10-09 11:15:00 | 192.44 | STOP_HIT | 1.00 | -3.37% |
| BUY | retest2 | 2023-09-26 10:15:00 | 199.03 | 2023-10-09 11:15:00 | 192.44 | STOP_HIT | 1.00 | -3.31% |
| BUY | retest2 | 2023-09-29 09:15:00 | 201.25 | 2023-10-09 11:15:00 | 192.44 | STOP_HIT | 1.00 | -4.38% |
| BUY | retest2 | 2023-09-29 14:45:00 | 196.75 | 2023-10-09 11:15:00 | 192.44 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2023-10-03 15:15:00 | 196.20 | 2023-10-16 15:15:00 | 188.28 | STOP_HIT | 1.00 | -4.04% |
| BUY | retest2 | 2023-10-04 15:00:00 | 196.36 | 2023-10-16 15:15:00 | 188.28 | STOP_HIT | 1.00 | -4.11% |
| SELL | retest2 | 2023-10-27 13:15:00 | 192.31 | 2023-11-16 15:15:00 | 193.25 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2023-11-02 10:00:00 | 192.20 | 2023-11-16 15:15:00 | 193.25 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2023-11-02 11:00:00 | 190.96 | 2023-11-16 15:15:00 | 193.25 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2023-11-02 14:00:00 | 191.43 | 2023-11-16 15:15:00 | 193.25 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2023-11-03 13:45:00 | 192.19 | 2023-11-17 09:15:00 | 193.94 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2023-11-07 12:00:00 | 191.66 | 2023-11-17 09:15:00 | 193.94 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2023-11-13 10:15:00 | 191.78 | 2023-11-17 09:15:00 | 193.94 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2023-11-16 14:45:00 | 192.11 | 2023-11-17 09:15:00 | 193.94 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2023-11-17 13:45:00 | 189.90 | 2023-12-19 14:15:00 | 196.23 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2023-11-17 15:00:00 | 189.33 | 2023-12-19 14:15:00 | 196.23 | STOP_HIT | 1.00 | -3.64% |
| SELL | retest2 | 2023-11-20 11:00:00 | 189.61 | 2023-12-19 14:15:00 | 196.23 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest2 | 2023-11-22 15:00:00 | 189.98 | 2023-12-27 12:15:00 | 197.26 | STOP_HIT | 1.00 | -3.83% |
| SELL | retest2 | 2023-12-06 10:30:00 | 185.94 | 2023-12-27 12:15:00 | 197.26 | STOP_HIT | 1.00 | -6.09% |
| SELL | retest2 | 2023-12-06 15:15:00 | 186.03 | 2023-12-27 12:15:00 | 197.26 | STOP_HIT | 1.00 | -6.04% |
| SELL | retest2 | 2023-12-07 09:45:00 | 186.04 | 2023-12-27 12:15:00 | 197.26 | STOP_HIT | 1.00 | -6.03% |
| SELL | retest2 | 2023-12-07 10:45:00 | 186.33 | 2023-12-27 12:15:00 | 197.26 | STOP_HIT | 1.00 | -5.87% |
| SELL | retest2 | 2023-12-08 09:15:00 | 190.16 | 2023-12-27 12:15:00 | 197.26 | STOP_HIT | 1.00 | -3.73% |
| SELL | retest2 | 2023-12-12 11:15:00 | 192.39 | 2023-12-27 12:15:00 | 197.26 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2023-12-12 11:45:00 | 191.89 | 2023-12-27 12:15:00 | 197.26 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2023-12-20 12:15:00 | 192.20 | 2023-12-27 12:15:00 | 197.26 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2023-12-26 11:15:00 | 191.25 | 2023-12-29 13:15:00 | 192.85 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2023-12-26 12:30:00 | 191.16 | 2023-12-29 13:15:00 | 192.85 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2023-12-26 14:00:00 | 191.19 | 2023-12-29 13:15:00 | 192.85 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2023-12-27 10:45:00 | 191.28 | 2023-12-29 13:15:00 | 192.85 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-06-03 13:15:00 | 221.65 | 2024-06-04 09:15:00 | 210.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 14:15:00 | 221.75 | 2024-06-04 09:15:00 | 210.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 13:15:00 | 221.65 | 2024-06-04 12:15:00 | 199.49 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-03 14:15:00 | 221.75 | 2024-06-04 12:15:00 | 199.58 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 212.75 | 2024-06-04 12:15:00 | 202.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 212.75 | 2024-06-14 14:15:00 | 224.98 | STOP_HIT | 0.50 | -5.75% |
| SELL | retest2 | 2024-06-19 12:45:00 | 220.25 | 2024-06-20 13:15:00 | 225.01 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2024-06-24 11:00:00 | 216.00 | 2024-06-25 09:15:00 | 228.61 | STOP_HIT | 1.00 | -5.84% |
| SELL | retest2 | 2024-06-24 13:15:00 | 217.59 | 2024-06-25 09:15:00 | 228.61 | STOP_HIT | 1.00 | -5.06% |
| SELL | retest2 | 2024-06-25 11:30:00 | 218.38 | 2024-07-10 14:15:00 | 225.23 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2024-06-25 12:45:00 | 217.64 | 2024-07-10 14:15:00 | 225.23 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest2 | 2024-06-27 10:45:00 | 216.00 | 2024-07-10 14:15:00 | 225.23 | STOP_HIT | 1.00 | -4.27% |
| SELL | retest2 | 2024-06-28 11:00:00 | 215.00 | 2024-07-10 14:15:00 | 225.23 | STOP_HIT | 1.00 | -4.76% |
| SELL | retest2 | 2024-06-28 14:15:00 | 216.00 | 2024-07-10 14:15:00 | 225.23 | STOP_HIT | 1.00 | -4.27% |
| SELL | retest2 | 2024-07-01 10:30:00 | 216.18 | 2024-07-10 14:15:00 | 225.23 | STOP_HIT | 1.00 | -4.19% |
| SELL | retest2 | 2024-07-11 09:15:00 | 222.10 | 2024-07-12 14:15:00 | 228.26 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2024-07-11 14:00:00 | 222.99 | 2024-07-12 14:15:00 | 228.26 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2024-07-12 09:30:00 | 221.65 | 2024-07-12 14:15:00 | 228.26 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2024-07-15 09:15:00 | 222.82 | 2024-07-19 10:15:00 | 211.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-15 09:15:00 | 222.82 | 2024-08-01 14:15:00 | 225.05 | STOP_HIT | 0.50 | -1.00% |
| SELL | retest2 | 2024-07-23 12:15:00 | 210.05 | 2024-08-01 14:15:00 | 225.05 | STOP_HIT | 1.00 | -7.14% |
| SELL | retest2 | 2024-07-23 13:00:00 | 210.77 | 2024-08-01 14:15:00 | 225.05 | STOP_HIT | 1.00 | -6.78% |
| SELL | retest2 | 2024-07-23 14:15:00 | 210.97 | 2024-08-01 14:15:00 | 225.05 | STOP_HIT | 1.00 | -6.67% |
| SELL | retest2 | 2024-07-24 14:15:00 | 210.90 | 2024-08-01 14:15:00 | 225.05 | STOP_HIT | 1.00 | -6.71% |
| SELL | retest2 | 2024-08-02 09:15:00 | 217.79 | 2024-08-05 12:15:00 | 206.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-02 09:15:00 | 217.79 | 2024-08-21 09:15:00 | 220.63 | STOP_HIT | 0.50 | -1.30% |
| SELL | retest2 | 2024-08-21 09:30:00 | 221.55 | 2024-08-21 15:15:00 | 210.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-21 09:30:00 | 221.55 | 2024-08-21 15:15:00 | 211.75 | STOP_HIT | 0.50 | 4.42% |
| SELL | retest2 | 2025-01-27 09:15:00 | 186.77 | 2025-01-28 09:15:00 | 177.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-27 09:15:00 | 186.77 | 2025-01-29 09:15:00 | 168.09 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-27 13:00:00 | 186.89 | 2025-03-27 13:15:00 | 168.20 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-27 13:30:00 | 171.95 | 2025-03-27 14:15:00 | 163.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-27 13:30:00 | 171.95 | 2025-03-28 09:15:00 | 175.53 | STOP_HIT | 0.50 | -2.08% |
| BUY | retest2 | 2025-07-08 11:30:00 | 168.64 | 2025-07-24 09:15:00 | 185.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-10 09:15:00 | 169.28 | 2025-07-24 09:15:00 | 186.21 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-02 13:15:00 | 185.40 | 2026-01-05 11:15:00 | 188.51 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-01-05 10:00:00 | 185.37 | 2026-01-05 11:15:00 | 188.51 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2026-01-07 09:15:00 | 185.61 | 2026-01-07 14:15:00 | 187.48 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2026-01-07 10:00:00 | 185.62 | 2026-01-07 14:15:00 | 187.48 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-01-07 15:15:00 | 184.90 | 2026-01-20 09:15:00 | 175.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 10:45:00 | 185.60 | 2026-01-20 09:15:00 | 176.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 15:15:00 | 184.90 | 2026-01-22 10:15:00 | 167.04 | TARGET_HIT | 0.50 | 9.66% |
| SELL | retest2 | 2026-01-09 10:45:00 | 185.60 | 2026-01-23 14:15:00 | 166.41 | TARGET_HIT | 0.50 | 10.34% |
| SELL | retest2 | 2026-04-15 09:45:00 | 185.89 | 2026-04-17 15:15:00 | 182.00 | STOP_HIT | 1.00 | 2.09% |
| SELL | retest2 | 2026-04-15 13:00:00 | 185.92 | 2026-04-17 15:15:00 | 182.00 | STOP_HIT | 1.00 | 2.11% |
