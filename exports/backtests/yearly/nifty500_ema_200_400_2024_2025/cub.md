# City Union Bank Ltd. (CUB)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 258.95
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 5 |
| ALERT3 | 43 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 34 |
| PARTIAL | 0 |
| TARGET_HIT | 11 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 34 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 21
- **Target hits / Stop hits / Partials:** 11 / 23 / 0
- **Avg / median % per leg:** 1.48% / -1.44%
- **Sum % (uncompounded):** 50.23%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 10 | 50.0% | 10 | 10 | 0 | 3.37% | 67.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 20 | 10 | 50.0% | 10 | 10 | 0 | 3.37% | 67.5% |
| SELL (all) | 14 | 3 | 21.4% | 1 | 13 | 0 | -1.23% | -17.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 3 | 21.4% | 1 | 13 | 0 | -1.23% | -17.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 34 | 13 | 38.2% | 11 | 23 | 0 | 1.48% | 50.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 11:15:00 | 154.50 | 163.97 | 164.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 13:15:00 | 154.22 | 163.78 | 163.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-22 09:15:00 | 168.11 | 159.80 | 161.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-22 09:15:00 | 168.11 | 159.80 | 161.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 168.11 | 159.80 | 161.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:45:00 | 169.00 | 159.80 | 161.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 172.17 | 159.92 | 161.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-22 11:00:00 | 172.17 | 159.92 | 161.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 12:15:00 | 174.94 | 163.19 | 163.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 13:15:00 | 176.15 | 163.32 | 163.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 12:15:00 | 169.74 | 170.23 | 167.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-13 13:00:00 | 169.74 | 170.23 | 167.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 168.06 | 170.53 | 167.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:30:00 | 167.78 | 170.53 | 167.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 169.59 | 170.52 | 167.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:30:00 | 167.40 | 170.52 | 167.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 174.51 | 179.37 | 175.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 174.51 | 179.37 | 175.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 177.00 | 179.35 | 175.13 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 12:15:00 | 165.88 | 173.23 | 173.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 14:15:00 | 163.72 | 171.84 | 172.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 171.60 | 171.28 | 172.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 10:00:00 | 171.60 | 171.28 | 172.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 172.70 | 171.29 | 172.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:45:00 | 173.21 | 171.29 | 172.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 172.55 | 171.31 | 172.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:30:00 | 172.73 | 171.31 | 172.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 13:15:00 | 172.96 | 171.33 | 172.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 14:00:00 | 172.96 | 171.33 | 172.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 172.98 | 171.34 | 172.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 14:45:00 | 173.10 | 171.34 | 172.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 172.30 | 171.39 | 172.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 10:30:00 | 172.65 | 171.39 | 172.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 11:15:00 | 172.65 | 171.40 | 172.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 12:00:00 | 172.65 | 171.40 | 172.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 12:15:00 | 172.13 | 171.41 | 172.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 13:15:00 | 170.75 | 171.41 | 172.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 15:00:00 | 171.77 | 171.40 | 172.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 15:15:00 | 175.50 | 171.44 | 172.18 | SL hit (close>static) qty=1.00 sl=172.69 alert=retest2 |

### Cycle 4 — BUY (started 2025-02-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 10:15:00 | 173.63 | 172.76 | 172.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 12:15:00 | 174.05 | 172.77 | 172.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 09:15:00 | 172.58 | 172.81 | 172.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 09:15:00 | 172.58 | 172.81 | 172.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 172.58 | 172.81 | 172.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:00:00 | 172.58 | 172.81 | 172.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 171.74 | 172.80 | 172.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 11:00:00 | 171.74 | 172.80 | 172.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 11:15:00 | 170.64 | 172.77 | 172.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 12:00:00 | 170.64 | 172.77 | 172.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2025-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 13:15:00 | 170.31 | 172.73 | 172.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 168.27 | 172.67 | 172.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 15:15:00 | 156.50 | 156.33 | 161.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-21 09:15:00 | 162.10 | 156.41 | 161.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 162.10 | 156.41 | 161.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 09:30:00 | 162.00 | 156.41 | 161.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 10:15:00 | 162.00 | 156.47 | 161.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-21 13:15:00 | 160.26 | 156.58 | 161.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-27 14:15:00 | 144.23 | 157.24 | 161.04 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 10:15:00 | 182.40 | 162.75 | 162.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 11:15:00 | 184.25 | 162.97 | 162.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 10:15:00 | 192.86 | 193.07 | 184.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 11:00:00 | 192.86 | 193.07 | 184.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 206.15 | 208.99 | 201.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:45:00 | 210.60 | 208.91 | 202.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 15:00:00 | 209.27 | 211.08 | 205.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 12:30:00 | 209.65 | 210.97 | 205.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 13:00:00 | 209.34 | 210.97 | 205.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 206.99 | 211.76 | 206.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:00:00 | 206.99 | 211.76 | 206.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 207.50 | 211.72 | 206.90 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-28 11:15:00 | 200.09 | 210.50 | 206.61 | SL hit (close<static) qty=1.00 sl=200.30 alert=retest2 |

### Cycle 7 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 246.70 | 275.28 | 275.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 15:15:00 | 246.30 | 275.00 | 275.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 257.50 | 252.87 | 260.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 14:15:00 | 255.00 | 252.98 | 260.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 14:15:00 | 255.00 | 252.98 | 260.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 14:45:00 | 256.30 | 252.98 | 260.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 15:15:00 | 259.51 | 252.97 | 259.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:15:00 | 263.48 | 252.97 | 259.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 265.98 | 253.10 | 259.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:30:00 | 266.00 | 253.10 | 259.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 269.89 | 253.27 | 259.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 11:00:00 | 269.89 | 253.27 | 259.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 261.03 | 256.56 | 260.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 13:30:00 | 260.46 | 256.60 | 260.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 14:00:00 | 260.75 | 256.60 | 260.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 265.62 | 256.81 | 260.32 | SL hit (close>static) qty=1.00 sl=262.68 alert=retest2 |

### Cycle 8 — BUY (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 11:15:00 | 269.10 | 263.15 | 263.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 13:15:00 | 269.85 | 263.28 | 263.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 10:15:00 | 262.00 | 263.44 | 263.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 10:15:00 | 262.00 | 263.44 | 263.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 262.00 | 263.44 | 263.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:00:00 | 262.00 | 263.44 | 263.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 261.80 | 263.43 | 263.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:45:00 | 261.75 | 263.43 | 263.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-13 13:00:00 | 149.20 | 2024-05-21 13:15:00 | 144.70 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest2 | 2024-05-13 13:45:00 | 149.60 | 2024-05-21 13:15:00 | 144.70 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest2 | 2024-06-07 10:45:00 | 147.40 | 2024-06-20 09:15:00 | 162.14 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-07 11:30:00 | 148.30 | 2024-06-20 09:15:00 | 163.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-07 13:30:00 | 147.50 | 2024-06-20 09:15:00 | 162.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-10 09:15:00 | 148.15 | 2024-06-20 09:15:00 | 162.97 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-14 14:15:00 | 151.60 | 2024-06-20 11:15:00 | 166.76 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-14 15:15:00 | 151.70 | 2024-06-20 11:15:00 | 166.87 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-30 13:15:00 | 170.75 | 2025-01-30 15:15:00 | 175.50 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2025-01-30 15:00:00 | 171.77 | 2025-01-30 15:15:00 | 175.50 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-02-01 15:15:00 | 170.22 | 2025-02-03 09:15:00 | 173.61 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-02-03 12:30:00 | 171.90 | 2025-02-03 14:15:00 | 172.90 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-02-06 11:15:00 | 175.53 | 2025-02-07 10:15:00 | 173.63 | STOP_HIT | 1.00 | 1.08% |
| SELL | retest2 | 2025-02-06 11:45:00 | 175.31 | 2025-02-07 10:15:00 | 173.63 | STOP_HIT | 1.00 | 0.96% |
| SELL | retest2 | 2025-03-21 13:15:00 | 160.26 | 2025-03-27 14:15:00 | 144.23 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-02 14:00:00 | 161.40 | 2025-04-03 11:15:00 | 162.84 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-04-02 14:45:00 | 161.69 | 2025-04-03 11:15:00 | 162.84 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-04-07 09:15:00 | 156.88 | 2025-04-11 10:15:00 | 162.82 | STOP_HIT | 1.00 | -3.79% |
| SELL | retest2 | 2025-04-08 10:30:00 | 156.95 | 2025-04-11 15:15:00 | 167.40 | STOP_HIT | 1.00 | -6.66% |
| SELL | retest2 | 2025-04-09 10:00:00 | 158.12 | 2025-04-11 15:15:00 | 167.40 | STOP_HIT | 1.00 | -5.87% |
| BUY | retest2 | 2025-07-31 11:45:00 | 210.60 | 2025-08-28 11:15:00 | 200.09 | STOP_HIT | 1.00 | -4.99% |
| BUY | retest2 | 2025-08-11 15:00:00 | 209.27 | 2025-08-28 11:15:00 | 200.09 | STOP_HIT | 1.00 | -4.39% |
| BUY | retest2 | 2025-08-12 12:30:00 | 209.65 | 2025-08-28 11:15:00 | 200.09 | STOP_HIT | 1.00 | -4.56% |
| BUY | retest2 | 2025-08-12 13:00:00 | 209.34 | 2025-08-28 11:15:00 | 200.09 | STOP_HIT | 1.00 | -4.42% |
| BUY | retest2 | 2025-09-19 12:30:00 | 209.00 | 2025-09-22 14:15:00 | 206.00 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-09-22 12:15:00 | 209.43 | 2025-09-22 14:15:00 | 206.00 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-09-24 09:30:00 | 208.88 | 2025-10-20 12:15:00 | 229.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-26 10:15:00 | 208.64 | 2025-10-20 12:15:00 | 229.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-13 09:15:00 | 269.75 | 2026-01-28 15:15:00 | 294.03 | TARGET_HIT | 1.00 | 9.00% |
| BUY | retest2 | 2026-01-20 10:00:00 | 267.30 | 2026-01-30 09:15:00 | 296.73 | TARGET_HIT | 1.00 | 11.01% |
| BUY | retest2 | 2026-03-05 09:15:00 | 267.75 | 2026-03-06 14:15:00 | 260.70 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2026-03-05 10:00:00 | 266.50 | 2026-03-06 14:15:00 | 260.70 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2026-04-24 13:30:00 | 260.46 | 2026-04-27 09:15:00 | 265.62 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2026-04-24 14:00:00 | 260.75 | 2026-04-27 09:15:00 | 265.62 | STOP_HIT | 1.00 | -1.87% |
