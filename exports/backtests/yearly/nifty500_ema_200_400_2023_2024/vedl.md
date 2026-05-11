# Vedanta Ltd. (VEDL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 297.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT2_SKIP | 7 |
| ALERT3 | 41 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 29 |
| PARTIAL | 1 |
| TARGET_HIT | 2 |
| STOP_HIT | 27 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 30 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 22
- **Target hits / Stop hits / Partials:** 2 / 27 / 1
- **Avg / median % per leg:** -1.23% / -1.71%
- **Sum % (uncompounded):** -36.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 5 | 29.4% | 2 | 15 | 0 | -1.60% | -27.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 17 | 5 | 29.4% | 2 | 15 | 0 | -1.60% | -27.2% |
| SELL (all) | 13 | 3 | 23.1% | 0 | 12 | 1 | -0.73% | -9.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 13 | 3 | 23.1% | 0 | 12 | 1 | -0.73% | -9.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 30 | 8 | 26.7% | 2 | 27 | 1 | -1.23% | -36.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-03 14:15:00 | 104.18 | 104.99 | 104.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-04 13:15:00 | 104.08 | 104.94 | 104.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-06 13:15:00 | 104.87 | 104.83 | 104.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-06 14:15:00 | 104.89 | 104.83 | 104.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 14:15:00 | 104.89 | 104.83 | 104.91 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-18 10:15:00 | 106.52 | 104.97 | 104.96 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 13:15:00 | 101.65 | 104.94 | 104.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-03 09:15:00 | 95.09 | 104.33 | 104.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-16 12:15:00 | 86.46 | 85.74 | 89.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-03 09:15:00 | 87.43 | 84.11 | 87.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 09:15:00 | 87.43 | 84.11 | 87.10 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2023-12-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 11:15:00 | 93.22 | 88.27 | 88.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 09:15:00 | 94.19 | 88.52 | 88.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 10:15:00 | 97.30 | 97.63 | 94.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 14:15:00 | 94.40 | 97.55 | 94.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 14:15:00 | 94.40 | 97.55 | 94.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 136.24 | 109.04 | 104.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-26 09:15:00 | 149.86 | 124.95 | 114.99 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 12:15:00 | 170.54 | 173.71 | 173.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 13:15:00 | 168.82 | 173.66 | 173.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 09:15:00 | 171.91 | 171.06 | 172.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 09:15:00 | 171.91 | 171.06 | 172.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 171.91 | 171.06 | 172.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:00:00 | 171.91 | 171.06 | 172.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 170.75 | 171.05 | 172.17 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2024-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 11:15:00 | 183.46 | 173.05 | 173.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 12:15:00 | 186.09 | 173.18 | 173.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 14:15:00 | 178.56 | 180.60 | 177.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-20 15:00:00 | 178.56 | 180.60 | 177.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 13:15:00 | 176.76 | 180.48 | 177.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 14:00:00 | 176.76 | 180.48 | 177.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 14:15:00 | 177.42 | 180.45 | 177.53 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 166.84 | 175.34 | 175.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 11:15:00 | 166.63 | 175.26 | 175.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-17 09:15:00 | 170.49 | 170.17 | 172.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-17 10:00:00 | 170.49 | 170.17 | 172.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 172.32 | 170.18 | 172.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 11:00:00 | 172.32 | 170.18 | 172.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 11:15:00 | 172.60 | 170.21 | 172.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 12:00:00 | 172.60 | 170.21 | 172.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 12:15:00 | 173.16 | 170.24 | 172.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 13:00:00 | 173.16 | 170.24 | 172.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 13:15:00 | 173.18 | 170.27 | 172.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 13:45:00 | 173.16 | 170.27 | 172.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 171.99 | 170.33 | 172.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 09:30:00 | 174.94 | 170.33 | 172.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 172.28 | 170.35 | 172.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:45:00 | 172.53 | 170.35 | 172.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 172.04 | 170.37 | 172.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 13:00:00 | 172.04 | 170.37 | 172.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 170.21 | 166.41 | 169.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:00:00 | 170.21 | 166.41 | 169.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 170.15 | 166.45 | 169.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 165.17 | 166.61 | 169.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 156.91 | 165.79 | 168.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-20 15:15:00 | 163.05 | 162.57 | 166.15 | SL hit (close>ema200) qty=0.50 sl=162.57 alert=retest2 |

### Cycle 8 — BUY (started 2025-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 10:15:00 | 174.03 | 165.72 | 165.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-26 09:15:00 | 176.12 | 166.18 | 165.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 09:15:00 | 165.36 | 168.18 | 167.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 09:15:00 | 165.36 | 168.18 | 167.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 165.36 | 168.18 | 167.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:45:00 | 164.94 | 168.18 | 167.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 165.54 | 168.15 | 167.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 11:45:00 | 166.72 | 168.13 | 167.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 09:15:00 | 155.34 | 167.87 | 166.96 | SL hit (close<static) qty=1.00 sl=164.49 alert=retest2 |

### Cycle 9 — SELL (started 2025-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 12:15:00 | 137.66 | 165.85 | 165.97 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 11:15:00 | 168.61 | 161.11 | 161.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 12:15:00 | 169.98 | 162.02 | 161.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 09:15:00 | 161.01 | 162.32 | 161.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 161.01 | 162.32 | 161.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 161.01 | 162.32 | 161.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 13:15:00 | 162.60 | 162.29 | 161.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 09:15:00 | 163.99 | 162.28 | 161.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 10:00:00 | 162.60 | 162.29 | 161.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 12:30:00 | 163.16 | 162.30 | 161.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 162.57 | 162.30 | 161.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:45:00 | 162.12 | 162.30 | 161.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 162.62 | 162.31 | 161.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 161.91 | 162.31 | 161.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 165.32 | 166.99 | 164.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:30:00 | 164.01 | 166.99 | 164.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 164.70 | 166.97 | 164.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 15:00:00 | 164.70 | 166.97 | 164.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 164.76 | 166.95 | 164.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:15:00 | 164.42 | 166.95 | 164.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 165.58 | 166.93 | 164.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 10:15:00 | 166.63 | 166.93 | 164.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 12:45:00 | 166.35 | 166.94 | 164.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 166.37 | 166.96 | 165.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 11:15:00 | 163.50 | 169.29 | 166.94 | SL hit (close<static) qty=1.00 sl=163.93 alert=retest2 |

### Cycle 11 — SELL (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 15:15:00 | 161.46 | 166.20 | 166.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 15:15:00 | 161.24 | 165.51 | 165.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 166.03 | 165.03 | 165.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 166.03 | 165.03 | 165.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 166.03 | 165.03 | 165.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:45:00 | 165.97 | 165.03 | 165.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 165.82 | 165.04 | 165.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 11:15:00 | 165.24 | 165.04 | 165.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 09:15:00 | 167.81 | 164.70 | 165.35 | SL hit (close>static) qty=1.00 sl=166.59 alert=retest2 |

### Cycle 12 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 173.07 | 165.27 | 165.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 12:15:00 | 175.49 | 165.46 | 165.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 167.08 | 167.46 | 166.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-26 14:00:00 | 167.08 | 167.46 | 166.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 255.36 | 259.89 | 249.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 10:30:00 | 257.51 | 259.84 | 249.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 256.12 | 259.43 | 249.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 10:15:00 | 256.12 | 259.39 | 249.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 241.80 | 258.80 | 249.59 | SL hit (close<static) qty=1.00 sl=248.75 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-12 09:15:00 | 136.24 | 2024-04-26 09:15:00 | 149.86 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 165.17 | 2025-02-12 09:15:00 | 156.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 165.17 | 2025-02-20 15:15:00 | 163.05 | STOP_HIT | 0.50 | 1.28% |
| BUY | retest2 | 2025-04-03 11:45:00 | 166.72 | 2025-04-04 09:15:00 | 155.34 | STOP_HIT | 1.00 | -6.83% |
| BUY | retest2 | 2025-06-02 13:15:00 | 162.60 | 2025-07-09 11:15:00 | 163.50 | STOP_HIT | 1.00 | 0.55% |
| BUY | retest2 | 2025-06-03 09:15:00 | 163.99 | 2025-07-09 11:15:00 | 163.50 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-06-03 10:00:00 | 162.60 | 2025-07-09 11:15:00 | 163.50 | STOP_HIT | 1.00 | 0.55% |
| BUY | retest2 | 2025-06-03 12:30:00 | 163.16 | 2025-07-17 12:15:00 | 166.31 | STOP_HIT | 1.00 | 1.93% |
| BUY | retest2 | 2025-06-20 10:15:00 | 166.63 | 2025-07-25 11:15:00 | 166.40 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2025-06-20 12:45:00 | 166.35 | 2025-07-28 12:15:00 | 163.18 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-06-26 09:15:00 | 166.37 | 2025-07-31 15:15:00 | 159.04 | STOP_HIT | 1.00 | -4.41% |
| BUY | retest2 | 2025-07-11 09:30:00 | 166.40 | 2025-07-31 15:15:00 | 159.04 | STOP_HIT | 1.00 | -4.42% |
| BUY | retest2 | 2025-07-17 09:15:00 | 168.03 | 2025-07-31 15:15:00 | 159.04 | STOP_HIT | 1.00 | -5.35% |
| BUY | retest2 | 2025-07-21 09:15:00 | 168.91 | 2025-07-31 15:15:00 | 159.04 | STOP_HIT | 1.00 | -5.84% |
| SELL | retest2 | 2025-08-13 11:15:00 | 165.24 | 2025-08-19 09:15:00 | 167.81 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-08-20 12:30:00 | 165.00 | 2025-08-20 14:15:00 | 166.78 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-08-26 09:15:00 | 161.95 | 2025-09-05 13:15:00 | 165.17 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-09-04 11:30:00 | 165.43 | 2025-09-05 13:15:00 | 165.17 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2025-09-04 14:15:00 | 164.08 | 2025-09-05 13:15:00 | 165.17 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-09-05 10:00:00 | 164.23 | 2025-09-05 13:15:00 | 165.17 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-09-05 10:45:00 | 164.23 | 2025-09-05 14:15:00 | 167.04 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-09-05 11:15:00 | 164.19 | 2025-09-05 14:15:00 | 167.04 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-09-11 12:30:00 | 163.16 | 2025-09-12 09:15:00 | 166.76 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-09-11 13:00:00 | 163.16 | 2025-09-12 09:15:00 | 166.76 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-09-11 13:30:00 | 163.11 | 2025-09-12 09:15:00 | 166.76 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2026-03-19 10:30:00 | 257.51 | 2026-03-23 09:15:00 | 241.80 | STOP_HIT | 1.00 | -6.10% |
| BUY | retest2 | 2026-03-20 09:15:00 | 256.12 | 2026-03-23 09:15:00 | 241.80 | STOP_HIT | 1.00 | -5.59% |
| BUY | retest2 | 2026-03-20 10:15:00 | 256.12 | 2026-03-23 09:15:00 | 241.80 | STOP_HIT | 1.00 | -5.59% |
| BUY | retest2 | 2026-04-01 09:45:00 | 256.31 | 2026-04-02 09:15:00 | 246.57 | STOP_HIT | 1.00 | -3.80% |
| BUY | retest2 | 2026-04-02 11:30:00 | 249.10 | 2026-04-08 09:15:00 | 274.01 | TARGET_HIT | 1.00 | 10.00% |
