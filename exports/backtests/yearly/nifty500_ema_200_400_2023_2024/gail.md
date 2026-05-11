# GAIL (India) Ltd. (GAIL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 166.59
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT2_SKIP | 2 |
| ALERT3 | 43 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 28 |
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 31 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 23
- **Target hits / Stop hits / Partials:** 3 / 25 / 3
- **Avg / median % per leg:** 0.15% / -1.25%
- **Sum % (uncompounded):** 4.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 2 | 18.2% | 2 | 9 | 0 | -0.75% | -8.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 2 | 18.2% | 2 | 9 | 0 | -0.75% | -8.2% |
| SELL (all) | 20 | 6 | 30.0% | 1 | 16 | 3 | 0.65% | 12.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 20 | 6 | 30.0% | 1 | 16 | 3 | 0.65% | 12.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 31 | 8 | 25.8% | 3 | 25 | 3 | 0.15% | 4.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 13:15:00 | 109.30 | 107.76 | 107.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-20 09:15:00 | 110.70 | 107.83 | 107.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-14 09:15:00 | 113.20 | 113.71 | 111.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-14 09:30:00 | 113.05 | 113.71 | 111.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 12:15:00 | 112.15 | 113.61 | 111.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-18 15:00:00 | 112.85 | 113.59 | 111.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-21 10:15:00 | 112.90 | 113.57 | 111.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2023-09-05 09:15:00 | 124.14 | 115.92 | 113.68 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 10:15:00 | 211.55 | 224.78 | 224.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 11:15:00 | 210.71 | 224.64 | 224.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-07 09:15:00 | 214.50 | 214.49 | 218.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-07 09:30:00 | 215.08 | 214.49 | 218.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 207.76 | 201.90 | 208.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:00:00 | 207.76 | 201.90 | 208.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 208.96 | 201.97 | 208.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:45:00 | 208.76 | 201.97 | 208.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 208.95 | 202.04 | 208.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 12:00:00 | 208.95 | 202.04 | 208.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 12:15:00 | 209.40 | 202.11 | 208.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 13:00:00 | 209.40 | 202.11 | 208.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 15:15:00 | 208.88 | 202.31 | 208.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:15:00 | 211.66 | 202.31 | 208.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 212.13 | 202.41 | 208.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 10:00:00 | 212.13 | 202.41 | 208.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 212.00 | 202.51 | 208.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 10:30:00 | 213.32 | 202.51 | 208.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 15:15:00 | 208.74 | 203.37 | 208.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:15:00 | 208.13 | 203.37 | 208.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 209.08 | 203.43 | 208.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 11:45:00 | 207.99 | 203.52 | 208.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 09:15:00 | 197.59 | 203.58 | 207.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-06 10:15:00 | 187.19 | 197.45 | 202.70 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2025-04-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 13:15:00 | 195.66 | 174.88 | 174.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 196.06 | 187.15 | 183.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 09:15:00 | 187.40 | 188.65 | 184.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-04 09:45:00 | 186.99 | 188.65 | 184.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 189.03 | 190.53 | 186.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 14:45:00 | 190.03 | 190.50 | 186.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 10:15:00 | 190.30 | 190.50 | 186.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 10:45:00 | 190.47 | 190.49 | 186.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 11:30:00 | 190.54 | 190.50 | 186.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 187.03 | 190.46 | 187.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:00:00 | 187.03 | 190.46 | 187.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 186.28 | 190.42 | 187.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:45:00 | 185.84 | 190.42 | 187.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 185.71 | 190.37 | 187.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 14:00:00 | 185.71 | 190.37 | 187.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-19 09:15:00 | 183.83 | 190.22 | 187.04 | SL hit (close<static) qty=1.00 sl=185.17 alert=retest2 |

### Cycle 4 — SELL (started 2025-07-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 15:15:00 | 179.90 | 186.55 | 186.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 177.90 | 185.79 | 186.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 179.40 | 176.82 | 179.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 179.40 | 176.82 | 179.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 179.40 | 176.82 | 179.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:45:00 | 180.09 | 176.82 | 179.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 179.85 | 176.85 | 179.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:30:00 | 179.80 | 176.85 | 179.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 11:15:00 | 180.07 | 176.89 | 179.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 12:00:00 | 180.07 | 176.89 | 179.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 12:15:00 | 180.00 | 176.92 | 179.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 12:45:00 | 179.75 | 176.92 | 179.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 179.31 | 176.94 | 179.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 12:45:00 | 178.36 | 177.08 | 179.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 10:15:00 | 178.45 | 176.12 | 178.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:00:00 | 178.58 | 176.14 | 178.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 181.10 | 176.32 | 178.83 | SL hit (close>static) qty=1.00 sl=180.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 11:15:00 | 184.00 | 178.84 | 178.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 12:15:00 | 184.25 | 179.17 | 179.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 11:15:00 | 179.35 | 179.71 | 179.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 11:15:00 | 179.35 | 179.71 | 179.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 179.35 | 179.71 | 179.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:30:00 | 179.13 | 179.71 | 179.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 179.63 | 179.71 | 179.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:30:00 | 179.01 | 179.71 | 179.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 179.22 | 179.71 | 179.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 14:00:00 | 179.22 | 179.71 | 179.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 178.78 | 179.70 | 179.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:00:00 | 178.78 | 179.70 | 179.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 179.10 | 179.69 | 179.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:15:00 | 177.68 | 179.69 | 179.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 179.65 | 179.67 | 179.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 11:15:00 | 179.95 | 179.67 | 179.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 15:00:00 | 180.27 | 181.56 | 180.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 09:15:00 | 183.34 | 181.54 | 180.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 09:15:00 | 174.03 | 181.82 | 180.79 | SL hit (close<static) qty=1.00 sl=177.76 alert=retest2 |

### Cycle 6 — SELL (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 09:15:00 | 171.09 | 179.88 | 179.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 11:15:00 | 170.44 | 179.70 | 179.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 09:15:00 | 172.89 | 172.69 | 174.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-02 10:15:00 | 173.37 | 172.69 | 174.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 175.28 | 172.71 | 174.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:00:00 | 175.28 | 172.71 | 174.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 175.04 | 172.74 | 174.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:30:00 | 174.99 | 172.74 | 174.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 175.34 | 172.76 | 174.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:00:00 | 175.34 | 172.76 | 174.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 175.88 | 172.79 | 174.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:00:00 | 175.88 | 172.79 | 174.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 174.36 | 172.86 | 174.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 10:30:00 | 173.70 | 172.88 | 174.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 11:15:00 | 173.93 | 172.88 | 174.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:15:00 | 165.01 | 172.25 | 174.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:15:00 | 165.23 | 172.25 | 174.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 14:15:00 | 168.31 | 167.13 | 170.60 | SL hit (close>ema200) qty=0.50 sl=167.13 alert=retest2 |

### Cycle 7 — BUY (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 13:15:00 | 163.76 | 158.11 | 158.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 164.67 | 158.29 | 158.18 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-08-18 15:00:00 | 112.85 | 2023-09-05 09:15:00 | 124.14 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-21 10:15:00 | 112.90 | 2023-09-05 09:15:00 | 124.19 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-10 11:45:00 | 207.99 | 2024-12-18 09:15:00 | 197.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-10 11:45:00 | 207.99 | 2025-01-06 10:15:00 | 187.19 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-06-13 14:45:00 | 190.03 | 2025-06-19 09:15:00 | 183.83 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2025-06-16 10:15:00 | 190.30 | 2025-06-19 09:15:00 | 183.83 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2025-06-16 10:45:00 | 190.47 | 2025-06-19 09:15:00 | 183.83 | STOP_HIT | 1.00 | -3.49% |
| BUY | retest2 | 2025-06-16 11:30:00 | 190.54 | 2025-06-19 09:15:00 | 183.83 | STOP_HIT | 1.00 | -3.52% |
| BUY | retest2 | 2025-06-27 09:15:00 | 188.66 | 2025-07-09 12:15:00 | 186.15 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-07-24 09:15:00 | 187.92 | 2025-07-25 09:15:00 | 185.30 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-09-03 12:45:00 | 178.36 | 2025-09-12 09:15:00 | 181.10 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-09-11 10:15:00 | 178.45 | 2025-09-12 09:15:00 | 181.10 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-09-11 11:00:00 | 178.58 | 2025-09-12 09:15:00 | 181.10 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-09-12 13:45:00 | 178.48 | 2025-09-15 15:15:00 | 180.40 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-09-24 09:15:00 | 177.42 | 2025-10-07 09:15:00 | 179.12 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-10-06 09:30:00 | 176.98 | 2025-10-07 09:15:00 | 179.12 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-10-07 10:30:00 | 177.65 | 2025-10-07 14:15:00 | 179.54 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-10-07 11:15:00 | 177.24 | 2025-10-07 14:15:00 | 179.54 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-10-13 09:15:00 | 176.97 | 2025-10-13 11:15:00 | 179.26 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-10-14 14:00:00 | 174.40 | 2025-10-16 11:15:00 | 179.06 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-10-15 09:30:00 | 176.85 | 2025-10-16 11:15:00 | 179.06 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-10-15 12:45:00 | 177.17 | 2025-10-16 11:15:00 | 179.06 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-10-16 10:00:00 | 177.21 | 2025-10-16 11:15:00 | 179.06 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-10-17 09:45:00 | 177.03 | 2025-10-20 09:15:00 | 178.86 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-11-07 11:15:00 | 179.95 | 2025-11-28 09:15:00 | 174.03 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2025-11-25 15:00:00 | 180.27 | 2025-11-28 09:15:00 | 174.03 | STOP_HIT | 1.00 | -3.46% |
| BUY | retest2 | 2025-11-26 09:15:00 | 183.34 | 2025-11-28 09:15:00 | 174.03 | STOP_HIT | 1.00 | -5.08% |
| SELL | retest2 | 2026-01-05 10:30:00 | 173.70 | 2026-01-08 11:15:00 | 165.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 11:15:00 | 173.93 | 2026-01-08 11:15:00 | 165.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 10:30:00 | 173.70 | 2026-01-28 14:15:00 | 168.31 | STOP_HIT | 0.50 | 3.10% |
| SELL | retest2 | 2026-01-05 11:15:00 | 173.93 | 2026-01-28 14:15:00 | 168.31 | STOP_HIT | 0.50 | 3.23% |
