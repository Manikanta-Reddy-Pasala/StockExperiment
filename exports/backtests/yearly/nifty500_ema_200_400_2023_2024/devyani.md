# Devyani International Ltd. (DEVYANI)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 118.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT2_SKIP | 5 |
| ALERT3 | 94 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 68 |
| PARTIAL | 24 |
| TARGET_HIT | 23 |
| STOP_HIT | 45 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 92 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 48 / 44
- **Target hits / Stop hits / Partials:** 23 / 45 / 24
- **Avg / median % per leg:** 2.76% / 5.00%
- **Sum % (uncompounded):** 254.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 3 | 15.0% | 3 | 17 | 0 | -0.49% | -9.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 20 | 3 | 15.0% | 3 | 17 | 0 | -0.49% | -9.7% |
| SELL (all) | 72 | 45 | 62.5% | 20 | 28 | 24 | 3.67% | 264.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 72 | 45 | 62.5% | 20 | 28 | 24 | 3.67% | 264.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 92 | 48 | 52.2% | 23 | 45 | 24 | 2.76% | 254.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 14:15:00 | 179.90 | 201.58 | 201.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 09:15:00 | 178.30 | 188.80 | 193.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-07 11:15:00 | 185.95 | 185.07 | 190.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-07 11:30:00 | 186.00 | 185.07 | 190.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 09:15:00 | 186.80 | 185.15 | 190.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-08 13:45:00 | 186.15 | 185.23 | 190.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-08 15:15:00 | 186.00 | 185.25 | 190.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-11 09:30:00 | 185.00 | 185.26 | 190.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-11 15:00:00 | 185.40 | 185.29 | 189.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 193.10 | 185.03 | 189.02 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-12-19 09:15:00 | 193.10 | 185.03 | 189.02 | SL hit (close>static) qty=1.00 sl=190.30 alert=retest2 |

### Cycle 2 — BUY (started 2024-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 12:15:00 | 180.05 | 160.88 | 160.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 14:15:00 | 181.00 | 168.76 | 166.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 09:15:00 | 172.00 | 172.49 | 169.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 09:15:00 | 172.00 | 172.49 | 169.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 172.00 | 172.49 | 169.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 09:30:00 | 171.25 | 172.49 | 169.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 10:15:00 | 168.38 | 172.45 | 169.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 11:00:00 | 168.38 | 172.45 | 169.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 11:15:00 | 172.20 | 172.45 | 169.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 11:30:00 | 168.19 | 172.45 | 169.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 168.91 | 173.29 | 170.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:00:00 | 168.91 | 173.29 | 170.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 169.74 | 173.26 | 170.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:30:00 | 168.80 | 173.26 | 170.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 11:15:00 | 168.35 | 173.21 | 170.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 11:45:00 | 168.54 | 173.21 | 170.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 12:15:00 | 170.30 | 173.18 | 170.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 14:00:00 | 171.00 | 173.16 | 170.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 14:15:00 | 167.93 | 172.93 | 170.06 | SL hit (close<static) qty=1.00 sl=168.18 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 162.67 | 180.73 | 180.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 10:15:00 | 161.39 | 180.54 | 180.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 15:15:00 | 176.00 | 174.68 | 177.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-12 09:15:00 | 172.83 | 174.68 | 177.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 11:15:00 | 173.85 | 168.22 | 171.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 11:45:00 | 174.81 | 168.22 | 171.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 12:15:00 | 174.45 | 168.28 | 171.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 13:00:00 | 174.45 | 168.28 | 171.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 13:15:00 | 171.00 | 168.67 | 171.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 13:45:00 | 171.53 | 168.67 | 171.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 172.14 | 168.70 | 171.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 11:15:00 | 170.66 | 168.79 | 171.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 14:45:00 | 170.52 | 168.49 | 171.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 09:15:00 | 169.87 | 168.52 | 171.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 11:00:00 | 170.41 | 168.55 | 171.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 12:15:00 | 171.38 | 168.60 | 171.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 13:00:00 | 171.38 | 168.60 | 171.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 13:15:00 | 170.79 | 168.62 | 171.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 09:15:00 | 168.58 | 168.66 | 171.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 10:45:00 | 170.18 | 168.68 | 171.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 13:45:00 | 170.25 | 168.72 | 171.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 09:15:00 | 173.30 | 168.81 | 171.21 | SL hit (close>static) qty=1.00 sl=173.20 alert=retest2 |

### Cycle 4 — BUY (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 14:15:00 | 183.02 | 173.13 | 173.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 09:15:00 | 184.57 | 173.88 | 173.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 13:15:00 | 180.30 | 180.57 | 177.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-10 14:00:00 | 180.30 | 180.57 | 177.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 13:15:00 | 177.17 | 180.55 | 177.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 14:00:00 | 177.17 | 180.55 | 177.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 14:15:00 | 177.09 | 180.51 | 177.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 14:30:00 | 176.77 | 180.51 | 177.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 15:15:00 | 178.00 | 180.49 | 177.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-14 09:15:00 | 178.22 | 180.49 | 177.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 180.47 | 180.49 | 177.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-14 11:00:00 | 181.56 | 180.50 | 177.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-14 13:15:00 | 175.60 | 180.46 | 177.53 | SL hit (close<static) qty=1.00 sl=176.31 alert=retest2 |

### Cycle 5 — SELL (started 2025-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 09:15:00 | 169.58 | 176.64 | 176.66 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 10:15:00 | 192.61 | 176.81 | 176.74 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 09:15:00 | 166.46 | 176.90 | 176.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-13 12:15:00 | 164.24 | 176.55 | 176.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 10:15:00 | 174.09 | 172.54 | 174.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-20 11:00:00 | 174.09 | 172.54 | 174.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 11:15:00 | 178.12 | 172.59 | 174.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 12:00:00 | 178.12 | 172.59 | 174.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 12:15:00 | 178.25 | 172.65 | 174.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 12:45:00 | 181.10 | 172.65 | 174.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 176.18 | 172.87 | 174.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:30:00 | 175.80 | 172.87 | 174.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 176.70 | 172.91 | 174.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:45:00 | 176.81 | 172.91 | 174.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 10:15:00 | 179.96 | 173.16 | 174.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 11:00:00 | 179.96 | 173.16 | 174.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 177.68 | 173.29 | 174.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:45:00 | 177.99 | 173.29 | 174.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 175.87 | 173.31 | 174.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 12:30:00 | 173.01 | 173.50 | 174.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 14:00:00 | 172.90 | 173.50 | 174.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 15:00:00 | 173.01 | 173.49 | 174.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 12:15:00 | 164.36 | 173.22 | 174.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 12:15:00 | 164.25 | 173.22 | 174.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 12:15:00 | 164.36 | 173.22 | 174.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-06 09:15:00 | 174.64 | 171.75 | 173.64 | SL hit (close>ema200) qty=0.50 sl=171.75 alert=retest2 |

### Cycle 8 — BUY (started 2025-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 11:15:00 | 177.04 | 165.09 | 165.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 09:15:00 | 177.99 | 165.68 | 165.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 168.24 | 168.54 | 166.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 09:15:00 | 168.24 | 168.54 | 166.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 168.24 | 168.54 | 166.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:30:00 | 167.61 | 168.54 | 166.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 172.00 | 174.73 | 171.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 15:00:00 | 172.00 | 174.73 | 171.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 171.33 | 174.50 | 171.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:00:00 | 171.33 | 174.50 | 171.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 170.42 | 174.46 | 171.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 11:30:00 | 171.63 | 172.56 | 170.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 14:45:00 | 171.50 | 172.50 | 170.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:45:00 | 171.79 | 172.47 | 170.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:30:00 | 171.74 | 172.61 | 171.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 171.00 | 172.58 | 171.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 171.00 | 172.58 | 171.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 169.96 | 172.56 | 171.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:45:00 | 169.98 | 172.56 | 171.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 169.00 | 172.52 | 171.09 | SL hit (close<static) qty=1.00 sl=169.47 alert=retest2 |

### Cycle 9 — SELL (started 2025-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 14:15:00 | 166.28 | 170.15 | 170.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 09:15:00 | 165.82 | 170.07 | 170.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 172.91 | 169.83 | 169.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 172.91 | 169.83 | 169.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 172.91 | 169.83 | 169.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 170.66 | 169.83 | 169.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 173.31 | 169.87 | 170.00 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 12:15:00 | 173.73 | 170.14 | 170.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 175.65 | 170.46 | 170.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 12:15:00 | 172.25 | 172.37 | 171.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 13:15:00 | 171.65 | 172.36 | 171.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 171.65 | 172.36 | 171.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 13:45:00 | 172.02 | 172.36 | 171.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 171.30 | 172.35 | 171.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:30:00 | 171.76 | 172.35 | 171.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 171.85 | 172.34 | 171.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:15:00 | 172.47 | 172.34 | 171.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 171.17 | 172.33 | 171.44 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 162.55 | 170.69 | 170.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 12:15:00 | 162.25 | 170.60 | 170.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 10:15:00 | 165.74 | 163.56 | 166.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 10:15:00 | 165.74 | 163.56 | 166.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 165.74 | 163.56 | 166.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:00:00 | 165.74 | 163.56 | 166.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 166.84 | 163.59 | 166.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:45:00 | 167.85 | 163.59 | 166.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 170.90 | 163.66 | 166.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:00:00 | 170.90 | 163.66 | 166.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 168.79 | 163.71 | 166.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 14:45:00 | 167.05 | 163.74 | 166.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 09:15:00 | 175.99 | 164.24 | 166.69 | SL hit (close>static) qty=1.00 sl=171.95 alert=retest2 |

### Cycle 12 — BUY (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 11:15:00 | 175.42 | 168.54 | 168.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 09:15:00 | 180.05 | 168.91 | 168.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 10:15:00 | 175.88 | 176.14 | 173.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-19 11:00:00 | 175.88 | 176.14 | 173.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 173.22 | 176.14 | 173.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:00:00 | 173.22 | 176.14 | 173.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 173.48 | 176.12 | 173.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:15:00 | 171.31 | 176.12 | 173.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 169.06 | 175.98 | 173.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:30:00 | 169.23 | 175.98 | 173.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2025-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 15:15:00 | 165.00 | 171.74 | 171.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 13:15:00 | 164.24 | 171.41 | 171.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 14:15:00 | 139.12 | 137.97 | 146.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-22 15:00:00 | 139.12 | 137.97 | 146.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 145.33 | 138.19 | 146.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 14:00:00 | 140.63 | 141.02 | 145.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 13:15:00 | 133.60 | 140.21 | 144.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-20 11:15:00 | 126.57 | 137.65 | 142.67 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-12-08 13:45:00 | 186.15 | 2023-12-19 09:15:00 | 193.10 | STOP_HIT | 1.00 | -3.73% |
| SELL | retest2 | 2023-12-08 15:15:00 | 186.00 | 2023-12-19 09:15:00 | 193.10 | STOP_HIT | 1.00 | -3.82% |
| SELL | retest2 | 2023-12-11 09:30:00 | 185.00 | 2023-12-19 09:15:00 | 193.10 | STOP_HIT | 1.00 | -4.38% |
| SELL | retest2 | 2023-12-11 15:00:00 | 185.40 | 2023-12-19 09:15:00 | 193.10 | STOP_HIT | 1.00 | -4.15% |
| SELL | retest2 | 2023-12-20 15:15:00 | 188.90 | 2023-12-27 09:15:00 | 192.60 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2023-12-22 13:30:00 | 188.95 | 2023-12-27 09:15:00 | 192.60 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2023-12-22 14:00:00 | 188.85 | 2023-12-27 09:15:00 | 192.60 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2023-12-26 09:15:00 | 188.90 | 2024-01-18 09:15:00 | 179.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-26 13:00:00 | 189.85 | 2024-01-18 09:15:00 | 179.50 | PARTIAL | 0.50 | 5.45% |
| SELL | retest2 | 2023-12-26 13:45:00 | 189.95 | 2024-01-18 09:15:00 | 179.45 | PARTIAL | 0.50 | 5.53% |
| SELL | retest2 | 2023-12-26 15:00:00 | 189.90 | 2024-01-18 09:15:00 | 180.31 | PARTIAL | 0.50 | 5.05% |
| SELL | retest2 | 2024-01-02 10:15:00 | 189.80 | 2024-01-18 09:15:00 | 179.93 | PARTIAL | 0.50 | 5.20% |
| SELL | retest2 | 2024-01-04 11:15:00 | 189.40 | 2024-01-18 09:15:00 | 179.64 | PARTIAL | 0.50 | 5.15% |
| SELL | retest2 | 2024-01-04 12:15:00 | 189.10 | 2024-01-18 09:15:00 | 179.98 | PARTIAL | 0.50 | 4.82% |
| SELL | retest2 | 2024-01-05 09:45:00 | 189.45 | 2024-01-23 11:15:00 | 179.41 | PARTIAL | 0.50 | 5.30% |
| SELL | retest2 | 2024-01-05 10:15:00 | 188.80 | 2024-01-23 11:15:00 | 179.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-05 14:15:00 | 187.50 | 2024-01-23 14:15:00 | 178.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-08 14:15:00 | 187.15 | 2024-01-23 14:15:00 | 177.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-11 15:00:00 | 187.00 | 2024-01-23 14:15:00 | 177.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-16 10:15:00 | 187.10 | 2024-01-23 14:15:00 | 177.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-26 09:15:00 | 188.90 | 2024-02-05 09:15:00 | 170.01 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-12-26 13:00:00 | 189.85 | 2024-02-05 09:15:00 | 170.06 | TARGET_HIT | 0.50 | 10.43% |
| SELL | retest2 | 2023-12-26 13:45:00 | 189.95 | 2024-02-05 09:15:00 | 169.97 | TARGET_HIT | 0.50 | 10.52% |
| SELL | retest2 | 2023-12-26 15:00:00 | 189.90 | 2024-02-05 09:15:00 | 170.01 | TARGET_HIT | 0.50 | 10.47% |
| SELL | retest2 | 2024-01-02 10:15:00 | 189.80 | 2024-02-05 09:15:00 | 170.82 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-01-04 11:15:00 | 189.40 | 2024-02-05 09:15:00 | 170.46 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-01-04 12:15:00 | 189.10 | 2024-02-05 09:15:00 | 170.19 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-01-05 09:45:00 | 189.45 | 2024-02-05 09:15:00 | 170.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-01-05 10:15:00 | 188.80 | 2024-02-05 09:15:00 | 169.92 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-01-05 14:15:00 | 187.50 | 2024-02-05 09:15:00 | 168.75 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-01-08 14:15:00 | 187.15 | 2024-02-05 09:15:00 | 168.44 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-01-11 15:00:00 | 187.00 | 2024-02-05 09:15:00 | 168.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-01-16 10:15:00 | 187.10 | 2024-02-05 09:15:00 | 168.39 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-04-19 09:15:00 | 160.15 | 2024-04-22 14:15:00 | 163.05 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-04-19 14:15:00 | 161.70 | 2024-04-22 14:15:00 | 163.05 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-04-22 10:00:00 | 161.95 | 2024-04-22 14:15:00 | 163.05 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2024-04-22 11:00:00 | 161.80 | 2024-04-22 14:15:00 | 163.05 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-05-06 13:15:00 | 162.85 | 2024-05-07 11:15:00 | 165.60 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-05-06 14:00:00 | 163.00 | 2024-05-07 11:15:00 | 165.60 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-05-07 10:15:00 | 162.90 | 2024-05-07 11:15:00 | 165.60 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-05-07 11:15:00 | 163.00 | 2024-05-07 11:15:00 | 165.60 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-05-07 15:15:00 | 161.15 | 2024-05-14 12:15:00 | 153.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-08 10:15:00 | 161.05 | 2024-05-14 12:15:00 | 153.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-09 10:30:00 | 161.35 | 2024-05-14 12:15:00 | 153.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-09 12:45:00 | 161.20 | 2024-05-14 12:15:00 | 153.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-13 10:15:00 | 157.80 | 2024-05-14 12:15:00 | 149.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-07 15:15:00 | 161.15 | 2024-06-04 11:15:00 | 145.22 | TARGET_HIT | 0.50 | 9.89% |
| SELL | retest2 | 2024-05-08 10:15:00 | 161.05 | 2024-06-04 11:15:00 | 145.08 | TARGET_HIT | 0.50 | 9.92% |
| SELL | retest2 | 2024-05-09 10:30:00 | 161.35 | 2024-06-04 12:15:00 | 145.03 | TARGET_HIT | 0.50 | 10.11% |
| SELL | retest2 | 2024-05-09 12:45:00 | 161.20 | 2024-06-04 12:15:00 | 144.95 | TARGET_HIT | 0.50 | 10.08% |
| SELL | retest2 | 2024-05-13 10:15:00 | 157.80 | 2024-06-05 12:15:00 | 156.55 | STOP_HIT | 0.50 | 0.79% |
| SELL | retest2 | 2024-06-05 14:15:00 | 158.05 | 2024-06-06 09:15:00 | 166.90 | STOP_HIT | 1.00 | -5.60% |
| BUY | retest2 | 2024-08-12 14:00:00 | 171.00 | 2024-08-13 14:15:00 | 167.93 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2024-08-16 11:30:00 | 171.00 | 2024-08-23 10:15:00 | 188.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-19 09:15:00 | 171.99 | 2024-08-23 10:15:00 | 189.19 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-20 09:30:00 | 170.69 | 2024-08-23 10:15:00 | 187.76 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-12 11:15:00 | 170.66 | 2024-12-20 09:15:00 | 173.30 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-12-17 14:45:00 | 170.52 | 2024-12-20 09:15:00 | 173.30 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-12-18 09:15:00 | 169.87 | 2024-12-20 09:15:00 | 173.30 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2024-12-18 11:00:00 | 170.41 | 2024-12-20 09:15:00 | 173.30 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2024-12-19 09:15:00 | 168.58 | 2024-12-20 09:15:00 | 173.30 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2024-12-19 10:45:00 | 170.18 | 2024-12-20 09:15:00 | 173.30 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-12-19 13:45:00 | 170.25 | 2024-12-20 09:15:00 | 173.30 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-01-14 11:00:00 | 181.56 | 2025-01-14 13:15:00 | 175.60 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest2 | 2025-01-15 09:45:00 | 182.58 | 2025-01-24 09:15:00 | 175.60 | STOP_HIT | 1.00 | -3.82% |
| BUY | retest2 | 2025-01-17 09:45:00 | 183.61 | 2025-01-24 09:15:00 | 175.60 | STOP_HIT | 1.00 | -4.36% |
| BUY | retest2 | 2025-01-22 13:00:00 | 181.60 | 2025-01-24 09:15:00 | 175.60 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2025-02-27 12:30:00 | 173.01 | 2025-02-28 12:15:00 | 164.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 14:00:00 | 172.90 | 2025-02-28 12:15:00 | 164.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 15:00:00 | 173.01 | 2025-02-28 12:15:00 | 164.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 12:30:00 | 173.01 | 2025-03-06 09:15:00 | 174.64 | STOP_HIT | 0.50 | -0.94% |
| SELL | retest2 | 2025-02-27 14:00:00 | 172.90 | 2025-03-06 09:15:00 | 174.64 | STOP_HIT | 0.50 | -1.01% |
| SELL | retest2 | 2025-02-27 15:00:00 | 173.01 | 2025-03-06 09:15:00 | 174.64 | STOP_HIT | 0.50 | -0.94% |
| SELL | retest2 | 2025-03-06 14:45:00 | 172.30 | 2025-03-10 14:15:00 | 163.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-06 14:45:00 | 172.30 | 2025-03-11 09:15:00 | 155.07 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-10 11:00:00 | 168.18 | 2025-03-11 09:15:00 | 159.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-10 11:00:00 | 168.18 | 2025-03-13 12:15:00 | 151.36 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-06-06 11:30:00 | 171.63 | 2025-06-12 13:15:00 | 169.00 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-06-06 14:45:00 | 171.50 | 2025-06-12 13:15:00 | 169.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-06-09 09:45:00 | 171.79 | 2025-06-12 13:15:00 | 169.00 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-06-12 09:30:00 | 171.74 | 2025-06-12 13:15:00 | 169.00 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-06-17 13:00:00 | 168.58 | 2025-06-23 09:15:00 | 163.12 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2025-06-19 14:00:00 | 168.67 | 2025-06-23 09:15:00 | 163.12 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2025-06-20 11:30:00 | 168.22 | 2025-06-23 09:15:00 | 163.12 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2025-06-23 12:30:00 | 168.23 | 2025-06-23 13:15:00 | 165.99 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-06-25 12:30:00 | 172.10 | 2025-06-30 11:15:00 | 169.23 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-06-26 11:00:00 | 171.98 | 2025-06-30 11:15:00 | 169.23 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-06-27 09:15:00 | 171.64 | 2025-06-30 11:15:00 | 169.23 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-06-27 09:45:00 | 171.61 | 2025-06-30 11:15:00 | 169.23 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-08-19 14:45:00 | 167.05 | 2025-08-21 09:15:00 | 175.99 | STOP_HIT | 1.00 | -5.35% |
| SELL | retest2 | 2026-01-05 14:00:00 | 140.63 | 2026-01-09 13:15:00 | 133.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 14:00:00 | 140.63 | 2026-01-20 11:15:00 | 126.57 | TARGET_HIT | 0.50 | 10.00% |
