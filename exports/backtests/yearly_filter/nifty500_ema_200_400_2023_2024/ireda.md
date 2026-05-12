# Indian Renewable Energy Development Agency Ltd. (IREDA)

## Backtest Summary

- **Window:** 2023-11-29 09:15:00 → 2026-05-11 15:15:00 (4224 bars)
- **Last close:** 130.92
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 4 |
| ALERT3 | 40 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 42 |
| PARTIAL | 20 |
| TARGET_HIT | 11 |
| STOP_HIT | 31 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 62 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 39 / 23
- **Target hits / Stop hits / Partials:** 11 / 31 / 20
- **Avg / median % per leg:** 2.58% / 4.83%
- **Sum % (uncompounded):** 159.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 62 | 39 | 62.9% | 11 | 31 | 20 | 2.58% | 160.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 62 | 39 | 62.9% | 11 | 31 | 20 | 2.58% | 160.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 62 | 39 | 62.9% | 11 | 31 | 20 | 2.58% | 160.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 11:15:00 | 220.60 | 230.39 | 230.42 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-10-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 12:15:00 | 233.27 | 230.43 | 230.42 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 09:15:00 | 224.88 | 230.36 | 230.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 10:15:00 | 223.89 | 230.30 | 230.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 207.61 | 201.17 | 210.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 09:15:00 | 207.61 | 201.17 | 210.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 207.61 | 201.17 | 210.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:15:00 | 209.85 | 201.17 | 210.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 207.01 | 201.23 | 210.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 12:30:00 | 204.94 | 201.32 | 210.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 09:15:00 | 205.50 | 201.51 | 210.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 09:30:00 | 206.00 | 201.87 | 209.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 12:15:00 | 206.02 | 201.97 | 209.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 211.96 | 202.24 | 209.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:00:00 | 211.96 | 202.24 | 209.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 210.75 | 202.33 | 209.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 11:30:00 | 209.97 | 202.40 | 209.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-04 14:15:00 | 217.19 | 202.71 | 209.61 | SL hit (close>static) qty=1.00 sl=213.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 12:15:00 | 219.27 | 212.30 | 212.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-08 13:15:00 | 222.82 | 212.41 | 212.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 09:15:00 | 211.50 | 213.15 | 212.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 09:15:00 | 211.50 | 213.15 | 212.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 211.50 | 213.15 | 212.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 09:30:00 | 210.18 | 213.15 | 212.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 211.49 | 213.13 | 212.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:30:00 | 210.76 | 213.13 | 212.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 12:15:00 | 206.20 | 213.03 | 212.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 12:30:00 | 206.45 | 213.03 | 212.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2025-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 14:15:00 | 199.55 | 212.21 | 212.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 198.08 | 209.73 | 210.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 09:15:00 | 206.89 | 203.25 | 206.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 09:15:00 | 206.89 | 203.25 | 206.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 206.89 | 203.25 | 206.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 09:30:00 | 207.82 | 203.25 | 206.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 205.92 | 203.27 | 206.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 10:45:00 | 207.91 | 203.27 | 206.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 202.07 | 203.26 | 206.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:30:00 | 204.30 | 203.26 | 206.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 172.20 | 160.24 | 173.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 09:30:00 | 176.26 | 160.24 | 173.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 171.70 | 160.35 | 173.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 11:30:00 | 169.84 | 160.46 | 173.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 14:15:00 | 170.60 | 160.67 | 173.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 15:15:00 | 168.10 | 160.77 | 173.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-27 09:15:00 | 161.35 | 161.23 | 173.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-27 09:15:00 | 165.30 | 161.23 | 173.23 | SL hit (close>static) qty=0.50 sl=161.23 alert=retest2 |

### Cycle 6 — BUY (started 2025-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 14:15:00 | 175.49 | 169.87 | 169.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 10:15:00 | 177.19 | 170.50 | 170.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 169.80 | 173.60 | 171.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 169.80 | 173.60 | 171.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 169.80 | 173.60 | 171.95 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 10:15:00 | 165.85 | 170.62 | 170.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 161.07 | 169.00 | 169.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 10:15:00 | 148.14 | 147.62 | 152.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 11:00:00 | 148.14 | 147.62 | 152.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 153.82 | 147.66 | 152.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:00:00 | 153.82 | 147.66 | 152.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 152.63 | 147.70 | 152.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:30:00 | 153.62 | 147.70 | 152.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 152.50 | 147.91 | 152.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:30:00 | 152.85 | 147.91 | 152.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 152.40 | 148.00 | 152.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:45:00 | 152.80 | 148.00 | 152.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 152.35 | 148.04 | 152.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 13:45:00 | 151.74 | 151.17 | 153.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:30:00 | 151.87 | 150.66 | 152.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 10:15:00 | 153.38 | 150.69 | 152.61 | SL hit (close>static) qty=1.00 sl=152.75 alert=retest2 |

### Cycle 8 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 137.81 | 125.76 | 125.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 139.34 | 126.02 | 125.84 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-11-28 12:30:00 | 204.94 | 2024-12-04 14:15:00 | 217.19 | STOP_HIT | 1.00 | -5.98% |
| SELL | retest2 | 2024-11-29 09:15:00 | 205.50 | 2024-12-04 14:15:00 | 217.19 | STOP_HIT | 1.00 | -5.69% |
| SELL | retest2 | 2024-12-03 09:30:00 | 206.00 | 2024-12-04 14:15:00 | 217.19 | STOP_HIT | 1.00 | -5.43% |
| SELL | retest2 | 2024-12-03 12:15:00 | 206.02 | 2024-12-04 14:15:00 | 217.19 | STOP_HIT | 1.00 | -5.42% |
| SELL | retest2 | 2024-12-04 11:30:00 | 209.97 | 2024-12-04 14:15:00 | 217.19 | STOP_HIT | 1.00 | -3.44% |
| SELL | retest2 | 2024-12-18 09:45:00 | 209.39 | 2024-12-26 13:15:00 | 198.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-18 09:45:00 | 209.39 | 2024-12-30 14:15:00 | 218.87 | STOP_HIT | 0.50 | -4.53% |
| SELL | retest2 | 2025-03-25 11:30:00 | 169.84 | 2025-03-27 09:15:00 | 161.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-25 11:30:00 | 169.84 | 2025-03-27 09:15:00 | 165.30 | STOP_HIT | 0.50 | 2.67% |
| SELL | retest2 | 2025-03-25 14:15:00 | 170.60 | 2025-03-27 09:15:00 | 162.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-25 14:15:00 | 170.60 | 2025-03-27 09:15:00 | 165.30 | STOP_HIT | 0.50 | 3.11% |
| SELL | retest2 | 2025-03-25 15:15:00 | 168.10 | 2025-03-28 15:15:00 | 159.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-25 15:15:00 | 168.10 | 2025-04-02 15:15:00 | 161.80 | STOP_HIT | 0.50 | 3.75% |
| SELL | retest2 | 2025-04-25 09:45:00 | 169.10 | 2025-05-06 14:15:00 | 160.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-25 15:15:00 | 166.74 | 2025-05-07 09:15:00 | 158.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-28 14:15:00 | 167.14 | 2025-05-07 09:15:00 | 158.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 09:15:00 | 165.58 | 2025-05-07 09:15:00 | 157.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 09:45:00 | 167.16 | 2025-05-07 09:15:00 | 158.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-25 09:45:00 | 169.10 | 2025-05-12 09:15:00 | 165.47 | STOP_HIT | 0.50 | 2.15% |
| SELL | retest2 | 2025-04-25 15:15:00 | 166.74 | 2025-05-12 09:15:00 | 165.47 | STOP_HIT | 0.50 | 0.76% |
| SELL | retest2 | 2025-04-28 14:15:00 | 167.14 | 2025-05-12 09:15:00 | 165.47 | STOP_HIT | 0.50 | 1.00% |
| SELL | retest2 | 2025-04-30 09:15:00 | 165.58 | 2025-05-12 09:15:00 | 165.47 | STOP_HIT | 0.50 | 0.07% |
| SELL | retest2 | 2025-04-30 09:45:00 | 167.16 | 2025-05-12 09:15:00 | 165.47 | STOP_HIT | 0.50 | 1.01% |
| SELL | retest2 | 2025-05-13 09:15:00 | 166.94 | 2025-05-16 09:15:00 | 171.40 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-05-13 10:15:00 | 167.85 | 2025-05-16 09:15:00 | 171.40 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-05-13 12:45:00 | 167.30 | 2025-05-16 09:15:00 | 171.40 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-05-14 10:30:00 | 167.81 | 2025-05-16 09:15:00 | 171.40 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-05-14 13:45:00 | 166.21 | 2025-05-16 09:15:00 | 171.40 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2025-05-15 10:15:00 | 166.27 | 2025-05-16 09:15:00 | 171.40 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2025-05-15 11:00:00 | 166.25 | 2025-05-16 09:15:00 | 171.40 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2025-09-25 13:45:00 | 151.74 | 2025-10-01 10:15:00 | 153.38 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-10-01 09:30:00 | 151.87 | 2025-10-01 10:15:00 | 153.38 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-10-01 13:15:00 | 151.80 | 2025-10-01 14:15:00 | 152.87 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-10-06 10:15:00 | 151.80 | 2025-10-07 13:15:00 | 153.36 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-10-08 09:15:00 | 151.84 | 2025-10-14 10:15:00 | 154.54 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-10-08 09:45:00 | 151.75 | 2025-10-14 10:15:00 | 154.54 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-10-17 09:45:00 | 152.03 | 2025-10-20 15:15:00 | 153.42 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-10-17 11:30:00 | 152.02 | 2025-10-20 15:15:00 | 153.42 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-10-28 14:00:00 | 151.17 | 2025-10-29 15:15:00 | 156.43 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2025-11-04 11:30:00 | 151.46 | 2025-11-24 09:15:00 | 143.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 13:15:00 | 151.28 | 2025-11-24 09:15:00 | 143.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 10:30:00 | 151.38 | 2025-11-24 09:15:00 | 143.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-13 12:30:00 | 150.19 | 2025-11-24 14:15:00 | 142.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-13 13:15:00 | 150.31 | 2025-11-24 14:15:00 | 142.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-14 11:00:00 | 149.98 | 2025-11-24 14:15:00 | 142.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 13:00:00 | 150.21 | 2025-11-24 14:15:00 | 142.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 11:30:00 | 151.46 | 2025-12-03 14:15:00 | 136.31 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-04 13:15:00 | 151.28 | 2025-12-03 14:15:00 | 136.24 | TARGET_HIT | 0.50 | 9.94% |
| SELL | retest2 | 2025-11-12 10:30:00 | 151.38 | 2025-12-05 09:15:00 | 136.15 | TARGET_HIT | 0.50 | 10.06% |
| SELL | retest2 | 2025-11-13 12:30:00 | 150.19 | 2025-12-05 09:15:00 | 135.17 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-13 13:15:00 | 150.31 | 2025-12-05 09:15:00 | 135.28 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-14 11:00:00 | 149.98 | 2025-12-05 09:15:00 | 134.98 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-17 13:00:00 | 150.21 | 2025-12-05 09:15:00 | 135.19 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-14 14:15:00 | 138.15 | 2026-01-20 12:15:00 | 131.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 15:00:00 | 138.16 | 2026-01-20 12:15:00 | 131.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 09:30:00 | 137.87 | 2026-01-20 12:15:00 | 131.21 | PARTIAL | 0.50 | 4.83% |
| SELL | retest2 | 2026-01-16 11:00:00 | 138.12 | 2026-01-20 13:15:00 | 130.98 | PARTIAL | 0.50 | 5.17% |
| SELL | retest2 | 2026-01-14 14:15:00 | 138.15 | 2026-02-01 12:15:00 | 124.34 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-14 15:00:00 | 138.16 | 2026-02-01 12:15:00 | 124.34 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-16 09:30:00 | 137.87 | 2026-02-01 12:15:00 | 124.31 | TARGET_HIT | 0.50 | 9.84% |
| SELL | retest2 | 2026-01-16 11:00:00 | 138.12 | 2026-02-13 09:15:00 | 124.08 | TARGET_HIT | 0.50 | 10.16% |
| SELL | retest2 | 2026-04-13 09:15:00 | 120.11 | 2026-04-13 12:15:00 | 123.29 | STOP_HIT | 1.00 | -2.65% |
