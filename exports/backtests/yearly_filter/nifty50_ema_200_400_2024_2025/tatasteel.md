# TATASTEEL (TATASTEEL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 214.60
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
| ALERT2 | 6 |
| ALERT2_SKIP | 2 |
| ALERT3 | 35 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 27 |
| PARTIAL | 1 |
| TARGET_HIT | 11 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 17
- **Target hits / Stop hits / Partials:** 10 / 17 / 1
- **Avg / median % per leg:** 2.14% / -0.90%
- **Sum % (uncompounded):** 59.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 9 | 42.9% | 9 | 12 | 0 | 2.38% | 49.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 21 | 9 | 42.9% | 9 | 12 | 0 | 2.38% | 49.9% |
| SELL (all) | 7 | 2 | 28.6% | 1 | 5 | 1 | 1.44% | 10.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 7 | 2 | 28.6% | 1 | 5 | 1 | 1.44% | 10.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 28 | 11 | 39.3% | 10 | 17 | 1 | 2.14% | 60.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 09:15:00 | 163.41 | 168.00 | 168.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-29 13:15:00 | 162.57 | 167.82 | 167.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-13 09:15:00 | 154.60 | 153.65 | 157.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-13 10:00:00 | 154.60 | 153.65 | 157.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 159.06 | 153.21 | 156.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-24 10:00:00 | 159.06 | 153.21 | 156.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 159.10 | 153.27 | 156.41 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-07 10:15:00 | 164.53 | 158.70 | 158.70 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 15:15:00 | 155.10 | 158.78 | 158.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 154.25 | 158.73 | 158.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 12:15:00 | 153.29 | 153.27 | 155.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-06 13:00:00 | 153.29 | 153.27 | 155.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 156.37 | 153.30 | 155.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 11:45:00 | 154.10 | 153.33 | 155.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 09:15:00 | 146.39 | 152.90 | 155.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-11-13 14:15:00 | 138.69 | 151.25 | 154.01 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-03-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-07 15:15:00 | 151.50 | 137.99 | 137.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-10 09:15:00 | 153.13 | 138.14 | 138.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 144.79 | 149.70 | 145.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 144.79 | 149.70 | 145.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 144.79 | 149.70 | 145.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 144.79 | 149.70 | 145.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 145.34 | 149.66 | 145.49 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2025-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 11:15:00 | 139.60 | 142.36 | 142.37 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 15:15:00 | 145.90 | 142.27 | 142.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 147.74 | 142.47 | 142.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 154.85 | 155.31 | 151.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 10:00:00 | 154.85 | 155.31 | 151.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 152.08 | 155.16 | 151.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 10:45:00 | 153.14 | 154.92 | 151.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 10:30:00 | 152.41 | 154.28 | 151.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 13:15:00 | 152.36 | 154.24 | 151.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 12:30:00 | 152.37 | 154.08 | 151.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-03 09:15:00 | 168.45 | 156.31 | 153.34 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 12:15:00 | 161.79 | 170.51 | 170.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 14:15:00 | 160.68 | 170.33 | 170.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 169.49 | 169.38 | 169.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 169.49 | 169.38 | 169.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 169.49 | 169.38 | 169.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:30:00 | 169.52 | 169.38 | 169.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 170.05 | 169.39 | 169.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:00:00 | 170.05 | 169.39 | 169.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 170.54 | 169.40 | 169.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:45:00 | 170.72 | 169.40 | 169.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 170.01 | 169.73 | 170.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 14:15:00 | 169.53 | 169.73 | 170.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 169.40 | 169.76 | 170.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 13:00:00 | 169.51 | 169.76 | 170.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 12:00:00 | 169.59 | 169.64 | 169.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 170.12 | 169.64 | 169.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 10:00:00 | 170.12 | 169.64 | 169.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 170.15 | 169.64 | 169.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:00:00 | 170.15 | 169.64 | 169.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 171.04 | 169.66 | 169.99 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-23 11:15:00 | 171.04 | 169.66 | 169.99 | SL hit (close>static) qty=1.00 sl=170.65 alert=retest2 |

### Cycle 8 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 179.75 | 170.27 | 170.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 10:15:00 | 181.45 | 171.06 | 170.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 10:15:00 | 182.79 | 184.32 | 179.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-02 11:00:00 | 182.79 | 184.32 | 179.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 190.89 | 201.25 | 193.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 09:15:00 | 193.75 | 200.63 | 193.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 10:30:00 | 193.96 | 200.50 | 193.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 12:30:00 | 193.92 | 200.38 | 193.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 13:30:00 | 193.92 | 199.52 | 193.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 193.71 | 199.46 | 193.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 14:45:00 | 193.11 | 199.46 | 193.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 193.50 | 199.40 | 193.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 09:15:00 | 189.18 | 199.40 | 193.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 185.67 | 199.26 | 193.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 185.67 | 199.26 | 193.70 | SL hit (close<static) qty=1.00 sl=187.03 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-11-07 11:45:00 | 154.10 | 2024-11-11 09:15:00 | 146.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 11:45:00 | 154.10 | 2024-11-13 14:15:00 | 138.69 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-06-16 10:45:00 | 153.14 | 2025-07-03 09:15:00 | 168.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-20 10:30:00 | 152.41 | 2025-07-03 09:15:00 | 167.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-20 13:15:00 | 152.36 | 2025-07-03 09:15:00 | 167.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-23 12:30:00 | 152.37 | 2025-07-03 09:15:00 | 167.61 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-11 14:30:00 | 158.59 | 2025-08-14 09:15:00 | 157.15 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-08-12 09:15:00 | 160.85 | 2025-08-14 09:15:00 | 157.15 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-08-19 13:15:00 | 158.69 | 2025-08-26 10:15:00 | 156.88 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-08-20 09:45:00 | 158.75 | 2025-08-26 10:15:00 | 156.88 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-09-03 09:15:00 | 161.82 | 2025-10-28 09:15:00 | 178.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-12-16 14:15:00 | 169.53 | 2025-12-23 11:15:00 | 171.04 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-12-18 09:15:00 | 169.40 | 2025-12-23 11:15:00 | 171.04 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-12-18 13:00:00 | 169.51 | 2025-12-23 11:15:00 | 171.04 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-12-22 12:00:00 | 169.59 | 2025-12-23 11:15:00 | 171.04 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-12-24 14:00:00 | 170.44 | 2025-12-29 09:15:00 | 172.70 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-03-10 09:15:00 | 193.75 | 2026-03-13 09:15:00 | 185.67 | STOP_HIT | 1.00 | -4.17% |
| BUY | retest2 | 2026-03-10 10:30:00 | 193.96 | 2026-03-13 09:15:00 | 185.67 | STOP_HIT | 1.00 | -4.27% |
| BUY | retest2 | 2026-03-10 12:30:00 | 193.92 | 2026-03-13 09:15:00 | 185.67 | STOP_HIT | 1.00 | -4.25% |
| BUY | retest2 | 2026-03-12 13:30:00 | 193.92 | 2026-03-13 09:15:00 | 185.67 | STOP_HIT | 1.00 | -4.25% |
| BUY | retest2 | 2026-03-20 09:15:00 | 197.67 | 2026-03-23 09:15:00 | 187.50 | STOP_HIT | 1.00 | -5.14% |
| BUY | retest2 | 2026-03-20 15:00:00 | 196.70 | 2026-03-23 09:15:00 | 187.50 | STOP_HIT | 1.00 | -4.68% |
| BUY | retest2 | 2026-03-25 13:30:00 | 196.49 | 2026-03-30 09:15:00 | 190.00 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest2 | 2026-04-01 09:15:00 | 197.70 | 2026-04-02 09:15:00 | 188.82 | STOP_HIT | 1.00 | -4.49% |
| BUY | retest2 | 2026-04-06 12:45:00 | 195.45 | 2026-04-27 11:15:00 | 215.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 13:45:00 | 195.44 | 2026-04-27 11:15:00 | 214.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 11:15:00 | 195.54 | 2026-04-28 09:15:00 | 215.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 11:45:00 | 195.59 | 2026-04-28 09:15:00 | 215.15 | TARGET_HIT | 1.00 | 10.00% |
