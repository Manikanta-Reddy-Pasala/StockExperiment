# FSN E-Commerce Ventures Ltd. (NYKAA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 273.00
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
| ALERT2 | 7 |
| ALERT2_SKIP | 3 |
| ALERT3 | 21 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 20 |
| PARTIAL | 4 |
| TARGET_HIT | 8 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 12
- **Target hits / Stop hits / Partials:** 8 / 12 / 4
- **Avg / median % per leg:** 3.18% / 5.00%
- **Sum % (uncompounded):** 76.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 7 | 58.3% | 7 | 5 | 0 | 4.86% | 58.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 7 | 58.3% | 7 | 5 | 0 | 4.86% | 58.3% |
| SELL (all) | 12 | 5 | 41.7% | 1 | 7 | 4 | 1.51% | 18.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 5 | 41.7% | 1 | 7 | 4 | 1.51% | 18.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 24 | 12 | 50.0% | 8 | 12 | 4 | 3.18% | 76.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 09:15:00 | 146.15 | 165.18 | 165.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 11:15:00 | 145.15 | 164.79 | 165.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-01 09:15:00 | 157.40 | 157.19 | 160.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-01 09:15:00 | 157.40 | 157.19 | 160.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 157.40 | 157.19 | 160.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 09:45:00 | 158.85 | 157.19 | 160.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 158.90 | 157.26 | 160.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-06 09:15:00 | 158.00 | 157.55 | 160.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 12:45:00 | 158.30 | 157.51 | 159.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-12 09:15:00 | 157.25 | 157.65 | 159.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 14:15:00 | 150.10 | 157.27 | 159.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 14:15:00 | 150.38 | 157.27 | 159.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 14:15:00 | 149.39 | 157.27 | 159.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-21 09:15:00 | 158.40 | 155.57 | 158.18 | SL hit (close>ema200) qty=0.50 sl=155.57 alert=retest2 |

### Cycle 2 — BUY (started 2024-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 10:15:00 | 177.15 | 159.87 | 159.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 11:15:00 | 178.60 | 160.05 | 159.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 09:15:00 | 164.25 | 166.07 | 163.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-19 09:15:00 | 164.25 | 166.07 | 163.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 164.25 | 166.07 | 163.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-19 09:30:00 | 162.70 | 166.07 | 163.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 10:15:00 | 164.40 | 166.06 | 163.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-19 11:30:00 | 165.45 | 166.06 | 163.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-23 09:15:00 | 182.00 | 171.79 | 168.70 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 11:15:00 | 182.00 | 194.97 | 194.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 179.46 | 194.31 | 194.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-07 11:15:00 | 186.55 | 186.53 | 189.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-07 11:45:00 | 187.71 | 186.53 | 189.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 190.29 | 186.59 | 189.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:00:00 | 190.29 | 186.59 | 189.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 192.33 | 186.65 | 189.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 09:15:00 | 189.48 | 186.65 | 189.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 12:15:00 | 180.01 | 186.26 | 189.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-11-13 14:15:00 | 170.53 | 185.29 | 188.69 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 11:15:00 | 180.00 | 169.64 | 169.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 11:15:00 | 183.00 | 172.56 | 171.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 11:15:00 | 195.00 | 196.06 | 189.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-02 11:45:00 | 194.72 | 196.06 | 189.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 193.04 | 196.68 | 191.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 193.69 | 196.68 | 191.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 12:45:00 | 193.32 | 196.58 | 191.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 10:15:00 | 193.55 | 196.48 | 191.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 12:15:00 | 193.83 | 196.32 | 191.50 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-09 14:15:00 | 213.06 | 201.07 | 196.47 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 14:15:00 | 234.65 | 253.34 | 253.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 15:15:00 | 234.00 | 253.15 | 253.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 247.64 | 247.56 | 250.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 12:15:00 | 248.14 | 247.56 | 250.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 251.55 | 247.51 | 250.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 14:00:00 | 251.55 | 247.51 | 250.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 251.63 | 247.55 | 250.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 15:00:00 | 251.63 | 247.55 | 250.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 258.50 | 247.69 | 250.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 258.50 | 247.69 | 250.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 09:15:00 | 276.25 | 252.16 | 252.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 11:15:00 | 280.28 | 252.68 | 252.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 09:15:00 | 261.67 | 262.06 | 258.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-24 09:30:00 | 261.17 | 262.06 | 258.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 12:15:00 | 259.31 | 261.98 | 258.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:00:00 | 259.31 | 261.98 | 258.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 261.75 | 262.79 | 259.00 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2026-03-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 12:15:00 | 235.25 | 256.54 | 256.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 233.75 | 251.85 | 254.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 09:15:00 | 254.22 | 247.16 | 250.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 09:15:00 | 254.22 | 247.16 | 250.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 254.22 | 247.16 | 250.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:45:00 | 252.70 | 247.16 | 250.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 252.95 | 247.21 | 250.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 12:00:00 | 251.74 | 247.26 | 250.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 13:30:00 | 251.82 | 247.36 | 250.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:15:00 | 245.73 | 247.46 | 250.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-09 09:15:00 | 256.16 | 247.95 | 250.97 | SL hit (close>static) qty=1.00 sl=255.09 alert=retest2 |

### Cycle 8 — BUY (started 2026-04-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 14:15:00 | 264.88 | 253.28 | 253.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 10:15:00 | 267.12 | 255.88 | 254.67 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-03-06 09:15:00 | 158.00 | 2024-03-13 14:15:00 | 150.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-07 12:45:00 | 158.30 | 2024-03-13 14:15:00 | 150.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-12 09:15:00 | 157.25 | 2024-03-13 14:15:00 | 149.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-06 09:15:00 | 158.00 | 2024-03-21 09:15:00 | 158.40 | STOP_HIT | 0.50 | -0.25% |
| SELL | retest2 | 2024-03-07 12:45:00 | 158.30 | 2024-03-21 09:15:00 | 158.40 | STOP_HIT | 0.50 | -0.06% |
| SELL | retest2 | 2024-03-12 09:15:00 | 157.25 | 2024-03-21 09:15:00 | 158.40 | STOP_HIT | 0.50 | -0.73% |
| SELL | retest2 | 2024-03-21 10:00:00 | 158.40 | 2024-03-22 13:15:00 | 163.40 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest2 | 2024-04-19 11:30:00 | 165.45 | 2024-05-23 09:15:00 | 182.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-28 13:30:00 | 165.20 | 2024-05-29 12:15:00 | 162.40 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-05-29 09:30:00 | 164.95 | 2024-05-29 12:15:00 | 162.40 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-06-03 09:15:00 | 167.10 | 2024-06-03 12:15:00 | 162.95 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2024-06-06 14:00:00 | 169.35 | 2024-06-14 09:15:00 | 164.00 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest2 | 2024-06-10 14:30:00 | 168.80 | 2024-06-14 09:15:00 | 164.00 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2024-06-14 14:15:00 | 172.38 | 2024-07-30 09:15:00 | 189.62 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-08 09:15:00 | 189.48 | 2024-11-12 12:15:00 | 180.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 09:15:00 | 189.48 | 2024-11-13 14:15:00 | 170.53 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-06-13 10:15:00 | 193.69 | 2025-07-09 14:15:00 | 213.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-13 12:45:00 | 193.32 | 2025-07-09 14:15:00 | 212.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-16 10:15:00 | 193.55 | 2025-07-09 14:15:00 | 212.91 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-17 12:15:00 | 193.83 | 2025-07-09 14:15:00 | 213.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-13 09:15:00 | 212.48 | 2025-08-26 14:15:00 | 233.73 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-06 12:00:00 | 251.74 | 2026-04-09 09:15:00 | 256.16 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2026-04-06 13:30:00 | 251.82 | 2026-04-09 09:15:00 | 256.16 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2026-04-07 09:15:00 | 245.73 | 2026-04-09 09:15:00 | 256.16 | STOP_HIT | 1.00 | -4.24% |
