# Manappuram Finance Ltd. (MANAPPURAM)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 315.05
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT2_SKIP | 1 |
| ALERT3 | 22 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 29 |
| PARTIAL | 5 |
| TARGET_HIT | 7 |
| STOP_HIT | 27 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 39 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 25
- **Target hits / Stop hits / Partials:** 7 / 27 / 5
- **Avg / median % per leg:** 0.63% / -1.36%
- **Sum % (uncompounded):** 24.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 34 | 14 | 41.2% | 7 | 22 | 5 | 1.20% | 40.8% |
| BUY @ 2nd Alert (retest1) | 10 | 7 | 70.0% | 1 | 4 | 5 | 3.45% | 34.5% |
| BUY @ 3rd Alert (retest2) | 24 | 7 | 29.2% | 6 | 18 | 0 | 0.26% | 6.3% |
| SELL (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -3.25% | -16.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -3.25% | -16.2% |
| retest1 (combined) | 10 | 7 | 70.0% | 1 | 4 | 5 | 3.45% | 34.5% |
| retest2 (combined) | 29 | 7 | 24.1% | 6 | 23 | 0 | -0.34% | -9.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-02 11:15:00 | 135.30 | 139.82 | 139.82 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-06 13:15:00 | 143.10 | 139.82 | 139.82 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 11:15:00 | 135.90 | 139.82 | 139.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-10 12:15:00 | 135.20 | 139.78 | 139.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-13 15:15:00 | 144.80 | 139.58 | 139.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-13 15:15:00 | 144.80 | 139.58 | 139.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 15:15:00 | 144.80 | 139.58 | 139.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-15 09:15:00 | 151.55 | 139.58 | 139.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 152.65 | 139.71 | 139.76 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2023-11-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 10:15:00 | 151.30 | 139.82 | 139.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-16 09:15:00 | 161.25 | 140.56 | 140.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-10 09:15:00 | 167.35 | 168.74 | 161.05 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-16 11:15:00 | 177.55 | 169.66 | 162.57 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-19 10:15:00 | 176.90 | 170.26 | 163.56 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-19 11:00:00 | 176.65 | 170.32 | 163.63 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-25 14:15:00 | 176.80 | 171.38 | 165.17 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-30 09:15:00 | 186.43 | 172.23 | 165.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-30 09:15:00 | 185.75 | 172.23 | 165.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-30 09:15:00 | 185.48 | 172.23 | 165.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-30 09:15:00 | 185.64 | 172.23 | 165.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-02-12 11:15:00 | 176.75 | 178.02 | 171.03 | SL hit (close<ema200) qty=0.50 sl=178.02 alert=retest1 |

### Cycle 5 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 169.80 | 181.56 | 181.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 10:15:00 | 169.10 | 181.43 | 181.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 13:15:00 | 177.90 | 177.57 | 179.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-06 13:45:00 | 178.30 | 177.57 | 179.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 178.50 | 177.57 | 179.37 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2024-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 13:15:00 | 192.04 | 180.62 | 180.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 09:15:00 | 192.78 | 180.93 | 180.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 13:15:00 | 204.80 | 206.03 | 197.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-23 13:30:00 | 204.46 | 206.03 | 197.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 10:15:00 | 199.38 | 208.39 | 200.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 11:00:00 | 199.38 | 208.39 | 200.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 11:15:00 | 202.30 | 208.33 | 200.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-06 09:15:00 | 205.71 | 208.01 | 200.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-06 10:45:00 | 203.56 | 207.92 | 200.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-06 14:15:00 | 196.69 | 207.61 | 200.84 | SL hit (close<static) qty=1.00 sl=196.94 alert=retest2 |

### Cycle 7 — SELL (started 2024-10-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 14:15:00 | 189.67 | 204.74 | 204.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 187.99 | 204.42 | 204.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 163.08 | 159.45 | 170.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-03 10:00:00 | 163.08 | 159.45 | 170.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 15:15:00 | 169.30 | 160.25 | 169.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:15:00 | 167.13 | 160.25 | 169.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 166.41 | 160.31 | 169.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 11:45:00 | 165.60 | 160.44 | 169.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 11:15:00 | 171.71 | 161.47 | 169.64 | SL hit (close>static) qty=1.00 sl=170.95 alert=retest2 |

### Cycle 8 — BUY (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 11:15:00 | 188.79 | 174.33 | 174.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 14:15:00 | 193.62 | 179.77 | 177.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-27 15:15:00 | 183.05 | 184.70 | 180.95 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-28 09:15:00 | 187.60 | 184.70 | 180.95 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 09:15:00 | 196.98 | 185.16 | 181.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-02-05 09:15:00 | 206.36 | 189.89 | 184.66 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 9 — SELL (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 15:15:00 | 267.50 | 294.08 | 294.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 10:15:00 | 264.80 | 293.52 | 293.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 269.50 | 266.27 | 275.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 11:00:00 | 269.50 | 266.27 | 275.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 271.20 | 266.31 | 274.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 12:15:00 | 269.50 | 266.39 | 274.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 09:45:00 | 269.90 | 266.52 | 274.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 12:30:00 | 270.40 | 266.65 | 274.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 13:30:00 | 269.55 | 266.98 | 273.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 274.20 | 267.09 | 273.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:00:00 | 274.20 | 267.09 | 273.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 278.30 | 267.21 | 273.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-21 10:15:00 | 278.30 | 267.21 | 273.79 | SL hit (close>static) qty=1.00 sl=274.75 alert=retest2 |

### Cycle 10 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 303.90 | 278.58 | 278.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 15:15:00 | 307.20 | 279.39 | 278.98 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-01-16 11:15:00 | 177.55 | 2024-01-30 09:15:00 | 186.43 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-01-19 10:15:00 | 176.90 | 2024-01-30 09:15:00 | 185.75 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-01-19 11:00:00 | 176.65 | 2024-01-30 09:15:00 | 185.48 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-01-25 14:15:00 | 176.80 | 2024-01-30 09:15:00 | 185.64 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-01-16 11:15:00 | 177.55 | 2024-02-12 11:15:00 | 176.75 | STOP_HIT | 0.50 | -0.45% |
| BUY | retest1 | 2024-01-19 10:15:00 | 176.90 | 2024-02-12 11:15:00 | 176.75 | STOP_HIT | 0.50 | -0.08% |
| BUY | retest1 | 2024-01-19 11:00:00 | 176.65 | 2024-02-12 11:15:00 | 176.75 | STOP_HIT | 0.50 | 0.06% |
| BUY | retest1 | 2024-01-25 14:15:00 | 176.80 | 2024-02-12 11:15:00 | 176.75 | STOP_HIT | 0.50 | -0.03% |
| BUY | retest2 | 2024-02-14 10:45:00 | 177.50 | 2024-03-05 09:15:00 | 195.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-29 15:15:00 | 177.15 | 2024-03-05 09:15:00 | 194.87 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-01 09:15:00 | 181.10 | 2024-04-04 09:15:00 | 199.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-14 09:15:00 | 177.70 | 2024-05-18 12:15:00 | 181.50 | STOP_HIT | 1.00 | 2.14% |
| BUY | retest2 | 2024-05-16 14:15:00 | 183.80 | 2024-05-18 12:15:00 | 181.50 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-05-17 15:15:00 | 184.00 | 2024-05-18 12:15:00 | 181.50 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-05-18 11:30:00 | 184.15 | 2024-05-30 09:15:00 | 169.80 | STOP_HIT | 1.00 | -7.79% |
| BUY | retest2 | 2024-08-06 09:15:00 | 205.71 | 2024-08-06 14:15:00 | 196.69 | STOP_HIT | 1.00 | -4.38% |
| BUY | retest2 | 2024-08-06 10:45:00 | 203.56 | 2024-08-06 14:15:00 | 196.69 | STOP_HIT | 1.00 | -3.37% |
| BUY | retest2 | 2024-08-09 09:30:00 | 202.44 | 2024-08-16 10:15:00 | 197.87 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2024-08-09 11:00:00 | 203.60 | 2024-08-16 10:15:00 | 197.87 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2024-08-14 14:30:00 | 201.81 | 2024-08-30 09:15:00 | 222.04 | TARGET_HIT | 1.00 | 10.02% |
| BUY | retest2 | 2024-08-14 15:00:00 | 201.60 | 2024-09-26 09:15:00 | 198.85 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-08-16 14:00:00 | 201.85 | 2024-09-26 09:15:00 | 198.85 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-08-16 14:45:00 | 202.53 | 2024-09-26 09:15:00 | 198.85 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2024-08-19 09:15:00 | 203.15 | 2024-09-26 09:15:00 | 198.85 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2024-08-20 09:30:00 | 202.13 | 2024-09-26 09:15:00 | 198.85 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2024-09-11 15:15:00 | 202.00 | 2024-10-01 09:15:00 | 196.27 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2024-09-25 14:30:00 | 202.10 | 2024-10-01 09:15:00 | 196.27 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2024-12-05 11:45:00 | 165.60 | 2024-12-09 11:15:00 | 171.71 | STOP_HIT | 1.00 | -3.69% |
| BUY | retest1 | 2025-01-28 09:15:00 | 187.60 | 2025-01-29 09:15:00 | 196.98 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-01-28 09:15:00 | 187.60 | 2025-02-05 09:15:00 | 206.36 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-12 11:45:00 | 193.30 | 2025-02-14 09:15:00 | 181.89 | STOP_HIT | 1.00 | -5.90% |
| BUY | retest2 | 2025-02-12 13:15:00 | 193.24 | 2025-02-14 09:15:00 | 181.89 | STOP_HIT | 1.00 | -5.87% |
| BUY | retest2 | 2025-02-13 09:15:00 | 195.02 | 2025-02-14 09:15:00 | 181.89 | STOP_HIT | 1.00 | -6.73% |
| BUY | retest2 | 2025-02-17 14:00:00 | 193.58 | 2025-02-27 10:15:00 | 212.94 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-19 11:15:00 | 197.76 | 2025-03-20 09:15:00 | 217.54 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-15 12:15:00 | 269.50 | 2026-04-21 10:15:00 | 278.30 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2026-04-16 09:45:00 | 269.90 | 2026-04-21 10:15:00 | 278.30 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2026-04-16 12:30:00 | 270.40 | 2026-04-21 10:15:00 | 278.30 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2026-04-20 13:30:00 | 269.55 | 2026-04-21 10:15:00 | 278.30 | STOP_HIT | 1.00 | -3.25% |
