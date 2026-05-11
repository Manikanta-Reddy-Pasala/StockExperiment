# Canara Bank (CANBK)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 134.13
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 30 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 30 |
| PARTIAL | 6 |
| TARGET_HIT | 4 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 13 / 19
- **Target hits / Stop hits / Partials:** 4 / 22 / 6
- **Avg / median % per leg:** 1.47% / -0.39%
- **Sum % (uncompounded):** 47.18%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 4 | 21.1% | 4 | 15 | 0 | 0.88% | 16.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 19 | 4 | 21.1% | 4 | 15 | 0 | 0.88% | 16.7% |
| SELL (all) | 13 | 9 | 69.2% | 0 | 7 | 6 | 2.35% | 30.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 13 | 9 | 69.2% | 0 | 7 | 6 | 2.35% | 30.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 32 | 13 | 40.6% | 4 | 22 | 6 | 1.47% | 47.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 14:15:00 | 112.80 | 116.61 | 116.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-24 11:15:00 | 111.98 | 116.45 | 116.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-29 09:15:00 | 116.79 | 115.76 | 116.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-29 09:15:00 | 116.79 | 115.76 | 116.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 116.79 | 115.76 | 116.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-31 13:30:00 | 115.00 | 115.77 | 116.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 10:00:00 | 114.88 | 115.74 | 116.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 10:30:00 | 114.64 | 115.73 | 116.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 109.25 | 115.20 | 115.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 109.14 | 115.20 | 115.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 108.91 | 115.20 | 115.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-20 12:15:00 | 111.55 | 111.49 | 113.41 | SL hit (close>ema200) qty=0.50 sl=111.49 alert=retest2 |

### Cycle 2 — BUY (started 2025-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 12:15:00 | 99.04 | 91.13 | 91.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 13:15:00 | 99.49 | 91.21 | 91.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 14:15:00 | 92.33 | 93.97 | 92.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 14:15:00 | 92.33 | 93.97 | 92.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 92.33 | 93.97 | 92.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:45:00 | 92.17 | 93.97 | 92.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 92.09 | 93.95 | 92.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:15:00 | 94.00 | 93.95 | 92.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 93.51 | 93.93 | 92.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 11:15:00 | 93.75 | 93.93 | 92.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 12:15:00 | 93.88 | 93.93 | 92.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 15:15:00 | 93.89 | 93.92 | 92.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 13:15:00 | 98.05 | 93.92 | 92.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-13 09:15:00 | 103.13 | 94.73 | 93.32 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 12:15:00 | 107.72 | 108.72 | 108.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 09:15:00 | 107.47 | 108.69 | 108.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 109.71 | 108.42 | 108.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 109.71 | 108.42 | 108.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 109.71 | 108.42 | 108.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:00:00 | 109.71 | 108.42 | 108.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 110.72 | 108.44 | 108.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:30:00 | 110.73 | 108.44 | 108.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 12:15:00 | 112.47 | 108.71 | 108.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 113.15 | 109.21 | 108.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 143.26 | 143.63 | 135.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 12:45:00 | 142.98 | 143.63 | 135.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 148.10 | 152.17 | 147.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 09:30:00 | 150.95 | 148.43 | 147.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 10:45:00 | 150.80 | 148.46 | 147.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 11:15:00 | 151.07 | 148.46 | 147.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:30:00 | 151.35 | 148.77 | 147.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 148.36 | 151.65 | 149.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:00:00 | 148.36 | 151.65 | 149.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 147.20 | 151.61 | 149.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:45:00 | 146.55 | 151.61 | 149.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-04 11:15:00 | 145.80 | 151.55 | 149.21 | SL hit (close<static) qty=1.00 sl=146.62 alert=retest2 |

### Cycle 5 — SELL (started 2026-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 15:15:00 | 134.70 | 147.46 | 147.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 134.26 | 147.33 | 147.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 137.19 | 137.14 | 141.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 11:00:00 | 137.19 | 137.14 | 141.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 141.55 | 137.58 | 140.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 10:30:00 | 141.30 | 137.61 | 140.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 12:15:00 | 141.47 | 137.65 | 140.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 15:15:00 | 141.43 | 137.77 | 140.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 11:15:00 | 141.31 | 137.89 | 140.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 14:15:00 | 141.15 | 138.00 | 140.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 14:45:00 | 141.30 | 138.00 | 140.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 15:15:00 | 141.07 | 138.03 | 140.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 09:15:00 | 141.39 | 138.03 | 140.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 141.59 | 138.06 | 140.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 10:00:00 | 141.59 | 138.06 | 140.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 10:15:00 | 141.31 | 138.09 | 140.95 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-20 10:15:00 | 144.05 | 138.39 | 141.01 | SL hit (close>static) qty=1.00 sl=143.67 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-07-31 13:30:00 | 115.00 | 2024-08-05 09:15:00 | 109.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-01 10:00:00 | 114.88 | 2024-08-05 09:15:00 | 109.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-01 10:30:00 | 114.64 | 2024-08-05 09:15:00 | 108.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-31 13:30:00 | 115.00 | 2024-08-20 12:15:00 | 111.55 | STOP_HIT | 0.50 | 3.00% |
| SELL | retest2 | 2024-08-01 10:00:00 | 114.88 | 2024-08-20 12:15:00 | 111.55 | STOP_HIT | 0.50 | 2.90% |
| SELL | retest2 | 2024-08-01 10:30:00 | 114.64 | 2024-08-20 12:15:00 | 111.55 | STOP_HIT | 0.50 | 2.70% |
| BUY | retest2 | 2025-05-07 11:15:00 | 93.75 | 2025-05-13 09:15:00 | 103.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-07 12:15:00 | 93.88 | 2025-05-13 09:15:00 | 103.27 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-07 15:15:00 | 93.89 | 2025-05-13 09:15:00 | 103.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-08 13:15:00 | 98.05 | 2025-05-16 10:15:00 | 107.86 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-06 12:45:00 | 109.36 | 2025-08-06 14:15:00 | 108.75 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-08-11 09:15:00 | 109.00 | 2025-08-11 11:15:00 | 108.71 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-08-11 12:45:00 | 109.38 | 2025-08-25 14:15:00 | 108.78 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-08-13 12:00:00 | 109.06 | 2025-08-25 14:15:00 | 108.78 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-08-14 09:30:00 | 109.21 | 2025-08-25 14:15:00 | 108.78 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-08-14 10:15:00 | 109.69 | 2025-08-25 14:15:00 | 108.78 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-08-14 11:45:00 | 109.29 | 2025-08-25 14:15:00 | 108.78 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-08-14 13:30:00 | 109.23 | 2025-08-26 09:15:00 | 107.82 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-08-18 09:15:00 | 109.75 | 2025-08-26 09:15:00 | 107.82 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-08-18 13:15:00 | 109.64 | 2025-08-26 09:15:00 | 107.82 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-08-25 14:00:00 | 109.35 | 2025-08-26 09:15:00 | 107.82 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2026-02-18 09:30:00 | 150.95 | 2026-03-04 11:15:00 | 145.80 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest2 | 2026-02-18 10:45:00 | 150.80 | 2026-03-04 11:15:00 | 145.80 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2026-02-18 11:15:00 | 151.07 | 2026-03-04 11:15:00 | 145.80 | STOP_HIT | 1.00 | -3.49% |
| BUY | retest2 | 2026-02-20 09:30:00 | 151.35 | 2026-03-04 11:15:00 | 145.80 | STOP_HIT | 1.00 | -3.67% |
| SELL | retest2 | 2026-04-15 10:30:00 | 141.30 | 2026-04-20 10:15:00 | 144.05 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2026-04-15 12:15:00 | 141.47 | 2026-04-20 10:15:00 | 144.05 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2026-04-15 15:15:00 | 141.43 | 2026-04-20 10:15:00 | 144.05 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2026-04-16 11:15:00 | 141.31 | 2026-04-20 10:15:00 | 144.05 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-04-23 14:00:00 | 140.53 | 2026-04-30 10:15:00 | 133.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-24 09:15:00 | 140.11 | 2026-04-30 10:15:00 | 133.68 | PARTIAL | 0.50 | 4.59% |
| SELL | retest2 | 2026-04-24 14:45:00 | 140.72 | 2026-04-30 10:15:00 | 133.86 | PARTIAL | 0.50 | 4.87% |
