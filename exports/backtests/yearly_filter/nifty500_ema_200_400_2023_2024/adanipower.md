# Adani Power Ltd. (ADANIPOWER)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 225.02
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
| ALERT2 | 7 |
| ALERT2_SKIP | 5 |
| ALERT3 | 35 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 20 |
| PARTIAL | 6 |
| TARGET_HIT | 9 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 9
- **Target hits / Stop hits / Partials:** 9 / 11 / 6
- **Avg / median % per leg:** 3.90% / 5.00%
- **Sum % (uncompounded):** 101.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 5 | 35.7% | 5 | 9 | 0 | 2.11% | 29.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 5 | 35.7% | 5 | 9 | 0 | 2.11% | 29.5% |
| SELL (all) | 12 | 12 | 100.0% | 4 | 2 | 6 | 6.00% | 72.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 12 | 100.0% | 4 | 2 | 6 | 6.00% | 72.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 26 | 17 | 65.4% | 9 | 11 | 6 | 3.90% | 101.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 15:15:00 | 128.00 | 138.34 | 138.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 09:15:00 | 127.21 | 138.23 | 138.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 09:15:00 | 134.27 | 132.63 | 134.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 09:15:00 | 134.27 | 132.63 | 134.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 134.27 | 132.63 | 134.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 09:30:00 | 135.09 | 132.63 | 134.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 134.86 | 132.65 | 134.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 11:15:00 | 134.77 | 132.65 | 134.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 135.03 | 132.67 | 134.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 13:15:00 | 134.23 | 132.69 | 134.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 14:00:00 | 134.41 | 132.71 | 134.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 11:15:00 | 127.52 | 132.48 | 134.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 11:15:00 | 127.69 | 132.48 | 134.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-20 14:15:00 | 133.01 | 132.30 | 134.42 | SL hit (close>ema200) qty=0.50 sl=132.30 alert=retest2 |

### Cycle 2 — BUY (started 2025-04-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 14:15:00 | 109.25 | 102.94 | 102.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 10:15:00 | 109.96 | 103.13 | 103.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 14:15:00 | 106.10 | 107.15 | 105.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-30 14:45:00 | 106.61 | 107.15 | 105.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 15:15:00 | 105.80 | 107.14 | 105.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:15:00 | 106.77 | 107.14 | 105.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 106.92 | 107.14 | 105.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 10:15:00 | 107.90 | 107.14 | 105.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-02 15:15:00 | 104.74 | 107.05 | 105.42 | SL hit (close<static) qty=1.00 sl=104.81 alert=retest2 |

### Cycle 3 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 140.23 | 145.13 | 145.14 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 11:15:00 | 149.48 | 145.16 | 145.14 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 10:15:00 | 140.26 | 145.14 | 145.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 10:15:00 | 140.18 | 144.26 | 144.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 142.20 | 139.91 | 142.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 142.20 | 139.91 | 142.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 142.20 | 139.91 | 142.06 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 12:15:00 | 149.50 | 143.83 | 143.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 14:15:00 | 150.95 | 143.96 | 143.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 143.38 | 144.39 | 144.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 143.38 | 144.39 | 144.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 143.38 | 144.39 | 144.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 143.65 | 144.39 | 144.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 143.30 | 144.38 | 144.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 09:45:00 | 144.12 | 144.08 | 143.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 12:30:00 | 144.38 | 144.07 | 143.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 13:00:00 | 144.00 | 144.07 | 143.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 14:30:00 | 144.00 | 144.07 | 143.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 142.90 | 144.06 | 143.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:00:00 | 142.90 | 144.06 | 143.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 142.55 | 144.04 | 143.94 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-18 10:15:00 | 142.55 | 144.04 | 143.94 | SL hit (close<static) qty=1.00 sl=142.61 alert=retest2 |

### Cycle 7 — SELL (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 10:15:00 | 141.02 | 143.82 | 143.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 13:15:00 | 140.39 | 143.71 | 143.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 09:15:00 | 143.27 | 141.58 | 142.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 143.27 | 141.58 | 142.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 143.27 | 141.58 | 142.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 143.67 | 141.58 | 142.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 143.90 | 141.60 | 142.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:00:00 | 143.90 | 141.60 | 142.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 155.64 | 143.37 | 143.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 12:15:00 | 155.76 | 143.49 | 143.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 11:15:00 | 145.25 | 145.60 | 144.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-23 12:00:00 | 145.25 | 145.60 | 144.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-19 13:15:00 | 46.39 | 2023-05-23 09:15:00 | 51.03 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-09-16 13:15:00 | 134.23 | 2024-09-19 11:15:00 | 127.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-16 14:00:00 | 134.41 | 2024-09-19 11:15:00 | 127.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-16 13:15:00 | 134.23 | 2024-09-20 14:15:00 | 133.01 | STOP_HIT | 0.50 | 0.91% |
| SELL | retest2 | 2024-09-16 14:00:00 | 134.41 | 2024-09-20 14:15:00 | 133.01 | STOP_HIT | 0.50 | 1.04% |
| SELL | retest2 | 2024-09-24 09:15:00 | 134.20 | 2024-10-03 09:15:00 | 127.81 | PARTIAL | 0.50 | 4.76% |
| SELL | retest2 | 2024-09-24 11:00:00 | 134.29 | 2024-10-03 11:15:00 | 127.49 | PARTIAL | 0.50 | 5.06% |
| SELL | retest2 | 2024-09-24 14:45:00 | 134.43 | 2024-10-03 11:15:00 | 127.58 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2024-09-25 09:30:00 | 134.54 | 2024-10-03 11:15:00 | 127.71 | PARTIAL | 0.50 | 5.08% |
| SELL | retest2 | 2024-09-24 09:15:00 | 134.20 | 2024-10-18 09:15:00 | 120.78 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-24 11:00:00 | 134.29 | 2024-10-18 09:15:00 | 120.86 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-24 14:45:00 | 134.43 | 2024-10-18 09:15:00 | 120.99 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-25 09:30:00 | 134.54 | 2024-10-18 09:15:00 | 121.09 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-05-02 10:15:00 | 107.90 | 2025-05-02 15:15:00 | 104.74 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2025-05-05 09:45:00 | 107.80 | 2025-05-08 14:15:00 | 104.62 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2025-05-06 15:00:00 | 107.40 | 2025-05-08 14:15:00 | 104.62 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2025-05-07 11:30:00 | 107.53 | 2025-05-08 14:15:00 | 104.62 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2025-05-29 09:15:00 | 110.86 | 2025-06-10 12:15:00 | 121.95 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-29 12:45:00 | 111.31 | 2025-06-10 12:15:00 | 121.95 | TARGET_HIT | 1.00 | 9.56% |
| BUY | retest2 | 2025-05-30 09:15:00 | 110.86 | 2025-06-10 12:15:00 | 121.68 | TARGET_HIT | 1.00 | 9.76% |
| BUY | retest2 | 2025-05-30 13:15:00 | 110.62 | 2025-06-20 14:15:00 | 105.94 | STOP_HIT | 1.00 | -4.23% |
| BUY | retest2 | 2025-06-25 09:15:00 | 110.25 | 2025-06-27 10:15:00 | 121.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-17 09:45:00 | 144.12 | 2026-02-18 10:15:00 | 142.55 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2026-02-17 12:30:00 | 144.38 | 2026-02-18 10:15:00 | 142.55 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2026-02-17 13:00:00 | 144.00 | 2026-02-18 10:15:00 | 142.55 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2026-02-17 14:30:00 | 144.00 | 2026-02-18 10:15:00 | 142.55 | STOP_HIT | 1.00 | -1.01% |
