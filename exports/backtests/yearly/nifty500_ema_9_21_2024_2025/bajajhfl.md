# Bajaj Housing Finance Ltd. (BAJAJHFL)

## Backtest Summary

- **Window:** 2024-09-16 09:15:00 → 2026-05-08 15:15:00 (2839 bars)
- **Last close:** 87.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 94 |
| ALERT1 | 70 |
| ALERT2 | 69 |
| ALERT2_SKIP | 28 |
| ALERT3 | 164 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 83 |
| PARTIAL | 21 |
| TARGET_HIT | 2 |
| STOP_HIT | 86 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 109 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 58 / 51
- **Target hits / Stop hits / Partials:** 2 / 86 / 21
- **Avg / median % per leg:** 1.63% / 0.18%
- **Sum % (uncompounded):** 177.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 8 | 26.7% | 0 | 30 | 0 | 0.17% | 5.2% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.70% | -1.4% |
| BUY @ 3rd Alert (retest2) | 28 | 8 | 28.6% | 0 | 28 | 0 | 0.23% | 6.6% |
| SELL (all) | 79 | 50 | 63.3% | 2 | 56 | 21 | 2.18% | 172.0% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.02% | -3.0% |
| SELL @ 3rd Alert (retest2) | 76 | 50 | 65.8% | 2 | 53 | 21 | 2.30% | 175.0% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.89% | -4.4% |
| retest2 (combined) | 104 | 58 | 55.8% | 2 | 81 | 21 | 1.75% | 181.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 10:15:00 | 158.10 | 169.41 | 170.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 11:15:00 | 156.29 | 166.79 | 169.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 166.48 | 163.70 | 166.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 09:15:00 | 166.48 | 163.70 | 166.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 166.48 | 163.70 | 166.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:00:00 | 166.48 | 163.70 | 166.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 165.60 | 164.08 | 166.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 11:45:00 | 163.52 | 164.08 | 166.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 13:30:00 | 163.55 | 164.16 | 165.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 09:30:00 | 163.47 | 163.49 | 165.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 09:15:00 | 155.34 | 158.58 | 160.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 09:15:00 | 155.37 | 158.58 | 160.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 09:15:00 | 155.30 | 158.58 | 160.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-26 09:15:00 | 157.64 | 155.74 | 157.84 | SL hit (close>ema200) qty=0.50 sl=155.74 alert=retest2 |

### Cycle 2 — BUY (started 2024-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 15:15:00 | 149.63 | 147.33 | 147.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 157.16 | 149.30 | 147.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 09:15:00 | 153.29 | 153.59 | 151.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 09:15:00 | 153.29 | 153.59 | 151.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 153.29 | 153.59 | 151.37 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-10-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 14:15:00 | 150.75 | 151.13 | 151.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 09:15:00 | 144.30 | 149.74 | 150.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 09:15:00 | 141.00 | 140.69 | 143.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 09:15:00 | 141.00 | 140.69 | 143.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 141.00 | 140.69 | 143.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:30:00 | 142.95 | 140.69 | 143.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 138.68 | 140.16 | 141.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 13:00:00 | 138.05 | 139.22 | 140.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 15:15:00 | 138.00 | 138.86 | 140.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 09:45:00 | 138.16 | 139.00 | 139.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 11:30:00 | 138.15 | 138.62 | 139.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 137.59 | 137.59 | 138.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 10:15:00 | 136.76 | 137.59 | 138.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 11:45:00 | 136.95 | 137.49 | 138.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 10:15:00 | 136.99 | 136.81 | 137.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 12:00:00 | 136.60 | 136.84 | 137.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 136.79 | 136.83 | 137.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:30:00 | 137.69 | 136.83 | 137.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 15:15:00 | 136.95 | 136.82 | 137.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 09:15:00 | 133.87 | 136.82 | 137.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-24 14:15:00 | 131.25 | 133.76 | 135.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-24 14:15:00 | 131.24 | 133.76 | 135.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-24 15:15:00 | 131.15 | 133.30 | 134.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-24 15:15:00 | 131.10 | 133.30 | 134.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:15:00 | 129.92 | 132.39 | 134.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:15:00 | 130.10 | 132.39 | 134.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:15:00 | 130.14 | 132.39 | 134.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:15:00 | 129.77 | 132.39 | 134.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-25 11:15:00 | 132.20 | 131.89 | 133.81 | SL hit (close>ema200) qty=0.50 sl=131.89 alert=retest2 |

### Cycle 4 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 135.94 | 132.56 | 132.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 17:15:00 | 136.96 | 134.63 | 133.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 134.62 | 135.17 | 134.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 134.62 | 135.17 | 134.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 134.62 | 135.17 | 134.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 134.62 | 135.17 | 134.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 135.26 | 135.09 | 134.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 13:45:00 | 135.57 | 134.83 | 134.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 15:00:00 | 135.60 | 134.98 | 134.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 09:15:00 | 135.71 | 135.01 | 134.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 11:30:00 | 135.75 | 135.41 | 135.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 137.20 | 138.00 | 137.22 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-08 13:15:00 | 135.44 | 136.65 | 136.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 135.44 | 136.65 | 136.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 133.75 | 135.72 | 136.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 130.30 | 130.03 | 131.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-18 09:15:00 | 128.84 | 129.48 | 130.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 128.84 | 129.48 | 130.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 15:00:00 | 127.63 | 128.61 | 129.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 15:00:00 | 127.61 | 128.16 | 128.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 11:30:00 | 127.51 | 127.06 | 127.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 12:30:00 | 127.50 | 127.04 | 127.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 130.37 | 127.69 | 127.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 130.37 | 127.69 | 127.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 139.47 | 132.34 | 130.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 09:15:00 | 135.30 | 135.55 | 133.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-29 10:00:00 | 135.30 | 135.55 | 133.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 134.85 | 135.47 | 134.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 14:30:00 | 135.94 | 135.12 | 134.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 09:15:00 | 137.07 | 135.03 | 134.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 10:15:00 | 140.98 | 142.57 | 142.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 10:15:00 | 140.98 | 142.57 | 142.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 11:15:00 | 140.49 | 142.15 | 142.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 10:15:00 | 141.00 | 140.99 | 141.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-11 10:45:00 | 141.01 | 140.99 | 141.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 13:15:00 | 141.80 | 141.20 | 141.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 13:45:00 | 141.99 | 141.20 | 141.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 141.36 | 141.23 | 141.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 14:45:00 | 141.70 | 141.23 | 141.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 15:15:00 | 141.20 | 141.23 | 141.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 09:15:00 | 133.75 | 141.23 | 141.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 14:15:00 | 127.06 | 128.14 | 129.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-19 10:15:00 | 128.59 | 128.04 | 129.22 | SL hit (close>ema200) qty=0.50 sl=128.04 alert=retest2 |

### Cycle 8 — BUY (started 2024-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 11:15:00 | 127.29 | 126.56 | 126.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 09:15:00 | 129.51 | 127.38 | 126.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 13:15:00 | 127.50 | 127.87 | 127.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-30 14:00:00 | 127.50 | 127.87 | 127.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 127.21 | 127.74 | 127.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 127.21 | 127.74 | 127.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 128.45 | 127.88 | 127.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:15:00 | 127.40 | 127.88 | 127.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 126.88 | 127.68 | 127.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 10:00:00 | 126.88 | 127.68 | 127.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 127.32 | 127.61 | 127.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 09:15:00 | 128.45 | 127.41 | 127.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 12:45:00 | 127.66 | 127.69 | 127.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 09:15:00 | 127.00 | 127.37 | 127.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 09:15:00 | 127.00 | 127.37 | 127.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-02 11:15:00 | 126.48 | 127.11 | 127.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 09:15:00 | 126.82 | 126.74 | 127.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 09:15:00 | 126.82 | 126.74 | 127.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 126.82 | 126.74 | 127.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 09:30:00 | 127.63 | 126.74 | 127.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 12:15:00 | 126.45 | 126.67 | 126.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-03 14:15:00 | 126.29 | 126.61 | 126.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-03 15:00:00 | 126.10 | 126.51 | 126.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 119.98 | 121.05 | 122.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 119.79 | 121.05 | 122.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 12:15:00 | 113.66 | 116.61 | 118.84 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 10 — BUY (started 2025-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 13:15:00 | 111.87 | 109.32 | 109.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 09:15:00 | 115.09 | 111.04 | 109.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 114.61 | 115.02 | 113.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 13:45:00 | 114.55 | 115.02 | 113.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 117.42 | 117.00 | 115.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 116.77 | 117.00 | 115.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 115.63 | 116.72 | 115.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 115.63 | 116.72 | 115.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 116.91 | 116.76 | 115.80 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2025-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 13:15:00 | 114.81 | 115.49 | 115.50 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 09:15:00 | 115.90 | 115.57 | 115.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 117.47 | 115.98 | 115.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 14:15:00 | 115.08 | 116.30 | 116.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 14:15:00 | 115.08 | 116.30 | 116.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 14:15:00 | 115.08 | 116.30 | 116.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 15:00:00 | 115.08 | 116.30 | 116.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 15:15:00 | 115.29 | 116.10 | 115.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 09:15:00 | 115.46 | 116.10 | 115.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-06 11:15:00 | 115.50 | 115.90 | 115.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2025-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 11:15:00 | 115.50 | 115.90 | 115.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 12:15:00 | 115.29 | 115.78 | 115.86 | Break + close below crossover candle low |

### Cycle 14 — BUY (started 2025-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 14:15:00 | 117.44 | 115.99 | 115.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 15:15:00 | 121.24 | 117.04 | 116.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 09:15:00 | 116.88 | 118.85 | 118.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 09:15:00 | 116.88 | 118.85 | 118.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 116.88 | 118.85 | 118.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:00:00 | 116.88 | 118.85 | 118.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 116.85 | 118.45 | 117.96 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2025-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 13:15:00 | 116.23 | 117.43 | 117.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 112.27 | 116.19 | 116.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 112.70 | 112.17 | 113.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:00:00 | 112.70 | 112.17 | 113.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 113.36 | 112.41 | 113.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:45:00 | 113.29 | 112.41 | 113.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 113.95 | 112.86 | 113.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 113.95 | 112.86 | 113.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 113.50 | 112.99 | 113.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 11:30:00 | 113.33 | 113.03 | 113.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 107.66 | 109.82 | 111.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 109.44 | 109.14 | 110.24 | SL hit (close>ema200) qty=0.50 sl=109.14 alert=retest2 |

### Cycle 16 — BUY (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 10:15:00 | 111.58 | 109.58 | 109.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 11:15:00 | 113.60 | 110.38 | 109.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 15:15:00 | 113.19 | 113.46 | 112.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-24 09:15:00 | 112.57 | 113.46 | 112.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 113.27 | 113.43 | 112.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 10:30:00 | 114.16 | 113.54 | 112.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 14:00:00 | 113.58 | 113.73 | 112.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 14:30:00 | 113.75 | 113.69 | 112.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 09:15:00 | 114.40 | 113.60 | 113.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 114.52 | 113.78 | 113.15 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-27 11:15:00 | 111.29 | 112.97 | 113.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2025-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 11:15:00 | 111.29 | 112.97 | 113.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 108.59 | 111.39 | 112.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 13:15:00 | 110.62 | 108.26 | 109.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 13:15:00 | 110.62 | 108.26 | 109.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 110.62 | 108.26 | 109.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 14:00:00 | 110.62 | 108.26 | 109.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 110.25 | 108.66 | 109.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 14:45:00 | 110.17 | 108.66 | 109.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 110.10 | 109.41 | 109.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 11:30:00 | 109.89 | 109.37 | 109.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-04 14:15:00 | 110.85 | 109.72 | 109.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2025-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 14:15:00 | 110.85 | 109.72 | 109.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 09:15:00 | 112.41 | 110.42 | 110.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 11:15:00 | 116.00 | 116.03 | 114.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 12:00:00 | 116.00 | 116.03 | 114.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 115.49 | 115.83 | 115.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 15:00:00 | 115.49 | 115.83 | 115.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 115.01 | 115.66 | 115.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 113.99 | 115.66 | 115.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 113.75 | 115.28 | 114.94 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2025-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 12:15:00 | 113.56 | 114.57 | 114.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 15:15:00 | 113.30 | 114.17 | 114.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 11:15:00 | 114.15 | 114.10 | 114.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 12:00:00 | 114.15 | 114.10 | 114.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 115.72 | 114.42 | 114.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 13:00:00 | 115.72 | 114.42 | 114.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — BUY (started 2025-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 13:15:00 | 115.59 | 114.66 | 114.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 14:15:00 | 116.10 | 114.95 | 114.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 12:15:00 | 115.40 | 115.57 | 115.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-13 13:00:00 | 115.40 | 115.57 | 115.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 115.35 | 115.53 | 115.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 15:00:00 | 115.35 | 115.53 | 115.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 115.10 | 115.45 | 115.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:15:00 | 116.42 | 115.45 | 115.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 115.33 | 115.42 | 115.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 11:15:00 | 117.21 | 115.52 | 115.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 12:45:00 | 117.40 | 116.39 | 115.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 13:15:00 | 120.54 | 121.68 | 121.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2025-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 13:15:00 | 120.54 | 121.68 | 121.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 15:15:00 | 118.90 | 120.90 | 121.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 12:15:00 | 118.21 | 117.58 | 118.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 13:00:00 | 118.21 | 117.58 | 118.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 22 — BUY (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 14:15:00 | 126.70 | 119.48 | 119.38 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 121.03 | 122.25 | 122.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 120.66 | 121.93 | 122.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 118.76 | 118.24 | 119.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 120.47 | 118.24 | 119.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 119.99 | 118.59 | 119.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 118.82 | 118.71 | 119.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 13:15:00 | 119.20 | 118.47 | 118.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 09:30:00 | 119.18 | 118.48 | 118.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 15:15:00 | 119.60 | 119.01 | 118.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2025-04-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 15:15:00 | 119.60 | 119.01 | 118.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 124.20 | 120.05 | 119.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 15:15:00 | 128.48 | 128.94 | 127.22 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:30:00 | 129.34 | 128.97 | 127.39 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 10:15:00 | 130.70 | 128.97 | 127.39 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 129.11 | 130.75 | 130.01 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 129.11 | 130.75 | 130.01 | SL hit (close<ema400) qty=1.00 sl=130.01 alert=retest1 |

### Cycle 25 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 125.28 | 131.09 | 131.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 13:15:00 | 122.71 | 123.68 | 124.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 123.66 | 123.16 | 124.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-02 10:15:00 | 123.80 | 123.16 | 124.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 122.68 | 122.47 | 122.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 09:15:00 | 121.86 | 122.46 | 122.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 115.77 | 119.22 | 120.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-07 13:15:00 | 118.85 | 118.80 | 120.06 | SL hit (close>ema200) qty=0.50 sl=118.80 alert=retest2 |

### Cycle 26 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 120.86 | 119.08 | 118.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 121.19 | 120.15 | 119.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 09:15:00 | 122.86 | 122.92 | 122.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 10:00:00 | 122.86 | 122.92 | 122.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 124.20 | 125.09 | 124.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 13:00:00 | 124.20 | 125.09 | 124.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 123.74 | 124.82 | 124.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 123.74 | 124.82 | 124.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 124.14 | 124.52 | 124.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 124.14 | 124.52 | 124.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 123.05 | 124.22 | 124.33 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 13:15:00 | 124.22 | 124.13 | 124.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 15:15:00 | 124.68 | 124.29 | 124.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 10:15:00 | 124.36 | 124.38 | 124.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 10:15:00 | 124.36 | 124.38 | 124.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 124.36 | 124.38 | 124.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 10:30:00 | 124.38 | 124.38 | 124.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 124.20 | 124.35 | 124.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:00:00 | 124.20 | 124.35 | 124.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 124.30 | 124.34 | 124.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 13:30:00 | 124.40 | 124.34 | 124.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 14:30:00 | 124.40 | 124.32 | 124.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 09:15:00 | 123.74 | 124.19 | 124.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 09:15:00 | 123.74 | 124.19 | 124.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 13:15:00 | 123.10 | 123.48 | 123.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 122.98 | 122.52 | 122.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 122.98 | 122.52 | 122.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 122.98 | 122.52 | 122.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:00:00 | 122.98 | 122.52 | 122.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 122.40 | 122.50 | 122.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 11:15:00 | 122.16 | 122.50 | 122.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 13:15:00 | 123.27 | 122.51 | 122.78 | SL hit (close>static) qty=1.00 sl=122.99 alert=retest2 |

### Cycle 30 — BUY (started 2025-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 15:15:00 | 124.05 | 123.01 | 122.97 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 122.57 | 122.96 | 122.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 09:15:00 | 121.57 | 122.58 | 122.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 09:15:00 | 122.40 | 122.05 | 122.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 09:15:00 | 122.40 | 122.05 | 122.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 122.40 | 122.05 | 122.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:15:00 | 122.88 | 122.05 | 122.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 123.45 | 122.33 | 122.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:45:00 | 123.19 | 122.33 | 122.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 123.20 | 122.50 | 122.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 12:15:00 | 123.05 | 122.50 | 122.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 12:15:00 | 122.87 | 122.58 | 122.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 12:15:00 | 122.87 | 122.58 | 122.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 09:15:00 | 124.08 | 122.95 | 122.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 09:15:00 | 126.70 | 126.79 | 125.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 09:30:00 | 126.50 | 126.79 | 125.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 125.95 | 126.33 | 125.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 13:30:00 | 125.94 | 126.33 | 125.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 125.85 | 126.23 | 125.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:15:00 | 125.70 | 126.23 | 125.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 125.70 | 126.12 | 125.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 126.57 | 126.12 | 125.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 13:15:00 | 124.85 | 126.06 | 125.92 | SL hit (close<static) qty=1.00 sl=125.60 alert=retest2 |

### Cycle 33 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 125.28 | 125.77 | 125.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 10:15:00 | 124.61 | 125.44 | 125.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 121.90 | 121.65 | 122.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 121.90 | 121.65 | 122.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 122.00 | 121.77 | 122.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 122.22 | 121.77 | 122.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 119.73 | 119.00 | 119.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 119.73 | 119.00 | 119.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 119.47 | 119.10 | 119.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 118.39 | 119.10 | 119.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 120.72 | 119.37 | 119.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 120.72 | 119.37 | 119.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 11:15:00 | 121.38 | 120.06 | 119.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 10:15:00 | 121.40 | 121.50 | 120.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 11:00:00 | 121.40 | 121.50 | 120.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 121.97 | 122.10 | 121.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 12:00:00 | 121.97 | 122.10 | 121.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 121.86 | 122.05 | 121.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:00:00 | 121.86 | 122.05 | 121.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 121.91 | 122.02 | 121.86 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 121.32 | 121.79 | 121.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 121.05 | 121.50 | 121.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 14:15:00 | 121.52 | 121.33 | 121.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 14:15:00 | 121.52 | 121.33 | 121.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 121.52 | 121.33 | 121.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 15:00:00 | 121.52 | 121.33 | 121.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 121.24 | 121.31 | 121.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:15:00 | 121.02 | 121.31 | 121.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 121.45 | 121.34 | 121.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:45:00 | 121.35 | 121.34 | 121.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 121.50 | 121.37 | 121.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:30:00 | 121.50 | 121.37 | 121.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 121.22 | 121.34 | 121.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 13:00:00 | 121.10 | 121.29 | 121.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 14:15:00 | 121.05 | 121.27 | 121.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 15:15:00 | 121.10 | 121.25 | 121.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 09:15:00 | 122.99 | 121.57 | 121.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 09:15:00 | 122.99 | 121.57 | 121.49 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 11:15:00 | 120.99 | 121.58 | 121.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 120.43 | 121.03 | 121.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 120.92 | 120.79 | 121.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 15:00:00 | 120.92 | 120.79 | 121.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 121.52 | 120.94 | 121.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:00:00 | 121.52 | 120.94 | 121.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 121.76 | 121.10 | 121.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:30:00 | 121.90 | 121.10 | 121.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 121.13 | 121.12 | 121.16 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 09:15:00 | 121.62 | 121.26 | 121.22 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 13:15:00 | 121.07 | 121.21 | 121.21 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2025-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 14:15:00 | 121.80 | 121.33 | 121.27 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 120.78 | 121.23 | 121.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 09:15:00 | 120.10 | 120.77 | 120.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 120.90 | 120.30 | 120.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 120.90 | 120.30 | 120.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 120.90 | 120.30 | 120.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:30:00 | 121.10 | 120.30 | 120.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 121.40 | 120.52 | 120.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 121.40 | 120.52 | 120.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 121.40 | 120.84 | 120.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 10:15:00 | 122.54 | 121.34 | 121.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 11:15:00 | 122.60 | 122.61 | 122.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 12:00:00 | 122.60 | 122.61 | 122.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 122.09 | 122.57 | 122.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:30:00 | 122.03 | 122.57 | 122.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 121.81 | 122.41 | 122.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 121.88 | 122.41 | 122.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 15:15:00 | 121.81 | 122.04 | 122.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 09:15:00 | 121.36 | 121.90 | 122.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 09:15:00 | 121.72 | 121.62 | 121.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 121.72 | 121.62 | 121.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 121.72 | 121.62 | 121.78 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 09:15:00 | 123.00 | 121.89 | 121.83 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 09:15:00 | 120.86 | 121.89 | 121.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 12:15:00 | 120.00 | 121.21 | 121.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 14:15:00 | 115.82 | 115.69 | 116.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 14:30:00 | 115.98 | 115.69 | 116.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 115.86 | 115.82 | 116.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:30:00 | 116.63 | 115.82 | 116.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 113.80 | 113.44 | 114.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:30:00 | 113.85 | 113.44 | 114.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 113.97 | 113.73 | 114.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 113.97 | 113.73 | 114.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 114.47 | 113.87 | 114.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 114.49 | 113.87 | 114.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 114.27 | 113.95 | 114.07 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2025-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 14:15:00 | 114.24 | 114.11 | 114.11 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 113.29 | 113.96 | 114.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 11:15:00 | 112.96 | 113.34 | 113.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 112.97 | 112.89 | 113.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 112.97 | 112.89 | 113.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 113.45 | 113.00 | 113.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:45:00 | 112.51 | 112.94 | 113.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 13:00:00 | 112.75 | 112.83 | 113.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 13:45:00 | 112.75 | 112.78 | 113.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 15:00:00 | 112.67 | 112.47 | 112.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 112.89 | 112.56 | 112.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:15:00 | 113.40 | 112.56 | 112.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 113.21 | 112.69 | 112.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 10:45:00 | 112.49 | 112.64 | 112.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 12:15:00 | 113.15 | 112.80 | 112.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 12:15:00 | 113.15 | 112.80 | 112.79 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 14:15:00 | 112.37 | 112.71 | 112.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 11:15:00 | 112.25 | 112.60 | 112.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 09:15:00 | 113.54 | 111.35 | 111.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 113.54 | 111.35 | 111.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 113.54 | 111.35 | 111.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:30:00 | 114.01 | 111.35 | 111.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 113.40 | 111.76 | 112.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 11:15:00 | 112.77 | 111.76 | 112.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 13:15:00 | 113.00 | 112.30 | 112.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 13:15:00 | 113.00 | 112.30 | 112.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 117.15 | 113.38 | 112.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 15:15:00 | 114.75 | 114.76 | 113.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 09:15:00 | 113.64 | 114.76 | 113.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 113.75 | 114.56 | 113.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:30:00 | 113.40 | 114.56 | 113.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 114.27 | 114.50 | 113.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 10:30:00 | 114.50 | 114.24 | 114.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 11:45:00 | 114.36 | 114.24 | 114.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 12:15:00 | 114.46 | 114.24 | 114.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 13:30:00 | 114.40 | 114.27 | 114.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 114.35 | 114.28 | 114.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:15:00 | 113.85 | 114.28 | 114.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 113.80 | 114.19 | 114.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 113.69 | 114.19 | 114.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 113.71 | 114.09 | 114.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:00:00 | 113.71 | 114.09 | 114.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-21 11:15:00 | 113.52 | 113.98 | 114.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 11:15:00 | 113.52 | 113.98 | 114.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 113.07 | 113.79 | 113.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 113.21 | 113.13 | 113.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 113.21 | 113.13 | 113.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 113.21 | 113.13 | 113.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:15:00 | 113.35 | 113.13 | 113.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 113.47 | 113.20 | 113.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:30:00 | 113.65 | 113.20 | 113.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 113.49 | 113.26 | 113.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:15:00 | 113.54 | 113.26 | 113.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 113.45 | 113.30 | 113.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 14:15:00 | 113.30 | 113.31 | 113.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 15:15:00 | 113.28 | 113.35 | 113.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 15:15:00 | 113.10 | 112.22 | 112.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 113.10 | 112.22 | 112.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 113.83 | 112.54 | 112.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 113.10 | 113.24 | 112.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 10:45:00 | 113.17 | 113.24 | 112.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 112.96 | 113.19 | 112.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 12:00:00 | 112.96 | 113.19 | 112.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 113.10 | 113.17 | 112.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 12:45:00 | 112.94 | 113.17 | 112.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 113.45 | 113.24 | 113.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:15:00 | 114.27 | 113.24 | 113.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 11:15:00 | 112.85 | 113.17 | 113.04 | SL hit (close<static) qty=1.00 sl=113.01 alert=retest2 |

### Cycle 53 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 111.80 | 112.87 | 112.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 111.35 | 112.27 | 112.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 112.14 | 112.11 | 112.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-05 14:00:00 | 112.14 | 112.11 | 112.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 112.21 | 112.13 | 112.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:30:00 | 112.23 | 112.13 | 112.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 113.00 | 112.33 | 112.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 15:00:00 | 112.19 | 112.46 | 112.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 112.10 | 112.45 | 112.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 10:00:00 | 112.05 | 112.37 | 112.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 13:15:00 | 112.20 | 112.30 | 112.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 112.52 | 112.28 | 112.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:30:00 | 112.91 | 112.28 | 112.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-11 10:15:00 | 112.39 | 112.30 | 112.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 10:15:00 | 112.39 | 112.30 | 112.30 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 11:15:00 | 112.18 | 112.28 | 112.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 14:15:00 | 111.95 | 112.16 | 112.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 09:15:00 | 112.17 | 112.15 | 112.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 112.17 | 112.15 | 112.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 112.17 | 112.15 | 112.21 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 113.82 | 112.45 | 112.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 10:15:00 | 115.89 | 113.84 | 113.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 13:15:00 | 115.49 | 115.62 | 114.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 14:00:00 | 115.49 | 115.62 | 114.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 114.92 | 115.45 | 114.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:30:00 | 114.95 | 115.45 | 114.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 115.05 | 115.37 | 114.96 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 10:15:00 | 114.38 | 114.79 | 114.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 11:15:00 | 114.23 | 114.68 | 114.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 14:15:00 | 114.59 | 114.56 | 114.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 14:15:00 | 114.59 | 114.56 | 114.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 114.59 | 114.56 | 114.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 114.59 | 114.56 | 114.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 114.75 | 114.59 | 114.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 09:15:00 | 114.34 | 114.59 | 114.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 09:45:00 | 114.32 | 114.55 | 114.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 111.55 | 110.99 | 110.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 111.55 | 110.99 | 110.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 10:15:00 | 111.85 | 111.26 | 111.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 13:15:00 | 111.40 | 111.43 | 111.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 13:30:00 | 111.36 | 111.43 | 111.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 111.27 | 111.52 | 111.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 14:00:00 | 111.27 | 111.52 | 111.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 111.83 | 111.58 | 111.42 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2025-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 13:15:00 | 111.21 | 111.38 | 111.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 15:15:00 | 111.04 | 111.26 | 111.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 108.25 | 107.90 | 108.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:00:00 | 108.25 | 107.90 | 108.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 108.22 | 108.07 | 108.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:30:00 | 108.29 | 108.07 | 108.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 109.89 | 108.43 | 108.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 109.89 | 108.43 | 108.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 15:15:00 | 111.20 | 108.98 | 108.81 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2025-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 13:15:00 | 109.04 | 109.33 | 109.33 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 110.72 | 109.54 | 109.42 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2025-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 14:15:00 | 109.84 | 109.98 | 110.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 09:15:00 | 109.53 | 109.87 | 109.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 109.46 | 109.33 | 109.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-29 10:00:00 | 109.46 | 109.33 | 109.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 109.97 | 109.46 | 109.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 109.97 | 109.46 | 109.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 110.30 | 109.63 | 109.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:30:00 | 110.28 | 109.63 | 109.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 110.30 | 109.76 | 109.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 13:15:00 | 110.45 | 109.90 | 109.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 13:15:00 | 110.50 | 110.53 | 110.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 14:00:00 | 110.50 | 110.53 | 110.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 110.16 | 110.52 | 110.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 110.16 | 110.52 | 110.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 109.97 | 110.41 | 110.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 109.97 | 110.41 | 110.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 110.26 | 110.35 | 110.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 14:30:00 | 110.55 | 110.30 | 110.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 109.80 | 110.19 | 110.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 109.80 | 110.19 | 110.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 10:15:00 | 109.55 | 109.85 | 110.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 15:15:00 | 109.60 | 109.59 | 109.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 09:15:00 | 109.29 | 109.59 | 109.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 108.70 | 109.41 | 109.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 108.50 | 109.36 | 109.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 10:15:00 | 106.20 | 105.56 | 105.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — BUY (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 10:15:00 | 106.20 | 105.56 | 105.52 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 09:15:00 | 105.25 | 105.48 | 105.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 09:15:00 | 104.66 | 105.11 | 105.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 93.98 | 93.96 | 95.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 11:00:00 | 93.98 | 93.96 | 95.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 95.70 | 94.31 | 95.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 95.70 | 94.31 | 95.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 95.21 | 94.49 | 95.14 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 96.26 | 95.51 | 95.44 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 12:15:00 | 95.67 | 95.76 | 95.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 09:15:00 | 94.61 | 95.50 | 95.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 15:15:00 | 96.79 | 95.24 | 95.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 15:15:00 | 96.79 | 95.24 | 95.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 96.79 | 95.24 | 95.37 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 13:15:00 | 95.76 | 95.45 | 95.44 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 09:15:00 | 94.76 | 95.31 | 95.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 15:15:00 | 94.54 | 94.84 | 95.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 94.97 | 94.70 | 94.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 94.97 | 94.70 | 94.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 94.97 | 94.70 | 94.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:45:00 | 94.98 | 94.70 | 94.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 94.92 | 94.74 | 94.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:45:00 | 94.94 | 94.74 | 94.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 94.60 | 94.72 | 94.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 14:15:00 | 94.48 | 94.72 | 94.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 14:15:00 | 95.17 | 94.81 | 94.93 | SL hit (close>static) qty=1.00 sl=95.00 alert=retest2 |

### Cycle 72 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 95.93 | 95.09 | 95.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 96.08 | 95.39 | 95.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 12:15:00 | 95.91 | 95.93 | 95.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 12:30:00 | 95.89 | 95.93 | 95.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 95.75 | 95.88 | 95.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:15:00 | 95.67 | 95.88 | 95.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 95.59 | 95.82 | 95.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:30:00 | 95.56 | 95.82 | 95.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 95.25 | 95.71 | 95.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:00:00 | 95.25 | 95.71 | 95.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2025-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 11:15:00 | 95.32 | 95.63 | 95.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 95.19 | 95.46 | 95.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 14:15:00 | 94.25 | 94.14 | 94.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 15:00:00 | 94.25 | 94.14 | 94.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 94.40 | 94.14 | 94.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 94.40 | 94.14 | 94.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 94.38 | 94.19 | 94.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 94.73 | 94.19 | 94.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 94.87 | 94.33 | 94.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:30:00 | 95.15 | 94.33 | 94.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 10:15:00 | 96.34 | 94.73 | 94.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 11:15:00 | 97.30 | 96.69 | 96.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 97.02 | 97.10 | 96.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 09:30:00 | 96.91 | 97.10 | 96.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 96.99 | 97.02 | 96.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:30:00 | 96.97 | 97.02 | 96.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 96.19 | 96.83 | 96.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:45:00 | 96.10 | 96.83 | 96.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 96.01 | 96.67 | 96.61 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 11:15:00 | 95.89 | 96.51 | 96.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 12:15:00 | 95.78 | 96.37 | 96.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 15:15:00 | 94.00 | 93.91 | 94.63 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-12 09:15:00 | 93.01 | 93.91 | 94.63 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-12 10:45:00 | 93.38 | 93.71 | 94.41 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-12 15:00:00 | 93.43 | 93.59 | 94.13 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 93.52 | 93.57 | 94.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:15:00 | 94.15 | 93.57 | 94.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 94.00 | 93.66 | 94.02 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-13 14:15:00 | 94.22 | 93.88 | 94.02 | SL hit (close>ema400) qty=1.00 sl=94.02 alert=retest1 |

### Cycle 76 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 91.70 | 89.10 | 88.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 93.05 | 90.90 | 90.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 90.67 | 91.01 | 90.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 12:00:00 | 90.67 | 91.01 | 90.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 89.97 | 90.80 | 90.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 90.30 | 90.80 | 90.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 89.92 | 90.62 | 90.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:45:00 | 89.83 | 90.62 | 90.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 88.30 | 89.70 | 89.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 11:15:00 | 87.25 | 89.21 | 89.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 90.07 | 89.15 | 89.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 90.07 | 89.15 | 89.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 90.07 | 89.15 | 89.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 90.07 | 89.15 | 89.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 89.80 | 89.28 | 89.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 91.20 | 89.28 | 89.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 91.52 | 89.73 | 89.68 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 10:15:00 | 90.45 | 91.04 | 91.07 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 91.34 | 91.07 | 91.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 11:15:00 | 92.30 | 91.55 | 91.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 91.15 | 91.63 | 91.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 91.15 | 91.63 | 91.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 91.15 | 91.63 | 91.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 91.09 | 91.63 | 91.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 91.05 | 91.51 | 91.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 91.05 | 91.51 | 91.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 12:15:00 | 91.01 | 91.35 | 91.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 90.35 | 91.05 | 91.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 09:15:00 | 88.85 | 88.83 | 89.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 88.85 | 88.83 | 89.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 88.85 | 88.83 | 89.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:45:00 | 88.65 | 88.83 | 88.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 84.22 | 86.63 | 87.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 14:15:00 | 85.40 | 85.16 | 86.03 | SL hit (close>ema200) qty=0.50 sl=85.16 alert=retest2 |

### Cycle 82 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 84.56 | 83.96 | 83.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 84.78 | 84.12 | 83.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 84.07 | 84.19 | 84.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 10:15:00 | 84.07 | 84.19 | 84.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 84.07 | 84.19 | 84.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 10:45:00 | 84.22 | 84.19 | 84.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 84.16 | 84.18 | 84.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:45:00 | 83.90 | 84.18 | 84.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 83.96 | 84.14 | 84.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 83.96 | 84.14 | 84.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 83.94 | 84.10 | 84.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:45:00 | 83.94 | 84.10 | 84.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 83.65 | 84.01 | 84.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 83.65 | 84.01 | 84.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 15:15:00 | 83.66 | 83.94 | 83.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 82.28 | 83.61 | 83.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 83.85 | 83.57 | 83.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 11:15:00 | 83.85 | 83.57 | 83.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 83.85 | 83.57 | 83.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:00:00 | 83.85 | 83.57 | 83.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 83.65 | 83.58 | 83.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:45:00 | 83.15 | 83.54 | 83.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 10:00:00 | 82.80 | 81.64 | 81.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 11:15:00 | 82.50 | 81.98 | 81.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 82.50 | 81.98 | 81.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 83.09 | 82.32 | 82.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 82.09 | 82.51 | 82.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 82.09 | 82.51 | 82.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 82.09 | 82.51 | 82.26 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 81.40 | 82.06 | 82.11 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 82.91 | 82.16 | 82.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 11:15:00 | 83.12 | 82.36 | 82.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-20 13:15:00 | 81.93 | 82.29 | 82.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 13:15:00 | 81.93 | 82.29 | 82.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 81.93 | 82.29 | 82.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 14:00:00 | 81.93 | 82.29 | 82.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 14:15:00 | 81.40 | 82.11 | 82.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 78.96 | 81.37 | 81.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 14:15:00 | 78.75 | 78.44 | 79.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 15:00:00 | 78.75 | 78.44 | 79.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 81.14 | 79.04 | 79.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 80.90 | 79.04 | 79.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 13:15:00 | 79.76 | 79.51 | 79.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 13:30:00 | 79.60 | 79.51 | 79.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 14:15:00 | 79.14 | 79.43 | 79.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 09:15:00 | 77.89 | 79.35 | 79.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 74.00 | 76.42 | 77.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 76.40 | 74.63 | 75.88 | SL hit (close>ema200) qty=0.50 sl=74.63 alert=retest2 |

### Cycle 88 — BUY (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 15:15:00 | 77.31 | 76.46 | 76.40 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 75.96 | 76.36 | 76.36 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 11:15:00 | 76.59 | 76.36 | 76.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 12:15:00 | 76.90 | 76.47 | 76.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 83.85 | 83.88 | 82.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:45:00 | 83.89 | 83.88 | 82.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 84.84 | 85.45 | 84.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 85.00 | 85.45 | 84.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 85.05 | 85.53 | 84.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 15:00:00 | 85.03 | 85.23 | 84.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 15:15:00 | 90.04 | 90.59 | 90.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 90.04 | 90.59 | 90.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 89.92 | 90.46 | 90.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 15:15:00 | 89.62 | 89.46 | 89.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:15:00 | 90.82 | 89.46 | 89.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 90.85 | 89.74 | 90.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 91.10 | 89.74 | 90.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 91.27 | 90.32 | 90.24 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 11:15:00 | 89.50 | 90.40 | 90.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 09:15:00 | 88.83 | 89.65 | 90.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 09:15:00 | 87.20 | 87.12 | 87.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-05 09:45:00 | 87.44 | 87.12 | 87.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 87.00 | 86.73 | 87.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:15:00 | 86.67 | 86.83 | 87.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 87.76 | 87.04 | 87.09 | SL hit (close>static) qty=1.00 sl=87.55 alert=retest2 |

### Cycle 94 — BUY (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 10:15:00 | 88.34 | 87.30 | 87.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 13:15:00 | 88.96 | 87.99 | 87.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 14:15:00 | 87.90 | 87.97 | 87.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 15:00:00 | 87.90 | 87.97 | 87.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 87.49 | 87.85 | 87.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 87.38 | 87.85 | 87.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 87.40 | 87.76 | 87.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:00:00 | 87.40 | 87.76 | 87.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 87.60 | 87.56 | 87.54 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-09-20 11:45:00 | 163.52 | 2024-09-25 09:15:00 | 155.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-20 13:30:00 | 163.55 | 2024-09-25 09:15:00 | 155.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-23 09:30:00 | 163.47 | 2024-09-25 09:15:00 | 155.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-20 11:45:00 | 163.52 | 2024-09-26 09:15:00 | 157.64 | STOP_HIT | 0.50 | 3.60% |
| SELL | retest2 | 2024-09-20 13:30:00 | 163.55 | 2024-09-26 09:15:00 | 157.64 | STOP_HIT | 0.50 | 3.61% |
| SELL | retest2 | 2024-09-23 09:30:00 | 163.47 | 2024-09-26 09:15:00 | 157.64 | STOP_HIT | 0.50 | 3.57% |
| SELL | retest2 | 2024-10-17 13:00:00 | 138.05 | 2024-10-24 14:15:00 | 131.25 | PARTIAL | 0.50 | 4.92% |
| SELL | retest2 | 2024-10-17 15:15:00 | 138.00 | 2024-10-24 14:15:00 | 131.24 | PARTIAL | 0.50 | 4.90% |
| SELL | retest2 | 2024-10-21 09:45:00 | 138.16 | 2024-10-24 15:15:00 | 131.15 | PARTIAL | 0.50 | 5.08% |
| SELL | retest2 | 2024-10-21 11:30:00 | 138.15 | 2024-10-24 15:15:00 | 131.10 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2024-10-22 10:15:00 | 136.76 | 2024-10-25 09:15:00 | 129.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 11:45:00 | 136.95 | 2024-10-25 09:15:00 | 130.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-23 10:15:00 | 136.99 | 2024-10-25 09:15:00 | 130.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-23 12:00:00 | 136.60 | 2024-10-25 09:15:00 | 129.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 13:00:00 | 138.05 | 2024-10-25 11:15:00 | 132.20 | STOP_HIT | 0.50 | 4.24% |
| SELL | retest2 | 2024-10-17 15:15:00 | 138.00 | 2024-10-25 11:15:00 | 132.20 | STOP_HIT | 0.50 | 4.20% |
| SELL | retest2 | 2024-10-21 09:45:00 | 138.16 | 2024-10-25 11:15:00 | 132.20 | STOP_HIT | 0.50 | 4.31% |
| SELL | retest2 | 2024-10-21 11:30:00 | 138.15 | 2024-10-25 11:15:00 | 132.20 | STOP_HIT | 0.50 | 4.31% |
| SELL | retest2 | 2024-10-22 10:15:00 | 136.76 | 2024-10-25 11:15:00 | 132.20 | STOP_HIT | 0.50 | 3.33% |
| SELL | retest2 | 2024-10-22 11:45:00 | 136.95 | 2024-10-25 11:15:00 | 132.20 | STOP_HIT | 0.50 | 3.47% |
| SELL | retest2 | 2024-10-23 10:15:00 | 136.99 | 2024-10-25 11:15:00 | 132.20 | STOP_HIT | 0.50 | 3.50% |
| SELL | retest2 | 2024-10-23 12:00:00 | 136.60 | 2024-10-25 11:15:00 | 132.20 | STOP_HIT | 0.50 | 3.22% |
| SELL | retest2 | 2024-10-24 09:15:00 | 133.87 | 2024-10-30 10:15:00 | 135.94 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-11-05 13:45:00 | 135.57 | 2024-11-08 13:15:00 | 135.44 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2024-11-05 15:00:00 | 135.60 | 2024-11-08 13:15:00 | 135.44 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2024-11-06 09:15:00 | 135.71 | 2024-11-08 13:15:00 | 135.44 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2024-11-06 11:30:00 | 135.75 | 2024-11-08 13:15:00 | 135.44 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2024-11-18 15:00:00 | 127.63 | 2024-11-25 09:15:00 | 130.37 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2024-11-19 15:00:00 | 127.61 | 2024-11-25 09:15:00 | 130.37 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2024-11-22 11:30:00 | 127.51 | 2024-11-25 09:15:00 | 130.37 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2024-11-22 12:30:00 | 127.50 | 2024-11-25 09:15:00 | 130.37 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2024-12-02 14:30:00 | 135.94 | 2024-12-10 10:15:00 | 140.98 | STOP_HIT | 1.00 | 3.71% |
| BUY | retest2 | 2024-12-03 09:15:00 | 137.07 | 2024-12-10 10:15:00 | 140.98 | STOP_HIT | 1.00 | 2.85% |
| SELL | retest2 | 2024-12-12 09:15:00 | 133.75 | 2024-12-18 14:15:00 | 127.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 09:15:00 | 133.75 | 2024-12-19 10:15:00 | 128.59 | STOP_HIT | 0.50 | 3.86% |
| BUY | retest2 | 2025-01-01 09:15:00 | 128.45 | 2025-01-02 09:15:00 | 127.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-01-01 12:45:00 | 127.66 | 2025-01-02 09:15:00 | 127.00 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-01-03 14:15:00 | 126.29 | 2025-01-10 09:15:00 | 119.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-03 15:00:00 | 126.10 | 2025-01-10 09:15:00 | 119.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-03 14:15:00 | 126.29 | 2025-01-13 12:15:00 | 113.66 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-03 15:00:00 | 126.10 | 2025-01-13 13:15:00 | 113.49 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-06 09:15:00 | 115.46 | 2025-02-06 11:15:00 | 115.50 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2025-02-13 11:30:00 | 113.33 | 2025-02-17 09:15:00 | 107.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 11:30:00 | 113.33 | 2025-02-17 14:15:00 | 109.44 | STOP_HIT | 0.50 | 3.43% |
| BUY | retest2 | 2025-02-24 10:30:00 | 114.16 | 2025-02-27 11:15:00 | 111.29 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2025-02-24 14:00:00 | 113.58 | 2025-02-27 11:15:00 | 111.29 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-02-24 14:30:00 | 113.75 | 2025-02-27 11:15:00 | 111.29 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-02-25 09:15:00 | 114.40 | 2025-02-27 11:15:00 | 111.29 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2025-03-04 11:30:00 | 109.89 | 2025-03-04 14:15:00 | 110.85 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-03-18 11:15:00 | 117.21 | 2025-03-25 13:15:00 | 120.54 | STOP_HIT | 1.00 | 2.84% |
| BUY | retest2 | 2025-03-18 12:45:00 | 117.40 | 2025-03-25 13:15:00 | 120.54 | STOP_HIT | 1.00 | 2.67% |
| SELL | retest2 | 2025-04-08 10:30:00 | 118.82 | 2025-04-11 15:15:00 | 119.60 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-04-09 13:15:00 | 119.20 | 2025-04-11 15:15:00 | 119.60 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-04-11 09:30:00 | 119.18 | 2025-04-11 15:15:00 | 119.60 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-04-21 09:30:00 | 129.34 | 2025-04-23 09:15:00 | 129.11 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-04-21 10:15:00 | 130.70 | 2025-04-23 09:15:00 | 129.11 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-04-23 12:30:00 | 130.50 | 2025-04-25 09:15:00 | 125.28 | STOP_HIT | 1.00 | -4.00% |
| SELL | retest2 | 2025-05-06 09:15:00 | 121.86 | 2025-05-07 09:15:00 | 115.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 09:15:00 | 121.86 | 2025-05-07 13:15:00 | 118.85 | STOP_HIT | 0.50 | 2.47% |
| SELL | retest2 | 2025-05-08 09:45:00 | 122.08 | 2025-05-09 09:15:00 | 115.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 10:15:00 | 121.63 | 2025-05-09 09:15:00 | 115.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 09:45:00 | 122.08 | 2025-05-09 15:15:00 | 117.60 | STOP_HIT | 0.50 | 3.67% |
| SELL | retest2 | 2025-05-08 10:15:00 | 121.63 | 2025-05-09 15:15:00 | 117.60 | STOP_HIT | 0.50 | 3.31% |
| BUY | retest2 | 2025-05-26 13:30:00 | 124.40 | 2025-05-27 09:15:00 | 123.74 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-05-26 14:30:00 | 124.40 | 2025-05-27 09:15:00 | 123.74 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-06-02 11:15:00 | 122.16 | 2025-06-02 13:15:00 | 123.27 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-06-05 12:15:00 | 123.05 | 2025-06-05 12:15:00 | 122.87 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2025-06-11 09:15:00 | 126.57 | 2025-06-11 13:15:00 | 124.85 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-06-23 09:15:00 | 118.39 | 2025-06-24 09:15:00 | 120.72 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-07-03 13:00:00 | 121.10 | 2025-07-04 09:15:00 | 122.99 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-07-03 14:15:00 | 121.05 | 2025-07-04 09:15:00 | 122.99 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-07-03 15:15:00 | 121.10 | 2025-07-04 09:15:00 | 122.99 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-08-08 09:45:00 | 112.51 | 2025-08-12 12:15:00 | 113.15 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-08-08 13:00:00 | 112.75 | 2025-08-12 12:15:00 | 113.15 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-08-08 13:45:00 | 112.75 | 2025-08-12 12:15:00 | 113.15 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-08-11 15:00:00 | 112.67 | 2025-08-12 12:15:00 | 113.15 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-08-12 10:45:00 | 112.49 | 2025-08-12 12:15:00 | 113.15 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-08-14 11:15:00 | 112.77 | 2025-08-14 13:15:00 | 113.00 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2025-08-20 10:30:00 | 114.50 | 2025-08-21 11:15:00 | 113.52 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-08-20 11:45:00 | 114.36 | 2025-08-21 11:15:00 | 113.52 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-08-20 12:15:00 | 114.46 | 2025-08-21 11:15:00 | 113.52 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-08-20 13:30:00 | 114.40 | 2025-08-21 11:15:00 | 113.52 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-08-25 14:15:00 | 113.30 | 2025-09-01 15:15:00 | 113.10 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2025-08-25 15:15:00 | 113.28 | 2025-09-01 15:15:00 | 113.10 | STOP_HIT | 1.00 | 0.16% |
| BUY | retest2 | 2025-09-04 09:15:00 | 114.27 | 2025-09-04 11:15:00 | 112.85 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-09-08 15:00:00 | 112.19 | 2025-09-11 10:15:00 | 112.39 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2025-09-09 09:15:00 | 112.10 | 2025-09-11 10:15:00 | 112.39 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-09-09 10:00:00 | 112.05 | 2025-09-11 10:15:00 | 112.39 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-09-10 13:15:00 | 112.20 | 2025-09-11 10:15:00 | 112.39 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-09-22 09:15:00 | 114.34 | 2025-10-01 15:15:00 | 111.55 | STOP_HIT | 1.00 | 2.44% |
| SELL | retest2 | 2025-09-22 09:45:00 | 114.32 | 2025-10-01 15:15:00 | 111.55 | STOP_HIT | 1.00 | 2.42% |
| BUY | retest2 | 2025-10-31 14:30:00 | 110.55 | 2025-11-03 09:15:00 | 109.80 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-11-10 09:15:00 | 108.50 | 2025-11-27 10:15:00 | 106.20 | STOP_HIT | 1.00 | 2.12% |
| SELL | retest2 | 2025-12-18 14:15:00 | 94.48 | 2025-12-18 14:15:00 | 95.17 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest1 | 2026-01-12 09:15:00 | 93.01 | 2026-01-13 14:15:00 | 94.22 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest1 | 2026-01-12 10:45:00 | 93.38 | 2026-01-13 14:15:00 | 94.22 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest1 | 2026-01-12 15:00:00 | 93.43 | 2026-01-13 14:15:00 | 94.22 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-01-14 15:15:00 | 93.40 | 2026-01-21 10:15:00 | 88.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 15:15:00 | 93.40 | 2026-01-22 09:15:00 | 90.31 | STOP_HIT | 0.50 | 3.31% |
| SELL | retest2 | 2026-02-19 11:45:00 | 88.65 | 2026-03-02 09:15:00 | 84.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 11:45:00 | 88.65 | 2026-03-02 14:15:00 | 85.40 | STOP_HIT | 0.50 | 3.67% |
| SELL | retest2 | 2026-03-13 09:45:00 | 83.15 | 2026-03-18 11:15:00 | 82.50 | STOP_HIT | 1.00 | 0.78% |
| SELL | retest2 | 2026-03-18 10:00:00 | 82.80 | 2026-03-18 11:15:00 | 82.50 | STOP_HIT | 1.00 | 0.36% |
| SELL | retest2 | 2026-03-27 09:15:00 | 77.89 | 2026-03-30 09:15:00 | 74.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-27 09:15:00 | 77.89 | 2026-04-01 09:15:00 | 76.40 | STOP_HIT | 0.50 | 1.91% |
| BUY | retest2 | 2026-04-13 10:15:00 | 85.00 | 2026-04-23 15:15:00 | 90.04 | STOP_HIT | 1.00 | 5.93% |
| BUY | retest2 | 2026-04-13 10:45:00 | 85.05 | 2026-04-23 15:15:00 | 90.04 | STOP_HIT | 1.00 | 5.87% |
| BUY | retest2 | 2026-04-13 15:00:00 | 85.03 | 2026-04-23 15:15:00 | 90.04 | STOP_HIT | 1.00 | 5.89% |
| SELL | retest2 | 2026-05-06 12:15:00 | 86.67 | 2026-05-07 09:15:00 | 87.76 | STOP_HIT | 1.00 | -1.26% |
