# Meesho Ltd. (MEESHO)

## Backtest Summary

- **Window:** 2025-12-10 09:15:00 → 2026-05-08 15:15:00 (700 bars)
- **Last close:** 200.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 28 |
| ALERT1 | 22 |
| ALERT2 | 22 |
| ALERT2_SKIP | 8 |
| ALERT3 | 49 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 25 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 31 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 21
- **Target hits / Stop hits / Partials:** 2 / 25 / 4
- **Avg / median % per leg:** 0.05% / -1.89%
- **Sum % (uncompounded):** 1.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 2 | 13.3% | 2 | 13 | 0 | -0.54% | -8.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 2 | 13.3% | 2 | 13 | 0 | -0.54% | -8.1% |
| SELL (all) | 16 | 8 | 50.0% | 0 | 12 | 4 | 0.61% | 9.8% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.43% | -4.9% |
| SELL @ 3rd Alert (retest2) | 14 | 8 | 57.1% | 0 | 10 | 4 | 1.05% | 14.7% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.43% | -4.9% |
| retest2 (combined) | 29 | 10 | 34.5% | 2 | 23 | 4 | 0.22% | 6.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 09:15:00 | 187.30 | 170.78 | 168.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-17 09:15:00 | 197.49 | 182.05 | 176.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 14:15:00 | 225.10 | 225.36 | 215.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-19 14:45:00 | 226.25 | 225.36 | 215.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 213.63 | 221.06 | 216.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 11:00:00 | 213.63 | 221.06 | 216.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 11:15:00 | 209.23 | 218.69 | 215.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 12:00:00 | 209.23 | 218.69 | 215.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 13:15:00 | 201.68 | 212.57 | 213.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 09:15:00 | 194.68 | 205.86 | 209.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 193.29 | 191.60 | 198.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:30:00 | 195.50 | 191.60 | 198.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 197.22 | 195.66 | 197.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 10:45:00 | 193.40 | 196.03 | 196.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:45:00 | 193.68 | 195.55 | 196.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-31 09:15:00 | 183.73 | 188.16 | 191.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-31 09:15:00 | 184.00 | 188.16 | 191.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-01 14:15:00 | 181.90 | 181.50 | 184.43 | SL hit (close>ema200) qty=0.50 sl=181.50 alert=retest2 |

### Cycle 3 — BUY (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 12:15:00 | 184.70 | 182.93 | 182.87 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 15:15:00 | 182.30 | 182.85 | 182.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 09:15:00 | 181.50 | 182.58 | 182.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 14:15:00 | 182.34 | 182.17 | 182.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-06 14:45:00 | 182.25 | 182.17 | 182.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 181.69 | 182.07 | 182.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 09:15:00 | 173.13 | 182.07 | 182.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 164.47 | 167.98 | 171.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-09 09:15:00 | 171.00 | 167.98 | 171.62 | SL hit (close>static) qty=0.50 sl=167.98 alert=retest2 |

### Cycle 5 — BUY (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-20 09:15:00 | 168.83 | 164.19 | 163.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-20 14:15:00 | 169.50 | 166.90 | 165.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 15:15:00 | 166.20 | 166.76 | 165.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-21 09:15:00 | 167.39 | 166.76 | 165.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 169.33 | 167.27 | 165.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-21 11:00:00 | 170.26 | 167.87 | 166.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-21 11:30:00 | 170.77 | 168.71 | 166.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 11:45:00 | 170.40 | 171.38 | 169.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 10:15:00 | 165.81 | 168.81 | 169.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 165.81 | 168.81 | 169.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 12:15:00 | 164.42 | 167.55 | 168.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 09:15:00 | 166.12 | 165.72 | 167.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 10:15:00 | 167.33 | 166.04 | 167.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 167.33 | 166.04 | 167.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 167.33 | 166.04 | 167.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 168.97 | 166.63 | 167.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:00:00 | 168.97 | 166.63 | 167.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 167.72 | 166.85 | 167.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 15:15:00 | 167.10 | 167.66 | 167.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 09:15:00 | 169.72 | 167.99 | 167.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 09:15:00 | 169.72 | 167.99 | 167.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 171.70 | 169.26 | 168.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 165.45 | 171.30 | 170.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 165.45 | 171.30 | 170.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 165.45 | 171.30 | 170.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:30:00 | 165.45 | 171.30 | 170.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 165.45 | 169.19 | 169.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 157.18 | 165.02 | 167.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 09:15:00 | 153.82 | 150.24 | 152.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 153.82 | 150.24 | 152.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 153.82 | 150.24 | 152.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:30:00 | 154.59 | 150.24 | 152.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 152.92 | 150.77 | 152.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 11:15:00 | 152.12 | 150.77 | 152.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 12:30:00 | 152.27 | 151.47 | 152.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 13:45:00 | 152.50 | 151.62 | 152.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 14:45:00 | 152.10 | 151.52 | 152.64 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 156.87 | 152.66 | 152.97 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 156.87 | 152.66 | 152.97 | SL hit (close>static) qty=1.00 sl=154.48 alert=retest2 |

### Cycle 9 — BUY (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 10:15:00 | 157.64 | 153.66 | 153.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 13:15:00 | 158.20 | 156.69 | 155.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-09 15:15:00 | 156.50 | 156.77 | 155.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 09:15:00 | 155.29 | 156.77 | 155.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 155.43 | 156.50 | 155.74 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 14:15:00 | 154.50 | 155.75 | 155.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 14:15:00 | 153.93 | 155.42 | 155.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 13:15:00 | 153.44 | 153.35 | 154.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 14:00:00 | 153.44 | 153.35 | 154.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 154.10 | 153.26 | 153.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 11:00:00 | 154.10 | 153.26 | 153.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 154.41 | 153.49 | 153.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 12:00:00 | 154.41 | 153.49 | 153.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 153.99 | 153.71 | 153.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 13:30:00 | 154.19 | 153.71 | 153.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 153.75 | 153.72 | 153.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:30:00 | 154.09 | 153.72 | 153.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 154.25 | 153.82 | 153.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 154.15 | 153.82 | 153.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 153.86 | 153.83 | 153.98 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 09:15:00 | 157.72 | 153.95 | 153.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 10:15:00 | 159.58 | 157.28 | 155.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 157.00 | 158.28 | 156.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 15:00:00 | 157.00 | 158.28 | 156.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 156.40 | 157.91 | 156.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 155.62 | 157.91 | 156.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 157.61 | 157.85 | 156.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:15:00 | 158.29 | 157.85 | 156.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 11:30:00 | 158.15 | 157.89 | 157.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 13:15:00 | 158.11 | 157.87 | 157.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:30:00 | 157.99 | 158.16 | 157.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 157.93 | 158.12 | 157.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:45:00 | 157.50 | 158.12 | 157.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 157.63 | 158.02 | 157.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:45:00 | 157.50 | 158.02 | 157.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 158.50 | 158.11 | 157.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 13:15:00 | 158.96 | 158.11 | 157.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 15:15:00 | 158.70 | 158.10 | 157.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 09:15:00 | 155.75 | 157.73 | 157.65 | SL hit (close<static) qty=1.00 sl=157.45 alert=retest2 |

### Cycle 12 — SELL (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 10:15:00 | 154.99 | 157.18 | 157.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 11:15:00 | 153.51 | 156.45 | 157.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 15:15:00 | 156.50 | 155.61 | 156.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 15:15:00 | 156.50 | 155.61 | 156.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 156.50 | 155.61 | 156.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:15:00 | 157.12 | 155.61 | 156.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 157.45 | 155.98 | 156.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:45:00 | 157.74 | 155.98 | 156.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 158.50 | 156.48 | 156.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 11:00:00 | 158.50 | 156.48 | 156.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 11:15:00 | 158.26 | 156.84 | 156.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 13:15:00 | 159.16 | 157.42 | 157.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 10:15:00 | 158.10 | 158.25 | 157.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 10:15:00 | 158.10 | 158.25 | 157.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 158.10 | 158.25 | 157.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:45:00 | 158.19 | 158.25 | 157.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 157.95 | 158.19 | 157.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:45:00 | 157.94 | 158.19 | 157.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 157.96 | 158.15 | 157.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 13:45:00 | 157.43 | 158.15 | 157.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 160.00 | 158.52 | 157.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 15:15:00 | 160.90 | 158.52 | 157.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 11:15:00 | 160.65 | 159.31 | 158.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 12:00:00 | 160.32 | 159.51 | 158.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 13:00:00 | 160.30 | 159.67 | 158.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 158.50 | 159.64 | 159.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 156.40 | 159.64 | 159.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 157.29 | 159.17 | 158.88 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 157.29 | 159.17 | 158.88 | SL hit (close<static) qty=1.00 sl=157.78 alert=retest2 |

### Cycle 14 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 155.75 | 158.48 | 158.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 154.91 | 157.77 | 158.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 15:15:00 | 156.50 | 156.25 | 157.25 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:15:00 | 152.45 | 156.25 | 157.25 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 154.98 | 152.20 | 154.03 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 154.98 | 152.20 | 154.03 | SL hit (close>ema400) qty=1.00 sl=154.03 alert=retest1 |

### Cycle 15 — BUY (started 2026-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 14:15:00 | 158.70 | 155.31 | 154.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 15:15:00 | 159.90 | 156.23 | 155.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 14:15:00 | 158.92 | 158.93 | 157.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-06 15:00:00 | 158.92 | 158.93 | 157.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 16 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 143.54 | 155.90 | 156.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-10 09:15:00 | 139.42 | 145.19 | 149.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-11 09:15:00 | 141.04 | 140.95 | 144.71 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 11:00:00 | 137.98 | 140.35 | 144.10 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 140.75 | 140.00 | 142.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 141.48 | 140.00 | 142.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 141.24 | 140.16 | 141.73 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-12 13:15:00 | 142.40 | 140.61 | 141.79 | SL hit (close>ema400) qty=1.00 sl=141.79 alert=retest1 |

### Cycle 17 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 10:15:00 | 148.24 | 137.55 | 137.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 148.88 | 144.18 | 141.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 145.47 | 146.78 | 144.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 145.47 | 146.78 | 144.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 145.47 | 146.78 | 144.30 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 140.71 | 143.85 | 144.25 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 10:15:00 | 150.50 | 144.69 | 144.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 152.62 | 149.75 | 147.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 150.26 | 150.44 | 148.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 15:00:00 | 150.26 | 150.44 | 148.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 149.66 | 150.05 | 148.68 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 143.51 | 147.72 | 148.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 142.53 | 146.09 | 147.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 144.66 | 143.62 | 145.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-01 09:45:00 | 144.98 | 143.62 | 145.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 145.00 | 143.98 | 145.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 12:00:00 | 145.00 | 143.98 | 145.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 146.58 | 144.50 | 145.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 12:30:00 | 146.68 | 144.50 | 145.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 13:15:00 | 147.23 | 145.05 | 145.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 13:30:00 | 146.80 | 145.05 | 145.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 15:15:00 | 146.50 | 145.73 | 145.73 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 143.78 | 145.34 | 145.55 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 147.25 | 145.50 | 145.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 09:15:00 | 149.60 | 146.56 | 145.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 149.00 | 149.30 | 148.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:00:00 | 149.00 | 149.30 | 148.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 13:15:00 | 148.51 | 149.24 | 148.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 13:45:00 | 148.87 | 149.24 | 148.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 148.35 | 149.06 | 148.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 151.11 | 148.86 | 148.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-09 09:15:00 | 166.22 | 155.48 | 152.16 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2026-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 12:15:00 | 158.49 | 161.49 | 161.63 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 11:15:00 | 162.10 | 161.49 | 161.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 13:15:00 | 162.69 | 161.85 | 161.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 160.13 | 162.04 | 161.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 09:15:00 | 160.13 | 162.04 | 161.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 160.13 | 162.04 | 161.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 10:00:00 | 160.13 | 162.04 | 161.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 161.85 | 162.00 | 161.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 11:15:00 | 162.10 | 162.00 | 161.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-22 12:15:00 | 178.31 | 172.85 | 169.06 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 11:15:00 | 177.11 | 178.35 | 178.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 12:15:00 | 176.29 | 177.94 | 178.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 09:15:00 | 186.27 | 177.76 | 177.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 186.27 | 177.76 | 177.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 186.27 | 177.76 | 177.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:30:00 | 181.60 | 177.76 | 177.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 10:15:00 | 188.21 | 179.85 | 178.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 11:15:00 | 192.20 | 182.32 | 180.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 13:15:00 | 214.42 | 216.34 | 207.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 14:00:00 | 214.42 | 216.34 | 207.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 204.01 | 213.87 | 206.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 15:00:00 | 204.01 | 213.87 | 206.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 205.50 | 212.20 | 206.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:30:00 | 198.24 | 209.04 | 205.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 196.80 | 206.59 | 204.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:30:00 | 195.46 | 206.59 | 204.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2026-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 12:15:00 | 194.15 | 201.96 | 202.99 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-12-29 10:45:00 | 193.40 | 2025-12-31 09:15:00 | 183.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-29 11:45:00 | 193.68 | 2025-12-31 09:15:00 | 184.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-29 10:45:00 | 193.40 | 2026-01-01 14:15:00 | 181.90 | STOP_HIT | 0.50 | 5.95% |
| SELL | retest2 | 2025-12-29 11:45:00 | 193.68 | 2026-01-01 14:15:00 | 181.90 | STOP_HIT | 0.50 | 6.08% |
| SELL | retest2 | 2026-01-07 09:15:00 | 173.13 | 2026-01-09 09:15:00 | 164.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 09:15:00 | 173.13 | 2026-01-09 09:15:00 | 171.00 | STOP_HIT | 0.50 | 1.23% |
| BUY | retest2 | 2026-01-21 11:00:00 | 170.26 | 2026-01-27 10:15:00 | 165.81 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2026-01-21 11:30:00 | 170.77 | 2026-01-27 10:15:00 | 165.81 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2026-01-22 11:45:00 | 170.40 | 2026-01-27 10:15:00 | 165.81 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2026-01-28 15:15:00 | 167.10 | 2026-01-29 09:15:00 | 169.72 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2026-02-05 11:15:00 | 152.12 | 2026-02-06 09:15:00 | 156.87 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2026-02-05 12:30:00 | 152.27 | 2026-02-06 09:15:00 | 156.87 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2026-02-05 13:45:00 | 152.50 | 2026-02-06 09:15:00 | 156.87 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2026-02-05 14:45:00 | 152.10 | 2026-02-06 09:15:00 | 156.87 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest2 | 2026-02-20 10:15:00 | 158.29 | 2026-02-24 09:15:00 | 155.75 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2026-02-20 11:30:00 | 158.15 | 2026-02-24 09:15:00 | 155.75 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2026-02-20 13:15:00 | 158.11 | 2026-02-24 10:15:00 | 154.99 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2026-02-23 09:30:00 | 157.99 | 2026-02-24 10:15:00 | 154.99 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2026-02-23 13:15:00 | 158.96 | 2026-02-24 10:15:00 | 154.99 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2026-02-23 15:15:00 | 158.70 | 2026-02-24 10:15:00 | 154.99 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2026-02-26 15:15:00 | 160.90 | 2026-03-02 09:15:00 | 157.29 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2026-02-27 11:15:00 | 160.65 | 2026-03-02 09:15:00 | 157.29 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2026-02-27 12:00:00 | 160.32 | 2026-03-02 09:15:00 | 157.29 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2026-02-27 13:00:00 | 160.30 | 2026-03-02 09:15:00 | 157.29 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest1 | 2026-03-04 09:15:00 | 152.45 | 2026-03-05 09:15:00 | 154.98 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest1 | 2026-03-11 11:00:00 | 137.98 | 2026-03-12 13:15:00 | 142.40 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2026-03-13 11:45:00 | 138.30 | 2026-03-16 09:15:00 | 131.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 11:45:00 | 138.30 | 2026-03-16 14:15:00 | 135.00 | STOP_HIT | 0.50 | 2.39% |
| SELL | retest2 | 2026-03-17 09:30:00 | 138.20 | 2026-03-17 10:15:00 | 148.24 | STOP_HIT | 1.00 | -7.26% |
| BUY | retest2 | 2026-04-08 09:15:00 | 151.11 | 2026-04-09 09:15:00 | 166.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-20 11:15:00 | 162.10 | 2026-04-22 12:15:00 | 178.31 | TARGET_HIT | 1.00 | 10.00% |
