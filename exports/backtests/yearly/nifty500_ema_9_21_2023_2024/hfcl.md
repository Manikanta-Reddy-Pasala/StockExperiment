# HFCL Ltd. (HFCL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 139.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 207 |
| ALERT1 | 144 |
| ALERT2 | 141 |
| ALERT2_SKIP | 67 |
| ALERT3 | 399 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 10 |
| ENTRY2 | 181 |
| PARTIAL | 38 |
| TARGET_HIT | 29 |
| STOP_HIT | 162 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 229 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 107 / 122
- **Target hits / Stop hits / Partials:** 29 / 162 / 38
- **Avg / median % per leg:** 1.70% / -0.15%
- **Sum % (uncompounded):** 388.42%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 81 | 40 | 49.4% | 20 | 56 | 5 | 2.42% | 195.9% |
| BUY @ 2nd Alert (retest1) | 10 | 10 | 100.0% | 0 | 5 | 5 | 4.26% | 42.6% |
| BUY @ 3rd Alert (retest2) | 71 | 30 | 42.3% | 20 | 51 | 0 | 2.16% | 153.4% |
| SELL (all) | 148 | 67 | 45.3% | 9 | 106 | 33 | 1.30% | 192.5% |
| SELL @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 5 | 1 | 0.24% | 1.5% |
| SELL @ 3rd Alert (retest2) | 142 | 66 | 46.5% | 9 | 101 | 32 | 1.35% | 191.0% |
| retest1 (combined) | 16 | 11 | 68.8% | 0 | 10 | 6 | 2.75% | 44.0% |
| retest2 (combined) | 213 | 96 | 45.1% | 29 | 152 | 32 | 1.62% | 344.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 14:15:00 | 67.85 | 66.44 | 66.44 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 11:15:00 | 66.10 | 66.82 | 66.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-18 13:15:00 | 65.80 | 66.21 | 66.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-22 14:15:00 | 65.35 | 64.76 | 65.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 14:15:00 | 65.35 | 64.76 | 65.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 14:15:00 | 65.35 | 64.76 | 65.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 14:45:00 | 65.95 | 64.76 | 65.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 15:15:00 | 64.95 | 64.80 | 65.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-23 09:15:00 | 65.00 | 64.80 | 65.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 09:15:00 | 64.75 | 64.79 | 65.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-23 15:00:00 | 64.65 | 64.80 | 65.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-24 10:15:00 | 64.60 | 64.69 | 64.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-24 11:45:00 | 64.55 | 64.69 | 64.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-31 14:15:00 | 64.70 | 63.65 | 63.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2023-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 14:15:00 | 64.70 | 63.65 | 63.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-01 09:15:00 | 65.25 | 64.14 | 63.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-01 15:15:00 | 64.35 | 64.51 | 64.21 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-02 09:15:00 | 64.80 | 64.51 | 64.21 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-02 10:30:00 | 64.85 | 64.60 | 64.30 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-02 12:00:00 | 64.75 | 64.63 | 64.34 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 12:15:00 | 64.45 | 64.59 | 64.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-02 13:00:00 | 64.45 | 64.59 | 64.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 13:15:00 | 64.75 | 64.62 | 64.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-05 09:15:00 | 67.30 | 64.62 | 64.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-05 09:15:00 | 67.99 | 65.07 | 64.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-05 14:15:00 | 68.04 | 66.52 | 65.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-05 14:15:00 | 68.09 | 66.52 | 65.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2023-06-06 15:15:00 | 66.90 | 66.97 | 66.41 | SL hit (close<ema200) qty=0.50 sl=66.97 alert=retest1 |

### Cycle 4 — SELL (started 2023-06-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-14 14:15:00 | 69.75 | 70.13 | 70.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-15 12:15:00 | 69.55 | 69.88 | 70.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-19 09:15:00 | 69.15 | 68.89 | 69.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-19 09:15:00 | 69.15 | 68.89 | 69.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 09:15:00 | 69.15 | 68.89 | 69.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-19 13:15:00 | 68.45 | 68.90 | 69.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-19 15:15:00 | 68.40 | 68.78 | 69.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-20 12:45:00 | 68.45 | 68.64 | 68.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-20 14:15:00 | 68.45 | 68.66 | 68.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 09:15:00 | 68.60 | 68.51 | 68.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-21 10:45:00 | 68.25 | 68.39 | 68.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-21 12:45:00 | 67.85 | 68.23 | 68.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-26 09:15:00 | 65.03 | 65.98 | 66.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-26 09:15:00 | 64.98 | 65.98 | 66.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-26 09:15:00 | 65.03 | 65.98 | 66.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-26 09:15:00 | 65.03 | 65.98 | 66.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-06-27 09:15:00 | 65.50 | 65.37 | 65.97 | SL hit (close>ema200) qty=0.50 sl=65.37 alert=retest2 |

### Cycle 5 — BUY (started 2023-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-03 09:15:00 | 68.15 | 65.67 | 65.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-03 11:15:00 | 68.90 | 66.76 | 66.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-04 12:15:00 | 67.55 | 67.78 | 67.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-04 13:00:00 | 67.55 | 67.78 | 67.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 67.35 | 67.63 | 67.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-05 09:30:00 | 67.35 | 67.63 | 67.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 10:15:00 | 67.15 | 67.54 | 67.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-05 11:00:00 | 67.15 | 67.54 | 67.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 11:15:00 | 67.15 | 67.46 | 67.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-05 12:00:00 | 67.15 | 67.46 | 67.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 12:15:00 | 67.10 | 67.39 | 67.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-05 12:30:00 | 67.05 | 67.39 | 67.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 13:15:00 | 67.00 | 67.31 | 67.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-05 13:30:00 | 67.10 | 67.31 | 67.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 14:15:00 | 67.15 | 67.28 | 67.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-05 14:45:00 | 67.05 | 67.28 | 67.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 15:15:00 | 67.05 | 67.23 | 67.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-06 09:30:00 | 67.30 | 67.27 | 67.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-06 10:45:00 | 67.25 | 67.24 | 67.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-06 13:30:00 | 67.95 | 67.27 | 67.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-07 10:15:00 | 66.65 | 67.24 | 67.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2023-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 10:15:00 | 66.65 | 67.24 | 67.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 09:15:00 | 65.75 | 66.63 | 66.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 09:15:00 | 66.40 | 65.92 | 66.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 09:15:00 | 66.40 | 65.92 | 66.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 66.40 | 65.92 | 66.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 10:00:00 | 66.40 | 65.92 | 66.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 10:15:00 | 66.05 | 65.94 | 66.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-11 13:00:00 | 65.85 | 65.93 | 66.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-17 11:15:00 | 65.95 | 64.78 | 64.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2023-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 11:15:00 | 65.95 | 64.78 | 64.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-20 09:15:00 | 66.35 | 65.83 | 65.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-20 13:15:00 | 65.65 | 65.86 | 65.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-20 13:15:00 | 65.65 | 65.86 | 65.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 13:15:00 | 65.65 | 65.86 | 65.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 13:30:00 | 65.75 | 65.86 | 65.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 14:15:00 | 65.35 | 65.76 | 65.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 14:45:00 | 65.20 | 65.76 | 65.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 15:15:00 | 65.20 | 65.64 | 65.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-21 09:15:00 | 65.15 | 65.64 | 65.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2023-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 10:15:00 | 65.00 | 65.44 | 65.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 14:15:00 | 64.80 | 65.15 | 65.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-24 09:15:00 | 65.15 | 65.10 | 65.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 09:15:00 | 65.15 | 65.10 | 65.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 65.15 | 65.10 | 65.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-24 11:45:00 | 64.55 | 65.01 | 65.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-24 12:15:00 | 64.70 | 65.01 | 65.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-24 13:30:00 | 64.70 | 64.92 | 65.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-26 09:15:00 | 65.35 | 65.13 | 65.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2023-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 09:15:00 | 65.35 | 65.13 | 65.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-26 10:15:00 | 65.70 | 65.24 | 65.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-26 14:15:00 | 64.95 | 65.33 | 65.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-26 14:15:00 | 64.95 | 65.33 | 65.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 14:15:00 | 64.95 | 65.33 | 65.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-26 15:00:00 | 64.95 | 65.33 | 65.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 15:15:00 | 64.80 | 65.22 | 65.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 09:15:00 | 64.45 | 65.22 | 65.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2023-07-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 09:15:00 | 64.40 | 65.06 | 65.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-28 11:15:00 | 63.55 | 64.00 | 64.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-28 13:15:00 | 64.10 | 63.95 | 64.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-28 13:15:00 | 64.10 | 63.95 | 64.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 13:15:00 | 64.10 | 63.95 | 64.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-28 14:00:00 | 64.10 | 63.95 | 64.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 14:15:00 | 63.85 | 63.93 | 64.28 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2023-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-01 13:15:00 | 65.35 | 64.29 | 64.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 14:15:00 | 66.00 | 64.63 | 64.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 13:15:00 | 64.40 | 65.22 | 64.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 13:15:00 | 64.40 | 65.22 | 64.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 13:15:00 | 64.40 | 65.22 | 64.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 14:00:00 | 64.40 | 65.22 | 64.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 14:15:00 | 65.25 | 65.23 | 64.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-03 09:15:00 | 65.50 | 65.20 | 64.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-03 11:30:00 | 65.65 | 65.23 | 65.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-03 12:00:00 | 65.40 | 65.23 | 65.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-03 14:15:00 | 65.50 | 65.11 | 64.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 14:15:00 | 65.50 | 65.19 | 65.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-04 09:15:00 | 66.30 | 65.29 | 65.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-04 14:00:00 | 65.80 | 65.77 | 65.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-07 09:15:00 | 68.70 | 65.66 | 65.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2023-08-07 10:15:00 | 72.05 | 68.11 | 66.65 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2023-08-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 15:15:00 | 70.05 | 70.67 | 70.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 11:15:00 | 69.40 | 70.18 | 70.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 09:15:00 | 68.70 | 68.31 | 68.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-16 09:15:00 | 68.70 | 68.31 | 68.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 09:15:00 | 68.70 | 68.31 | 68.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 13:15:00 | 67.20 | 67.80 | 68.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 14:30:00 | 67.30 | 67.64 | 68.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-21 10:00:00 | 67.25 | 67.50 | 67.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-21 10:45:00 | 67.15 | 67.44 | 67.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 09:15:00 | 68.35 | 67.43 | 67.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 10:00:00 | 68.35 | 67.43 | 67.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 10:15:00 | 67.90 | 67.53 | 67.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 10:30:00 | 68.15 | 67.53 | 67.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-08-22 14:15:00 | 67.90 | 67.75 | 67.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2023-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 14:15:00 | 67.90 | 67.75 | 67.74 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-22 15:15:00 | 67.45 | 67.69 | 67.71 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-23 09:15:00 | 68.15 | 67.78 | 67.75 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 14:15:00 | 67.50 | 67.72 | 67.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-23 15:15:00 | 67.15 | 67.61 | 67.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-24 09:15:00 | 67.70 | 67.63 | 67.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-24 09:15:00 | 67.70 | 67.63 | 67.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 09:15:00 | 67.70 | 67.63 | 67.69 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2023-08-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-24 11:15:00 | 68.95 | 67.91 | 67.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-25 09:15:00 | 71.50 | 69.02 | 68.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-28 15:15:00 | 70.95 | 71.06 | 70.36 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-29 09:30:00 | 72.35 | 71.35 | 70.55 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-29 12:30:00 | 71.80 | 71.50 | 70.84 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 13:15:00 | 70.85 | 71.37 | 70.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-29 14:00:00 | 70.85 | 71.37 | 70.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 14:15:00 | 71.50 | 71.40 | 70.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-29 14:45:00 | 70.70 | 71.40 | 70.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 15:15:00 | 70.95 | 71.31 | 70.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-30 09:15:00 | 71.65 | 71.31 | 70.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-30 10:15:00 | 71.75 | 71.27 | 70.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-30 11:00:00 | 71.65 | 71.34 | 70.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-31 11:15:00 | 75.39 | 73.30 | 72.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-31 13:15:00 | 75.97 | 74.11 | 72.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2023-09-01 13:15:00 | 74.90 | 75.76 | 74.53 | SL hit (close<ema200) qty=0.50 sl=75.76 alert=retest1 |

### Cycle 18 — SELL (started 2023-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-08 10:15:00 | 76.75 | 77.79 | 77.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-08 15:15:00 | 76.25 | 76.86 | 77.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-11 09:15:00 | 77.10 | 76.90 | 77.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-11 09:15:00 | 77.10 | 76.90 | 77.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 09:15:00 | 77.10 | 76.90 | 77.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-11 10:15:00 | 77.90 | 76.90 | 77.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 10:15:00 | 77.40 | 77.00 | 77.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-11 12:45:00 | 77.00 | 77.10 | 77.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-11 13:15:00 | 76.95 | 77.10 | 77.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 09:15:00 | 73.15 | 76.58 | 77.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 09:15:00 | 73.10 | 76.58 | 77.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-12 09:30:00 | 75.10 | 76.58 | 77.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 15:15:00 | 71.34 | 73.56 | 75.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-09-13 11:15:00 | 73.10 | 73.08 | 74.47 | SL hit (close>ema200) qty=0.50 sl=73.08 alert=retest2 |

### Cycle 19 — BUY (started 2023-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-18 09:15:00 | 76.80 | 74.14 | 74.07 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2023-09-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 12:15:00 | 73.85 | 74.21 | 74.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 09:15:00 | 73.70 | 74.01 | 74.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-21 14:15:00 | 73.80 | 73.57 | 73.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 14:15:00 | 73.80 | 73.57 | 73.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 14:15:00 | 73.80 | 73.57 | 73.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 15:00:00 | 73.80 | 73.57 | 73.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 15:15:00 | 73.70 | 73.60 | 73.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 09:15:00 | 73.40 | 73.60 | 73.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 72.45 | 73.37 | 73.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-25 10:00:00 | 71.70 | 72.54 | 73.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-26 11:45:00 | 72.20 | 72.18 | 72.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-26 12:30:00 | 72.00 | 72.14 | 72.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-27 10:15:00 | 74.35 | 72.74 | 72.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2023-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 10:15:00 | 74.35 | 72.74 | 72.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 14:15:00 | 76.20 | 74.20 | 73.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-29 15:15:00 | 76.70 | 76.86 | 76.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-03 09:15:00 | 76.75 | 76.86 | 76.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 09:15:00 | 75.40 | 76.70 | 76.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 10:00:00 | 75.40 | 76.70 | 76.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 10:15:00 | 75.80 | 76.52 | 76.35 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-10-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 11:15:00 | 74.90 | 76.19 | 76.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 12:15:00 | 74.45 | 75.84 | 76.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-06 09:15:00 | 74.55 | 74.47 | 75.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-06 09:30:00 | 74.45 | 74.47 | 75.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 14:15:00 | 75.25 | 74.53 | 74.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 14:45:00 | 75.50 | 74.53 | 74.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 15:15:00 | 75.50 | 74.73 | 74.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-09 09:15:00 | 72.80 | 74.73 | 74.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-12 11:15:00 | 73.60 | 73.45 | 73.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2023-10-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 11:15:00 | 73.60 | 73.45 | 73.44 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2023-10-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 12:15:00 | 73.05 | 73.37 | 73.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-12 13:15:00 | 72.80 | 73.25 | 73.35 | Break + close below crossover candle low |

### Cycle 25 — BUY (started 2023-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-13 09:15:00 | 75.50 | 73.62 | 73.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-16 14:15:00 | 75.90 | 74.98 | 74.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-17 10:15:00 | 75.15 | 75.25 | 74.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-17 11:00:00 | 75.15 | 75.25 | 74.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 12:15:00 | 75.00 | 75.19 | 74.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-17 13:00:00 | 75.00 | 75.19 | 74.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 13:15:00 | 74.75 | 75.10 | 74.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-17 13:30:00 | 74.80 | 75.10 | 74.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 14:15:00 | 74.70 | 75.02 | 74.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-17 14:30:00 | 74.70 | 75.02 | 74.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 15:15:00 | 74.25 | 74.87 | 74.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-18 09:45:00 | 75.00 | 74.84 | 74.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-18 10:15:00 | 73.25 | 74.53 | 74.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2023-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 10:15:00 | 73.25 | 74.53 | 74.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 11:15:00 | 72.85 | 74.19 | 74.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 09:15:00 | 74.15 | 73.63 | 74.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 09:15:00 | 74.15 | 73.63 | 74.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 09:15:00 | 74.15 | 73.63 | 74.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 10:00:00 | 74.15 | 73.63 | 74.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 10:15:00 | 73.70 | 73.64 | 73.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-19 13:00:00 | 73.35 | 73.57 | 73.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-19 14:00:00 | 73.40 | 73.53 | 73.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-20 13:15:00 | 69.68 | 71.22 | 72.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-20 13:15:00 | 69.73 | 71.22 | 72.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2023-10-23 09:15:00 | 66.02 | 69.43 | 71.25 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 27 — BUY (started 2023-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 11:15:00 | 65.70 | 64.89 | 64.85 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-11-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 12:15:00 | 64.60 | 65.14 | 65.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 14:15:00 | 64.35 | 64.93 | 65.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 09:15:00 | 65.15 | 64.89 | 65.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 09:15:00 | 65.15 | 64.89 | 65.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 09:15:00 | 65.15 | 64.89 | 65.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 09:45:00 | 65.30 | 64.89 | 65.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 10:15:00 | 65.05 | 64.92 | 65.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 10:45:00 | 65.15 | 64.92 | 65.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 11:15:00 | 64.85 | 64.91 | 65.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 11:30:00 | 64.95 | 64.91 | 65.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 12:15:00 | 64.95 | 64.91 | 65.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 12:30:00 | 65.00 | 64.91 | 65.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 13:15:00 | 64.95 | 64.92 | 64.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 13:45:00 | 64.90 | 64.92 | 64.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 14:15:00 | 65.00 | 64.94 | 64.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 14:45:00 | 64.90 | 64.94 | 64.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 15:15:00 | 64.75 | 64.90 | 64.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-03 09:15:00 | 66.65 | 64.90 | 64.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2023-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 09:15:00 | 67.35 | 65.39 | 65.19 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-11-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 10:15:00 | 65.95 | 66.15 | 66.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 14:15:00 | 65.50 | 65.90 | 66.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 14:15:00 | 65.60 | 65.48 | 65.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 14:15:00 | 65.60 | 65.48 | 65.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 14:15:00 | 65.60 | 65.48 | 65.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 15:00:00 | 65.60 | 65.48 | 65.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 66.35 | 65.69 | 65.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-12 19:00:00 | 66.35 | 65.69 | 65.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2023-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 10:15:00 | 66.45 | 65.93 | 65.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 09:15:00 | 68.75 | 66.55 | 66.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-17 10:15:00 | 69.25 | 69.35 | 68.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-17 11:15:00 | 69.20 | 69.35 | 68.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 13:15:00 | 69.10 | 69.24 | 68.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-17 13:30:00 | 68.80 | 69.24 | 68.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 14:15:00 | 68.80 | 69.15 | 68.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-17 14:30:00 | 68.95 | 69.15 | 68.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 15:15:00 | 68.70 | 69.06 | 68.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 09:30:00 | 68.40 | 68.92 | 68.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 10:15:00 | 68.00 | 68.73 | 68.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 11:00:00 | 68.00 | 68.73 | 68.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2023-11-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 12:15:00 | 67.70 | 68.37 | 68.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-21 13:15:00 | 67.45 | 67.79 | 68.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 09:15:00 | 67.60 | 66.98 | 67.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-23 09:15:00 | 67.60 | 66.98 | 67.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 67.60 | 66.98 | 67.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 09:30:00 | 67.95 | 66.98 | 67.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 10:15:00 | 67.25 | 67.03 | 67.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 11:30:00 | 67.10 | 67.03 | 67.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 14:30:00 | 67.05 | 67.08 | 67.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-24 10:15:00 | 67.85 | 67.28 | 67.32 | SL hit (close>static) qty=1.00 sl=67.65 alert=retest2 |

### Cycle 33 — BUY (started 2023-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 11:15:00 | 67.65 | 67.35 | 67.35 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 12:15:00 | 66.90 | 67.26 | 67.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-24 13:15:00 | 66.75 | 67.16 | 67.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 10:15:00 | 67.05 | 67.01 | 67.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 10:15:00 | 67.05 | 67.01 | 67.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 10:15:00 | 67.05 | 67.01 | 67.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 10:30:00 | 67.25 | 67.01 | 67.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 13:15:00 | 66.85 | 66.96 | 67.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-28 15:00:00 | 66.65 | 66.90 | 67.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-29 09:15:00 | 67.65 | 67.02 | 67.07 | SL hit (close>static) qty=1.00 sl=67.20 alert=retest2 |

### Cycle 35 — BUY (started 2023-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 11:15:00 | 67.40 | 67.16 | 67.13 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-11-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-29 15:15:00 | 66.75 | 67.08 | 67.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-30 13:15:00 | 66.55 | 66.87 | 66.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-01 09:15:00 | 67.00 | 66.83 | 66.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-01 09:15:00 | 67.00 | 66.83 | 66.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 09:15:00 | 67.00 | 66.83 | 66.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-01 13:00:00 | 66.80 | 66.85 | 66.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-01 15:15:00 | 66.70 | 66.86 | 66.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-04 09:15:00 | 67.70 | 67.00 | 66.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2023-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 09:15:00 | 67.70 | 67.00 | 66.96 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2023-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 14:15:00 | 66.85 | 67.25 | 67.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-06 12:15:00 | 66.60 | 67.01 | 67.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-07 09:15:00 | 68.20 | 67.05 | 67.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-07 09:15:00 | 68.20 | 67.05 | 67.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 09:15:00 | 68.20 | 67.05 | 67.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-07 10:00:00 | 68.20 | 67.05 | 67.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2023-12-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 10:15:00 | 69.25 | 67.49 | 67.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-07 11:15:00 | 71.15 | 68.22 | 67.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-08 09:15:00 | 69.35 | 69.48 | 68.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-08 10:00:00 | 69.35 | 69.48 | 68.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 12:15:00 | 68.40 | 69.18 | 68.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 13:00:00 | 68.40 | 69.18 | 68.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 13:15:00 | 68.30 | 69.00 | 68.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 13:30:00 | 67.90 | 69.00 | 68.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 15:15:00 | 68.40 | 68.76 | 68.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-11 09:15:00 | 69.05 | 68.76 | 68.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-11 10:45:00 | 68.75 | 68.80 | 68.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-11 13:30:00 | 68.55 | 68.70 | 68.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-11 15:15:00 | 68.55 | 68.65 | 68.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 15:15:00 | 68.55 | 68.63 | 68.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-12 09:30:00 | 68.65 | 68.60 | 68.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-12 10:15:00 | 68.55 | 68.59 | 68.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2023-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 10:15:00 | 68.55 | 68.59 | 68.59 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2023-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 11:15:00 | 68.95 | 68.66 | 68.62 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2023-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 13:15:00 | 68.35 | 68.56 | 68.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-12 14:15:00 | 67.75 | 68.39 | 68.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 12:15:00 | 68.45 | 68.14 | 68.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 12:15:00 | 68.45 | 68.14 | 68.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 12:15:00 | 68.45 | 68.14 | 68.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-13 12:45:00 | 68.30 | 68.14 | 68.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 13:15:00 | 68.05 | 68.12 | 68.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-13 14:45:00 | 67.90 | 68.11 | 68.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-14 14:30:00 | 67.90 | 68.08 | 68.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-15 09:15:00 | 70.60 | 68.54 | 68.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2023-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 09:15:00 | 70.60 | 68.54 | 68.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-15 10:15:00 | 71.90 | 69.21 | 68.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 09:15:00 | 77.45 | 77.61 | 75.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-19 09:45:00 | 77.90 | 77.61 | 75.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 76.15 | 79.59 | 78.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 13:30:00 | 77.40 | 79.59 | 78.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 14:15:00 | 74.50 | 78.58 | 77.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 15:00:00 | 74.50 | 78.58 | 77.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2023-12-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-21 10:15:00 | 76.85 | 77.49 | 77.50 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2023-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 09:15:00 | 82.40 | 78.15 | 77.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 09:15:00 | 83.90 | 81.40 | 79.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 11:15:00 | 84.30 | 84.74 | 82.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-27 12:00:00 | 84.30 | 84.74 | 82.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 09:15:00 | 81.75 | 83.79 | 83.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 10:00:00 | 81.75 | 83.79 | 83.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 10:15:00 | 82.05 | 83.44 | 83.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 10:30:00 | 81.90 | 83.44 | 83.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 12:15:00 | 82.30 | 83.08 | 82.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 13:00:00 | 82.30 | 83.08 | 82.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2023-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-29 09:15:00 | 82.35 | 82.77 | 82.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-29 13:15:00 | 81.85 | 82.46 | 82.65 | Break + close below crossover candle low |

### Cycle 47 — BUY (started 2023-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-29 14:15:00 | 84.50 | 82.87 | 82.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-01 09:15:00 | 86.90 | 83.87 | 83.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-01 14:15:00 | 84.80 | 85.17 | 84.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-01 15:00:00 | 84.80 | 85.17 | 84.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 15:15:00 | 84.25 | 84.99 | 84.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 09:15:00 | 85.80 | 84.99 | 84.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-02 10:15:00 | 83.35 | 84.56 | 84.22 | SL hit (close<static) qty=1.00 sl=84.10 alert=retest2 |

### Cycle 48 — SELL (started 2024-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 09:15:00 | 89.05 | 90.68 | 90.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-12 14:15:00 | 88.05 | 88.94 | 89.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-15 10:15:00 | 88.20 | 88.19 | 88.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-15 10:15:00 | 88.20 | 88.19 | 88.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 10:15:00 | 88.20 | 88.19 | 88.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-15 10:45:00 | 88.60 | 88.19 | 88.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 14:15:00 | 88.50 | 88.11 | 88.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-15 15:00:00 | 88.50 | 88.11 | 88.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 09:15:00 | 89.10 | 88.32 | 88.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-16 09:30:00 | 88.85 | 88.32 | 88.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 10:15:00 | 88.50 | 88.36 | 88.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-16 11:15:00 | 88.20 | 88.36 | 88.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-17 14:15:00 | 83.79 | 85.36 | 86.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-18 11:15:00 | 84.80 | 84.76 | 85.80 | SL hit (close>ema200) qty=0.50 sl=84.76 alert=retest2 |

### Cycle 49 — BUY (started 2024-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 15:15:00 | 90.70 | 86.26 | 85.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-20 09:15:00 | 95.00 | 88.01 | 86.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 13:15:00 | 96.60 | 97.16 | 94.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-23 14:00:00 | 96.60 | 97.16 | 94.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 14:15:00 | 93.45 | 96.42 | 94.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 15:00:00 | 93.45 | 96.42 | 94.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 15:15:00 | 94.30 | 95.99 | 94.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-24 09:15:00 | 96.20 | 95.99 | 94.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-01-24 14:15:00 | 105.82 | 99.09 | 96.38 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2024-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 15:15:00 | 101.00 | 102.52 | 102.58 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2024-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 14:15:00 | 102.80 | 102.53 | 102.52 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 10:15:00 | 101.80 | 102.43 | 102.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-01 11:15:00 | 100.85 | 102.11 | 102.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-01 14:15:00 | 103.00 | 101.79 | 102.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 14:15:00 | 103.00 | 101.79 | 102.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 14:15:00 | 103.00 | 101.79 | 102.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-01 15:00:00 | 103.00 | 101.79 | 102.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 15:15:00 | 102.15 | 101.86 | 102.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-02 09:15:00 | 100.00 | 101.86 | 102.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-05 13:15:00 | 101.45 | 99.45 | 99.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-05 13:45:00 | 101.25 | 99.71 | 100.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-06 14:15:00 | 103.60 | 99.98 | 99.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2024-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 14:15:00 | 103.60 | 99.98 | 99.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-07 14:15:00 | 106.00 | 102.88 | 101.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-09 09:15:00 | 102.25 | 106.01 | 104.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 09:15:00 | 102.25 | 106.01 | 104.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 102.25 | 106.01 | 104.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 09:45:00 | 100.90 | 106.01 | 104.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 10:15:00 | 100.35 | 104.88 | 104.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 11:00:00 | 100.35 | 104.88 | 104.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2024-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 12:15:00 | 101.70 | 103.74 | 103.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 09:15:00 | 97.05 | 101.70 | 102.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 14:15:00 | 96.95 | 95.62 | 97.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-13 15:00:00 | 96.95 | 95.62 | 97.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 09:15:00 | 97.60 | 96.18 | 97.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 09:45:00 | 97.80 | 96.18 | 97.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 10:15:00 | 97.80 | 96.50 | 97.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 11:00:00 | 97.80 | 96.50 | 97.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 11:15:00 | 97.50 | 96.70 | 97.52 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2024-02-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 09:15:00 | 101.00 | 98.44 | 98.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 14:15:00 | 101.60 | 100.13 | 99.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 14:15:00 | 100.20 | 100.79 | 100.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-16 15:00:00 | 100.20 | 100.79 | 100.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 15:15:00 | 100.60 | 100.75 | 100.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-19 11:45:00 | 104.50 | 101.54 | 100.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-02-21 11:15:00 | 114.95 | 110.69 | 107.83 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2024-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 11:15:00 | 110.60 | 112.20 | 112.24 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2024-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-26 14:15:00 | 115.70 | 112.60 | 112.39 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 10:15:00 | 109.15 | 112.45 | 112.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 11:15:00 | 107.45 | 111.45 | 112.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 12:15:00 | 106.65 | 106.39 | 108.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 13:00:00 | 106.65 | 106.39 | 108.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 108.25 | 107.06 | 108.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 09:30:00 | 108.90 | 107.06 | 108.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 10:15:00 | 108.10 | 107.27 | 108.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 10:30:00 | 108.40 | 107.27 | 108.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 11:15:00 | 109.70 | 107.76 | 108.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 12:00:00 | 109.70 | 107.76 | 108.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 12:15:00 | 109.85 | 108.18 | 108.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 12:45:00 | 110.75 | 108.18 | 108.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 15:15:00 | 108.00 | 108.42 | 108.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 09:15:00 | 108.55 | 108.42 | 108.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 09:15:00 | 108.60 | 108.46 | 108.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-04 09:15:00 | 106.75 | 108.42 | 108.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-04 12:15:00 | 109.15 | 108.57 | 108.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2024-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-04 12:15:00 | 109.15 | 108.57 | 108.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-04 14:15:00 | 110.35 | 108.96 | 108.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 10:15:00 | 108.50 | 109.06 | 108.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-05 10:15:00 | 108.50 | 109.06 | 108.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 10:15:00 | 108.50 | 109.06 | 108.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 10:45:00 | 108.65 | 109.06 | 108.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 11:15:00 | 109.10 | 109.07 | 108.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 13:00:00 | 110.10 | 109.27 | 108.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 14:45:00 | 109.65 | 109.03 | 108.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-05 15:15:00 | 107.70 | 108.76 | 108.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2024-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 15:15:00 | 107.70 | 108.76 | 108.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 09:15:00 | 104.35 | 107.88 | 108.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 09:15:00 | 88.20 | 87.43 | 91.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 10:00:00 | 88.20 | 87.43 | 91.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 13:15:00 | 90.00 | 88.59 | 90.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 13:45:00 | 90.45 | 88.59 | 90.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 14:15:00 | 90.35 | 88.94 | 90.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 15:00:00 | 90.35 | 88.94 | 90.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 15:15:00 | 90.35 | 89.22 | 90.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 09:15:00 | 91.90 | 89.22 | 90.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 89.50 | 89.28 | 90.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 11:45:00 | 87.80 | 88.91 | 90.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 13:00:00 | 88.30 | 88.79 | 90.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 14:45:00 | 88.30 | 88.97 | 89.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-21 11:15:00 | 89.55 | 88.80 | 88.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2024-03-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 11:15:00 | 89.55 | 88.80 | 88.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 09:15:00 | 91.40 | 89.61 | 89.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 10:15:00 | 91.25 | 91.60 | 90.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-26 11:00:00 | 91.25 | 91.60 | 90.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 11:15:00 | 91.10 | 91.50 | 90.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 11:30:00 | 91.10 | 91.50 | 90.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 14:15:00 | 90.65 | 91.38 | 90.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 15:00:00 | 90.65 | 91.38 | 90.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 15:15:00 | 90.40 | 91.19 | 90.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 09:15:00 | 91.90 | 91.19 | 90.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-08 13:15:00 | 96.95 | 97.92 | 97.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2024-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 13:15:00 | 96.95 | 97.92 | 97.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 12:15:00 | 96.20 | 97.23 | 97.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-09 15:15:00 | 97.20 | 96.92 | 97.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 15:15:00 | 97.20 | 96.92 | 97.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 15:15:00 | 97.20 | 96.92 | 97.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-10 09:15:00 | 96.35 | 96.92 | 97.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-10 10:00:00 | 96.65 | 96.87 | 97.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-12 09:15:00 | 99.15 | 96.94 | 96.99 | SL hit (close>static) qty=1.00 sl=98.00 alert=retest2 |

### Cycle 63 — BUY (started 2024-04-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-12 10:15:00 | 99.10 | 97.37 | 97.18 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2024-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 10:15:00 | 94.70 | 96.90 | 97.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-16 12:15:00 | 94.35 | 95.12 | 95.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-19 13:15:00 | 92.35 | 92.29 | 93.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-19 13:30:00 | 92.30 | 92.29 | 93.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 92.55 | 92.17 | 93.03 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 15:15:00 | 94.80 | 93.32 | 93.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 09:15:00 | 97.60 | 94.18 | 93.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 14:15:00 | 96.50 | 96.57 | 95.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-24 15:00:00 | 96.50 | 96.57 | 95.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 97.55 | 96.75 | 96.01 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2024-04-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 14:15:00 | 95.80 | 96.35 | 96.36 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-29 12:15:00 | 97.00 | 96.34 | 96.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-29 13:15:00 | 97.35 | 96.54 | 96.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-03 10:15:00 | 102.30 | 102.52 | 101.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-03 10:30:00 | 102.00 | 102.52 | 101.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 11:15:00 | 101.40 | 102.30 | 101.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 11:30:00 | 101.40 | 102.30 | 101.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 12:15:00 | 101.50 | 102.14 | 101.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 12:45:00 | 101.30 | 102.14 | 101.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 13:15:00 | 103.85 | 102.48 | 101.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 14:30:00 | 104.65 | 102.18 | 101.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-03 15:15:00 | 99.60 | 101.66 | 101.30 | SL hit (close<static) qty=1.00 sl=101.30 alert=retest2 |

### Cycle 68 — SELL (started 2024-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 09:15:00 | 97.95 | 100.92 | 100.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 10:15:00 | 96.40 | 98.34 | 99.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 97.10 | 97.04 | 98.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 09:45:00 | 96.80 | 97.04 | 98.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 14:15:00 | 92.60 | 92.35 | 93.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 15:00:00 | 92.60 | 92.35 | 93.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 94.20 | 92.78 | 93.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:30:00 | 94.85 | 92.78 | 93.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 10:15:00 | 95.40 | 93.31 | 93.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 10:45:00 | 95.45 | 93.31 | 93.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 12:15:00 | 95.15 | 94.01 | 93.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 14:15:00 | 96.25 | 94.70 | 94.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 12:15:00 | 96.20 | 96.28 | 95.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 13:00:00 | 96.20 | 96.28 | 95.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 95.60 | 96.14 | 95.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:30:00 | 95.80 | 96.14 | 95.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 96.15 | 96.14 | 95.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 11:00:00 | 96.60 | 96.15 | 95.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 11:45:00 | 96.75 | 96.18 | 95.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 12:15:00 | 96.50 | 96.18 | 95.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 13:00:00 | 96.45 | 96.23 | 95.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 13:15:00 | 100.90 | 101.56 | 100.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 13:45:00 | 101.00 | 101.56 | 100.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 14:15:00 | 100.60 | 101.37 | 100.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 14:30:00 | 100.55 | 101.37 | 100.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 15:15:00 | 100.50 | 101.19 | 100.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:15:00 | 100.50 | 101.19 | 100.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 100.90 | 100.89 | 100.74 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-24 12:15:00 | 99.90 | 100.56 | 100.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2024-05-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 12:15:00 | 99.90 | 100.56 | 100.61 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 14:15:00 | 102.25 | 100.80 | 100.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 09:15:00 | 107.30 | 102.39 | 101.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 13:15:00 | 106.25 | 106.61 | 105.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-28 14:00:00 | 106.25 | 106.61 | 105.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 104.65 | 106.04 | 105.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:15:00 | 104.85 | 106.04 | 105.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 104.55 | 105.74 | 105.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 11:15:00 | 106.15 | 105.65 | 105.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 12:00:00 | 105.85 | 105.69 | 105.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 12:45:00 | 105.70 | 105.66 | 105.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 13:30:00 | 105.70 | 105.54 | 105.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 104.70 | 105.37 | 105.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 15:00:00 | 104.70 | 105.37 | 105.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 15:15:00 | 104.45 | 105.19 | 105.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:15:00 | 103.40 | 105.19 | 105.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-05-30 09:15:00 | 102.25 | 104.60 | 104.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 102.25 | 104.60 | 104.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 11:15:00 | 102.20 | 103.81 | 104.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 102.65 | 101.56 | 102.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 13:15:00 | 102.65 | 101.56 | 102.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 102.65 | 101.56 | 102.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:00:00 | 102.65 | 101.56 | 102.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 103.60 | 101.97 | 102.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 103.60 | 101.97 | 102.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 103.45 | 102.26 | 102.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 105.85 | 102.26 | 102.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 104.95 | 103.37 | 103.16 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 09:15:00 | 99.15 | 102.94 | 103.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 10:15:00 | 92.70 | 100.89 | 102.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 12:15:00 | 90.40 | 89.71 | 93.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 12:30:00 | 90.00 | 89.71 | 93.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 94.65 | 90.94 | 93.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 94.65 | 90.94 | 93.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 94.80 | 91.71 | 93.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 11:30:00 | 93.80 | 92.04 | 93.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 15:00:00 | 94.00 | 92.87 | 93.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-07 10:15:00 | 95.75 | 93.93 | 93.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2024-06-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 10:15:00 | 95.75 | 93.93 | 93.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 14:15:00 | 97.50 | 95.46 | 94.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 09:15:00 | 121.12 | 122.73 | 118.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-19 09:30:00 | 121.51 | 122.73 | 118.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 15:15:00 | 123.95 | 125.47 | 123.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 09:15:00 | 124.25 | 125.47 | 123.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 123.64 | 125.11 | 123.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 09:30:00 | 124.80 | 125.11 | 123.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 123.18 | 124.72 | 123.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:45:00 | 122.96 | 124.72 | 123.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 11:15:00 | 123.70 | 124.52 | 123.74 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2024-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 15:15:00 | 121.30 | 122.98 | 123.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 09:15:00 | 118.88 | 122.16 | 122.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 11:15:00 | 116.49 | 116.08 | 117.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-26 12:00:00 | 116.49 | 116.08 | 117.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 115.90 | 115.48 | 116.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:30:00 | 115.99 | 115.48 | 116.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 114.22 | 113.23 | 114.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:45:00 | 113.89 | 113.23 | 114.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 116.70 | 113.92 | 114.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 11:00:00 | 116.70 | 113.92 | 114.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 114.20 | 113.98 | 114.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 12:15:00 | 113.77 | 113.98 | 114.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 13:30:00 | 114.05 | 114.10 | 114.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 14:30:00 | 113.90 | 113.93 | 114.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 15:00:00 | 113.29 | 113.93 | 114.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 117.21 | 114.43 | 114.76 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-01 09:15:00 | 117.21 | 114.43 | 114.76 | SL hit (close>static) qty=1.00 sl=116.80 alert=retest2 |

### Cycle 77 — BUY (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 11:15:00 | 116.52 | 115.24 | 115.10 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2024-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 15:15:00 | 114.48 | 115.04 | 115.06 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 09:15:00 | 115.27 | 115.09 | 115.08 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2024-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 10:15:00 | 114.91 | 115.05 | 115.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 11:15:00 | 113.78 | 114.80 | 114.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 10:15:00 | 114.95 | 113.82 | 114.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 10:15:00 | 114.95 | 113.82 | 114.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 114.95 | 113.82 | 114.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 11:00:00 | 114.95 | 113.82 | 114.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 11:15:00 | 114.40 | 113.94 | 114.25 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2024-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 14:15:00 | 119.20 | 115.27 | 114.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 125.26 | 117.87 | 116.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 10:15:00 | 130.20 | 130.33 | 127.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 14:15:00 | 126.90 | 129.20 | 128.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 126.90 | 129.20 | 128.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 15:00:00 | 126.90 | 129.20 | 128.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 15:15:00 | 126.50 | 128.66 | 127.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 09:15:00 | 127.25 | 128.66 | 127.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 09:15:00 | 123.81 | 127.69 | 127.58 | SL hit (close<static) qty=1.00 sl=126.00 alert=retest2 |

### Cycle 82 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 10:15:00 | 123.40 | 126.83 | 127.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 14:15:00 | 120.30 | 121.21 | 122.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 116.44 | 116.06 | 117.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 09:45:00 | 116.32 | 116.06 | 117.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 116.51 | 116.11 | 117.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:45:00 | 116.80 | 116.11 | 117.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 117.50 | 116.39 | 117.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 15:00:00 | 117.50 | 116.39 | 117.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 117.24 | 116.56 | 117.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 116.81 | 116.56 | 117.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 115.90 | 116.43 | 117.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 11:30:00 | 115.54 | 116.26 | 116.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 109.76 | 115.44 | 116.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 116.58 | 114.55 | 115.57 | SL hit (close>ema200) qty=0.50 sl=114.55 alert=retest2 |

### Cycle 83 — BUY (started 2024-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 13:15:00 | 118.85 | 116.02 | 115.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 09:15:00 | 121.55 | 118.14 | 117.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 14:15:00 | 119.01 | 119.24 | 118.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 14:15:00 | 119.01 | 119.24 | 118.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 119.01 | 119.24 | 118.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 14:30:00 | 118.17 | 119.24 | 118.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 117.95 | 118.99 | 118.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 09:45:00 | 119.13 | 119.00 | 118.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 11:15:00 | 121.62 | 118.98 | 118.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-30 09:15:00 | 131.04 | 128.12 | 124.79 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 128.00 | 132.67 | 133.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 10:15:00 | 124.30 | 130.99 | 132.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 128.37 | 126.76 | 129.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 128.37 | 126.76 | 129.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 128.37 | 126.76 | 129.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:45:00 | 129.35 | 126.76 | 129.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 126.95 | 125.62 | 127.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:15:00 | 129.00 | 125.62 | 127.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 129.05 | 126.31 | 127.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:45:00 | 130.95 | 126.31 | 127.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 128.77 | 126.80 | 127.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:45:00 | 129.12 | 126.80 | 127.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2024-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 13:15:00 | 131.50 | 128.00 | 127.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 14:15:00 | 136.29 | 129.66 | 128.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 14:15:00 | 135.67 | 136.82 | 133.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 15:00:00 | 135.67 | 136.82 | 133.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 10:15:00 | 136.49 | 136.21 | 134.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 10:45:00 | 138.41 | 135.07 | 134.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 12:15:00 | 139.14 | 135.62 | 134.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-14 11:15:00 | 133.08 | 136.47 | 136.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2024-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 11:15:00 | 133.08 | 136.47 | 136.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 13:15:00 | 132.32 | 135.11 | 136.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-19 09:15:00 | 135.50 | 132.91 | 133.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-19 09:15:00 | 135.50 | 132.91 | 133.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 135.50 | 132.91 | 133.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 10:00:00 | 135.50 | 132.91 | 133.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 10:15:00 | 135.13 | 133.36 | 133.95 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2024-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 13:15:00 | 135.80 | 134.60 | 134.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 14:15:00 | 137.21 | 135.12 | 134.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 12:15:00 | 145.47 | 145.74 | 143.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 13:00:00 | 145.47 | 145.74 | 143.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 144.46 | 145.46 | 143.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:00:00 | 144.46 | 145.46 | 143.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 144.83 | 145.10 | 143.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:45:00 | 144.60 | 145.10 | 143.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 14:15:00 | 146.25 | 147.82 | 146.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 15:00:00 | 146.25 | 147.82 | 146.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 15:15:00 | 146.04 | 147.46 | 146.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 09:15:00 | 147.39 | 147.46 | 146.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 09:45:00 | 146.67 | 147.27 | 146.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 11:15:00 | 146.38 | 147.00 | 146.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 11:45:00 | 146.30 | 146.87 | 146.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 12:15:00 | 147.06 | 146.91 | 146.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 12:30:00 | 146.06 | 146.91 | 146.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 13:15:00 | 147.90 | 147.11 | 146.72 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-28 15:15:00 | 145.80 | 146.85 | 146.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2024-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 15:15:00 | 145.80 | 146.85 | 146.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 10:15:00 | 143.35 | 145.98 | 146.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 13:15:00 | 146.57 | 145.45 | 146.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 13:15:00 | 146.57 | 145.45 | 146.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 13:15:00 | 146.57 | 145.45 | 146.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 13:45:00 | 146.58 | 145.45 | 146.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 146.40 | 145.64 | 146.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 15:15:00 | 147.50 | 145.64 | 146.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 147.50 | 146.01 | 146.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 147.60 | 146.01 | 146.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 147.10 | 146.23 | 146.28 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2024-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 10:15:00 | 147.09 | 146.40 | 146.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 14:15:00 | 149.01 | 147.06 | 146.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 09:15:00 | 147.44 | 147.44 | 146.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 10:00:00 | 147.44 | 147.44 | 146.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 147.38 | 147.43 | 146.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 14:45:00 | 148.07 | 147.18 | 146.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-02 15:15:00 | 146.50 | 147.04 | 146.94 | SL hit (close<static) qty=1.00 sl=146.75 alert=retest2 |

### Cycle 90 — SELL (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 09:15:00 | 150.93 | 153.72 | 153.98 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 12:15:00 | 154.39 | 153.40 | 153.34 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 12:15:00 | 153.01 | 153.42 | 153.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 13:15:00 | 147.71 | 152.28 | 152.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 12:15:00 | 150.16 | 149.32 | 150.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 12:15:00 | 150.16 | 149.32 | 150.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 150.16 | 149.32 | 150.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 13:00:00 | 150.16 | 149.32 | 150.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 13:15:00 | 150.99 | 149.66 | 150.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 13:45:00 | 151.44 | 149.66 | 150.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 150.50 | 149.82 | 150.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 15:00:00 | 150.50 | 149.82 | 150.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 152.70 | 150.43 | 150.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:45:00 | 154.25 | 150.43 | 150.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 10:15:00 | 152.31 | 150.80 | 151.03 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2024-09-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 12:15:00 | 152.24 | 151.37 | 151.26 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 09:15:00 | 150.03 | 151.03 | 151.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 10:15:00 | 149.40 | 150.71 | 150.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 09:15:00 | 152.56 | 150.53 | 150.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 09:15:00 | 152.56 | 150.53 | 150.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 152.56 | 150.53 | 150.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:30:00 | 152.57 | 150.53 | 150.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2024-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 10:15:00 | 153.59 | 151.14 | 150.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-18 09:15:00 | 154.88 | 153.29 | 152.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 10:15:00 | 154.27 | 156.39 | 154.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 10:15:00 | 154.27 | 156.39 | 154.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 154.27 | 156.39 | 154.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:00:00 | 154.27 | 156.39 | 154.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 153.57 | 155.82 | 154.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:30:00 | 152.52 | 155.82 | 154.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 13:15:00 | 153.06 | 154.96 | 154.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 13:45:00 | 153.19 | 154.96 | 154.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 154.94 | 154.96 | 154.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 09:15:00 | 159.23 | 154.95 | 154.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-24 13:15:00 | 157.19 | 159.37 | 159.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2024-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 13:15:00 | 157.19 | 159.37 | 159.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 14:15:00 | 156.44 | 158.78 | 159.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 10:15:00 | 157.91 | 157.82 | 158.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-25 10:45:00 | 157.99 | 157.82 | 158.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 157.50 | 157.51 | 158.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 13:45:00 | 157.72 | 157.51 | 158.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 158.88 | 157.78 | 158.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 15:00:00 | 158.88 | 157.78 | 158.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 157.66 | 157.76 | 158.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 09:15:00 | 156.80 | 157.76 | 158.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 09:15:00 | 148.96 | 152.16 | 154.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-01 09:15:00 | 150.83 | 150.17 | 152.03 | SL hit (close>ema200) qty=0.50 sl=150.17 alert=retest2 |

### Cycle 97 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 141.28 | 138.95 | 138.69 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 10:15:00 | 138.30 | 139.06 | 139.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 14:15:00 | 137.50 | 138.55 | 138.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 09:15:00 | 140.12 | 138.65 | 138.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 09:15:00 | 140.12 | 138.65 | 138.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 140.12 | 138.65 | 138.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 11:30:00 | 138.24 | 138.61 | 138.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 12:15:00 | 137.98 | 138.61 | 138.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 131.33 | 133.23 | 134.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 131.08 | 133.23 | 134.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-21 15:15:00 | 124.42 | 127.26 | 129.88 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 99 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 119.44 | 117.21 | 117.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 10:15:00 | 119.95 | 117.76 | 117.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 14:15:00 | 118.80 | 118.94 | 118.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 15:00:00 | 118.80 | 118.94 | 118.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 118.06 | 118.76 | 118.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:30:00 | 118.16 | 118.76 | 118.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 117.91 | 118.59 | 118.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 13:30:00 | 119.84 | 118.60 | 118.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 11:30:00 | 118.66 | 119.53 | 119.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 13:00:00 | 118.98 | 119.42 | 119.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 13:45:00 | 118.75 | 119.32 | 119.12 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 14:15:00 | 119.33 | 119.33 | 119.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 14:30:00 | 119.14 | 119.33 | 119.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 15:15:00 | 118.84 | 119.23 | 119.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:15:00 | 120.32 | 119.23 | 119.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 121.34 | 119.65 | 119.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 10:15:00 | 122.64 | 119.65 | 119.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 11:00:00 | 121.83 | 120.09 | 119.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 12:00:00 | 121.79 | 120.43 | 119.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-06 09:15:00 | 130.53 | 125.42 | 122.73 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2024-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 15:15:00 | 125.67 | 127.85 | 127.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 123.11 | 126.90 | 127.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 124.03 | 123.55 | 125.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 124.03 | 123.55 | 125.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 124.72 | 123.47 | 124.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:00:00 | 124.72 | 123.47 | 124.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 125.80 | 123.93 | 124.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:30:00 | 126.89 | 123.93 | 124.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 14:15:00 | 126.42 | 124.43 | 124.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 14:45:00 | 126.78 | 124.43 | 124.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2024-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 09:15:00 | 128.35 | 125.62 | 125.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 130.39 | 127.96 | 126.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 15:15:00 | 127.92 | 128.83 | 127.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 15:15:00 | 127.92 | 128.83 | 127.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 127.92 | 128.83 | 127.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:15:00 | 127.73 | 128.83 | 127.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 126.65 | 128.39 | 127.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:30:00 | 127.41 | 128.39 | 127.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 127.20 | 128.15 | 127.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 13:15:00 | 127.67 | 127.86 | 127.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-21 14:15:00 | 126.92 | 127.52 | 127.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2024-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 14:15:00 | 126.92 | 127.52 | 127.56 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2024-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 09:15:00 | 129.35 | 127.86 | 127.70 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-11-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 15:15:00 | 126.31 | 127.58 | 127.70 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2024-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 11:15:00 | 127.90 | 127.77 | 127.76 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-25 14:15:00 | 127.26 | 127.69 | 127.73 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2024-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 14:15:00 | 128.53 | 127.66 | 127.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 09:15:00 | 130.39 | 128.25 | 127.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 09:15:00 | 131.71 | 132.78 | 131.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-29 10:00:00 | 131.71 | 132.78 | 131.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 130.18 | 132.26 | 131.53 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2024-11-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 14:15:00 | 129.15 | 130.94 | 131.08 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 11:15:00 | 130.87 | 130.68 | 130.67 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2024-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 12:15:00 | 130.42 | 130.63 | 130.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 10:15:00 | 129.63 | 130.34 | 130.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 09:15:00 | 130.35 | 129.79 | 130.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 09:15:00 | 130.35 | 129.79 | 130.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 130.35 | 129.79 | 130.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:30:00 | 130.57 | 129.79 | 130.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 130.60 | 129.95 | 130.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 12:30:00 | 129.69 | 130.06 | 130.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 13:15:00 | 129.73 | 130.06 | 130.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 14:30:00 | 129.73 | 129.92 | 130.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 12:00:00 | 129.87 | 129.80 | 129.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-06 12:15:00 | 134.18 | 130.68 | 130.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2024-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 12:15:00 | 134.18 | 130.68 | 130.32 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 09:15:00 | 127.86 | 130.44 | 130.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 10:15:00 | 127.60 | 129.87 | 130.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 125.33 | 124.49 | 125.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 14:15:00 | 125.33 | 124.49 | 125.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 125.33 | 124.49 | 125.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 125.33 | 124.49 | 125.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 125.40 | 124.67 | 125.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 126.77 | 124.67 | 125.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 125.77 | 124.89 | 125.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 126.51 | 124.89 | 125.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 125.40 | 124.99 | 125.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 11:45:00 | 125.17 | 125.00 | 125.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 10:15:00 | 124.87 | 125.40 | 125.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 14:15:00 | 118.91 | 120.78 | 122.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 15:15:00 | 118.63 | 120.42 | 122.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-12-23 09:15:00 | 112.65 | 114.93 | 117.02 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 113 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 115.50 | 114.70 | 114.63 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2024-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 11:15:00 | 113.91 | 114.58 | 114.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-27 14:15:00 | 112.99 | 114.01 | 114.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 09:15:00 | 113.80 | 113.79 | 114.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-30 10:00:00 | 113.80 | 113.79 | 114.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 112.92 | 113.61 | 114.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 10:45:00 | 113.38 | 113.61 | 114.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 112.93 | 111.95 | 112.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:00:00 | 112.93 | 111.95 | 112.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 112.80 | 112.12 | 112.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:30:00 | 113.13 | 112.12 | 112.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 15:15:00 | 112.84 | 112.26 | 112.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:15:00 | 113.47 | 112.26 | 112.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 113.25 | 112.46 | 112.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:30:00 | 113.10 | 112.46 | 112.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 113.23 | 112.61 | 112.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 10:45:00 | 113.20 | 112.61 | 112.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 12:15:00 | 112.93 | 112.74 | 112.76 | EMA400 retest candle locked (from downside) |

### Cycle 115 — BUY (started 2025-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 13:15:00 | 113.35 | 112.86 | 112.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 14:15:00 | 114.40 | 113.17 | 112.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 113.20 | 113.27 | 113.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-02 10:00:00 | 113.20 | 113.27 | 113.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 112.95 | 113.21 | 113.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:30:00 | 113.09 | 113.21 | 113.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 112.78 | 113.12 | 113.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 11:45:00 | 112.61 | 113.12 | 113.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 113.70 | 114.54 | 114.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 112.28 | 114.54 | 114.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 112.32 | 114.10 | 113.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 112.50 | 114.10 | 113.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 109.93 | 113.26 | 113.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 11:15:00 | 109.65 | 112.54 | 113.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 12:15:00 | 109.25 | 109.20 | 110.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 12:45:00 | 109.00 | 109.20 | 110.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 98.19 | 97.58 | 98.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 15:00:00 | 97.11 | 98.09 | 98.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 09:15:00 | 100.90 | 98.57 | 98.80 | SL hit (close>static) qty=1.00 sl=100.50 alert=retest2 |

### Cycle 117 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 100.79 | 99.01 | 98.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 14:15:00 | 102.00 | 100.26 | 99.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 103.32 | 104.70 | 103.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 09:15:00 | 103.32 | 104.70 | 103.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 103.32 | 104.70 | 103.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 103.32 | 104.70 | 103.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 102.11 | 104.18 | 103.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 102.11 | 104.18 | 103.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 102.73 | 103.71 | 103.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 12:45:00 | 102.73 | 103.71 | 103.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 14:15:00 | 101.67 | 103.08 | 103.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 98.96 | 101.99 | 102.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 99.39 | 99.37 | 100.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 99.39 | 99.37 | 100.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 101.87 | 99.87 | 100.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 101.87 | 99.87 | 100.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 101.63 | 100.22 | 100.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:45:00 | 102.02 | 100.22 | 100.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 101.57 | 100.71 | 101.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 13:00:00 | 101.57 | 100.71 | 101.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 15:15:00 | 100.75 | 100.87 | 101.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:15:00 | 101.67 | 100.87 | 101.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 99.62 | 100.62 | 100.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 14:00:00 | 99.40 | 100.35 | 100.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:15:00 | 94.43 | 98.05 | 99.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 12:15:00 | 95.24 | 94.56 | 96.33 | SL hit (close>ema200) qty=0.50 sl=94.56 alert=retest2 |

### Cycle 119 — BUY (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 11:15:00 | 97.99 | 96.24 | 96.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 09:15:00 | 98.80 | 97.51 | 96.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 09:15:00 | 102.14 | 102.87 | 100.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-03 09:30:00 | 101.95 | 102.87 | 100.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 15:15:00 | 101.19 | 102.11 | 101.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:15:00 | 100.02 | 102.11 | 101.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 102.15 | 102.12 | 101.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 09:15:00 | 104.25 | 100.91 | 100.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-06 12:15:00 | 100.40 | 101.57 | 101.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2025-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 12:15:00 | 100.40 | 101.57 | 101.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 13:15:00 | 99.70 | 101.20 | 101.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 10:15:00 | 100.88 | 100.66 | 101.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-07 11:00:00 | 100.88 | 100.66 | 101.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 101.10 | 100.74 | 101.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:45:00 | 101.88 | 100.74 | 101.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 12:15:00 | 101.42 | 100.88 | 101.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 13:00:00 | 101.42 | 100.88 | 101.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 13:15:00 | 99.94 | 100.69 | 100.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 99.70 | 100.67 | 100.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 10:15:00 | 94.72 | 97.28 | 98.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-12 09:15:00 | 89.73 | 93.11 | 95.80 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 121 — BUY (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 13:15:00 | 88.11 | 86.59 | 86.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 89.07 | 87.08 | 86.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 10:15:00 | 89.39 | 89.91 | 88.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 11:00:00 | 89.39 | 89.91 | 88.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 89.27 | 89.69 | 89.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:45:00 | 89.19 | 89.69 | 89.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 89.63 | 89.67 | 89.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 14:30:00 | 89.12 | 89.67 | 89.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 89.01 | 89.54 | 89.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 87.39 | 89.54 | 89.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 87.40 | 89.11 | 88.92 | EMA400 retest candle locked (from upside) |

### Cycle 122 — SELL (started 2025-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 11:15:00 | 87.95 | 88.66 | 88.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 15:15:00 | 87.34 | 88.02 | 88.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 88.31 | 88.08 | 88.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 09:15:00 | 88.31 | 88.08 | 88.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 88.31 | 88.08 | 88.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 10:00:00 | 88.31 | 88.08 | 88.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 87.96 | 88.06 | 88.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 11:15:00 | 87.50 | 88.06 | 88.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 11:45:00 | 87.05 | 87.87 | 88.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 14:15:00 | 83.12 | 84.47 | 85.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 82.70 | 83.60 | 85.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-28 12:15:00 | 78.75 | 81.71 | 83.91 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 123 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 82.73 | 81.22 | 81.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 83.62 | 82.10 | 81.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 14:15:00 | 83.02 | 83.23 | 82.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 15:00:00 | 83.02 | 83.23 | 82.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 83.79 | 83.26 | 82.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 10:15:00 | 84.33 | 83.26 | 82.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 14:30:00 | 84.25 | 83.53 | 83.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:30:00 | 84.00 | 83.54 | 83.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 11:15:00 | 82.17 | 83.15 | 83.03 | SL hit (close<static) qty=1.00 sl=82.42 alert=retest2 |

### Cycle 124 — SELL (started 2025-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 12:15:00 | 82.01 | 82.92 | 82.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 13:15:00 | 81.52 | 82.64 | 82.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 15:15:00 | 80.80 | 80.50 | 81.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 09:15:00 | 80.11 | 80.50 | 81.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 79.13 | 80.23 | 81.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:30:00 | 81.08 | 80.23 | 81.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 79.42 | 79.31 | 80.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 13:00:00 | 78.21 | 78.97 | 79.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 14:00:00 | 78.19 | 78.81 | 79.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 10:00:00 | 78.02 | 78.29 | 79.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 11:00:00 | 77.81 | 78.20 | 78.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 79.34 | 78.18 | 78.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 79.34 | 78.18 | 78.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 78.97 | 78.34 | 78.63 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-18 12:15:00 | 79.96 | 78.90 | 78.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 12:15:00 | 79.96 | 78.90 | 78.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 13:15:00 | 80.16 | 79.16 | 78.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 81.25 | 81.38 | 80.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 81.25 | 81.38 | 80.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 83.98 | 82.06 | 81.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 09:15:00 | 84.74 | 82.96 | 82.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 09:45:00 | 84.29 | 83.32 | 82.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 12:15:00 | 81.60 | 82.98 | 83.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 12:15:00 | 81.60 | 82.98 | 83.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 14:15:00 | 81.24 | 82.49 | 82.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 09:15:00 | 82.26 | 82.23 | 82.59 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 13:30:00 | 80.81 | 81.57 | 82.14 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 14:15:00 | 80.80 | 81.57 | 82.14 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-27 11:30:00 | 80.77 | 80.87 | 81.51 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-27 13:45:00 | 80.63 | 80.71 | 81.32 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 81.30 | 80.83 | 81.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 81.30 | 80.83 | 81.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 81.38 | 80.94 | 81.33 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-27 15:15:00 | 81.38 | 80.94 | 81.33 | SL hit (close>ema400) qty=1.00 sl=81.33 alert=retest1 |

### Cycle 127 — BUY (started 2025-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 13:15:00 | 81.00 | 80.75 | 80.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 14:15:00 | 81.75 | 80.95 | 80.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 79.65 | 82.07 | 81.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 79.65 | 82.07 | 81.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 79.65 | 82.07 | 81.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 79.65 | 82.07 | 81.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 80.01 | 81.66 | 81.57 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 79.45 | 81.22 | 81.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 79.03 | 80.52 | 81.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 75.55 | 75.19 | 77.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 75.55 | 75.19 | 77.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 75.55 | 75.19 | 77.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 74.89 | 75.30 | 76.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 74.75 | 75.92 | 76.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 14:00:00 | 74.99 | 75.21 | 75.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 09:15:00 | 79.82 | 76.68 | 76.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 79.82 | 76.68 | 76.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 10:15:00 | 80.60 | 77.47 | 76.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 14:15:00 | 85.65 | 85.93 | 84.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 15:00:00 | 85.65 | 85.93 | 84.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 83.74 | 85.44 | 84.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 83.74 | 85.44 | 84.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 84.54 | 85.26 | 84.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 83.66 | 85.26 | 84.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 84.91 | 85.19 | 84.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 11:30:00 | 85.05 | 85.19 | 84.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 85.06 | 85.17 | 84.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 14:15:00 | 85.37 | 85.18 | 84.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 14:45:00 | 85.73 | 85.30 | 84.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 11:30:00 | 85.70 | 85.56 | 85.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 82.78 | 84.96 | 85.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 82.78 | 84.96 | 85.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 81.39 | 84.24 | 84.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 12:15:00 | 82.18 | 82.14 | 83.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 12:45:00 | 82.55 | 82.14 | 83.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 82.14 | 82.03 | 82.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 10:15:00 | 81.73 | 82.03 | 82.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 11:00:00 | 81.90 | 82.00 | 82.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 12:15:00 | 77.64 | 79.01 | 79.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 12:15:00 | 77.81 | 79.01 | 79.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-07 11:15:00 | 77.60 | 77.37 | 78.31 | SL hit (close>ema200) qty=0.50 sl=77.37 alert=retest2 |

### Cycle 131 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 81.07 | 76.74 | 76.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 12:15:00 | 82.83 | 81.45 | 79.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 11:15:00 | 88.80 | 88.93 | 86.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 12:00:00 | 88.80 | 88.93 | 86.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 87.86 | 89.49 | 88.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 87.85 | 89.49 | 88.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 87.50 | 89.09 | 88.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:30:00 | 87.60 | 89.09 | 88.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 86.80 | 88.29 | 88.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 86.40 | 87.91 | 88.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 15:15:00 | 85.50 | 85.36 | 86.29 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-22 10:15:00 | 84.80 | 85.26 | 86.16 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 85.17 | 84.68 | 85.33 | EMA400 retest candle locked (from downside) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-23 09:15:00 | 80.56 | 84.68 | 85.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-05-23 09:15:00 | 85.17 | 84.68 | 85.33 | SL hit (close>ema400) qty=0.50 sl=84.68 alert=retest1 |

### Cycle 133 — BUY (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 14:15:00 | 87.23 | 85.90 | 85.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 88.21 | 87.13 | 86.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 14:15:00 | 87.78 | 87.99 | 87.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 15:00:00 | 87.78 | 87.99 | 87.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 87.60 | 87.91 | 87.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 11:15:00 | 88.35 | 87.77 | 87.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 86.61 | 87.80 | 87.61 | SL hit (close<static) qty=1.00 sl=87.31 alert=retest2 |

### Cycle 134 — SELL (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 11:15:00 | 86.92 | 87.42 | 87.46 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 14:15:00 | 87.71 | 87.52 | 87.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 09:15:00 | 90.21 | 88.05 | 87.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 10:15:00 | 88.39 | 88.80 | 88.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 10:15:00 | 88.39 | 88.80 | 88.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 88.39 | 88.80 | 88.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:00:00 | 88.39 | 88.80 | 88.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 11:15:00 | 88.08 | 88.65 | 88.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:30:00 | 88.06 | 88.65 | 88.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 87.49 | 88.42 | 88.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:00:00 | 87.49 | 88.42 | 88.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 87.59 | 88.13 | 88.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 15:15:00 | 87.44 | 87.99 | 88.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 10:15:00 | 88.12 | 87.89 | 88.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 10:15:00 | 88.12 | 87.89 | 88.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 88.12 | 87.89 | 88.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:00:00 | 88.12 | 87.89 | 88.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 89.08 | 88.12 | 88.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:00:00 | 89.08 | 88.12 | 88.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2025-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 12:15:00 | 90.44 | 88.59 | 88.34 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 10:15:00 | 88.05 | 88.86 | 88.87 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 11:15:00 | 90.30 | 89.14 | 89.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 12:15:00 | 91.55 | 89.63 | 89.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 12:15:00 | 91.93 | 92.22 | 91.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 12:30:00 | 92.11 | 92.22 | 91.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 91.81 | 92.08 | 91.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 14:30:00 | 91.60 | 92.08 | 91.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 91.30 | 92.29 | 91.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 91.45 | 92.29 | 91.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 91.65 | 92.17 | 91.91 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 90.94 | 91.69 | 91.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 90.41 | 91.44 | 91.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 83.28 | 83.20 | 84.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-18 10:00:00 | 83.28 | 83.20 | 84.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 81.54 | 80.67 | 81.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 81.54 | 80.67 | 81.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 81.44 | 80.82 | 81.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:45:00 | 81.61 | 80.82 | 81.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 81.81 | 81.10 | 81.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 81.81 | 81.10 | 81.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 81.62 | 81.20 | 81.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:30:00 | 81.29 | 81.33 | 81.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 82.23 | 81.51 | 81.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:00:00 | 82.23 | 81.51 | 81.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 82.25 | 81.66 | 81.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 14:15:00 | 81.80 | 81.85 | 81.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 83.42 | 82.13 | 81.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 83.42 | 82.13 | 81.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 10:15:00 | 84.26 | 82.56 | 82.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 12:15:00 | 86.98 | 87.15 | 86.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 13:00:00 | 86.98 | 87.15 | 86.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 86.55 | 86.96 | 86.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:00:00 | 86.55 | 86.96 | 86.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 86.50 | 86.87 | 86.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:45:00 | 86.35 | 86.87 | 86.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 86.72 | 86.78 | 86.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:30:00 | 86.18 | 86.54 | 86.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 85.36 | 86.30 | 86.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 11:15:00 | 84.84 | 85.41 | 85.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 85.26 | 85.05 | 85.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 85.26 | 85.05 | 85.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 85.26 | 85.05 | 85.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:45:00 | 85.53 | 85.05 | 85.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 84.16 | 84.57 | 84.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 84.50 | 84.57 | 84.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 82.89 | 83.06 | 83.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 12:30:00 | 82.70 | 82.98 | 83.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 13:00:00 | 82.65 | 82.98 | 83.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 13:30:00 | 82.61 | 82.91 | 83.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 14:00:00 | 82.61 | 82.91 | 83.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 82.74 | 82.77 | 83.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:30:00 | 83.08 | 82.77 | 83.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 81.96 | 82.20 | 82.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:30:00 | 82.70 | 82.20 | 82.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 82.34 | 81.72 | 82.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:00:00 | 82.34 | 81.72 | 82.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 81.94 | 81.77 | 82.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:45:00 | 82.29 | 81.77 | 82.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 81.86 | 81.75 | 82.00 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 82.66 | 82.18 | 82.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 82.66 | 82.18 | 82.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 13:15:00 | 83.00 | 82.69 | 82.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 11:15:00 | 82.98 | 83.01 | 82.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 12:00:00 | 82.98 | 83.01 | 82.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 82.70 | 82.92 | 82.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 14:15:00 | 82.54 | 82.92 | 82.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 14:15:00 | 82.65 | 82.86 | 82.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 15:00:00 | 82.65 | 82.86 | 82.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 82.63 | 82.82 | 82.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:15:00 | 82.71 | 82.82 | 82.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 81.91 | 82.64 | 82.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 10:15:00 | 81.45 | 82.40 | 82.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 81.86 | 81.85 | 82.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 10:00:00 | 81.86 | 81.85 | 82.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 81.90 | 81.86 | 82.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 11:15:00 | 81.75 | 81.86 | 82.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 13:45:00 | 81.76 | 81.91 | 82.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 14:15:00 | 81.78 | 81.91 | 82.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 15:00:00 | 81.78 | 81.88 | 82.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 81.30 | 81.77 | 81.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:30:00 | 81.18 | 81.58 | 81.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 11:15:00 | 77.66 | 78.92 | 79.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 11:15:00 | 77.67 | 78.92 | 79.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 11:15:00 | 77.69 | 78.92 | 79.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 11:15:00 | 77.69 | 78.92 | 79.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 12:15:00 | 77.12 | 78.59 | 79.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 77.83 | 77.44 | 78.57 | SL hit (close>ema200) qty=0.50 sl=77.44 alert=retest2 |

### Cycle 145 — BUY (started 2025-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 12:15:00 | 72.72 | 72.15 | 72.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 13:15:00 | 73.00 | 72.32 | 72.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 11:15:00 | 74.50 | 75.03 | 74.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 11:15:00 | 74.50 | 75.03 | 74.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 74.50 | 75.03 | 74.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 74.50 | 75.03 | 74.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 74.04 | 74.83 | 74.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:00:00 | 74.04 | 74.83 | 74.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 73.41 | 74.55 | 74.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:00:00 | 73.41 | 74.55 | 74.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2025-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 14:15:00 | 72.80 | 74.20 | 74.21 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 12:15:00 | 74.46 | 73.96 | 73.93 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 72.99 | 73.90 | 73.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 13:15:00 | 71.99 | 73.01 | 73.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 14:15:00 | 70.03 | 69.95 | 70.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 15:00:00 | 70.03 | 69.95 | 70.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 71.11 | 70.17 | 70.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 71.11 | 70.17 | 70.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 70.96 | 70.33 | 70.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:30:00 | 71.43 | 70.33 | 70.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 71.10 | 70.48 | 70.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:15:00 | 71.33 | 70.48 | 70.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 71.33 | 70.65 | 70.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:30:00 | 71.29 | 70.65 | 70.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 71.34 | 71.07 | 71.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 11:15:00 | 71.56 | 71.17 | 71.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 12:15:00 | 71.01 | 71.14 | 71.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 12:15:00 | 71.01 | 71.14 | 71.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 12:15:00 | 71.01 | 71.14 | 71.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 12:45:00 | 71.03 | 71.14 | 71.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — SELL (started 2025-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 13:15:00 | 70.24 | 70.96 | 71.02 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2025-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 13:15:00 | 71.66 | 71.10 | 71.04 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 12:15:00 | 70.79 | 71.09 | 71.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 70.30 | 70.93 | 71.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 70.00 | 69.96 | 70.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 70.00 | 69.96 | 70.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 70.00 | 69.96 | 70.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 70.00 | 69.96 | 70.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 70.52 | 70.06 | 70.32 | EMA400 retest candle locked (from downside) |

### Cycle 153 — BUY (started 2025-09-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 14:15:00 | 70.91 | 70.44 | 70.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 15:15:00 | 71.65 | 70.68 | 70.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 70.67 | 70.68 | 70.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 70.67 | 70.68 | 70.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 70.67 | 70.68 | 70.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:30:00 | 70.68 | 70.68 | 70.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 70.23 | 70.59 | 70.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:00:00 | 70.23 | 70.59 | 70.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 70.22 | 70.51 | 70.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:00:00 | 70.22 | 70.51 | 70.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — SELL (started 2025-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 12:15:00 | 70.14 | 70.44 | 70.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 13:15:00 | 70.04 | 70.36 | 70.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 70.84 | 70.36 | 70.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 70.84 | 70.36 | 70.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 70.84 | 70.36 | 70.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:00:00 | 70.84 | 70.36 | 70.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 71.26 | 70.54 | 70.48 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 70.13 | 70.64 | 70.64 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 72.97 | 71.02 | 70.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 73.62 | 72.67 | 71.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 13:15:00 | 75.85 | 75.99 | 74.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 14:00:00 | 75.85 | 75.99 | 74.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 75.12 | 75.82 | 74.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 15:00:00 | 75.12 | 75.82 | 74.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 76.32 | 76.93 | 76.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:45:00 | 76.30 | 76.93 | 76.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 76.70 | 76.88 | 76.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 11:30:00 | 76.55 | 76.88 | 76.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 76.44 | 76.71 | 76.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:45:00 | 76.49 | 76.71 | 76.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 76.40 | 76.65 | 76.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:15:00 | 76.11 | 76.65 | 76.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 76.00 | 76.52 | 76.34 | EMA400 retest candle locked (from upside) |

### Cycle 158 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 11:15:00 | 75.89 | 76.22 | 76.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 74.90 | 75.75 | 75.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 14:15:00 | 75.06 | 74.96 | 75.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 15:00:00 | 75.06 | 74.96 | 75.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 75.70 | 75.11 | 75.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:15:00 | 74.78 | 75.11 | 75.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 75.70 | 75.23 | 75.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:15:00 | 75.94 | 75.23 | 75.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 75.40 | 75.26 | 75.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 11:15:00 | 75.30 | 75.26 | 75.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 12:00:00 | 75.25 | 75.26 | 75.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 14:15:00 | 75.70 | 75.50 | 75.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 14:15:00 | 75.70 | 75.50 | 75.49 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 15:15:00 | 75.31 | 75.46 | 75.48 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 09:15:00 | 76.61 | 75.69 | 75.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 11:15:00 | 76.79 | 76.07 | 75.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 14:15:00 | 76.38 | 76.45 | 76.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-25 15:00:00 | 76.38 | 76.45 | 76.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 74.26 | 75.97 | 75.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:30:00 | 74.05 | 75.97 | 75.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — SELL (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 10:15:00 | 74.07 | 75.59 | 75.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 73.28 | 75.13 | 75.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 75.24 | 73.85 | 74.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 75.24 | 73.85 | 74.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 75.24 | 73.85 | 74.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:00:00 | 75.24 | 73.85 | 74.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 74.30 | 73.94 | 74.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 73.91 | 73.85 | 74.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 15:00:00 | 73.99 | 74.04 | 74.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 12:15:00 | 74.79 | 73.61 | 73.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — BUY (started 2025-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 12:15:00 | 74.79 | 73.61 | 73.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 75.90 | 74.34 | 73.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 74.47 | 74.67 | 74.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 10:45:00 | 74.47 | 74.67 | 74.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 74.63 | 74.66 | 74.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:45:00 | 74.20 | 74.66 | 74.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 74.48 | 74.59 | 74.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 10:00:00 | 74.78 | 74.59 | 74.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 12:30:00 | 74.63 | 74.57 | 74.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 15:15:00 | 74.67 | 74.59 | 74.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 74.26 | 74.54 | 74.45 | SL hit (close<static) qty=1.00 sl=74.29 alert=retest2 |

### Cycle 164 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 73.40 | 74.31 | 74.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 73.13 | 73.84 | 74.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 73.86 | 73.74 | 74.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 73.86 | 73.74 | 74.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 73.86 | 73.74 | 74.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 11:00:00 | 73.74 | 73.74 | 73.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 12:30:00 | 73.84 | 73.88 | 74.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 77.04 | 74.60 | 74.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 77.04 | 74.60 | 74.30 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 15:15:00 | 74.78 | 75.42 | 75.46 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 76.79 | 75.69 | 75.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 11:15:00 | 77.11 | 76.15 | 75.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 12:15:00 | 77.01 | 77.11 | 76.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 13:00:00 | 77.01 | 77.11 | 76.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 77.24 | 77.33 | 76.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:45:00 | 77.19 | 77.33 | 76.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 77.20 | 77.31 | 76.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:30:00 | 76.98 | 77.31 | 76.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 75.95 | 77.03 | 76.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 75.95 | 77.03 | 76.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 76.36 | 76.90 | 76.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 74.98 | 76.90 | 76.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — SELL (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 14:15:00 | 75.45 | 76.61 | 76.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 15:15:00 | 75.21 | 76.33 | 76.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 12:15:00 | 76.21 | 75.98 | 76.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 12:15:00 | 76.21 | 75.98 | 76.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 76.21 | 75.98 | 76.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:00:00 | 76.21 | 75.98 | 76.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 76.26 | 76.03 | 76.28 | EMA400 retest candle locked (from downside) |

### Cycle 169 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 77.15 | 76.42 | 76.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 09:15:00 | 77.99 | 77.10 | 76.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 09:15:00 | 77.41 | 77.60 | 77.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 77.41 | 77.60 | 77.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 77.41 | 77.60 | 77.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:30:00 | 77.25 | 77.60 | 77.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 77.08 | 77.50 | 77.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:00:00 | 77.08 | 77.50 | 77.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 77.07 | 77.41 | 77.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:30:00 | 77.48 | 77.41 | 77.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 77.00 | 77.33 | 77.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:30:00 | 76.92 | 77.33 | 77.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 77.08 | 77.28 | 77.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:30:00 | 77.12 | 77.28 | 77.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — SELL (started 2025-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 14:15:00 | 76.76 | 77.18 | 77.19 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 15:15:00 | 77.25 | 77.19 | 77.19 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 09:15:00 | 76.35 | 77.02 | 77.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 12:15:00 | 76.20 | 76.65 | 76.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 10:15:00 | 77.11 | 76.30 | 76.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 10:15:00 | 77.11 | 76.30 | 76.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 77.11 | 76.30 | 76.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:45:00 | 77.12 | 76.30 | 76.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 77.40 | 76.52 | 76.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:45:00 | 77.22 | 76.52 | 76.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 77.03 | 76.69 | 76.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:30:00 | 77.04 | 76.69 | 76.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — BUY (started 2025-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 15:15:00 | 77.26 | 76.80 | 76.76 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 09:15:00 | 75.28 | 76.50 | 76.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 13:15:00 | 74.95 | 75.75 | 76.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 74.93 | 74.32 | 74.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 74.93 | 74.32 | 74.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 74.93 | 74.32 | 74.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 74.93 | 74.32 | 74.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 76.26 | 74.71 | 75.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:00:00 | 76.26 | 74.71 | 75.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 77.01 | 75.17 | 75.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:00:00 | 77.01 | 75.17 | 75.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — BUY (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 12:15:00 | 76.76 | 75.49 | 75.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 14:15:00 | 77.27 | 76.08 | 75.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 12:15:00 | 77.17 | 77.23 | 76.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 12:30:00 | 77.05 | 77.23 | 76.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 77.04 | 77.11 | 76.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:45:00 | 76.55 | 77.11 | 76.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 75.54 | 76.78 | 76.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 75.54 | 76.78 | 76.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 75.01 | 76.43 | 76.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:45:00 | 75.19 | 76.43 | 76.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 74.89 | 76.12 | 76.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 14:15:00 | 74.72 | 75.68 | 76.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 74.92 | 74.70 | 75.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:00:00 | 74.92 | 74.70 | 75.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 74.78 | 74.72 | 75.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 74.83 | 74.72 | 75.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 74.81 | 74.74 | 75.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:30:00 | 75.14 | 74.74 | 75.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 74.45 | 74.69 | 75.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 75.10 | 74.69 | 75.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 74.63 | 74.74 | 75.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 13:30:00 | 74.53 | 74.71 | 74.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 14:30:00 | 74.47 | 74.61 | 74.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 11:15:00 | 75.69 | 74.86 | 74.93 | SL hit (close>static) qty=1.00 sl=75.12 alert=retest2 |

### Cycle 177 — BUY (started 2025-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 12:15:00 | 75.56 | 75.00 | 74.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 13:15:00 | 76.82 | 75.36 | 75.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 13:15:00 | 77.21 | 77.30 | 76.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 13:30:00 | 77.32 | 77.30 | 76.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 76.59 | 77.19 | 76.60 | EMA400 retest candle locked (from upside) |

### Cycle 178 — SELL (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 15:15:00 | 75.60 | 76.31 | 76.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 74.94 | 75.86 | 76.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 11:15:00 | 74.05 | 73.91 | 74.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 11:30:00 | 74.15 | 73.91 | 74.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 71.82 | 71.21 | 71.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:30:00 | 71.77 | 71.21 | 71.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 71.61 | 71.29 | 71.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 12:30:00 | 71.41 | 71.36 | 71.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 14:30:00 | 71.46 | 71.47 | 71.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:00:00 | 71.37 | 71.60 | 71.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 09:15:00 | 72.49 | 70.99 | 71.16 | SL hit (close>static) qty=1.00 sl=72.00 alert=retest2 |

### Cycle 179 — BUY (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 12:15:00 | 71.56 | 71.33 | 71.30 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 15:15:00 | 71.05 | 71.29 | 71.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 09:15:00 | 70.63 | 71.16 | 71.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 71.31 | 69.79 | 70.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 71.31 | 69.79 | 70.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 71.31 | 69.79 | 70.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:00:00 | 71.31 | 69.79 | 70.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 71.43 | 70.12 | 70.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:30:00 | 71.59 | 70.12 | 70.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — BUY (started 2025-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 13:15:00 | 70.48 | 70.41 | 70.41 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 09:15:00 | 68.04 | 69.96 | 70.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 67.55 | 68.76 | 69.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 13:15:00 | 67.18 | 66.99 | 67.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 13:45:00 | 67.14 | 66.99 | 67.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 67.90 | 67.24 | 67.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:00:00 | 67.90 | 67.24 | 67.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 66.98 | 67.19 | 67.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 13:00:00 | 66.69 | 67.05 | 67.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 13:45:00 | 66.47 | 66.91 | 67.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 11:30:00 | 66.68 | 66.98 | 67.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 15:15:00 | 66.69 | 66.99 | 67.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 67.54 | 67.05 | 67.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:45:00 | 67.53 | 67.05 | 67.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 66.79 | 67.00 | 67.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:30:00 | 67.33 | 67.00 | 67.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 66.79 | 66.96 | 67.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:30:00 | 66.95 | 66.96 | 67.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 66.31 | 66.43 | 66.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 15:15:00 | 65.75 | 66.33 | 66.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 13:15:00 | 65.75 | 66.15 | 66.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 11:15:00 | 66.85 | 65.02 | 64.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — BUY (started 2025-12-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 11:15:00 | 66.85 | 65.02 | 64.82 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 64.23 | 65.07 | 65.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 12:15:00 | 64.01 | 64.52 | 64.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 09:15:00 | 68.04 | 64.18 | 64.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 68.04 | 64.18 | 64.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 68.04 | 64.18 | 64.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:00:00 | 68.04 | 64.18 | 64.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 65.94 | 64.53 | 64.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:30:00 | 65.04 | 64.59 | 64.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 13:15:00 | 65.60 | 64.71 | 64.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — BUY (started 2025-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 13:15:00 | 65.60 | 64.71 | 64.65 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 13:15:00 | 63.62 | 64.65 | 64.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 14:15:00 | 63.32 | 64.38 | 64.55 | Break + close below crossover candle low |

### Cycle 187 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 70.52 | 65.56 | 65.06 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 12:15:00 | 67.25 | 67.64 | 67.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 13:15:00 | 66.74 | 67.46 | 67.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 12:15:00 | 67.99 | 67.25 | 67.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 12:15:00 | 67.99 | 67.25 | 67.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 67.99 | 67.25 | 67.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 13:00:00 | 67.99 | 67.25 | 67.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 68.05 | 67.41 | 67.44 | EMA400 retest candle locked (from downside) |

### Cycle 189 — BUY (started 2026-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 14:15:00 | 68.60 | 67.64 | 67.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 15:15:00 | 69.75 | 68.07 | 67.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 15:15:00 | 69.39 | 69.40 | 68.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 09:15:00 | 68.55 | 69.40 | 68.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 68.43 | 69.21 | 68.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:30:00 | 68.60 | 69.21 | 68.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 67.55 | 68.87 | 68.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 67.55 | 68.87 | 68.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 67.19 | 68.26 | 68.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 66.92 | 67.99 | 68.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 64.03 | 63.90 | 65.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 64.03 | 63.90 | 65.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 64.03 | 63.90 | 65.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:45:00 | 63.13 | 63.49 | 64.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 10:15:00 | 67.32 | 64.86 | 64.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 191 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 67.32 | 64.86 | 64.52 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 64.63 | 65.52 | 65.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 64.25 | 65.26 | 65.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 62.90 | 62.82 | 63.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:30:00 | 63.28 | 62.82 | 63.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 64.15 | 61.47 | 61.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:00:00 | 64.15 | 61.47 | 61.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 64.12 | 62.00 | 61.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 64.85 | 62.57 | 62.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 62.92 | 63.56 | 62.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 62.92 | 63.56 | 62.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 62.92 | 63.56 | 62.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 62.92 | 63.56 | 62.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 62.98 | 63.45 | 62.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 62.92 | 63.45 | 62.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 62.90 | 63.34 | 62.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:00:00 | 62.90 | 63.34 | 62.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 63.30 | 63.33 | 62.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 14:00:00 | 63.41 | 63.35 | 62.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-01 10:15:00 | 69.75 | 67.71 | 66.14 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 194 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 63.77 | 65.79 | 65.84 | EMA200 below EMA400 |

### Cycle 195 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 68.35 | 66.23 | 65.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-05 09:15:00 | 70.90 | 69.31 | 68.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 14:15:00 | 70.99 | 71.23 | 70.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 14:45:00 | 71.12 | 71.23 | 70.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 71.28 | 71.16 | 70.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:30:00 | 71.20 | 71.16 | 70.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 71.90 | 72.27 | 71.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:45:00 | 72.00 | 72.27 | 71.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 71.42 | 72.10 | 71.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 71.42 | 72.10 | 71.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 71.55 | 71.99 | 71.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:45:00 | 71.64 | 71.99 | 71.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 71.72 | 71.89 | 71.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 14:00:00 | 71.72 | 71.89 | 71.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 71.55 | 71.82 | 71.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 14:45:00 | 71.68 | 71.82 | 71.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 71.70 | 71.79 | 71.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:15:00 | 70.85 | 71.79 | 71.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 70.70 | 71.58 | 71.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 10:15:00 | 70.28 | 71.32 | 71.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 13:15:00 | 71.88 | 71.19 | 71.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 13:15:00 | 71.88 | 71.19 | 71.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 71.88 | 71.19 | 71.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 13:45:00 | 71.77 | 71.19 | 71.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 71.67 | 71.29 | 71.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 15:15:00 | 71.40 | 71.29 | 71.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 11:15:00 | 71.36 | 70.27 | 70.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 71.36 | 70.27 | 70.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 13:15:00 | 72.80 | 70.96 | 70.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 72.70 | 72.98 | 72.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 11:30:00 | 72.62 | 72.98 | 72.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 71.80 | 72.72 | 72.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 71.80 | 72.72 | 72.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 70.75 | 72.33 | 72.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 70.56 | 72.33 | 72.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 71.00 | 72.06 | 72.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 13:15:00 | 70.52 | 71.23 | 71.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 14:15:00 | 69.22 | 69.17 | 69.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-24 15:00:00 | 69.22 | 69.17 | 69.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 69.60 | 69.25 | 69.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:15:00 | 69.73 | 69.25 | 69.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 69.86 | 69.38 | 69.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 11:45:00 | 69.40 | 69.87 | 69.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 09:15:00 | 69.03 | 69.56 | 69.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 65.93 | 68.61 | 69.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 12:15:00 | 65.58 | 67.31 | 68.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 14:15:00 | 67.16 | 67.00 | 67.99 | SL hit (close>ema200) qty=0.50 sl=67.00 alert=retest2 |

### Cycle 199 — BUY (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 11:15:00 | 68.95 | 67.34 | 67.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 14:15:00 | 69.65 | 68.18 | 67.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 15:15:00 | 69.40 | 69.63 | 68.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-09 09:15:00 | 66.20 | 69.63 | 68.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 66.31 | 68.97 | 68.69 | EMA400 retest candle locked (from upside) |

### Cycle 200 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 66.35 | 68.45 | 68.47 | EMA200 below EMA400 |

### Cycle 201 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 69.44 | 68.46 | 68.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 70.09 | 69.33 | 68.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 12:15:00 | 72.25 | 73.70 | 72.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 12:15:00 | 72.25 | 73.70 | 72.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 12:15:00 | 72.25 | 73.70 | 72.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 12:30:00 | 72.62 | 73.70 | 72.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 13:15:00 | 72.38 | 73.43 | 72.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 13:30:00 | 72.29 | 73.43 | 72.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 72.09 | 72.93 | 72.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:15:00 | 71.69 | 72.93 | 72.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — SELL (started 2026-03-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 10:15:00 | 70.11 | 72.01 | 72.21 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 73.30 | 71.86 | 71.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 11:15:00 | 74.27 | 72.34 | 71.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 14:15:00 | 72.62 | 72.81 | 72.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-18 15:00:00 | 72.62 | 72.81 | 72.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 71.48 | 72.53 | 72.26 | EMA400 retest candle locked (from upside) |

### Cycle 204 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 71.33 | 72.06 | 72.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 69.75 | 71.60 | 71.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 71.70 | 71.26 | 71.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 71.70 | 71.26 | 71.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 71.70 | 71.26 | 71.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 71.73 | 71.26 | 71.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 72.11 | 71.43 | 71.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 72.30 | 71.43 | 71.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 71.56 | 71.46 | 71.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 71.18 | 71.46 | 71.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:30:00 | 71.38 | 71.40 | 71.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 67.62 | 70.04 | 70.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 67.81 | 70.04 | 70.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 68.40 | 67.63 | 68.72 | SL hit (close>ema200) qty=0.50 sl=67.63 alert=retest2 |

### Cycle 205 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 71.34 | 69.49 | 69.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 71.91 | 69.97 | 69.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 15:15:00 | 70.50 | 71.07 | 70.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 15:15:00 | 70.50 | 71.07 | 70.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 70.50 | 71.07 | 70.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 70.44 | 71.07 | 70.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 69.86 | 70.83 | 70.57 | EMA400 retest candle locked (from upside) |

### Cycle 206 — SELL (started 2026-03-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 11:15:00 | 69.29 | 70.27 | 70.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 68.05 | 69.42 | 69.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 72.77 | 69.84 | 69.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 72.77 | 69.84 | 69.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 72.77 | 69.84 | 69.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 72.77 | 69.84 | 69.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 72.76 | 70.43 | 70.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 75.14 | 73.22 | 72.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 09:15:00 | 87.42 | 87.89 | 86.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 10:00:00 | 87.42 | 87.89 | 86.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 97.76 | 98.94 | 97.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:30:00 | 97.55 | 98.94 | 97.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 97.30 | 98.61 | 97.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:45:00 | 97.41 | 98.61 | 97.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 97.36 | 98.36 | 97.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:30:00 | 97.01 | 98.36 | 97.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 12:15:00 | 98.25 | 98.34 | 97.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 12:30:00 | 97.21 | 98.34 | 97.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 97.62 | 98.16 | 97.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 14:30:00 | 97.12 | 98.16 | 97.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 98.78 | 98.21 | 97.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:30:00 | 97.97 | 98.21 | 97.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 97.55 | 98.01 | 97.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:45:00 | 96.40 | 98.01 | 97.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 98.08 | 98.03 | 97.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 12:45:00 | 98.35 | 98.03 | 97.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 105.49 | 106.29 | 105.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 13:45:00 | 105.23 | 106.29 | 105.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 138.80 | 142.40 | 139.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:00:00 | 138.80 | 142.40 | 139.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 141.47 | 142.21 | 139.74 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-23 15:00:00 | 64.65 | 2023-05-31 14:15:00 | 64.70 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2023-05-24 10:15:00 | 64.60 | 2023-05-31 14:15:00 | 64.70 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2023-05-24 11:45:00 | 64.55 | 2023-05-31 14:15:00 | 64.70 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-06-02 09:15:00 | 64.80 | 2023-06-05 09:15:00 | 67.99 | PARTIAL | 0.50 | 4.92% |
| BUY | retest1 | 2023-06-02 10:30:00 | 64.85 | 2023-06-05 14:15:00 | 68.04 | PARTIAL | 0.50 | 4.92% |
| BUY | retest1 | 2023-06-02 12:00:00 | 64.75 | 2023-06-05 14:15:00 | 68.09 | PARTIAL | 0.50 | 5.16% |
| BUY | retest1 | 2023-06-02 09:15:00 | 64.80 | 2023-06-06 15:15:00 | 66.90 | STOP_HIT | 0.50 | 3.24% |
| BUY | retest1 | 2023-06-02 10:30:00 | 64.85 | 2023-06-06 15:15:00 | 66.90 | STOP_HIT | 0.50 | 3.16% |
| BUY | retest1 | 2023-06-02 12:00:00 | 64.75 | 2023-06-06 15:15:00 | 66.90 | STOP_HIT | 0.50 | 3.32% |
| BUY | retest2 | 2023-06-05 09:15:00 | 67.30 | 2023-06-14 14:15:00 | 69.75 | STOP_HIT | 1.00 | 3.64% |
| SELL | retest2 | 2023-06-19 13:15:00 | 68.45 | 2023-06-26 09:15:00 | 65.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-19 15:15:00 | 68.40 | 2023-06-26 09:15:00 | 64.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-20 12:45:00 | 68.45 | 2023-06-26 09:15:00 | 65.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-20 14:15:00 | 68.45 | 2023-06-26 09:15:00 | 65.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-19 13:15:00 | 68.45 | 2023-06-27 09:15:00 | 65.50 | STOP_HIT | 0.50 | 4.31% |
| SELL | retest2 | 2023-06-19 15:15:00 | 68.40 | 2023-06-27 09:15:00 | 65.50 | STOP_HIT | 0.50 | 4.24% |
| SELL | retest2 | 2023-06-20 12:45:00 | 68.45 | 2023-06-27 09:15:00 | 65.50 | STOP_HIT | 0.50 | 4.31% |
| SELL | retest2 | 2023-06-20 14:15:00 | 68.45 | 2023-06-27 09:15:00 | 65.50 | STOP_HIT | 0.50 | 4.31% |
| SELL | retest2 | 2023-06-21 10:45:00 | 68.25 | 2023-06-28 14:15:00 | 64.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-21 10:45:00 | 68.25 | 2023-06-30 10:15:00 | 65.25 | STOP_HIT | 0.50 | 4.40% |
| SELL | retest2 | 2023-06-21 12:45:00 | 67.85 | 2023-07-03 09:15:00 | 68.15 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2023-07-06 09:30:00 | 67.30 | 2023-07-07 10:15:00 | 66.65 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2023-07-06 10:45:00 | 67.25 | 2023-07-07 10:15:00 | 66.65 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2023-07-06 13:30:00 | 67.95 | 2023-07-07 10:15:00 | 66.65 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2023-07-11 13:00:00 | 65.85 | 2023-07-17 11:15:00 | 65.95 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2023-07-24 11:45:00 | 64.55 | 2023-07-26 09:15:00 | 65.35 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2023-07-24 12:15:00 | 64.70 | 2023-07-26 09:15:00 | 65.35 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2023-07-24 13:30:00 | 64.70 | 2023-07-26 09:15:00 | 65.35 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2023-08-03 09:15:00 | 65.50 | 2023-08-07 10:15:00 | 72.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-03 11:30:00 | 65.65 | 2023-08-07 10:15:00 | 72.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-03 12:00:00 | 65.40 | 2023-08-07 10:15:00 | 71.94 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-03 14:15:00 | 65.50 | 2023-08-07 10:15:00 | 72.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-04 09:15:00 | 66.30 | 2023-08-07 10:15:00 | 72.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-04 14:00:00 | 65.80 | 2023-08-07 10:15:00 | 72.38 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-07 09:15:00 | 68.70 | 2023-08-10 15:15:00 | 70.05 | STOP_HIT | 1.00 | 1.97% |
| SELL | retest2 | 2023-08-18 13:15:00 | 67.20 | 2023-08-22 14:15:00 | 67.90 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2023-08-18 14:30:00 | 67.30 | 2023-08-22 14:15:00 | 67.90 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2023-08-21 10:00:00 | 67.25 | 2023-08-22 14:15:00 | 67.90 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2023-08-21 10:45:00 | 67.15 | 2023-08-22 14:15:00 | 67.90 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest1 | 2023-08-29 09:30:00 | 72.35 | 2023-08-31 11:15:00 | 75.39 | PARTIAL | 0.50 | 4.20% |
| BUY | retest1 | 2023-08-29 12:30:00 | 71.80 | 2023-08-31 13:15:00 | 75.97 | PARTIAL | 0.50 | 5.80% |
| BUY | retest1 | 2023-08-29 09:30:00 | 72.35 | 2023-09-01 13:15:00 | 74.90 | STOP_HIT | 0.50 | 3.52% |
| BUY | retest1 | 2023-08-29 12:30:00 | 71.80 | 2023-09-01 13:15:00 | 74.90 | STOP_HIT | 0.50 | 4.32% |
| BUY | retest2 | 2023-08-30 09:15:00 | 71.65 | 2023-09-05 11:15:00 | 78.82 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-30 10:15:00 | 71.75 | 2023-09-05 11:15:00 | 78.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-30 11:00:00 | 71.65 | 2023-09-05 11:15:00 | 78.82 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-09-11 12:45:00 | 77.00 | 2023-09-12 09:15:00 | 73.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-11 13:15:00 | 76.95 | 2023-09-12 09:15:00 | 73.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-12 09:30:00 | 75.10 | 2023-09-12 15:15:00 | 71.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-11 12:45:00 | 77.00 | 2023-09-13 11:15:00 | 73.10 | STOP_HIT | 0.50 | 5.06% |
| SELL | retest2 | 2023-09-11 13:15:00 | 76.95 | 2023-09-13 11:15:00 | 73.10 | STOP_HIT | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-12 09:30:00 | 75.10 | 2023-09-13 11:15:00 | 73.10 | STOP_HIT | 0.50 | 2.66% |
| SELL | retest2 | 2023-09-25 10:00:00 | 71.70 | 2023-09-27 10:15:00 | 74.35 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2023-09-26 11:45:00 | 72.20 | 2023-09-27 10:15:00 | 74.35 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2023-09-26 12:30:00 | 72.00 | 2023-09-27 10:15:00 | 74.35 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2023-10-09 09:15:00 | 72.80 | 2023-10-12 11:15:00 | 73.60 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2023-10-18 09:45:00 | 75.00 | 2023-10-18 10:15:00 | 73.25 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2023-10-19 13:00:00 | 73.35 | 2023-10-20 13:15:00 | 69.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-19 14:00:00 | 73.40 | 2023-10-20 13:15:00 | 69.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-19 13:00:00 | 73.35 | 2023-10-23 09:15:00 | 66.02 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-10-19 14:00:00 | 73.40 | 2023-10-23 09:15:00 | 66.06 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2023-11-23 11:30:00 | 67.10 | 2023-11-24 10:15:00 | 67.85 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2023-11-23 14:30:00 | 67.05 | 2023-11-24 10:15:00 | 67.85 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2023-11-28 15:00:00 | 66.65 | 2023-11-29 09:15:00 | 67.65 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2023-12-01 13:00:00 | 66.80 | 2023-12-04 09:15:00 | 67.70 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2023-12-01 15:15:00 | 66.70 | 2023-12-04 09:15:00 | 67.70 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2023-12-11 09:15:00 | 69.05 | 2023-12-12 10:15:00 | 68.55 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2023-12-11 10:45:00 | 68.75 | 2023-12-12 10:15:00 | 68.55 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2023-12-11 13:30:00 | 68.55 | 2023-12-12 10:15:00 | 68.55 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2023-12-11 15:15:00 | 68.55 | 2023-12-12 10:15:00 | 68.55 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2023-12-12 09:30:00 | 68.65 | 2023-12-12 10:15:00 | 68.55 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2023-12-13 14:45:00 | 67.90 | 2023-12-15 09:15:00 | 70.60 | STOP_HIT | 1.00 | -3.98% |
| SELL | retest2 | 2023-12-14 14:30:00 | 67.90 | 2023-12-15 09:15:00 | 70.60 | STOP_HIT | 1.00 | -3.98% |
| BUY | retest2 | 2024-01-02 09:15:00 | 85.80 | 2024-01-02 10:15:00 | 83.35 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2024-01-02 13:30:00 | 85.35 | 2024-01-04 09:15:00 | 93.89 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-03 14:45:00 | 85.30 | 2024-01-04 09:15:00 | 93.83 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-01-16 11:15:00 | 88.20 | 2024-01-17 14:15:00 | 83.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-16 11:15:00 | 88.20 | 2024-01-18 11:15:00 | 84.80 | STOP_HIT | 0.50 | 3.85% |
| BUY | retest2 | 2024-01-24 09:15:00 | 96.20 | 2024-01-24 14:15:00 | 105.82 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-02-02 09:15:00 | 100.00 | 2024-02-06 14:15:00 | 103.60 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2024-02-05 13:15:00 | 101.45 | 2024-02-06 14:15:00 | 103.60 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2024-02-05 13:45:00 | 101.25 | 2024-02-06 14:15:00 | 103.60 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2024-02-19 11:45:00 | 104.50 | 2024-02-21 11:15:00 | 114.95 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-03-04 09:15:00 | 106.75 | 2024-03-04 12:15:00 | 109.15 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2024-03-05 13:00:00 | 110.10 | 2024-03-05 15:15:00 | 107.70 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2024-03-05 14:45:00 | 109.65 | 2024-03-05 15:15:00 | 107.70 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-03-15 11:45:00 | 87.80 | 2024-03-21 11:15:00 | 89.55 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2024-03-15 13:00:00 | 88.30 | 2024-03-21 11:15:00 | 89.55 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-03-19 14:45:00 | 88.30 | 2024-03-21 11:15:00 | 89.55 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2024-03-27 09:15:00 | 91.90 | 2024-04-08 13:15:00 | 96.95 | STOP_HIT | 1.00 | 5.50% |
| SELL | retest2 | 2024-04-10 09:15:00 | 96.35 | 2024-04-12 09:15:00 | 99.15 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2024-04-10 10:00:00 | 96.65 | 2024-04-12 09:15:00 | 99.15 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2024-05-03 14:30:00 | 104.65 | 2024-05-03 15:15:00 | 99.60 | STOP_HIT | 1.00 | -4.83% |
| BUY | retest2 | 2024-05-17 11:00:00 | 96.60 | 2024-05-24 12:15:00 | 99.90 | STOP_HIT | 1.00 | 3.42% |
| BUY | retest2 | 2024-05-17 11:45:00 | 96.75 | 2024-05-24 12:15:00 | 99.90 | STOP_HIT | 1.00 | 3.26% |
| BUY | retest2 | 2024-05-17 12:15:00 | 96.50 | 2024-05-24 12:15:00 | 99.90 | STOP_HIT | 1.00 | 3.52% |
| BUY | retest2 | 2024-05-17 13:00:00 | 96.45 | 2024-05-24 12:15:00 | 99.90 | STOP_HIT | 1.00 | 3.58% |
| BUY | retest2 | 2024-05-29 11:15:00 | 106.15 | 2024-05-30 09:15:00 | 102.25 | STOP_HIT | 1.00 | -3.67% |
| BUY | retest2 | 2024-05-29 12:00:00 | 105.85 | 2024-05-30 09:15:00 | 102.25 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2024-05-29 12:45:00 | 105.70 | 2024-05-30 09:15:00 | 102.25 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2024-05-29 13:30:00 | 105.70 | 2024-05-30 09:15:00 | 102.25 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2024-06-06 11:30:00 | 93.80 | 2024-06-07 10:15:00 | 95.75 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-06-06 15:00:00 | 94.00 | 2024-06-07 10:15:00 | 95.75 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2024-06-28 12:15:00 | 113.77 | 2024-07-01 09:15:00 | 117.21 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2024-06-28 13:30:00 | 114.05 | 2024-07-01 09:15:00 | 117.21 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2024-06-28 14:30:00 | 113.90 | 2024-07-01 09:15:00 | 117.21 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2024-06-28 15:00:00 | 113.29 | 2024-07-01 09:15:00 | 117.21 | STOP_HIT | 1.00 | -3.46% |
| BUY | retest2 | 2024-07-10 09:15:00 | 127.25 | 2024-07-10 09:15:00 | 123.81 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2024-07-23 11:30:00 | 115.54 | 2024-07-23 12:15:00 | 109.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-23 11:30:00 | 115.54 | 2024-07-24 09:15:00 | 116.58 | STOP_HIT | 0.50 | -0.90% |
| BUY | retest2 | 2024-07-26 09:45:00 | 119.13 | 2024-07-30 09:15:00 | 131.04 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-26 11:15:00 | 121.62 | 2024-07-30 13:15:00 | 133.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-12 10:45:00 | 138.41 | 2024-08-14 11:15:00 | 133.08 | STOP_HIT | 1.00 | -3.85% |
| BUY | retest2 | 2024-08-12 12:15:00 | 139.14 | 2024-08-14 11:15:00 | 133.08 | STOP_HIT | 1.00 | -4.36% |
| BUY | retest2 | 2024-08-27 09:15:00 | 147.39 | 2024-08-28 15:15:00 | 145.80 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-08-27 09:45:00 | 146.67 | 2024-08-28 15:15:00 | 145.80 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2024-08-27 11:15:00 | 146.38 | 2024-08-28 15:15:00 | 145.80 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2024-08-27 11:45:00 | 146.30 | 2024-08-28 15:15:00 | 145.80 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2024-09-02 14:45:00 | 148.07 | 2024-09-02 15:15:00 | 146.50 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-09-03 09:15:00 | 151.20 | 2024-09-09 09:15:00 | 150.93 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2024-09-20 09:15:00 | 159.23 | 2024-09-24 13:15:00 | 157.19 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-09-26 09:15:00 | 156.80 | 2024-09-30 09:15:00 | 148.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-26 09:15:00 | 156.80 | 2024-10-01 09:15:00 | 150.83 | STOP_HIT | 0.50 | 3.81% |
| SELL | retest2 | 2024-10-14 11:30:00 | 138.24 | 2024-10-18 09:15:00 | 131.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-14 12:15:00 | 137.98 | 2024-10-18 09:15:00 | 131.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-14 11:30:00 | 138.24 | 2024-10-21 15:15:00 | 124.42 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-14 12:15:00 | 137.98 | 2024-10-21 15:15:00 | 124.18 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-10-31 13:30:00 | 119.84 | 2024-11-06 09:15:00 | 130.53 | TARGET_HIT | 1.00 | 8.92% |
| BUY | retest2 | 2024-11-04 11:30:00 | 118.66 | 2024-11-06 09:15:00 | 130.88 | TARGET_HIT | 1.00 | 10.30% |
| BUY | retest2 | 2024-11-04 13:00:00 | 118.98 | 2024-11-06 09:15:00 | 130.62 | TARGET_HIT | 1.00 | 9.79% |
| BUY | retest2 | 2024-11-04 13:45:00 | 118.75 | 2024-11-11 12:15:00 | 131.82 | TARGET_HIT | 1.00 | 11.01% |
| BUY | retest2 | 2024-11-05 10:15:00 | 122.64 | 2024-11-12 15:15:00 | 125.67 | STOP_HIT | 1.00 | 2.47% |
| BUY | retest2 | 2024-11-05 11:00:00 | 121.83 | 2024-11-12 15:15:00 | 125.67 | STOP_HIT | 1.00 | 3.15% |
| BUY | retest2 | 2024-11-05 12:00:00 | 121.79 | 2024-11-12 15:15:00 | 125.67 | STOP_HIT | 1.00 | 3.19% |
| BUY | retest2 | 2024-11-21 13:15:00 | 127.67 | 2024-11-21 14:15:00 | 126.92 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2024-12-05 12:30:00 | 129.69 | 2024-12-06 12:15:00 | 134.18 | STOP_HIT | 1.00 | -3.46% |
| SELL | retest2 | 2024-12-05 13:15:00 | 129.73 | 2024-12-06 12:15:00 | 134.18 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2024-12-05 14:30:00 | 129.73 | 2024-12-06 12:15:00 | 134.18 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2024-12-06 12:00:00 | 129.87 | 2024-12-06 12:15:00 | 134.18 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2024-12-16 11:45:00 | 125.17 | 2024-12-18 14:15:00 | 118.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 10:15:00 | 124.87 | 2024-12-18 15:15:00 | 118.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 11:45:00 | 125.17 | 2024-12-23 09:15:00 | 112.65 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-17 10:15:00 | 124.87 | 2024-12-23 09:15:00 | 112.38 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-15 15:00:00 | 97.11 | 2025-01-16 09:15:00 | 100.90 | STOP_HIT | 1.00 | -3.90% |
| SELL | retest2 | 2025-01-24 14:00:00 | 99.40 | 2025-01-27 10:15:00 | 94.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 14:00:00 | 99.40 | 2025-01-28 12:15:00 | 95.24 | STOP_HIT | 0.50 | 4.19% |
| BUY | retest2 | 2025-02-05 09:15:00 | 104.25 | 2025-02-06 12:15:00 | 100.40 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest2 | 2025-02-10 09:15:00 | 99.70 | 2025-02-11 10:15:00 | 94.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 99.70 | 2025-02-12 09:15:00 | 89.73 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-25 11:15:00 | 87.50 | 2025-02-27 14:15:00 | 83.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 11:45:00 | 87.05 | 2025-02-28 09:15:00 | 82.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 11:15:00 | 87.50 | 2025-02-28 12:15:00 | 78.75 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-25 11:45:00 | 87.05 | 2025-03-03 09:15:00 | 78.34 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-03-07 10:15:00 | 84.33 | 2025-03-10 11:15:00 | 82.17 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2025-03-07 14:30:00 | 84.25 | 2025-03-10 11:15:00 | 82.17 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-03-10 09:30:00 | 84.00 | 2025-03-10 11:15:00 | 82.17 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-03-13 13:00:00 | 78.21 | 2025-03-18 12:15:00 | 79.96 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-03-13 14:00:00 | 78.19 | 2025-03-18 12:15:00 | 79.96 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-03-17 10:00:00 | 78.02 | 2025-03-18 12:15:00 | 79.96 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2025-03-17 11:00:00 | 77.81 | 2025-03-18 12:15:00 | 79.96 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2025-03-24 09:15:00 | 84.74 | 2025-03-25 12:15:00 | 81.60 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2025-03-24 09:45:00 | 84.29 | 2025-03-25 12:15:00 | 81.60 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest1 | 2025-03-26 13:30:00 | 80.81 | 2025-03-27 15:15:00 | 81.38 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest1 | 2025-03-26 14:15:00 | 80.80 | 2025-03-27 15:15:00 | 81.38 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest1 | 2025-03-27 11:30:00 | 80.77 | 2025-03-27 15:15:00 | 81.38 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest1 | 2025-03-27 13:45:00 | 80.63 | 2025-03-27 15:15:00 | 81.38 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-03-28 11:15:00 | 81.25 | 2025-04-02 13:15:00 | 81.00 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2025-04-08 10:30:00 | 74.89 | 2025-04-15 09:15:00 | 79.82 | STOP_HIT | 1.00 | -6.58% |
| SELL | retest2 | 2025-04-09 09:15:00 | 74.75 | 2025-04-15 09:15:00 | 79.82 | STOP_HIT | 1.00 | -6.78% |
| SELL | retest2 | 2025-04-09 14:00:00 | 74.99 | 2025-04-15 09:15:00 | 79.82 | STOP_HIT | 1.00 | -6.44% |
| BUY | retest2 | 2025-04-23 14:15:00 | 85.37 | 2025-04-25 09:15:00 | 82.78 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2025-04-23 14:45:00 | 85.73 | 2025-04-25 09:15:00 | 82.78 | STOP_HIT | 1.00 | -3.44% |
| BUY | retest2 | 2025-04-24 11:30:00 | 85.70 | 2025-04-25 09:15:00 | 82.78 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2025-04-29 10:15:00 | 81.73 | 2025-05-06 12:15:00 | 77.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-29 11:00:00 | 81.90 | 2025-05-06 12:15:00 | 77.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-29 10:15:00 | 81.73 | 2025-05-07 11:15:00 | 77.60 | STOP_HIT | 0.50 | 5.05% |
| SELL | retest2 | 2025-04-29 11:00:00 | 81.90 | 2025-05-07 11:15:00 | 77.60 | STOP_HIT | 0.50 | 5.25% |
| SELL | retest1 | 2025-05-22 10:15:00 | 84.80 | 2025-05-23 09:15:00 | 80.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-05-22 10:15:00 | 84.80 | 2025-05-23 09:15:00 | 85.17 | STOP_HIT | 0.50 | -0.44% |
| BUY | retest2 | 2025-05-29 11:15:00 | 88.35 | 2025-05-30 09:15:00 | 86.61 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-06-23 14:15:00 | 81.80 | 2025-06-24 09:15:00 | 83.42 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-07-09 12:30:00 | 82.70 | 2025-07-15 10:15:00 | 82.66 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2025-07-09 13:00:00 | 82.65 | 2025-07-15 10:15:00 | 82.66 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-07-09 13:30:00 | 82.61 | 2025-07-15 10:15:00 | 82.66 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-07-09 14:00:00 | 82.61 | 2025-07-15 10:15:00 | 82.66 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-07-21 11:15:00 | 81.75 | 2025-07-25 11:15:00 | 77.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-21 13:45:00 | 81.76 | 2025-07-25 11:15:00 | 77.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-21 14:15:00 | 81.78 | 2025-07-25 11:15:00 | 77.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-21 15:00:00 | 81.78 | 2025-07-25 11:15:00 | 77.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 10:30:00 | 81.18 | 2025-07-25 12:15:00 | 77.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-21 11:15:00 | 81.75 | 2025-07-28 09:15:00 | 77.83 | STOP_HIT | 0.50 | 4.80% |
| SELL | retest2 | 2025-07-21 13:45:00 | 81.76 | 2025-07-28 09:15:00 | 77.83 | STOP_HIT | 0.50 | 4.81% |
| SELL | retest2 | 2025-07-21 14:15:00 | 81.78 | 2025-07-28 09:15:00 | 77.83 | STOP_HIT | 0.50 | 4.83% |
| SELL | retest2 | 2025-07-21 15:00:00 | 81.78 | 2025-07-28 09:15:00 | 77.83 | STOP_HIT | 0.50 | 4.83% |
| SELL | retest2 | 2025-07-22 10:30:00 | 81.18 | 2025-07-28 09:15:00 | 77.83 | STOP_HIT | 0.50 | 4.13% |
| SELL | retest2 | 2025-09-24 11:15:00 | 75.30 | 2025-09-24 14:15:00 | 75.70 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-09-24 12:00:00 | 75.25 | 2025-09-24 14:15:00 | 75.70 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-09-29 11:30:00 | 73.91 | 2025-10-03 12:15:00 | 74.79 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-09-29 15:00:00 | 73.99 | 2025-10-03 12:15:00 | 74.79 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-10-07 10:00:00 | 74.78 | 2025-10-08 09:15:00 | 74.26 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-10-07 12:30:00 | 74.63 | 2025-10-08 09:15:00 | 74.26 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-10-07 15:15:00 | 74.67 | 2025-10-08 09:15:00 | 74.26 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-10-09 11:00:00 | 73.74 | 2025-10-10 09:15:00 | 77.04 | STOP_HIT | 1.00 | -4.48% |
| SELL | retest2 | 2025-10-09 12:30:00 | 73.84 | 2025-10-10 09:15:00 | 77.04 | STOP_HIT | 1.00 | -4.33% |
| SELL | retest2 | 2025-11-10 13:30:00 | 74.53 | 2025-11-11 11:15:00 | 75.69 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-11-10 14:30:00 | 74.47 | 2025-11-11 11:15:00 | 75.69 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-11-26 12:30:00 | 71.41 | 2025-12-01 09:15:00 | 72.49 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-11-26 14:30:00 | 71.46 | 2025-12-01 09:15:00 | 72.49 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-11-27 13:00:00 | 71.37 | 2025-12-01 09:15:00 | 72.49 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-12-01 12:00:00 | 71.45 | 2025-12-01 12:15:00 | 71.56 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-12-10 13:00:00 | 66.69 | 2025-12-23 11:15:00 | 66.85 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-12-10 13:45:00 | 66.47 | 2025-12-23 11:15:00 | 66.85 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-12-11 11:30:00 | 66.68 | 2025-12-23 11:15:00 | 66.85 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-12-11 15:15:00 | 66.69 | 2025-12-23 11:15:00 | 66.85 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-12-15 15:15:00 | 65.75 | 2025-12-23 11:15:00 | 66.85 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-12-16 13:15:00 | 65.75 | 2025-12-23 11:15:00 | 66.85 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-12-29 11:30:00 | 65.04 | 2025-12-29 13:15:00 | 65.60 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2026-01-13 11:45:00 | 63.13 | 2026-01-16 10:15:00 | 67.32 | STOP_HIT | 1.00 | -6.64% |
| BUY | retest2 | 2026-01-29 14:00:00 | 63.41 | 2026-02-01 10:15:00 | 69.75 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-12 15:15:00 | 71.40 | 2026-02-17 11:15:00 | 71.36 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2026-02-26 11:45:00 | 69.40 | 2026-03-02 09:15:00 | 65.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 09:15:00 | 69.03 | 2026-03-02 12:15:00 | 65.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 11:45:00 | 69.40 | 2026-03-02 14:15:00 | 67.16 | STOP_HIT | 0.50 | 3.23% |
| SELL | retest2 | 2026-02-27 09:15:00 | 69.03 | 2026-03-02 14:15:00 | 67.16 | STOP_HIT | 0.50 | 2.71% |
| SELL | retest2 | 2026-03-20 12:15:00 | 71.18 | 2026-03-23 10:15:00 | 67.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 13:30:00 | 71.38 | 2026-03-23 10:15:00 | 67.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 71.18 | 2026-03-24 12:15:00 | 68.40 | STOP_HIT | 0.50 | 3.91% |
| SELL | retest2 | 2026-03-20 13:30:00 | 71.38 | 2026-03-24 12:15:00 | 68.40 | STOP_HIT | 0.50 | 4.17% |
| SELL | retest2 | 2026-03-25 09:30:00 | 71.19 | 2026-03-25 10:15:00 | 71.34 | STOP_HIT | 1.00 | -0.21% |
