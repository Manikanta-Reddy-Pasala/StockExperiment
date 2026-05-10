# Pine Labs Ltd. (PINELABS)

## Backtest Summary

- **Window:** 2025-11-14 09:15:00 → 2026-05-08 15:15:00 (826 bars)
- **Last close:** 196.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 34 |
| ALERT1 | 24 |
| ALERT2 | 22 |
| ALERT2_SKIP | 12 |
| ALERT3 | 64 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 37 |
| PARTIAL | 10 |
| TARGET_HIT | 4 |
| STOP_HIT | 36 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 50 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 20
- **Target hits / Stop hits / Partials:** 4 / 36 / 10
- **Avg / median % per leg:** 2.03% / 0.19%
- **Sum % (uncompounded):** 101.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 16 | 66.7% | 4 | 17 | 3 | 1.81% | 43.5% |
| BUY @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 3 | 0 | 3 | 7.50% | 45.0% |
| BUY @ 3rd Alert (retest2) | 18 | 10 | 55.6% | 1 | 17 | 0 | -0.08% | -1.5% |
| SELL (all) | 26 | 14 | 53.8% | 0 | 19 | 7 | 2.22% | 57.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 26 | 14 | 53.8% | 0 | 19 | 7 | 2.22% | 57.8% |
| retest1 (combined) | 6 | 6 | 100.0% | 3 | 0 | 3 | 7.50% | 45.0% |
| retest2 (combined) | 44 | 24 | 54.5% | 1 | 36 | 7 | 1.28% | 56.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-11-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 12:15:00 | 240.34 | 237.84 | 237.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 13:15:00 | 246.51 | 239.57 | 238.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 09:15:00 | 239.90 | 241.17 | 239.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 239.90 | 241.17 | 239.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 239.90 | 241.17 | 239.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 239.90 | 241.17 | 239.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 241.02 | 241.14 | 239.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 09:30:00 | 242.66 | 241.08 | 240.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 11:00:00 | 242.20 | 241.83 | 241.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 11:45:00 | 242.20 | 241.76 | 241.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 12:30:00 | 242.30 | 241.90 | 241.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 242.74 | 244.02 | 242.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 14:00:00 | 245.37 | 243.73 | 242.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 12:00:00 | 245.40 | 244.23 | 243.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 14:15:00 | 244.77 | 243.70 | 243.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 15:00:00 | 250.73 | 245.11 | 244.07 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 246.98 | 247.05 | 245.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:15:00 | 249.34 | 247.05 | 245.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 248.22 | 247.28 | 246.11 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-04 14:15:00 | 242.53 | 245.87 | 245.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 14:15:00 | 242.53 | 245.87 | 245.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 14:15:00 | 242.53 | 245.87 | 245.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 14:15:00 | 242.53 | 245.87 | 245.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 14:15:00 | 242.53 | 245.87 | 245.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 14:15:00 | 242.53 | 245.87 | 245.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 14:15:00 | 242.53 | 245.87 | 245.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 14:15:00 | 242.53 | 245.87 | 245.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 14:15:00 | 242.53 | 245.87 | 245.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 15:15:00 | 240.60 | 244.81 | 245.40 | Break + close below crossover candle low |

### Cycle 3 — BUY (started 2025-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 11:15:00 | 250.62 | 245.70 | 245.64 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 241.44 | 245.37 | 245.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 240.29 | 244.35 | 245.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 12:15:00 | 241.50 | 241.36 | 242.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 13:00:00 | 241.50 | 241.36 | 242.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 242.84 | 241.43 | 242.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 242.84 | 241.43 | 242.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 242.06 | 241.56 | 242.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 245.18 | 241.56 | 242.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 241.18 | 241.48 | 242.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 14:30:00 | 240.67 | 242.57 | 242.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 09:45:00 | 239.30 | 242.08 | 242.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 13:00:00 | 240.63 | 241.63 | 242.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 15:00:00 | 240.80 | 241.15 | 241.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 239.33 | 240.70 | 241.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 11:45:00 | 238.34 | 240.06 | 241.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 14:15:00 | 228.64 | 231.14 | 233.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 14:15:00 | 227.34 | 231.14 | 233.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 14:15:00 | 228.60 | 231.14 | 233.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 14:15:00 | 228.76 | 231.14 | 233.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 15:15:00 | 226.42 | 230.21 | 233.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 15:15:00 | 226.48 | 225.56 | 228.57 | SL hit (close>ema200) qty=0.50 sl=225.56 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 15:15:00 | 226.48 | 225.56 | 228.57 | SL hit (close>ema200) qty=0.50 sl=225.56 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 15:15:00 | 226.48 | 225.56 | 228.57 | SL hit (close>ema200) qty=0.50 sl=225.56 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 15:15:00 | 226.48 | 225.56 | 228.57 | SL hit (close>ema200) qty=0.50 sl=225.56 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 15:15:00 | 226.48 | 225.56 | 228.57 | SL hit (close>ema200) qty=0.50 sl=225.56 alert=retest2 |

### Cycle 5 — BUY (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 12:15:00 | 231.76 | 226.99 | 226.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 243.51 | 231.16 | 228.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 10:15:00 | 232.69 | 232.84 | 230.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 10:30:00 | 232.65 | 232.84 | 230.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 232.84 | 232.77 | 231.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:15:00 | 233.21 | 232.77 | 231.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:45:00 | 233.58 | 233.01 | 231.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 10:45:00 | 235.19 | 233.80 | 232.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 14:00:00 | 233.15 | 233.92 | 232.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 233.50 | 233.85 | 232.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 246.82 | 233.85 | 232.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 13:00:00 | 235.23 | 236.18 | 234.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 14:30:00 | 235.55 | 235.45 | 234.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 237.66 | 235.36 | 234.50 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 236.25 | 237.06 | 236.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:00:00 | 236.25 | 237.06 | 236.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 235.76 | 236.80 | 236.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:00:00 | 235.76 | 236.80 | 236.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 235.50 | 236.54 | 236.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:30:00 | 235.47 | 236.54 | 236.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 235.50 | 235.91 | 235.86 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-30 11:15:00 | 235.64 | 235.80 | 235.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-30 11:15:00 | 235.64 | 235.80 | 235.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-30 11:15:00 | 235.64 | 235.80 | 235.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-30 11:15:00 | 235.64 | 235.80 | 235.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-30 11:15:00 | 235.64 | 235.80 | 235.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-30 11:15:00 | 235.64 | 235.80 | 235.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-30 11:15:00 | 235.64 | 235.80 | 235.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-30 11:15:00 | 235.64 | 235.80 | 235.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 11:15:00 | 235.64 | 235.80 | 235.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 14:15:00 | 235.46 | 235.73 | 235.78 | Break + close below crossover candle low |

### Cycle 7 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 238.00 | 236.14 | 235.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 14:15:00 | 239.70 | 237.18 | 236.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 236.33 | 237.32 | 236.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 236.33 | 237.32 | 236.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 236.33 | 237.32 | 236.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:00:00 | 236.33 | 237.32 | 236.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 236.34 | 237.12 | 236.72 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 13:15:00 | 235.85 | 236.47 | 236.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 15:15:00 | 235.39 | 235.64 | 235.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 14:15:00 | 237.07 | 235.20 | 235.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 14:15:00 | 237.07 | 235.20 | 235.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 237.07 | 235.20 | 235.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:00:00 | 237.07 | 235.20 | 235.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 15:15:00 | 238.10 | 235.78 | 235.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 09:15:00 | 239.79 | 236.58 | 236.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 09:15:00 | 238.22 | 238.55 | 237.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 238.22 | 238.55 | 237.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 238.22 | 238.55 | 237.52 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2026-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 14:15:00 | 235.52 | 237.06 | 237.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 234.71 | 236.53 | 236.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 11:15:00 | 216.67 | 216.30 | 219.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-14 12:00:00 | 216.67 | 216.30 | 219.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 224.54 | 217.95 | 219.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 13:00:00 | 224.54 | 217.95 | 219.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 230.78 | 220.51 | 220.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:00:00 | 230.78 | 220.51 | 220.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2026-01-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 14:15:00 | 227.51 | 221.91 | 221.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 236.00 | 225.22 | 223.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 234.99 | 238.40 | 234.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 09:15:00 | 234.99 | 238.40 | 234.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 234.99 | 238.40 | 234.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:30:00 | 235.71 | 238.40 | 234.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 234.05 | 237.53 | 234.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:45:00 | 234.12 | 237.53 | 234.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 237.20 | 237.47 | 235.02 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2026-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 15:15:00 | 229.49 | 233.26 | 233.61 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 09:15:00 | 234.50 | 232.57 | 232.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 11:15:00 | 238.89 | 234.19 | 233.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 12:15:00 | 232.80 | 237.61 | 236.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 12:15:00 | 232.80 | 237.61 | 236.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 232.80 | 237.61 | 236.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 12:30:00 | 234.86 | 237.61 | 236.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 236.10 | 237.31 | 236.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:30:00 | 232.71 | 237.31 | 236.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 240.00 | 237.85 | 236.52 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 230.05 | 235.73 | 235.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 11:15:00 | 229.27 | 234.44 | 235.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 09:15:00 | 236.24 | 231.99 | 233.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 09:15:00 | 236.24 | 231.99 | 233.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 236.24 | 231.99 | 233.37 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 13:15:00 | 237.69 | 234.64 | 234.31 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 14:15:00 | 228.25 | 233.37 | 233.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 11:15:00 | 225.93 | 230.43 | 232.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 14:15:00 | 228.12 | 224.34 | 226.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 14:15:00 | 228.12 | 224.34 | 226.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 228.12 | 224.34 | 226.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:45:00 | 229.36 | 224.34 | 226.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 229.00 | 225.27 | 226.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 10:30:00 | 224.95 | 225.54 | 226.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 11:30:00 | 224.90 | 225.37 | 226.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 09:15:00 | 224.10 | 225.76 | 226.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 09:30:00 | 225.03 | 223.13 | 224.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 11:15:00 | 223.56 | 223.50 | 224.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 15:15:00 | 222.00 | 224.04 | 224.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 226.53 | 224.21 | 224.30 | SL hit (close>static) qty=1.00 sl=224.92 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-04 10:15:00 | 225.97 | 224.56 | 224.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-04 10:15:00 | 225.97 | 224.56 | 224.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-04 10:15:00 | 225.97 | 224.56 | 224.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-04 10:15:00 | 225.97 | 224.56 | 224.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2026-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 10:15:00 | 225.97 | 224.56 | 224.45 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 223.15 | 224.58 | 224.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 12:15:00 | 221.29 | 223.52 | 224.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 09:15:00 | 223.47 | 221.97 | 223.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 223.47 | 221.97 | 223.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 223.47 | 221.97 | 223.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 11:30:00 | 219.82 | 221.53 | 222.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:15:00 | 208.83 | 215.28 | 217.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-10 14:15:00 | 213.69 | 213.34 | 216.02 | SL hit (close>ema200) qty=0.50 sl=213.34 alert=retest2 |

### Cycle 19 — BUY (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 09:15:00 | 205.48 | 203.17 | 203.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 12:15:00 | 207.09 | 204.71 | 204.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 10:15:00 | 205.38 | 206.13 | 205.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 10:15:00 | 205.38 | 206.13 | 205.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 205.38 | 206.13 | 205.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:45:00 | 205.45 | 206.13 | 205.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 205.08 | 205.92 | 205.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:00:00 | 205.08 | 205.92 | 205.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 202.91 | 205.32 | 204.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 13:15:00 | 202.13 | 205.32 | 204.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 13:15:00 | 202.00 | 204.66 | 204.71 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2026-02-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 12:15:00 | 208.73 | 204.78 | 204.55 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2026-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 14:15:00 | 203.00 | 205.28 | 205.32 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 206.74 | 205.43 | 205.37 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 10:15:00 | 202.95 | 204.93 | 205.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 12:15:00 | 199.45 | 203.53 | 204.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 13:15:00 | 192.50 | 192.32 | 195.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-27 14:00:00 | 192.50 | 192.32 | 195.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 176.12 | 173.96 | 176.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 179.86 | 173.96 | 176.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 176.31 | 174.43 | 176.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:30:00 | 177.71 | 174.43 | 176.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 176.30 | 174.80 | 176.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 11:30:00 | 176.68 | 174.80 | 176.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 176.35 | 175.11 | 176.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 13:15:00 | 176.76 | 175.11 | 176.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 176.53 | 175.40 | 176.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 13:45:00 | 177.03 | 175.40 | 176.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 176.20 | 175.56 | 176.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:15:00 | 177.58 | 175.56 | 176.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 177.58 | 175.96 | 176.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 170.27 | 175.96 | 176.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 09:15:00 | 161.76 | 165.88 | 168.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 14:15:00 | 163.11 | 161.05 | 163.12 | SL hit (close>ema200) qty=0.50 sl=161.05 alert=retest2 |

### Cycle 25 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 169.65 | 163.65 | 163.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 170.26 | 166.65 | 164.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 166.66 | 168.12 | 166.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 166.66 | 168.12 | 166.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 166.66 | 168.12 | 166.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:30:00 | 166.32 | 168.12 | 166.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 165.93 | 167.68 | 166.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:45:00 | 165.84 | 167.68 | 166.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 165.79 | 167.30 | 166.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 12:00:00 | 165.79 | 167.30 | 166.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 12:15:00 | 165.53 | 166.95 | 166.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 12:45:00 | 166.14 | 166.95 | 166.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 167.08 | 166.97 | 166.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 15:15:00 | 170.00 | 167.07 | 166.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 13:15:00 | 164.41 | 166.70 | 166.57 | SL hit (close<static) qty=1.00 sl=165.25 alert=retest2 |

### Cycle 26 — SELL (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 14:15:00 | 164.37 | 166.23 | 166.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 162.03 | 165.10 | 165.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 10:15:00 | 162.98 | 162.05 | 163.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:00:00 | 162.98 | 162.05 | 163.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 163.22 | 162.28 | 163.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 163.22 | 162.28 | 163.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 163.29 | 162.48 | 163.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 163.86 | 162.48 | 163.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 164.80 | 162.95 | 163.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 164.80 | 162.95 | 163.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 166.20 | 163.60 | 163.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 15:00:00 | 166.20 | 163.60 | 163.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 166.87 | 164.25 | 164.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 171.28 | 165.66 | 164.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 12:15:00 | 166.29 | 166.85 | 165.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 13:15:00 | 165.40 | 166.85 | 165.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 13:15:00 | 168.01 | 167.08 | 165.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 13:45:00 | 164.71 | 167.08 | 165.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 14:15:00 | 164.36 | 166.54 | 165.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 15:00:00 | 164.36 | 166.54 | 165.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 164.00 | 166.03 | 165.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 166.74 | 166.03 | 165.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 163.50 | 165.08 | 165.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 162.66 | 164.59 | 164.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 158.74 | 156.86 | 159.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 158.74 | 156.86 | 159.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 158.74 | 156.86 | 159.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:45:00 | 156.94 | 157.04 | 158.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:45:00 | 157.01 | 157.40 | 158.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:15:00 | 156.96 | 157.40 | 158.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 09:15:00 | 157.02 | 155.83 | 155.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-07 09:15:00 | 157.02 | 155.83 | 155.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-07 09:15:00 | 157.02 | 155.83 | 155.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2026-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 09:15:00 | 157.02 | 155.83 | 155.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 167.83 | 158.84 | 157.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 171.48 | 172.36 | 166.69 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 11:30:00 | 174.35 | 172.47 | 167.73 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 12:45:00 | 174.67 | 172.64 | 168.23 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 175.50 | 171.45 | 168.74 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 171.70 | 171.83 | 169.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 10:30:00 | 170.10 | 171.83 | 169.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 11:15:00 | 172.81 | 172.03 | 169.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 11:45:00 | 170.05 | 172.03 | 169.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 175.14 | 174.77 | 172.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 176.90 | 174.77 | 172.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 13:15:00 | 183.07 | 177.29 | 174.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 13:15:00 | 183.40 | 177.29 | 174.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 13:15:00 | 184.28 | 177.29 | 174.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-04-16 14:15:00 | 191.78 | 185.73 | 182.16 | Target hit (10%) qty=0.50 alert=retest1 |
| Target hit | 2026-04-17 10:15:00 | 192.14 | 188.49 | 184.39 | Target hit (10%) qty=0.50 alert=retest1 |
| Target hit | 2026-04-17 10:15:00 | 193.05 | 188.49 | 184.39 | Target hit (10%) qty=0.50 alert=retest1 |
| Target hit | 2026-04-17 10:15:00 | 194.59 | 188.49 | 184.39 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 194.60 | 195.39 | 195.41 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2026-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 15:15:00 | 196.00 | 195.50 | 195.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 11:15:00 | 196.38 | 195.84 | 195.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 10:15:00 | 203.03 | 203.88 | 201.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 11:00:00 | 203.03 | 203.88 | 201.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 200.87 | 203.17 | 201.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 12:30:00 | 201.20 | 203.17 | 201.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 197.65 | 202.06 | 201.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 13:45:00 | 196.90 | 202.06 | 201.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 195.90 | 200.19 | 200.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 194.25 | 199.00 | 199.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 12:15:00 | 195.90 | 195.82 | 197.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 13:00:00 | 195.90 | 195.82 | 197.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 195.04 | 195.70 | 196.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 10:45:00 | 194.39 | 195.56 | 196.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 11:30:00 | 194.54 | 195.43 | 196.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 14:30:00 | 194.53 | 195.23 | 196.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 15:00:00 | 194.60 | 195.23 | 196.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 195.33 | 195.37 | 195.97 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 198.37 | 196.02 | 196.16 | SL hit (close>static) qty=1.00 sl=197.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 198.37 | 196.02 | 196.16 | SL hit (close>static) qty=1.00 sl=197.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 198.37 | 196.02 | 196.16 | SL hit (close>static) qty=1.00 sl=197.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 198.37 | 196.02 | 196.16 | SL hit (close>static) qty=1.00 sl=197.00 alert=retest2 |

### Cycle 33 — BUY (started 2026-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 12:15:00 | 199.00 | 196.62 | 196.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 11:15:00 | 200.22 | 197.73 | 197.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 10:15:00 | 199.56 | 199.61 | 198.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 10:30:00 | 198.94 | 199.61 | 198.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 196.59 | 198.93 | 198.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:45:00 | 197.51 | 198.93 | 198.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 196.05 | 198.36 | 198.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:00:00 | 196.05 | 198.36 | 198.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2026-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 15:15:00 | 196.60 | 198.01 | 198.12 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-11-27 09:30:00 | 242.66 | 2025-12-04 14:15:00 | 242.53 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2025-11-28 11:00:00 | 242.20 | 2025-12-04 14:15:00 | 242.53 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2025-11-28 11:45:00 | 242.20 | 2025-12-04 14:15:00 | 242.53 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2025-11-28 12:30:00 | 242.30 | 2025-12-04 14:15:00 | 242.53 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-12-01 14:00:00 | 245.37 | 2025-12-04 14:15:00 | 242.53 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-12-02 12:00:00 | 245.40 | 2025-12-04 14:15:00 | 242.53 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-12-02 14:15:00 | 244.77 | 2025-12-04 14:15:00 | 242.53 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-12-02 15:00:00 | 250.73 | 2025-12-04 14:15:00 | 242.53 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2025-12-10 14:30:00 | 240.67 | 2025-12-16 14:15:00 | 228.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-11 09:45:00 | 239.30 | 2025-12-16 14:15:00 | 227.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-11 13:00:00 | 240.63 | 2025-12-16 14:15:00 | 228.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-11 15:00:00 | 240.80 | 2025-12-16 14:15:00 | 228.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 11:45:00 | 238.34 | 2025-12-16 15:15:00 | 226.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-10 14:30:00 | 240.67 | 2025-12-17 15:15:00 | 226.48 | STOP_HIT | 0.50 | 5.90% |
| SELL | retest2 | 2025-12-11 09:45:00 | 239.30 | 2025-12-17 15:15:00 | 226.48 | STOP_HIT | 0.50 | 5.36% |
| SELL | retest2 | 2025-12-11 13:00:00 | 240.63 | 2025-12-17 15:15:00 | 226.48 | STOP_HIT | 0.50 | 5.88% |
| SELL | retest2 | 2025-12-11 15:00:00 | 240.80 | 2025-12-17 15:15:00 | 226.48 | STOP_HIT | 0.50 | 5.95% |
| SELL | retest2 | 2025-12-12 11:45:00 | 238.34 | 2025-12-17 15:15:00 | 226.48 | STOP_HIT | 0.50 | 4.98% |
| BUY | retest2 | 2025-12-23 09:15:00 | 233.21 | 2025-12-30 11:15:00 | 235.64 | STOP_HIT | 1.00 | 1.04% |
| BUY | retest2 | 2025-12-23 09:45:00 | 233.58 | 2025-12-30 11:15:00 | 235.64 | STOP_HIT | 1.00 | 0.88% |
| BUY | retest2 | 2025-12-23 10:45:00 | 235.19 | 2025-12-30 11:15:00 | 235.64 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-12-23 14:00:00 | 233.15 | 2025-12-30 11:15:00 | 235.64 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest2 | 2025-12-24 09:15:00 | 246.82 | 2025-12-30 11:15:00 | 235.64 | STOP_HIT | 1.00 | -4.53% |
| BUY | retest2 | 2025-12-24 13:00:00 | 235.23 | 2025-12-30 11:15:00 | 235.64 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2025-12-24 14:30:00 | 235.55 | 2025-12-30 11:15:00 | 235.64 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2025-12-26 09:15:00 | 237.66 | 2025-12-30 11:15:00 | 235.64 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-02-01 10:30:00 | 224.95 | 2026-02-04 09:15:00 | 226.53 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2026-02-01 11:30:00 | 224.90 | 2026-02-04 10:15:00 | 225.97 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2026-02-02 09:15:00 | 224.10 | 2026-02-04 10:15:00 | 225.97 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2026-02-03 09:30:00 | 225.03 | 2026-02-04 10:15:00 | 225.97 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2026-02-03 15:15:00 | 222.00 | 2026-02-04 10:15:00 | 225.97 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2026-02-06 11:30:00 | 219.82 | 2026-02-10 10:15:00 | 208.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-06 11:30:00 | 219.82 | 2026-02-10 14:15:00 | 213.69 | STOP_HIT | 0.50 | 2.79% |
| SELL | retest2 | 2026-03-09 09:15:00 | 170.27 | 2026-03-13 09:15:00 | 161.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-09 09:15:00 | 170.27 | 2026-03-16 14:15:00 | 163.11 | STOP_HIT | 0.50 | 4.21% |
| BUY | retest2 | 2026-03-19 15:15:00 | 170.00 | 2026-03-20 13:15:00 | 164.41 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2026-04-01 11:45:00 | 156.94 | 2026-04-07 09:15:00 | 157.02 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2026-04-01 13:45:00 | 157.01 | 2026-04-07 09:15:00 | 157.02 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2026-04-01 14:15:00 | 156.96 | 2026-04-07 09:15:00 | 157.02 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest1 | 2026-04-09 11:30:00 | 174.35 | 2026-04-13 13:15:00 | 183.07 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-04-09 12:45:00 | 174.67 | 2026-04-13 13:15:00 | 183.40 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-04-10 09:15:00 | 175.50 | 2026-04-13 13:15:00 | 184.28 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-04-09 11:30:00 | 174.35 | 2026-04-16 14:15:00 | 191.78 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2026-04-09 12:45:00 | 174.67 | 2026-04-17 10:15:00 | 192.14 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2026-04-10 09:15:00 | 175.50 | 2026-04-17 10:15:00 | 193.05 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-04-13 10:15:00 | 176.90 | 2026-04-17 10:15:00 | 194.59 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-05-05 10:45:00 | 194.39 | 2026-05-06 11:15:00 | 198.37 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2026-05-05 11:30:00 | 194.54 | 2026-05-06 11:15:00 | 198.37 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2026-05-05 14:30:00 | 194.53 | 2026-05-06 11:15:00 | 198.37 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2026-05-05 15:00:00 | 194.60 | 2026-05-06 11:15:00 | 198.37 | STOP_HIT | 1.00 | -1.94% |
