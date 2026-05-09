# ONGC (ONGC)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 279.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 20 |
| PARTIAL | 0 |
| TARGET_HIT | 4 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 19
- **Target hits / Stop hits / Partials:** 4 / 19 / 0
- **Avg / median % per leg:** 0.39% / -1.14%
- **Sum % (uncompounded):** 9.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 4 | 26.7% | 4 | 11 | 0 | 1.31% | 19.6% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.31% | -9.9% |
| BUY @ 3rd Alert (retest2) | 12 | 4 | 33.3% | 4 | 8 | 0 | 2.47% | 29.6% |
| SELL (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.33% | -10.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.33% | -10.6% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.31% | -9.9% |
| retest2 (combined) | 20 | 4 | 20.0% | 4 | 16 | 0 | 0.95% | 19.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 10:15:00 | 251.84 | 241.87 | 241.84 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 11:15:00 | 236.99 | 242.11 | 242.12 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 13:15:00 | 247.86 | 242.07 | 242.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-12 09:15:00 | 251.65 | 242.27 | 242.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 10:15:00 | 244.03 | 246.21 | 244.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 10:15:00 | 244.03 | 246.21 | 244.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 244.03 | 246.21 | 244.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:45:00 | 245.05 | 246.21 | 244.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 245.02 | 246.20 | 244.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 245.37 | 245.62 | 244.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 10:15:00 | 243.56 | 245.59 | 244.35 | SL hit (close<static) qty=1.00 sl=244.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 10:15:00 | 245.63 | 245.07 | 244.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 11:00:00 | 245.94 | 245.08 | 244.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 12:15:00 | 243.95 | 245.07 | 244.21 | SL hit (close<static) qty=1.00 sl=244.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-03 12:15:00 | 243.95 | 245.07 | 244.21 | SL hit (close<static) qty=1.00 sl=244.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 12:45:00 | 245.42 | 244.20 | 243.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 244.29 | 244.49 | 244.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:00:00 | 244.29 | 244.49 | 244.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 244.49 | 244.49 | 244.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:00:00 | 244.49 | 244.49 | 244.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 244.70 | 244.50 | 244.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 15:15:00 | 245.30 | 244.50 | 244.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 242.61 | 244.49 | 244.15 | SL hit (close<static) qty=1.00 sl=244.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 242.61 | 244.49 | 244.15 | SL hit (close<static) qty=1.00 sl=244.10 alert=retest2 |

### Cycle 4 — SELL (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 15:15:00 | 241.10 | 243.84 | 243.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 09:15:00 | 240.29 | 243.80 | 243.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 12:15:00 | 239.82 | 239.52 | 241.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-13 13:00:00 | 239.82 | 239.52 | 241.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 240.43 | 239.03 | 240.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:45:00 | 240.85 | 239.03 | 240.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 241.38 | 237.78 | 239.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:00:00 | 241.38 | 237.78 | 239.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 241.86 | 237.82 | 239.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 11:00:00 | 241.86 | 237.82 | 239.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 239.43 | 237.93 | 239.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 13:15:00 | 238.97 | 238.03 | 239.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 14:00:00 | 239.02 | 238.04 | 239.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 09:15:00 | 237.16 | 238.06 | 239.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 12:45:00 | 238.84 | 236.10 | 237.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 241.52 | 236.24 | 237.82 | SL hit (close>static) qty=1.00 sl=239.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 241.52 | 236.24 | 237.82 | SL hit (close>static) qty=1.00 sl=239.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 241.52 | 236.24 | 237.82 | SL hit (close>static) qty=1.00 sl=239.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 241.52 | 236.24 | 237.82 | SL hit (close>static) qty=1.00 sl=239.95 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 237.99 | 236.55 | 237.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:45:00 | 238.02 | 236.55 | 237.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 238.10 | 236.57 | 237.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:15:00 | 238.75 | 236.57 | 237.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 240.10 | 236.60 | 237.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:00:00 | 240.10 | 236.60 | 237.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 10:15:00 | 243.74 | 238.90 | 238.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 246.35 | 239.42 | 239.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 10:15:00 | 247.20 | 248.70 | 245.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-11 10:45:00 | 247.40 | 248.70 | 245.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 247.75 | 249.14 | 245.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 13:45:00 | 250.00 | 248.89 | 246.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 10:15:00 | 250.00 | 248.89 | 246.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 13:45:00 | 249.70 | 248.91 | 246.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 13:15:00 | 244.95 | 248.64 | 246.21 | SL hit (close<static) qty=1.00 sl=245.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 13:15:00 | 244.95 | 248.64 | 246.21 | SL hit (close<static) qty=1.00 sl=245.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 13:15:00 | 244.95 | 248.64 | 246.21 | SL hit (close<static) qty=1.00 sl=245.10 alert=retest2 |

### Cycle 6 — SELL (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 09:15:00 | 240.20 | 244.86 | 244.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 13:15:00 | 238.73 | 244.64 | 244.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 13:15:00 | 239.61 | 238.70 | 241.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 14:00:00 | 239.61 | 238.70 | 241.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 240.18 | 238.74 | 241.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 11:30:00 | 238.32 | 238.74 | 241.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:30:00 | 238.47 | 238.73 | 241.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 241.55 | 238.75 | 240.96 | SL hit (close>static) qty=1.00 sl=241.11 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 241.55 | 238.75 | 240.96 | SL hit (close>static) qty=1.00 sl=241.11 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 09:45:00 | 238.17 | 238.84 | 240.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 14:15:00 | 241.88 | 238.91 | 240.85 | SL hit (close>static) qty=1.00 sl=241.11 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 13:30:00 | 238.40 | 239.00 | 240.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 240.05 | 238.05 | 240.13 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-13 11:15:00 | 241.58 | 238.11 | 240.14 | SL hit (close>static) qty=1.00 sl=241.11 alert=retest2 |

### Cycle 7 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 263.31 | 241.57 | 241.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 268.08 | 241.84 | 241.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-10 09:15:00 | 269.30 | 269.70 | 261.39 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 11:30:00 | 271.00 | 269.71 | 261.76 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 09:15:00 | 271.50 | 269.76 | 261.94 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:00:00 | 271.05 | 269.77 | 261.99 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 264.90 | 269.46 | 262.36 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 262.20 | 269.39 | 262.36 | SL hit (close<ema400) qty=1.00 sl=262.36 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 262.20 | 269.39 | 262.36 | SL hit (close<ema400) qty=1.00 sl=262.36 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 262.20 | 269.39 | 262.36 | SL hit (close<ema400) qty=1.00 sl=262.36 alert=retest1 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 14:15:00 | 266.35 | 268.30 | 262.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 267.15 | 268.23 | 262.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 10:15:00 | 268.70 | 268.20 | 262.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 267.80 | 268.22 | 262.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-01 09:15:00 | 292.99 | 270.50 | 264.85 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-28 09:15:00 | 293.87 | 280.80 | 274.04 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-28 10:15:00 | 295.57 | 280.95 | 274.15 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-28 10:15:00 | 294.58 | 280.95 | 274.15 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-30 09:15:00 | 245.37 | 2025-06-30 10:15:00 | 243.56 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-07-03 10:15:00 | 245.63 | 2025-07-03 12:15:00 | 243.95 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-07-03 11:00:00 | 245.94 | 2025-07-03 12:15:00 | 243.95 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-07-18 12:45:00 | 245.42 | 2025-07-25 09:15:00 | 242.61 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-07-24 15:15:00 | 245.30 | 2025-07-25 09:15:00 | 242.61 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-09-03 13:15:00 | 238.97 | 2025-09-25 09:15:00 | 241.52 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-09-03 14:00:00 | 239.02 | 2025-09-25 09:15:00 | 241.52 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-09-04 09:15:00 | 237.16 | 2025-09-25 09:15:00 | 241.52 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-09-24 12:45:00 | 238.84 | 2025-09-25 09:15:00 | 241.52 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-11-19 13:45:00 | 250.00 | 2025-11-24 13:15:00 | 244.95 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-11-20 10:15:00 | 250.00 | 2025-11-24 13:15:00 | 244.95 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-11-20 13:45:00 | 249.70 | 2025-11-24 13:15:00 | 244.95 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2026-01-01 11:30:00 | 238.32 | 2026-01-02 10:15:00 | 241.55 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2026-01-01 12:30:00 | 238.47 | 2026-01-02 10:15:00 | 241.55 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2026-01-05 09:45:00 | 238.17 | 2026-01-06 14:15:00 | 241.88 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2026-01-07 13:30:00 | 238.40 | 2026-01-13 11:15:00 | 241.58 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest1 | 2026-03-11 11:30:00 | 271.00 | 2026-03-16 10:15:00 | 262.20 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest1 | 2026-03-12 09:15:00 | 271.50 | 2026-03-16 10:15:00 | 262.20 | STOP_HIT | 1.00 | -3.43% |
| BUY | retest1 | 2026-03-12 10:00:00 | 271.05 | 2026-03-16 10:15:00 | 262.20 | STOP_HIT | 1.00 | -3.27% |
| BUY | retest2 | 2026-03-18 14:15:00 | 266.35 | 2026-04-01 09:15:00 | 292.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-19 09:15:00 | 267.15 | 2026-04-28 09:15:00 | 293.87 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-19 10:15:00 | 268.70 | 2026-04-28 10:15:00 | 295.57 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-23 09:15:00 | 267.80 | 2026-04-28 10:15:00 | 294.58 | TARGET_HIT | 1.00 | 10.00% |
