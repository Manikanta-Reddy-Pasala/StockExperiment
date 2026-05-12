# Canara Bank (CANBK)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 134.13
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 26 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 23 |
| PARTIAL | 3 |
| TARGET_HIT | 4 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 3 / 19
- **Target hits / Stop hits / Partials:** 0 / 19 / 3
- **Avg / median % per leg:** -0.75% / -1.29%
- **Sum % (uncompounded):** -16.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 0 | 0.0% | 0 | 15 | 0 | -1.55% | -23.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 0 | 0.0% | 0 | 15 | 0 | -1.55% | -23.3% |
| SELL (all) | 7 | 3 | 42.9% | 0 | 4 | 3 | 0.99% | 6.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 7 | 3 | 42.9% | 0 | 4 | 3 | 0.99% | 6.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 22 | 3 | 13.6% | 0 | 19 | 3 | -0.75% | -16.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-03 12:15:00)

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

### Cycle 2 — BUY (started 2025-09-11 12:15:00)

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

### Cycle 3 — SELL (started 2026-03-13 15:15:00)

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
