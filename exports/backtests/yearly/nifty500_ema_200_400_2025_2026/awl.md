# AWL Agri Business Ltd. (AWL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 206.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 4 |
| ALERT2_SKIP | 3 |
| ALERT3 | 29 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 36 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 36 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 34
- **Target hits / Stop hits / Partials:** 0 / 36 / 0
- **Avg / median % per leg:** -2.09% / -1.89%
- **Sum % (uncompounded):** -75.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.39% | -5.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.39% | -5.6% |
| SELL (all) | 32 | 2 | 6.2% | 0 | 32 | 0 | -2.18% | -69.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 32 | 2 | 6.2% | 0 | 32 | 0 | -2.18% | -69.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 36 | 2 | 5.6% | 0 | 36 | 0 | -2.09% | -75.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 13:15:00 | 279.70 | 265.36 | 265.34 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 12:15:00 | 253.45 | 265.97 | 265.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 13:15:00 | 252.95 | 265.84 | 265.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 15:15:00 | 260.20 | 259.72 | 262.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 09:15:00 | 258.45 | 259.72 | 262.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 262.10 | 259.75 | 262.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:45:00 | 262.45 | 259.75 | 262.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 261.70 | 259.77 | 262.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:30:00 | 262.35 | 259.77 | 262.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 262.00 | 259.82 | 262.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:15:00 | 260.70 | 259.82 | 262.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 260.85 | 259.83 | 262.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 09:15:00 | 258.20 | 259.88 | 262.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 12:00:00 | 258.85 | 259.83 | 262.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 15:00:00 | 258.30 | 259.82 | 262.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 15:15:00 | 258.80 | 259.67 | 261.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 260.10 | 257.39 | 260.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:45:00 | 260.20 | 257.39 | 260.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 263.70 | 257.45 | 260.33 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-03 11:15:00 | 263.70 | 257.45 | 260.33 | SL hit (close>static) qty=1.00 sl=263.40 alert=retest2 |

### Cycle 3 — BUY (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 15:15:00 | 265.80 | 261.06 | 261.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 12:15:00 | 266.60 | 261.20 | 261.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 14:15:00 | 261.70 | 263.21 | 262.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 14:15:00 | 261.70 | 263.21 | 262.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 261.70 | 263.21 | 262.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 15:00:00 | 261.70 | 263.21 | 262.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 260.50 | 263.18 | 262.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 262.25 | 263.18 | 262.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 12:15:00 | 259.85 | 263.07 | 262.22 | SL hit (close<static) qty=1.00 sl=260.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 10:15:00 | 246.40 | 265.16 | 265.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 11:15:00 | 245.35 | 261.09 | 263.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-11 11:15:00 | 196.28 | 196.17 | 210.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 09:15:00 | 196.18 | 184.24 | 193.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 196.18 | 184.24 | 193.11 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-14 14:30:00 | 267.00 | 2025-05-16 11:15:00 | 271.75 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-05-15 15:00:00 | 267.05 | 2025-05-16 11:15:00 | 271.75 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-05-20 09:15:00 | 266.65 | 2025-05-27 11:15:00 | 272.90 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-05-20 11:30:00 | 267.05 | 2025-05-27 11:15:00 | 272.90 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-05-27 13:45:00 | 267.85 | 2025-05-30 14:15:00 | 278.45 | STOP_HIT | 1.00 | -3.96% |
| SELL | retest2 | 2025-05-30 15:15:00 | 267.90 | 2025-06-10 09:15:00 | 268.60 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-06-02 10:15:00 | 270.45 | 2025-06-10 09:15:00 | 268.60 | STOP_HIT | 1.00 | 0.68% |
| SELL | retest2 | 2025-06-02 11:45:00 | 270.30 | 2025-06-10 09:15:00 | 268.60 | STOP_HIT | 1.00 | 0.63% |
| SELL | retest2 | 2025-06-05 09:45:00 | 266.85 | 2025-06-10 12:15:00 | 275.10 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2025-06-05 11:00:00 | 266.85 | 2025-06-10 12:15:00 | 275.10 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2025-06-09 10:00:00 | 266.70 | 2025-06-10 12:15:00 | 275.10 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2025-06-13 09:15:00 | 264.35 | 2025-07-10 09:15:00 | 264.35 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-07-09 14:15:00 | 262.00 | 2025-07-10 10:15:00 | 269.60 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2025-07-16 09:15:00 | 261.70 | 2025-07-17 09:15:00 | 277.20 | STOP_HIT | 1.00 | -5.92% |
| SELL | retest2 | 2025-07-16 12:30:00 | 261.50 | 2025-07-17 09:15:00 | 277.20 | STOP_HIT | 1.00 | -6.00% |
| SELL | retest2 | 2025-08-22 09:15:00 | 258.20 | 2025-09-03 11:15:00 | 263.70 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-08-22 12:00:00 | 258.85 | 2025-09-03 11:15:00 | 263.70 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-08-22 15:00:00 | 258.30 | 2025-09-03 11:15:00 | 263.70 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-08-25 15:15:00 | 258.80 | 2025-09-03 11:15:00 | 263.70 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-09-08 11:45:00 | 260.35 | 2025-09-12 09:15:00 | 261.50 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-09-09 10:00:00 | 260.50 | 2025-09-16 15:15:00 | 261.50 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-09-11 11:15:00 | 260.30 | 2025-09-16 15:15:00 | 261.50 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-09-11 12:15:00 | 260.55 | 2025-09-16 15:15:00 | 261.50 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-09-11 15:00:00 | 258.45 | 2025-09-16 15:15:00 | 261.50 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-09-15 10:00:00 | 258.75 | 2025-09-19 13:15:00 | 262.10 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-09-15 14:00:00 | 258.65 | 2025-09-22 11:15:00 | 267.60 | STOP_HIT | 1.00 | -3.46% |
| SELL | retest2 | 2025-09-15 15:00:00 | 258.00 | 2025-09-22 11:15:00 | 267.60 | STOP_HIT | 1.00 | -3.72% |
| SELL | retest2 | 2025-09-16 10:45:00 | 258.45 | 2025-09-22 11:15:00 | 267.60 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2025-09-19 10:45:00 | 258.50 | 2025-09-22 11:15:00 | 267.60 | STOP_HIT | 1.00 | -3.52% |
| SELL | retest2 | 2025-09-19 15:00:00 | 257.00 | 2025-09-22 11:15:00 | 267.60 | STOP_HIT | 1.00 | -4.12% |
| SELL | retest2 | 2025-09-24 09:15:00 | 256.90 | 2025-09-25 14:15:00 | 262.00 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-09-24 10:30:00 | 256.40 | 2025-09-25 14:15:00 | 262.00 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2025-10-20 09:15:00 | 262.25 | 2025-10-20 12:15:00 | 259.85 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-10-24 12:30:00 | 263.10 | 2025-11-28 12:15:00 | 259.05 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-10-27 10:30:00 | 262.90 | 2025-11-28 12:15:00 | 259.05 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-11-28 09:30:00 | 263.40 | 2025-11-28 12:15:00 | 259.05 | STOP_HIT | 1.00 | -1.65% |
