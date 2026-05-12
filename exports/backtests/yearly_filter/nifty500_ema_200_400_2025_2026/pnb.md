# Punjab National Bank (PNB)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 107.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 28 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 22 |
| PARTIAL | 4 |
| TARGET_HIT | 10 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 14 / 9
- **Target hits / Stop hits / Partials:** 10 / 9 / 4
- **Avg / median % per leg:** 4.58% / 5.04%
- **Sum % (uncompounded):** 105.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 10 | 52.6% | 10 | 9 | 0 | 4.49% | 85.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 19 | 10 | 52.6% | 10 | 9 | 0 | 4.49% | 85.3% |
| SELL (all) | 4 | 4 | 100.0% | 0 | 0 | 4 | 4.99% | 20.0% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 0 | 1 | 3.03% | 3.0% |
| SELL @ 3rd Alert (retest2) | 3 | 3 | 100.0% | 0 | 0 | 3 | 5.64% | 16.9% |
| retest1 (combined) | 1 | 1 | 100.0% | 0 | 0 | 1 | 3.03% | 3.0% |
| retest2 (combined) | 22 | 13 | 59.1% | 10 | 9 | 3 | 4.65% | 102.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 14:15:00 | 100.87 | 106.14 | 106.17 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 109.86 | 105.97 | 105.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 12:15:00 | 110.75 | 106.06 | 106.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 12:15:00 | 108.26 | 108.46 | 107.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-26 13:00:00 | 108.26 | 108.46 | 107.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 107.62 | 108.45 | 107.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:00:00 | 107.62 | 108.45 | 107.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 117.76 | 121.54 | 118.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:00:00 | 117.76 | 121.54 | 118.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 116.97 | 121.50 | 118.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:00:00 | 116.97 | 121.50 | 118.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 117.99 | 121.02 | 118.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:45:00 | 117.62 | 121.02 | 118.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 117.78 | 120.99 | 118.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:45:00 | 117.51 | 120.99 | 118.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 118.43 | 120.30 | 118.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:45:00 | 118.38 | 120.30 | 118.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 13:15:00 | 118.32 | 120.28 | 118.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 14:00:00 | 118.32 | 120.28 | 118.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 118.61 | 120.27 | 118.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 15:00:00 | 118.61 | 120.27 | 118.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 117.82 | 120.23 | 118.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 117.82 | 120.23 | 118.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 117.54 | 120.20 | 118.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 117.54 | 120.20 | 118.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 119.07 | 120.02 | 118.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:30:00 | 118.38 | 120.02 | 118.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 118.34 | 119.95 | 118.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:00:00 | 118.34 | 119.95 | 118.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 118.67 | 119.94 | 118.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 11:45:00 | 119.19 | 119.93 | 118.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 15:00:00 | 119.00 | 119.91 | 118.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 09:15:00 | 119.63 | 119.90 | 118.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 10:15:00 | 119.17 | 119.88 | 118.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 119.78 | 120.10 | 118.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 11:30:00 | 120.54 | 120.11 | 118.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 14:15:00 | 120.40 | 120.10 | 118.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 14:45:00 | 120.40 | 120.11 | 118.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:00:00 | 120.44 | 120.12 | 118.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-16 09:15:00 | 131.11 | 122.65 | 120.73 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 113.40 | 122.63 | 122.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 112.12 | 122.44 | 122.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 113.11 | 112.01 | 115.68 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 11:00:00 | 112.57 | 112.02 | 115.66 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 115.15 | 112.62 | 115.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:45:00 | 115.30 | 112.62 | 115.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 11:15:00 | 114.96 | 112.65 | 115.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 12:45:00 | 114.90 | 112.67 | 115.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 13:15:00 | 114.92 | 112.67 | 115.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 14:00:00 | 114.87 | 112.69 | 115.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 09:15:00 | 109.16 | 112.63 | 114.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 09:15:00 | 109.17 | 112.63 | 114.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 09:15:00 | 109.13 | 112.63 | 114.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 12:15:00 | 106.94 | 112.09 | 114.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-20 10:30:00 | 103.08 | 2025-07-01 13:15:00 | 113.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-20 14:30:00 | 103.05 | 2025-07-01 13:15:00 | 113.36 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-26 14:45:00 | 102.77 | 2025-08-29 14:15:00 | 100.87 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-12-18 11:45:00 | 119.19 | 2026-01-16 09:15:00 | 131.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-18 15:00:00 | 119.00 | 2026-01-16 09:15:00 | 130.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-19 09:15:00 | 119.63 | 2026-01-16 09:15:00 | 131.59 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-19 10:15:00 | 119.17 | 2026-01-16 09:15:00 | 131.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-29 11:30:00 | 120.54 | 2026-01-16 09:15:00 | 132.59 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-29 14:15:00 | 120.40 | 2026-01-16 09:15:00 | 132.44 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-29 14:45:00 | 120.40 | 2026-01-16 09:15:00 | 132.44 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-30 10:00:00 | 120.44 | 2026-01-16 09:15:00 | 132.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-28 12:00:00 | 123.43 | 2026-02-01 11:15:00 | 121.50 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2026-02-03 09:15:00 | 123.54 | 2026-02-06 12:15:00 | 121.44 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2026-02-03 09:45:00 | 123.89 | 2026-02-06 12:15:00 | 121.44 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-02-04 11:45:00 | 123.39 | 2026-02-06 12:15:00 | 121.44 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2026-02-06 15:00:00 | 123.00 | 2026-02-11 09:15:00 | 121.30 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-02-09 09:15:00 | 124.71 | 2026-02-11 09:15:00 | 121.30 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2026-02-11 11:15:00 | 122.73 | 2026-02-12 10:15:00 | 121.59 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2026-02-11 12:45:00 | 122.80 | 2026-02-12 10:15:00 | 121.59 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest1 | 2026-04-15 11:00:00 | 112.57 | 2026-04-30 09:15:00 | 109.16 | PARTIAL | 0.50 | 3.03% |
| SELL | retest2 | 2026-04-22 12:45:00 | 114.90 | 2026-04-30 09:15:00 | 109.17 | PARTIAL | 0.50 | 4.98% |
| SELL | retest2 | 2026-04-22 13:15:00 | 114.92 | 2026-04-30 09:15:00 | 109.13 | PARTIAL | 0.50 | 5.04% |
| SELL | retest2 | 2026-04-22 14:00:00 | 114.87 | 2026-05-05 12:15:00 | 106.94 | PARTIAL | 0.50 | 6.90% |
