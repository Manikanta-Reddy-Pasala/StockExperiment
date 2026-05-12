# Indian Renewable Energy Development Agency Ltd. (IREDA)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-11 15:15:00 (3605 bars)
- **Last close:** 130.92
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
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 22 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 21 |
| PARTIAL | 11 |
| TARGET_HIT | 11 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 10
- **Target hits / Stop hits / Partials:** 11 / 10 / 11
- **Avg / median % per leg:** 4.68% / 5.00%
- **Sum % (uncompounded):** 149.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 32 | 22 | 68.8% | 11 | 10 | 11 | 4.68% | 149.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 32 | 22 | 68.8% | 11 | 10 | 11 | 4.68% | 149.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 32 | 22 | 68.8% | 11 | 10 | 11 | 4.68% | 149.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 14:15:00 | 175.49 | 169.87 | 169.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 10:15:00 | 177.19 | 170.50 | 170.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 169.80 | 173.60 | 171.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 169.80 | 173.60 | 171.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 169.80 | 173.60 | 171.95 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 10:15:00 | 165.85 | 170.62 | 170.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 161.07 | 169.00 | 169.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 10:15:00 | 148.14 | 147.62 | 152.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 11:00:00 | 148.14 | 147.62 | 152.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 153.82 | 147.66 | 152.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:00:00 | 153.82 | 147.66 | 152.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 152.63 | 147.70 | 152.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:30:00 | 153.62 | 147.70 | 152.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 152.50 | 147.91 | 152.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:30:00 | 152.85 | 147.91 | 152.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 152.40 | 148.00 | 152.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:45:00 | 152.80 | 148.00 | 152.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 152.35 | 148.04 | 152.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 13:45:00 | 151.74 | 151.17 | 153.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:30:00 | 151.87 | 150.66 | 152.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 10:15:00 | 153.38 | 150.69 | 152.61 | SL hit (close>static) qty=1.00 sl=152.75 alert=retest2 |

### Cycle 3 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 137.81 | 125.76 | 125.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 139.34 | 126.02 | 125.84 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-25 13:45:00 | 151.74 | 2025-10-01 10:15:00 | 153.38 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-10-01 09:30:00 | 151.87 | 2025-10-01 10:15:00 | 153.38 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-10-01 13:15:00 | 151.80 | 2025-10-01 14:15:00 | 152.87 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-10-06 10:15:00 | 151.80 | 2025-10-07 13:15:00 | 153.36 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-10-08 09:15:00 | 151.84 | 2025-10-14 10:15:00 | 154.54 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-10-08 09:45:00 | 151.75 | 2025-10-14 10:15:00 | 154.54 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-10-17 09:45:00 | 152.03 | 2025-10-20 15:15:00 | 153.42 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-10-17 11:30:00 | 152.02 | 2025-10-20 15:15:00 | 153.42 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-10-28 14:00:00 | 151.17 | 2025-10-29 15:15:00 | 156.43 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2025-11-04 11:30:00 | 151.46 | 2025-11-24 09:15:00 | 143.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 13:15:00 | 151.28 | 2025-11-24 09:15:00 | 143.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 10:30:00 | 151.38 | 2025-11-24 09:15:00 | 143.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-13 12:30:00 | 150.19 | 2025-11-24 14:15:00 | 142.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-13 13:15:00 | 150.31 | 2025-11-24 14:15:00 | 142.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-14 11:00:00 | 149.98 | 2025-11-24 14:15:00 | 142.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 13:00:00 | 150.21 | 2025-11-24 14:15:00 | 142.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 11:30:00 | 151.46 | 2025-12-03 14:15:00 | 136.31 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-04 13:15:00 | 151.28 | 2025-12-03 14:15:00 | 136.24 | TARGET_HIT | 0.50 | 9.94% |
| SELL | retest2 | 2025-11-12 10:30:00 | 151.38 | 2025-12-05 09:15:00 | 136.15 | TARGET_HIT | 0.50 | 10.06% |
| SELL | retest2 | 2025-11-13 12:30:00 | 150.19 | 2025-12-05 09:15:00 | 135.17 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-13 13:15:00 | 150.31 | 2025-12-05 09:15:00 | 135.28 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-14 11:00:00 | 149.98 | 2025-12-05 09:15:00 | 134.98 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-17 13:00:00 | 150.21 | 2025-12-05 09:15:00 | 135.19 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-14 14:15:00 | 138.15 | 2026-01-20 12:15:00 | 131.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 15:00:00 | 138.16 | 2026-01-20 12:15:00 | 131.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 09:30:00 | 137.87 | 2026-01-20 12:15:00 | 131.21 | PARTIAL | 0.50 | 4.83% |
| SELL | retest2 | 2026-01-16 11:00:00 | 138.12 | 2026-01-20 13:15:00 | 130.98 | PARTIAL | 0.50 | 5.17% |
| SELL | retest2 | 2026-01-14 14:15:00 | 138.15 | 2026-02-01 12:15:00 | 124.34 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-14 15:00:00 | 138.16 | 2026-02-01 12:15:00 | 124.34 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-16 09:30:00 | 137.87 | 2026-02-01 12:15:00 | 124.31 | TARGET_HIT | 0.50 | 9.84% |
| SELL | retest2 | 2026-01-16 11:00:00 | 138.12 | 2026-02-13 09:15:00 | 124.08 | TARGET_HIT | 0.50 | 10.16% |
| SELL | retest2 | 2026-04-13 09:15:00 | 120.11 | 2026-04-13 12:15:00 | 123.29 | STOP_HIT | 1.00 | -2.65% |
