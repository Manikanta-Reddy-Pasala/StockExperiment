# JIOFIN (JIOFIN)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-11 15:15:00 (3605 bars)
- **Last close:** 239.98
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 11 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 11 |
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 8
- **Target hits / Stop hits / Partials:** 3 / 8 / 3
- **Avg / median % per leg:** 2.44% / -1.00%
- **Sum % (uncompounded):** 34.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.59% | -6.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.59% | -6.3% |
| SELL (all) | 10 | 6 | 60.0% | 3 | 4 | 3 | 4.05% | 40.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 6 | 60.0% | 3 | 4 | 3 | 4.05% | 40.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 6 | 42.9% | 3 | 8 | 3 | 2.44% | 34.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 09:15:00 | 294.50 | 311.01 | 311.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 291.90 | 305.13 | 306.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 302.30 | 300.48 | 303.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:45:00 | 301.75 | 300.48 | 303.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 301.55 | 298.92 | 301.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:30:00 | 301.60 | 298.92 | 301.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 302.10 | 298.95 | 301.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:00:00 | 302.10 | 298.95 | 301.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 302.00 | 298.98 | 301.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 14:30:00 | 301.30 | 299.01 | 301.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 300.50 | 299.04 | 301.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 13:00:00 | 301.00 | 299.11 | 301.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 10:00:00 | 300.80 | 299.15 | 301.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 304.30 | 299.10 | 301.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-07 10:15:00 | 304.30 | 299.10 | 301.79 | SL hit (close>static) qty=1.00 sl=302.10 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-09-16 11:30:00 | 317.45 | 2025-09-23 09:15:00 | 311.80 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-09-17 09:15:00 | 316.70 | 2025-09-23 09:15:00 | 311.80 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-09-18 09:15:00 | 316.65 | 2025-09-23 09:15:00 | 311.80 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-09-18 13:00:00 | 316.50 | 2025-09-23 09:15:00 | 311.80 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2026-01-02 14:30:00 | 301.30 | 2026-01-07 10:15:00 | 304.30 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-01-05 09:15:00 | 300.50 | 2026-01-07 10:15:00 | 304.30 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-01-05 13:00:00 | 301.00 | 2026-01-07 10:15:00 | 304.30 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-01-06 10:00:00 | 300.80 | 2026-01-07 10:15:00 | 304.30 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-01-07 13:15:00 | 302.90 | 2026-01-09 14:15:00 | 287.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 14:15:00 | 303.20 | 2026-01-09 14:15:00 | 288.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 301.70 | 2026-01-09 14:15:00 | 286.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 13:15:00 | 302.90 | 2026-01-20 09:15:00 | 272.61 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-07 14:15:00 | 303.20 | 2026-01-20 09:15:00 | 272.88 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 301.70 | 2026-01-20 09:15:00 | 271.53 | TARGET_HIT | 0.50 | 10.00% |
