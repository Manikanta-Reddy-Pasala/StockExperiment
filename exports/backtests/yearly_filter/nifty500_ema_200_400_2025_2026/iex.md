# Indian Energy Exchange Ltd. (IEX)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 134.07
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
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 26 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 22 |
| PARTIAL | 11 |
| TARGET_HIT | 6 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 33 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 15
- **Target hits / Stop hits / Partials:** 6 / 16 / 11
- **Avg / median % per leg:** 1.72% / 5.00%
- **Sum % (uncompounded):** 56.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 4 | 100.0% | 4 | 0 | 0 | 10.00% | 40.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 4 | 100.0% | 4 | 0 | 0 | 10.00% | 40.0% |
| SELL (all) | 29 | 14 | 48.3% | 2 | 16 | 11 | 0.58% | 16.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 29 | 14 | 48.3% | 2 | 16 | 11 | 0.58% | 16.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 33 | 18 | 54.5% | 6 | 16 | 11 | 1.72% | 56.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 14:15:00 | 143.63 | 191.33 | 191.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 09:15:00 | 140.38 | 190.37 | 191.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 147.28 | 146.60 | 156.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 10:00:00 | 147.28 | 146.60 | 156.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 148.15 | 140.97 | 146.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 12:00:00 | 148.15 | 140.97 | 146.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 147.42 | 141.03 | 146.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 13:15:00 | 147.08 | 141.03 | 146.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 10:45:00 | 146.59 | 141.32 | 146.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 12:45:00 | 147.31 | 141.44 | 146.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:00:00 | 147.39 | 141.88 | 146.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 148.07 | 142.05 | 147.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:15:00 | 148.60 | 142.05 | 147.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 148.67 | 142.11 | 147.00 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 148.67 | 142.11 | 147.00 | SL hit (close>static) qty=1.00 sl=148.49 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-27 10:15:00 | 191.14 | 2025-07-09 12:15:00 | 210.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-27 11:30:00 | 191.25 | 2025-07-09 12:15:00 | 210.38 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-27 12:15:00 | 191.52 | 2025-07-09 12:15:00 | 210.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-30 09:30:00 | 191.01 | 2025-07-09 12:15:00 | 210.11 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-24 13:15:00 | 147.08 | 2025-10-29 10:15:00 | 148.67 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-10-27 10:45:00 | 146.59 | 2025-10-29 10:15:00 | 148.67 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-10-27 12:45:00 | 147.31 | 2025-10-29 10:15:00 | 148.67 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-10-28 14:00:00 | 147.39 | 2025-10-29 10:15:00 | 148.67 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-10-30 11:30:00 | 146.15 | 2025-10-31 14:15:00 | 138.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 11:30:00 | 146.15 | 2025-11-20 09:15:00 | 140.75 | STOP_HIT | 0.50 | 3.69% |
| SELL | retest2 | 2025-12-01 11:00:00 | 146.20 | 2025-12-02 10:15:00 | 147.66 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-12-01 14:30:00 | 146.16 | 2025-12-02 10:15:00 | 147.66 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-12-02 09:15:00 | 145.65 | 2025-12-02 10:15:00 | 147.66 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-12-08 12:15:00 | 142.32 | 2025-12-26 09:15:00 | 135.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 10:15:00 | 142.21 | 2025-12-26 09:15:00 | 135.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 10:45:00 | 142.05 | 2025-12-26 09:15:00 | 134.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 11:45:00 | 142.70 | 2025-12-26 09:15:00 | 135.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-15 15:00:00 | 142.22 | 2025-12-26 09:15:00 | 135.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-23 11:15:00 | 142.03 | 2025-12-26 09:15:00 | 134.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-23 14:15:00 | 142.17 | 2025-12-26 09:15:00 | 135.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-24 09:45:00 | 142.10 | 2025-12-26 09:15:00 | 134.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-08 12:15:00 | 142.32 | 2026-01-06 14:15:00 | 151.88 | STOP_HIT | 0.50 | -6.72% |
| SELL | retest2 | 2025-12-12 10:15:00 | 142.21 | 2026-01-06 14:15:00 | 151.88 | STOP_HIT | 0.50 | -6.80% |
| SELL | retest2 | 2025-12-12 10:45:00 | 142.05 | 2026-01-06 14:15:00 | 151.88 | STOP_HIT | 0.50 | -6.92% |
| SELL | retest2 | 2025-12-12 11:45:00 | 142.70 | 2026-01-06 14:15:00 | 151.88 | STOP_HIT | 0.50 | -6.43% |
| SELL | retest2 | 2025-12-15 15:00:00 | 142.22 | 2026-01-06 14:15:00 | 151.88 | STOP_HIT | 0.50 | -6.79% |
| SELL | retest2 | 2025-12-23 11:15:00 | 142.03 | 2026-01-06 14:15:00 | 151.88 | STOP_HIT | 0.50 | -6.94% |
| SELL | retest2 | 2025-12-23 14:15:00 | 142.17 | 2026-01-06 14:15:00 | 151.88 | STOP_HIT | 0.50 | -6.83% |
| SELL | retest2 | 2025-12-24 09:45:00 | 142.10 | 2026-01-06 14:15:00 | 151.88 | STOP_HIT | 0.50 | -6.88% |
| SELL | retest2 | 2026-01-19 13:00:00 | 137.42 | 2026-01-20 13:15:00 | 130.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 14:45:00 | 137.77 | 2026-01-20 13:15:00 | 130.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 13:00:00 | 137.42 | 2026-02-01 12:15:00 | 123.68 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-19 14:45:00 | 137.77 | 2026-02-01 12:15:00 | 123.99 | TARGET_HIT | 0.50 | 10.00% |
