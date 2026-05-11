# Bank of Maharashtra (MAHABANK)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 83.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 19 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 17 |
| PARTIAL | 0 |
| TARGET_HIT | 6 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 11
- **Target hits / Stop hits / Partials:** 6 / 11 / 0
- **Avg / median % per leg:** 2.69% / -0.95%
- **Sum % (uncompounded):** 45.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 6 | 35.3% | 6 | 11 | 0 | 2.69% | 45.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 17 | 6 | 35.3% | 6 | 11 | 0 | 2.69% | 45.7% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 17 | 6 | 35.3% | 6 | 11 | 0 | 2.69% | 45.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 14:15:00 | 52.49 | 54.57 | 54.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 11:15:00 | 52.33 | 54.38 | 54.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 14:15:00 | 54.04 | 53.97 | 54.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 15:00:00 | 54.04 | 53.97 | 54.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 54.19 | 53.98 | 54.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:15:00 | 54.79 | 53.98 | 54.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 54.55 | 53.98 | 54.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:30:00 | 55.07 | 53.98 | 54.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 54.74 | 53.99 | 54.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:00:00 | 54.74 | 53.99 | 54.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 54.40 | 54.02 | 54.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:45:00 | 54.50 | 54.02 | 54.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 54.22 | 54.02 | 54.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:30:00 | 54.40 | 54.02 | 54.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 54.32 | 54.03 | 54.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:00:00 | 54.32 | 54.03 | 54.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 54.41 | 54.03 | 54.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:00:00 | 54.41 | 54.03 | 54.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 54.55 | 54.04 | 54.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:00:00 | 54.55 | 54.04 | 54.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2025-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 13:15:00 | 57.10 | 54.45 | 54.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 14:15:00 | 57.26 | 54.48 | 54.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 54.47 | 55.19 | 54.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 54.47 | 55.19 | 54.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 54.47 | 55.19 | 54.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:30:00 | 54.65 | 55.19 | 54.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 54.35 | 55.18 | 54.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:00:00 | 54.35 | 55.18 | 54.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 54.72 | 55.13 | 54.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:45:00 | 54.80 | 55.13 | 54.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 55.28 | 55.13 | 54.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 13:15:00 | 55.31 | 55.13 | 54.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 14:00:00 | 55.30 | 55.13 | 54.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 56.04 | 55.13 | 54.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 15:15:00 | 55.60 | 55.97 | 55.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 55.60 | 55.97 | 55.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 57.15 | 55.97 | 55.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-16 09:15:00 | 60.84 | 56.18 | 55.55 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-08-04 09:15:00 | 54.84 | 2025-08-06 14:15:00 | 53.98 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-08-04 12:00:00 | 54.45 | 2025-08-06 14:15:00 | 53.98 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-08-04 12:45:00 | 54.50 | 2025-08-06 14:15:00 | 53.98 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-08-06 12:15:00 | 54.43 | 2025-08-06 14:15:00 | 53.98 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-08-08 11:15:00 | 55.32 | 2025-08-14 14:15:00 | 54.40 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-08-08 12:15:00 | 55.29 | 2025-08-14 14:15:00 | 54.40 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-08-13 12:45:00 | 55.27 | 2025-08-14 14:15:00 | 54.40 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-08-18 09:45:00 | 55.24 | 2025-08-25 09:15:00 | 54.36 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-08-18 14:15:00 | 54.93 | 2025-08-25 09:15:00 | 54.36 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-08-19 09:45:00 | 54.98 | 2025-08-25 09:15:00 | 54.36 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-08-22 12:45:00 | 55.23 | 2025-08-25 09:15:00 | 54.36 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-09-29 13:15:00 | 55.31 | 2025-10-16 09:15:00 | 60.84 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-29 14:00:00 | 55.30 | 2025-10-16 09:15:00 | 60.83 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-30 09:15:00 | 56.04 | 2025-10-16 09:15:00 | 61.16 | TARGET_HIT | 1.00 | 9.14% |
| BUY | retest2 | 2025-10-14 15:15:00 | 55.60 | 2025-12-31 09:15:00 | 61.64 | TARGET_HIT | 1.00 | 10.87% |
| BUY | retest2 | 2025-10-15 09:15:00 | 57.15 | 2025-12-31 09:15:00 | 61.58 | TARGET_HIT | 1.00 | 7.75% |
| BUY | retest2 | 2025-12-09 10:00:00 | 55.98 | 2026-01-01 09:15:00 | 62.87 | TARGET_HIT | 1.00 | 12.30% |
