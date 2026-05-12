# Karur Vysya Bank Ltd. (KARURVYSYA)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 304.80
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
| ALERT3 | 9 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 6 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / Stop hits / Partials:** 1 / 5 / 0
- **Avg / median % per leg:** 0.42% / -1.00%
- **Sum % (uncompounded):** 2.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 1 | 16.7% | 1 | 5 | 0 | 0.42% | 2.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 1 | 16.7% | 1 | 5 | 0 | 0.42% | 2.5% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 6 | 1 | 16.7% | 1 | 5 | 0 | 0.42% | 2.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 10:15:00 | 183.18 | 178.61 | 178.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 185.48 | 178.92 | 178.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 12:15:00 | 183.10 | 183.27 | 181.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 13:00:00 | 183.10 | 183.27 | 181.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 213.20 | 217.29 | 213.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 12:00:00 | 213.20 | 217.29 | 213.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 12:15:00 | 212.50 | 217.24 | 213.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 13:00:00 | 212.50 | 217.24 | 213.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 13:15:00 | 213.05 | 217.20 | 213.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 11:15:00 | 214.35 | 217.02 | 213.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 09:15:00 | 213.22 | 216.86 | 213.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 10:15:00 | 210.18 | 216.75 | 213.09 | SL hit (close<static) qty=1.00 sl=212.25 alert=retest2 |

### Cycle 2 — SELL (started 2026-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-08 14:15:00 | 281.45 | 287.62 | 287.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-09 13:15:00 | 277.25 | 287.26 | 287.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 10:15:00 | 284.55 | 284.43 | 285.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-21 11:00:00 | 284.55 | 284.43 | 285.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 287.55 | 284.46 | 285.87 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 299.20 | 287.13 | 287.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 10:15:00 | 304.90 | 289.45 | 288.36 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-08-29 11:15:00 | 214.35 | 2025-09-01 10:15:00 | 210.18 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-09-01 09:15:00 | 213.22 | 2025-09-01 10:15:00 | 210.18 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-09-17 09:45:00 | 213.89 | 2025-09-24 14:15:00 | 211.76 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-09-23 09:15:00 | 213.43 | 2025-09-24 14:15:00 | 211.76 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-09-29 15:15:00 | 213.98 | 2025-09-30 09:15:00 | 209.05 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-10-06 10:00:00 | 213.86 | 2025-10-20 10:15:00 | 235.25 | TARGET_HIT | 1.00 | 10.00% |
