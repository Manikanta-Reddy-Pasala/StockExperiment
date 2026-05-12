# Bajaj Housing Finance Ltd. (BAJAJHFL)

## Backtest Summary

- **Window:** 2024-09-16 09:15:00 → 2026-05-11 15:15:00 (2846 bars)
- **Last close:** 86.29
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 2 |
| ALERT3 | 17 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 23 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 22
- **Target hits / Stop hits / Partials:** 1 / 22 / 1
- **Avg / median % per leg:** -0.18% / -0.66%
- **Sum % (uncompounded):** -4.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 0 | 0.0% | 0 | 10 | 0 | -0.77% | -7.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 0 | 0.0% | 0 | 10 | 0 | -0.77% | -7.7% |
| SELL (all) | 14 | 2 | 14.3% | 1 | 12 | 1 | 0.23% | 3.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 2 | 14.3% | 1 | 12 | 1 | 0.23% | 3.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 24 | 2 | 8.3% | 1 | 22 | 1 | -0.18% | -4.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 12:15:00 | 129.49 | 119.24 | 119.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 09:15:00 | 131.68 | 120.98 | 120.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 09:15:00 | 123.27 | 123.36 | 121.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 15:15:00 | 121.74 | 123.33 | 121.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 15:15:00 | 121.74 | 123.33 | 121.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 09:15:00 | 123.33 | 123.33 | 121.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 12:30:00 | 122.25 | 123.31 | 121.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-02 15:15:00 | 121.50 | 123.27 | 121.70 | SL hit (close<static) qty=1.00 sl=121.51 alert=retest2 |

### Cycle 2 — SELL (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 09:15:00 | 121.67 | 122.16 | 122.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 10:15:00 | 121.40 | 122.16 | 122.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 09:15:00 | 122.35 | 122.14 | 122.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 09:15:00 | 122.35 | 122.14 | 122.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 122.35 | 122.14 | 122.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 15:00:00 | 122.04 | 122.15 | 122.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 09:45:00 | 122.00 | 122.15 | 122.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 10:45:00 | 122.03 | 122.15 | 122.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 12:00:00 | 121.97 | 122.15 | 122.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 121.86 | 122.15 | 122.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 09:45:00 | 121.74 | 122.13 | 122.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 09:30:00 | 121.43 | 122.08 | 122.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 09:15:00 | 122.99 | 121.99 | 122.07 | SL hit (close>static) qty=1.00 sl=122.43 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-02 09:15:00 | 123.33 | 2025-05-02 15:15:00 | 121.50 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-05-02 12:30:00 | 122.25 | 2025-05-02 15:15:00 | 121.50 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-05-05 09:45:00 | 122.41 | 2025-05-06 09:15:00 | 120.67 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-05-05 12:00:00 | 122.38 | 2025-05-06 09:15:00 | 120.67 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-05-14 09:15:00 | 122.52 | 2025-06-13 12:15:00 | 121.20 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-06-04 10:00:00 | 121.57 | 2025-06-13 12:15:00 | 121.20 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-06-04 13:00:00 | 121.65 | 2025-06-13 12:15:00 | 121.20 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-06-13 09:45:00 | 121.80 | 2025-06-13 12:15:00 | 121.20 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-06-25 13:00:00 | 121.97 | 2025-06-26 09:15:00 | 121.67 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-06-26 09:15:00 | 121.97 | 2025-06-26 09:15:00 | 121.67 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-06-27 15:00:00 | 122.04 | 2025-07-04 09:15:00 | 122.99 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-06-30 09:45:00 | 122.00 | 2025-07-04 09:15:00 | 122.99 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-06-30 10:45:00 | 122.03 | 2025-07-16 10:15:00 | 122.54 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-06-30 12:00:00 | 121.97 | 2025-07-16 10:15:00 | 122.54 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-07-01 09:45:00 | 121.74 | 2025-07-16 10:15:00 | 122.54 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-07-02 09:30:00 | 121.43 | 2025-07-16 10:15:00 | 122.54 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-07-07 09:15:00 | 121.57 | 2025-07-16 10:15:00 | 122.54 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-07-09 10:45:00 | 121.76 | 2025-07-16 10:15:00 | 122.54 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-07-10 11:30:00 | 121.31 | 2025-07-16 14:15:00 | 123.31 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-07-11 09:15:00 | 121.45 | 2025-07-16 14:15:00 | 123.31 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-07-11 09:45:00 | 121.48 | 2025-07-16 14:15:00 | 123.31 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-07-15 11:45:00 | 121.41 | 2025-07-16 14:15:00 | 123.31 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-07-24 12:30:00 | 120.72 | 2025-07-29 09:15:00 | 114.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 12:30:00 | 120.72 | 2025-08-13 14:15:00 | 108.65 | TARGET_HIT | 0.50 | 10.00% |
