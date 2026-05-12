# Jindal Saw Ltd. (JINDALSAW)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 243.50
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
| ALERT2_SKIP | 1 |
| ALERT3 | 10 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 11 |
| PARTIAL | 0 |
| TARGET_HIT | 6 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 5
- **Target hits / Stop hits / Partials:** 6 / 5 / 0
- **Avg / median % per leg:** 4.76% / 10.00%
- **Sum % (uncompounded):** 52.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 6 | 54.5% | 6 | 5 | 0 | 4.76% | 52.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 6 | 54.5% | 6 | 5 | 0 | 4.76% | 52.4% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 11 | 6 | 54.5% | 6 | 5 | 0 | 4.76% | 52.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 13:15:00 | 189.53 | 174.46 | 174.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 191.20 | 177.11 | 175.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 180.15 | 182.13 | 178.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 12:15:00 | 180.56 | 182.08 | 179.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 180.56 | 182.08 | 179.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 13:15:00 | 180.74 | 182.08 | 179.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 14:15:00 | 178.26 | 182.02 | 179.02 | SL hit (close<static) qty=1.00 sl=179.00 alert=retest2 |

### Cycle 2 — SELL (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 13:15:00 | 165.30 | 177.69 | 177.69 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 11:15:00 | 188.06 | 177.77 | 177.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 13:15:00 | 190.20 | 178.01 | 177.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 183.78 | 184.87 | 181.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-23 10:00:00 | 183.78 | 184.87 | 181.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 182.50 | 184.85 | 181.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 09:15:00 | 184.95 | 184.57 | 181.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 180.17 | 184.52 | 181.71 | SL hit (close<static) qty=1.00 sl=181.62 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2026-02-19 13:15:00 | 180.74 | 2026-02-19 14:15:00 | 178.26 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-02-20 10:45:00 | 180.86 | 2026-02-23 10:15:00 | 178.95 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2026-02-20 11:30:00 | 181.13 | 2026-02-23 10:15:00 | 178.95 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-02-20 12:15:00 | 181.45 | 2026-02-23 10:15:00 | 178.95 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-03-24 09:15:00 | 184.95 | 2026-03-24 09:15:00 | 180.17 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2026-03-25 09:15:00 | 185.96 | 2026-04-09 09:15:00 | 204.56 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-27 12:15:00 | 184.65 | 2026-04-09 09:15:00 | 203.12 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-27 13:00:00 | 185.43 | 2026-04-09 09:15:00 | 203.97 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 15:00:00 | 195.50 | 2026-04-15 10:15:00 | 215.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 10:00:00 | 196.55 | 2026-04-15 10:15:00 | 216.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-08 09:15:00 | 195.82 | 2026-04-15 10:15:00 | 215.40 | TARGET_HIT | 1.00 | 10.00% |
