# Siemens Energy India Ltd. (ENRIN)

## Backtest Summary

- **Window:** 2025-06-19 09:15:00 → 2026-05-08 15:15:00 (1528 bars)
- **Last close:** 3186.00
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
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 7 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 5
- **Target hits / Stop hits / Partials:** 0 / 5 / 0
- **Avg / median % per leg:** -8.06% / -9.03%
- **Sum % (uncompounded):** -40.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -8.06% | -40.3% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -9.09% | -36.4% |
| SELL @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.92% | -3.9% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -9.09% | -36.4% |
| retest2 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.92% | -3.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 09:15:00 | 3127.20 | 3232.01 | 3232.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 13:15:00 | 3116.50 | 3215.11 | 3223.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 15:15:00 | 2508.00 | 2493.38 | 2703.64 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-30 09:15:00 | 2454.00 | 2493.38 | 2703.64 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-30 10:45:00 | 2452.40 | 2492.63 | 2701.17 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-30 13:45:00 | 2450.00 | 2491.24 | 2697.36 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-01 11:45:00 | 2454.10 | 2490.28 | 2691.79 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 2675.60 | 2504.08 | 2668.11 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 2675.60 | 2504.08 | 2668.11 | SL hit (close>ema400) qty=1.00 sl=2668.11 alert=retest1 |

### Cycle 2 — BUY (started 2026-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 11:15:00 | 2960.00 | 2739.92 | 2739.15 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2026-03-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 15:15:00 | 2599.00 | 2759.36 | 2759.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 2540.60 | 2747.73 | 2753.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-09 11:15:00 | 2770.30 | 2722.34 | 2738.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 11:15:00 | 2770.30 | 2722.34 | 2738.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 11:15:00 | 2770.30 | 2722.34 | 2738.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 12:00:00 | 2770.30 | 2722.34 | 2738.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 12:15:00 | 2748.70 | 2722.60 | 2738.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 13:30:00 | 2736.50 | 2722.87 | 2738.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 2843.70 | 2724.97 | 2739.79 | SL hit (close>static) qty=1.00 sl=2771.00 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 11:15:00 | 2892.00 | 2753.92 | 2753.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 2963.50 | 2761.77 | 2757.31 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-01-30 09:15:00 | 2454.00 | 2026-02-06 09:15:00 | 2675.60 | STOP_HIT | 1.00 | -9.03% |
| SELL | retest1 | 2026-01-30 10:45:00 | 2452.40 | 2026-02-06 09:15:00 | 2675.60 | STOP_HIT | 1.00 | -9.10% |
| SELL | retest1 | 2026-01-30 13:45:00 | 2450.00 | 2026-02-06 09:15:00 | 2675.60 | STOP_HIT | 1.00 | -9.21% |
| SELL | retest1 | 2026-02-01 11:45:00 | 2454.10 | 2026-02-06 09:15:00 | 2675.60 | STOP_HIT | 1.00 | -9.03% |
| SELL | retest2 | 2026-04-09 13:30:00 | 2736.50 | 2026-04-10 09:15:00 | 2843.70 | STOP_HIT | 1.00 | -3.92% |
