# HINDUNILVR (HINDUNILVR)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2025-11-10 09:15:00 → 2026-05-07 15:15:00 (847 bars)
- **Last close:** 2273.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 1 |
| PENDING | 3 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 1 |
| ENTRY2 | 0 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 1 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 1
- **Target hits / Stop hits / Partials:** 0 / 1 / 0
- **Avg / median % per leg:** -4.56% / -4.56%
- **Sum % (uncompounded):** -4.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -4.56% | -4.6% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -4.56% | -4.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -4.56% | -4.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 2280.00 | 2363.31 | 2363.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 14:15:00 | 2260.00 | 2351.82 | 2357.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 2162.70 | 2162.15 | 2230.76 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-04-09 12:15:00 | 2123.30 | 2160.56 | 2226.61 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-09 14:15:00 | 2133.90 | 2159.91 | 2225.62 | ENTRY1 sustain failed after 120m |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 2119.70 | 2158.96 | 2222.24 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-13 10:15:00 | 2136.30 | 2158.73 | 2221.82 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-13 12:15:00 | 2121.90 | 2158.11 | 2220.87 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 14:15:00 | 2129.00 | 2157.52 | 2219.95 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 10:15:00 | 2226.10 | 2157.30 | 2214.73 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-04-17 10:15:00 | 2226.10 | 2157.30 | 2214.73 | SL hit (close>ema400) qty=1.00 sl=2214.73 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-04-13 14:15:00 | 2129.00 | 2026-04-17 10:15:00 | 2226.10 | STOP_HIT | 1.00 | -4.56% |
