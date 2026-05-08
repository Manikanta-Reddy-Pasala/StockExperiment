# HINDUNILVR (HINDUNILVR)

## Backtest Summary

- **Window:** 2025-11-10 09:15:00 → 2026-05-08 15:15:00 (854 bars)
- **Last close:** 2286.00
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
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 1 |
| PENDING | 3 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 2 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 2
- **Target hits / Stop hits / Partials:** 0 / 2 / 0
- **Avg / median % per leg:** -4.79% / -4.62%
- **Sum % (uncompounded):** -9.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -4.79% | -9.6% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -4.79% | -9.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -4.79% | -9.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 2312.40 | 2357.95 | 2358.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 2301.00 | 2357.39 | 2357.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 2162.70 | 2162.25 | 2229.81 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-09 12:15:00 | 2123.30 | 2160.65 | 2225.71 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 13:15:00 | 2121.10 | 2160.26 | 2225.18 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 2119.70 | 2159.04 | 2221.39 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-13 10:15:00 | 2136.30 | 2158.81 | 2220.97 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-13 12:15:00 | 2121.90 | 2158.19 | 2220.03 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 13:15:00 | 2127.70 | 2157.88 | 2219.57 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 10:15:00 | 2226.10 | 2157.36 | 2213.97 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-17 10:15:00 | 2226.10 | 2157.36 | 2213.97 | SL hit (close>ema400) qty=1.00 sl=2213.97 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-17 10:15:00 | 2226.10 | 2157.36 | 2213.97 | SL hit (close>ema400) qty=1.00 sl=2213.97 alert=retest1 |

### Cycle 2 — BUY (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 09:15:00 | 2271.10 | 2249.01 | 2248.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 11:15:00 | 2293.00 | 2249.76 | 2249.36 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-04-09 13:15:00 | 2121.10 | 2026-04-17 10:15:00 | 2226.10 | STOP_HIT | 1.00 | -4.95% |
| SELL | retest1 | 2026-04-13 13:15:00 | 2127.70 | 2026-04-17 10:15:00 | 2226.10 | STOP_HIT | 1.00 | -4.62% |
