# SBILIFE (SBILIFE)

## Backtest Summary

- **Window:** 2024-10-07 09:15:00 → 2026-05-08 15:15:00 (2741 bars)
- **Last close:** 1871.10
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
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 10 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 9
- **Target hits / Stop hits / Partials:** 0 / 10 / 1
- **Avg / median % per leg:** -0.79% / -1.58%
- **Sum % (uncompounded):** -8.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.83% | -14.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.83% | -14.6% |
| SELL (all) | 3 | 2 | 66.7% | 0 | 2 | 1 | 1.99% | 6.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 3 | 2 | 66.7% | 0 | 2 | 1 | 1.99% | 6.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 11 | 2 | 18.2% | 0 | 10 | 1 | -0.79% | -8.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 15:15:00 | 1963.00 | 2013.11 | 2013.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 09:15:00 | 1940.10 | 2012.38 | 2012.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 1914.90 | 1898.14 | 1942.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 11:00:00 | 1914.90 | 1898.14 | 1942.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 1942.30 | 1899.91 | 1940.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 10:00:00 | 1942.30 | 1899.91 | 1940.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 1946.00 | 1900.37 | 1940.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 11:00:00 | 1946.00 | 1900.37 | 1940.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 11:15:00 | 1942.90 | 1900.80 | 1940.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 12:30:00 | 1933.00 | 1901.10 | 1940.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1960.00 | 1903.55 | 1939.71 | SL hit (close>static) qty=1.00 sl=1950.60 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 1920.00 | 1920.14 | 1943.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 11:15:00 | 1824.00 | 1914.98 | 1939.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 1874.70 | 1872.19 | 1908.02 | SL hit (close>ema200) qty=0.50 sl=1872.19 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2026-01-27 09:45:00 | 2030.00 | 2026-01-29 10:15:00 | 1999.90 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2026-01-27 11:30:00 | 2032.00 | 2026-01-29 10:15:00 | 1999.90 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2026-01-27 12:00:00 | 2029.80 | 2026-01-29 10:15:00 | 1999.90 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2026-02-03 09:15:00 | 2047.70 | 2026-02-03 14:15:00 | 2000.50 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2026-02-10 12:45:00 | 2012.00 | 2026-03-04 09:15:00 | 1972.00 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2026-02-12 10:30:00 | 2010.00 | 2026-03-04 09:15:00 | 1972.00 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2026-02-12 12:45:00 | 2011.20 | 2026-03-04 09:15:00 | 1972.00 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2026-03-02 10:45:00 | 2011.80 | 2026-03-04 09:15:00 | 1972.00 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2026-04-10 12:30:00 | 1933.00 | 2026-04-15 09:15:00 | 1960.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-04-21 09:15:00 | 1920.00 | 2026-04-23 11:15:00 | 1824.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-21 09:15:00 | 1920.00 | 2026-05-07 09:15:00 | 1874.70 | STOP_HIT | 0.50 | 2.36% |
