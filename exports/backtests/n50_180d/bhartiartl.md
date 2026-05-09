# BHARTIARTL (BHARTIARTL)

## Backtest Summary

- **Window:** 2024-10-07 09:15:00 → 2026-05-08 15:15:00 (2741 bars)
- **Last close:** 1834.70
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
| ALERT3 | 4 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 10 |
| PARTIAL | 4 |
| TARGET_HIT | 5 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 6
- **Target hits / Stop hits / Partials:** 4 / 6 / 4
- **Avg / median % per leg:** 3.95% / 5.00%
- **Sum % (uncompounded):** 55.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 8 | 57.1% | 4 | 6 | 4 | 3.95% | 55.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 8 | 57.1% | 4 | 6 | 4 | 3.95% | 55.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 8 | 57.1% | 4 | 6 | 4 | 3.95% | 55.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 09:15:00 | 2002.80 | 2056.72 | 2056.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 1997.40 | 2055.12 | 2055.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 10:15:00 | 2022.20 | 2019.76 | 2035.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-04 10:45:00 | 2021.40 | 2019.76 | 2035.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 2034.10 | 2018.71 | 2033.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:00:00 | 2034.10 | 2018.71 | 2033.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 2029.00 | 2018.82 | 2033.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 13:15:00 | 2016.90 | 2021.13 | 2034.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 13:45:00 | 2014.10 | 2021.03 | 2033.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 12:15:00 | 2016.10 | 2020.47 | 2033.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 14:45:00 | 2014.50 | 2020.21 | 2032.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 2030.00 | 2019.30 | 2031.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:30:00 | 2026.30 | 2019.36 | 2031.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 10:30:00 | 2026.70 | 2019.44 | 2031.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 11:15:00 | 2034.80 | 2019.59 | 2031.18 | SL hit (close>static) qty=1.00 sl=2034.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 11:15:00 | 2034.80 | 2019.59 | 2031.18 | SL hit (close>static) qty=1.00 sl=2034.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 11:15:00 | 2034.80 | 2019.59 | 2031.18 | SL hit (close>static) qty=1.00 sl=2034.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 11:15:00 | 2034.80 | 2019.59 | 2031.18 | SL hit (close>static) qty=1.00 sl=2034.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 11:15:00 | 2034.80 | 2019.59 | 2031.18 | SL hit (close>static) qty=1.00 sl=2032.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 11:15:00 | 2034.80 | 2019.59 | 2031.18 | SL hit (close>static) qty=1.00 sl=2032.20 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 13:45:00 | 2024.20 | 2019.69 | 2031.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:45:00 | 2021.50 | 2019.74 | 2030.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 2015.30 | 2019.84 | 2030.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 10:30:00 | 2011.80 | 2019.79 | 2030.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:15:00 | 2011.50 | 2019.79 | 2030.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:15:00 | 1922.99 | 2013.70 | 2026.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 12:15:00 | 1920.42 | 2006.81 | 2022.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 13:15:00 | 1911.21 | 2005.87 | 2021.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 13:15:00 | 1910.92 | 2005.87 | 2021.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 1821.78 | 1982.28 | 2007.44 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 1819.35 | 1982.28 | 2007.44 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-11 11:15:00 | 1810.62 | 1948.23 | 1984.81 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-11 11:15:00 | 1810.35 | 1948.23 | 1984.81 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-02-10 13:15:00 | 2016.90 | 2026-02-17 11:15:00 | 2034.80 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2026-02-10 13:45:00 | 2014.10 | 2026-02-17 11:15:00 | 2034.80 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2026-02-11 12:15:00 | 2016.10 | 2026-02-17 11:15:00 | 2034.80 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2026-02-12 14:45:00 | 2014.50 | 2026-02-17 11:15:00 | 2034.80 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2026-02-17 09:30:00 | 2026.30 | 2026-02-17 11:15:00 | 2034.80 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2026-02-17 10:30:00 | 2026.70 | 2026-02-17 11:15:00 | 2034.80 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2026-02-17 13:45:00 | 2024.20 | 2026-02-24 09:15:00 | 1922.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 09:45:00 | 2021.50 | 2026-02-25 12:15:00 | 1920.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 10:30:00 | 2011.80 | 2026-02-25 13:15:00 | 1911.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 11:15:00 | 2011.50 | 2026-02-25 13:15:00 | 1910.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 13:45:00 | 2024.20 | 2026-03-04 09:15:00 | 1821.78 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-18 09:45:00 | 2021.50 | 2026-03-04 09:15:00 | 1819.35 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-19 10:30:00 | 2011.80 | 2026-03-11 11:15:00 | 1810.62 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-19 11:15:00 | 2011.50 | 2026-03-11 11:15:00 | 1810.35 | TARGET_HIT | 0.50 | 10.00% |
