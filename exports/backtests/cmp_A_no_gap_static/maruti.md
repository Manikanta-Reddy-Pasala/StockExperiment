# MARUTI (MARUTI)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 13733.00
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
| ALERT2_SKIP | 1 |
| ALERT3 | 2 |
| PENDING | 10 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 4 |
| ENTRY2 | 5 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 1 / 5
- **Target hits / Stop hits / Partials:** 1 / 5 / 0
- **Avg / median % per leg:** -1.69% / -4.63%
- **Sum % (uncompounded):** -10.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 1 | 50.0% | 1 | 1 | 0 | 4.61% | 9.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 1 | 50.0% | 1 | 1 | 0 | 4.61% | 9.2% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.83% | -19.3% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.83% | -19.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.83% | -19.3% |
| retest2 (combined) | 2 | 1 | 50.0% | 1 | 1 | 0 | 4.61% | 9.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 13:15:00 | 12832.00 | 12530.85 | 12529.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 10:15:00 | 12893.00 | 12543.50 | 12536.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-03 09:15:00 | 15664.00 | 16004.21 | 15351.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 14:15:00 | 15362.00 | 15951.51 | 15362.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 15362.00 | 15951.51 | 15362.80 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-11-06 12:15:00 | 15423.00 | 15924.28 | 15363.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 13:15:00 | 15448.00 | 15919.54 | 15363.96 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 15326.00 | 15904.38 | 15364.60 | SL hit (close<static) qty=1.00 sl=15358.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-07 12:15:00 | 15435.00 | 15889.23 | 15365.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 13:15:00 | 15433.00 | 15884.69 | 15365.34 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Target hit | 2026-01-02 09:15:00 | 16976.30 | 16355.48 | 16042.89 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 13:15:00 | 14511.00 | 16031.42 | 16032.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 09:15:00 | 14425.00 | 15985.22 | 16009.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 13626.00 | 13304.14 | 14062.91 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 13180.00 | 13365.71 | 14019.73 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 10:15:00 | 13132.00 | 13363.39 | 14015.30 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-23 10:15:00 | 13183.00 | 13370.74 | 13876.94 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-23 11:15:00 | 13219.00 | 13369.23 | 13873.66 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-23 12:15:00 | 13163.00 | 13367.18 | 13870.11 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 13:15:00 | 13132.00 | 13364.84 | 13866.43 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-28 11:15:00 | 13136.00 | 13325.55 | 13800.26 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 12:15:00 | 13122.00 | 13323.53 | 13796.88 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-30 09:15:00 | 13015.00 | 13316.07 | 13767.69 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:15:00 | 13040.00 | 13313.33 | 13764.06 | SELL ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 13748.00 | 13312.84 | 13750.44 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-05-04 11:15:00 | 13525.00 | 13318.99 | 13749.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:15:00 | 13502.00 | 13320.81 | 13747.94 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-05-05 09:15:00 | 13448.00 | 13329.29 | 13743.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 10:15:00 | 13395.00 | 13329.94 | 13742.03 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-05-06 10:15:00 | 13433.00 | 13340.03 | 13732.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 11:15:00 | 13517.00 | 13341.79 | 13731.91 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 13740.00 | 13351.78 | 13729.22 | SL hit (close>ema400) qty=1.00 sl=13729.22 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 13740.00 | 13351.78 | 13729.22 | SL hit (close>ema400) qty=1.00 sl=13729.22 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 13740.00 | 13351.78 | 13729.22 | SL hit (close>ema400) qty=1.00 sl=13729.22 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 13740.00 | 13351.78 | 13729.22 | SL hit (close>ema400) qty=1.00 sl=13729.22 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-11-06 13:15:00 | 15448.00 | 2025-11-07 09:15:00 | 15326.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-11-07 13:15:00 | 15433.00 | 2026-01-02 09:15:00 | 16976.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2026-04-13 10:15:00 | 13132.00 | 2026-05-06 15:15:00 | 13740.00 | STOP_HIT | 1.00 | -4.63% |
| SELL | retest1 | 2026-04-23 13:15:00 | 13132.00 | 2026-05-06 15:15:00 | 13740.00 | STOP_HIT | 1.00 | -4.63% |
| SELL | retest1 | 2026-04-28 12:15:00 | 13122.00 | 2026-05-06 15:15:00 | 13740.00 | STOP_HIT | 1.00 | -4.71% |
| SELL | retest1 | 2026-04-30 10:15:00 | 13040.00 | 2026-05-06 15:15:00 | 13740.00 | STOP_HIT | 1.00 | -5.37% |
