# EICHERMOT (EICHERMOT)

## Backtest Summary

- **Window:** 2025-08-14 09:15:00 → 2026-05-08 15:30:00 (1238 bars)
- **Last close:** 7302.50
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
| ALERT2_SKIP | 1 |
| ALERT3 | 2 |
| PENDING | 5 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 5 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 5
- **Target hits / Stop hits / Partials:** 0 / 5 / 0
- **Avg / median % per leg:** -4.23% / -4.36%
- **Sum % (uncompounded):** -21.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -4.23% | -21.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -4.23% | -21.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -4.23% | -21.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 10:15:00 | 6874.50 | 7358.65 | 7359.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 6745.00 | 7329.92 | 7345.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 11:15:00 | 7072.00 | 7049.55 | 7180.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 09:15:00 | 7321.50 | 7059.41 | 7177.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 7321.50 | 7059.41 | 7177.48 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 7142.50 | 7079.32 | 7183.55 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 7115.00 | 7079.67 | 7183.21 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-22 09:15:00 | 7165.00 | 7104.19 | 7178.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 10:15:00 | 7141.50 | 7104.56 | 7178.40 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-23 09:15:00 | 7178.50 | 7110.58 | 7179.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:15:00 | 7094.50 | 7110.42 | 7178.85 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-28 10:15:00 | 7200.50 | 7115.60 | 7174.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:15:00 | 7065.50 | 7115.10 | 7174.09 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 7233.50 | 7114.90 | 7172.53 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-30 09:15:00 | 6984.00 | 7120.12 | 7173.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 10:15:00 | 7009.50 | 7119.02 | 7172.41 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-05-04 13:15:00 | 7308.00 | 7122.13 | 7171.34 | SL hit (close>static) qty=1.00 sl=7294.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 7403.50 | 7149.57 | 7181.63 | SL hit (close>static) qty=1.00 sl=7358.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 7403.50 | 7149.57 | 7181.63 | SL hit (close>static) qty=1.00 sl=7358.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 7403.50 | 7149.57 | 7181.63 | SL hit (close>static) qty=1.00 sl=7358.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 7403.50 | 7149.57 | 7181.63 | SL hit (close>static) qty=1.00 sl=7358.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-04-13 10:15:00 | 7115.00 | 2026-05-04 13:15:00 | 7308.00 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2026-04-22 10:15:00 | 7141.50 | 2026-05-07 09:15:00 | 7403.50 | STOP_HIT | 1.00 | -3.67% |
| SELL | retest2 | 2026-04-23 10:15:00 | 7094.50 | 2026-05-07 09:15:00 | 7403.50 | STOP_HIT | 1.00 | -4.36% |
| SELL | retest2 | 2026-04-28 11:15:00 | 7065.50 | 2026-05-07 09:15:00 | 7403.50 | STOP_HIT | 1.00 | -4.78% |
| SELL | retest2 | 2026-04-30 10:15:00 | 7009.50 | 2026-05-07 09:15:00 | 7403.50 | STOP_HIT | 1.00 | -5.62% |
