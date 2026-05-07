# EICHERMOT (EICHERMOT)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2025-11-10 09:15:00 → 2026-05-07 15:15:00 (847 bars)
- **Last close:** 7342.00
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
- **Avg / median % per leg:** -3.95% / -4.16%
- **Sum % (uncompounded):** -19.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -3.95% | -19.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -3.95% | -19.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -3.95% | -19.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 15:15:00 | 6825.00 | 7367.50 | 7368.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 6745.00 | 7328.98 | 7348.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 11:15:00 | 7072.00 | 7049.43 | 7182.43 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 09:15:00 | 7320.00 | 7058.99 | 7179.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 7320.00 | 7058.99 | 7179.57 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 7142.50 | 7079.03 | 7185.63 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 11:15:00 | 7124.50 | 7079.83 | 7184.97 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-04-22 09:15:00 | 7165.00 | 7108.48 | 7181.02 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 11:15:00 | 7180.00 | 7109.51 | 7180.82 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-04-23 09:15:00 | 7180.00 | 7114.52 | 7181.59 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 11:15:00 | 7108.00 | 7114.24 | 7180.79 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-04-28 10:15:00 | 7200.50 | 7118.91 | 7176.80 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 12:15:00 | 7088.00 | 7118.08 | 7175.80 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 7231.00 | 7118.00 | 7174.61 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-30 09:15:00 | 6982.50 | 7122.99 | 7175.23 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 11:15:00 | 7020.00 | 7120.84 | 7173.63 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-05-04 13:15:00 | 7308.50 | 7124.71 | 7173.25 | SL hit (close>static) qty=1.00 sl=7294.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 7403.50 | 7151.76 | 7183.38 | SL hit (close>static) qty=1.00 sl=7358.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 7403.50 | 7151.76 | 7183.38 | SL hit (close>static) qty=1.00 sl=7358.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 7403.50 | 7151.76 | 7183.38 | SL hit (close>static) qty=1.00 sl=7358.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 7403.50 | 7151.76 | 7183.38 | SL hit (close>static) qty=1.00 sl=7358.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-04-13 11:15:00 | 7124.50 | 2026-05-04 13:15:00 | 7308.50 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2026-04-22 11:15:00 | 7180.00 | 2026-05-07 09:15:00 | 7403.50 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2026-04-23 11:15:00 | 7108.00 | 2026-05-07 09:15:00 | 7403.50 | STOP_HIT | 1.00 | -4.16% |
| SELL | retest2 | 2026-04-28 12:15:00 | 7088.00 | 2026-05-07 09:15:00 | 7403.50 | STOP_HIT | 1.00 | -4.45% |
| SELL | retest2 | 2026-04-30 11:15:00 | 7020.00 | 2026-05-07 09:15:00 | 7403.50 | STOP_HIT | 1.00 | -5.46% |
