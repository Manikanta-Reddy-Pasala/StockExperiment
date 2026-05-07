# BPCL (BPCL)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2025-11-10 09:15:00 → 2026-05-07 15:15:00 (847 bars)
- **Last close:** 308.20
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
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 2 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 0
- **Avg / median % per leg:** -2.11% / -1.94%
- **Sum % (uncompounded):** -6.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.11% | -6.3% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.76% | -1.8% |
| BUY @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.28% | -4.6% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.76% | -1.8% |
| retest2 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.28% | -4.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 10:15:00 | 383.20 | 365.20 | 365.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 14:15:00 | 386.55 | 365.98 | 365.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-16 09:15:00 | 370.35 | 370.91 | 368.32 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-02-16 14:15:00 | 373.80 | 370.95 | 368.40 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 15:15:00 | 374.45 | 370.98 | 368.43 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 367.85 | 370.94 | 368.44 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-02-17 10:15:00 | 367.85 | 370.94 | 368.44 | SL hit (close<ema400) qty=1.00 sl=368.44 alert=retest1 |
| Cross detected — sustain check pending | 2026-02-17 12:15:00 | 371.70 | 370.93 | 368.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 13:15:00 | 373.15 | 370.95 | 368.48 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-19 15:15:00 | 365.90 | 371.53 | 368.98 | SL hit (close<static) qty=1.00 sl=366.55 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-23 10:15:00 | 375.05 | 371.09 | 368.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 11:15:00 | 372.20 | 371.10 | 368.88 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 362.45 | 373.35 | 370.51 | SL hit (close<static) qty=1.00 sl=366.55 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-16 15:15:00 | 374.45 | 2026-02-17 10:15:00 | 367.85 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2026-02-17 13:15:00 | 373.15 | 2026-02-19 15:15:00 | 365.90 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2026-02-23 11:15:00 | 372.20 | 2026-03-04 09:15:00 | 362.45 | STOP_HIT | 1.00 | -2.62% |
