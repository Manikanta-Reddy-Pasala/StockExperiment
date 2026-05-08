# COALINDIA (COALINDIA)

## Backtest Summary

- **Window:** 2025-08-14 09:15:00 → 2026-05-08 15:30:00 (1238 bars)
- **Last close:** 456.40
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
| ALERT3 | 1 |
| PENDING | 5 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 3 |
| PARTIAL | 0 |
| TARGET_HIT | 3 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / Stop hits / Partials:** 3 / 2 / 0
- **Avg / median % per leg:** 5.01% / 10.00%
- **Sum % (uncompounded):** 25.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 3 | 2 | 0 | 5.01% | 25.0% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.48% | -5.0% |
| BUY @ 3rd Alert (retest2) | 3 | 3 | 100.0% | 3 | 0 | 0 | 10.00% | 30.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.48% | -5.0% |
| retest2 (combined) | 3 | 3 | 100.0% | 3 | 0 | 0 | 10.00% | 30.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 13:15:00 | 402.50 | 384.20 | 384.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 09:15:00 | 405.35 | 384.76 | 384.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 416.35 | 418.97 | 407.40 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-02-02 14:15:00 | 423.85 | 418.91 | 407.65 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-02 15:15:00 | 422.80 | 418.95 | 407.73 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-11 13:15:00 | 421.45 | 422.93 | 412.32 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 14:15:00 | 422.65 | 422.93 | 412.37 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 412.25 | 422.60 | 412.67 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 412.25 | 422.60 | 412.67 | SL hit (close<ema400) qty=1.00 sl=412.67 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 412.25 | 422.60 | 412.67 | SL hit (close<ema400) qty=1.00 sl=412.67 alert=retest1 |
| Cross detected — sustain check pending | 2026-02-16 09:15:00 | 419.60 | 421.82 | 412.61 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 10:15:00 | 420.15 | 421.80 | 412.65 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-18 14:15:00 | 418.20 | 421.41 | 413.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 15:15:00 | 418.20 | 421.38 | 413.27 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-20 09:15:00 | 424.00 | 421.08 | 413.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:15:00 | 420.85 | 421.08 | 413.47 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Target hit | 2026-03-12 10:15:00 | 462.16 | 431.03 | 421.98 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-03-12 10:15:00 | 460.02 | 431.03 | 421.98 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-03-12 10:15:00 | 462.94 | 431.03 | 421.98 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-02 15:15:00 | 422.80 | 2026-02-13 09:15:00 | 412.25 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest1 | 2026-02-11 14:15:00 | 422.65 | 2026-02-13 09:15:00 | 412.25 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2026-02-16 10:15:00 | 420.15 | 2026-03-12 10:15:00 | 462.16 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-18 15:15:00 | 418.20 | 2026-03-12 10:15:00 | 460.02 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-20 10:15:00 | 420.85 | 2026-03-12 10:15:00 | 462.94 | TARGET_HIT | 1.00 | 10.00% |
