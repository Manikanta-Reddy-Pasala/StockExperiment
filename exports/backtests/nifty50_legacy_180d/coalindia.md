# COALINDIA (COALINDIA)

## Backtest Summary

- **Window:** 2025-08-14 09:15:00 → 2026-05-08 15:30:00 (1238 bars)
- **Last close:** 456.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty @ 15% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 15%, trail SL → EMA200
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
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 2
- **Target hits / Stop hits / Partials:** 0 / 5 / 3
- **Avg / median % per leg:** 8.30% / 9.18%
- **Sum % (uncompounded):** 66.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 6 | 75.0% | 0 | 5 | 3 | 8.30% | 66.4% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.48% | -5.0% |
| BUY @ 3rd Alert (retest2) | 6 | 6 | 100.0% | 0 | 3 | 3 | 11.89% | 71.4% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.48% | -5.0% |
| retest2 (combined) | 6 | 6 | 100.0% | 0 | 3 | 3 | 11.89% | 71.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 11:15:00 | 404.65 | 383.86 | 383.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 407.50 | 389.48 | 386.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 416.35 | 418.98 | 407.34 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-02-02 14:15:00 | 423.85 | 418.91 | 407.59 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-02 15:15:00 | 422.80 | 418.95 | 407.67 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-11 13:15:00 | 421.45 | 422.94 | 412.27 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 14:15:00 | 422.65 | 422.93 | 412.32 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 412.25 | 422.60 | 412.62 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 412.25 | 422.60 | 412.62 | SL hit (close<ema400) qty=1.00 sl=412.62 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 412.25 | 422.60 | 412.62 | SL hit (close<ema400) qty=1.00 sl=412.62 alert=retest1 |
| Cross detected — sustain check pending | 2026-02-16 09:15:00 | 419.60 | 421.82 | 412.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 10:15:00 | 420.15 | 421.80 | 412.60 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-18 14:15:00 | 418.20 | 421.42 | 413.20 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 15:15:00 | 418.20 | 421.39 | 413.22 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-20 09:15:00 | 424.00 | 421.08 | 413.39 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:15:00 | 420.85 | 421.08 | 413.43 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:15:00 | 483.17 | 447.69 | 440.28 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:15:00 | 480.93 | 447.69 | 440.28 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:15:00 | 483.98 | 447.69 | 440.28 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-08 11:15:00 | 456.60 | 456.82 | 446.84 | SL hit (close<ema200) qty=0.50 sl=456.82 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-08 11:15:00 | 456.60 | 456.82 | 446.84 | SL hit (close<ema200) qty=0.50 sl=456.82 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-08 11:15:00 | 456.60 | 456.82 | 446.84 | SL hit (close<ema200) qty=0.50 sl=456.82 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-02 15:15:00 | 422.80 | 2026-02-13 09:15:00 | 412.25 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest1 | 2026-02-11 14:15:00 | 422.65 | 2026-02-13 09:15:00 | 412.25 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2026-02-16 10:15:00 | 420.15 | 2026-04-29 10:15:00 | 483.17 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2026-02-18 15:15:00 | 418.20 | 2026-04-29 10:15:00 | 480.93 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2026-02-20 10:15:00 | 420.85 | 2026-04-29 10:15:00 | 483.98 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2026-02-16 10:15:00 | 420.15 | 2026-05-08 11:15:00 | 456.60 | STOP_HIT | 0.50 | 8.68% |
| BUY | retest2 | 2026-02-18 15:15:00 | 418.20 | 2026-05-08 11:15:00 | 456.60 | STOP_HIT | 0.50 | 9.18% |
| BUY | retest2 | 2026-02-20 10:15:00 | 420.85 | 2026-05-08 11:15:00 | 456.60 | STOP_HIT | 0.50 | 8.49% |
