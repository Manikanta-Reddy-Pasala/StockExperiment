# APOLLOHOSP (APOLLOHOSP)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 8100.00
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
| ALERT3 | 1 |
| PENDING | 8 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 0 |
| ENTRY2 | 4 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 2 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 0 / 2
- **Target hits / Stop hits / Partials:** 0 / 2 / 0
- **Avg / median % per leg:** -2.10% / -2.08%
- **Sum % (uncompounded):** -4.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.10% | -4.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.10% | -4.2% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.10% | -4.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 11:15:00 | 7643.00 | 7232.96 | 7231.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 12:15:00 | 7660.00 | 7237.21 | 7233.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 7513.50 | 7543.36 | 7428.83 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-03-13 11:15:00 | 7598.00 | 7543.91 | 7430.25 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-13 12:15:00 | 7516.00 | 7543.63 | 7430.68 | ENTRY1 sustain failed after 60m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 10:15:00 | 7460.50 | 7543.21 | 7433.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 7460.50 | 7543.21 | 7433.26 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-03-18 09:15:00 | 7571.00 | 7537.96 | 7437.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-18 10:15:00 | 7548.00 | 7538.06 | 7437.98 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-18 11:15:00 | 7558.50 | 7538.26 | 7438.59 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-18 12:15:00 | 7498.50 | 7537.86 | 7438.88 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-25 10:15:00 | 7573.00 | 7480.55 | 7422.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 11:15:00 | 7588.00 | 7481.62 | 7423.68 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-27 10:15:00 | 7600.50 | 7487.43 | 7428.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 11:15:00 | 7585.00 | 7488.40 | 7429.13 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 7427.50 | 7490.38 | 7431.60 | SL hit (close<static) qty=1.00 sl=7430.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 7427.50 | 7490.38 | 7431.60 | SL hit (close<static) qty=1.00 sl=7430.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-10 11:15:00 | 7554.50 | 7444.89 | 7418.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-10 12:15:00 | 7535.00 | 7445.79 | 7418.99 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-15 09:15:00 | 7648.50 | 7454.36 | 7424.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 10:15:00 | 7656.00 | 7456.37 | 7425.97 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-16 14:15:00 | 7550.00 | 7471.30 | 7435.30 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 15:15:00 | 7562.50 | 7472.21 | 7435.94 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2026-03-25 11:15:00 | 7588.00 | 2026-03-30 09:15:00 | 7427.50 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2026-03-27 11:15:00 | 7585.00 | 2026-03-30 09:15:00 | 7427.50 | STOP_HIT | 1.00 | -2.08% |
