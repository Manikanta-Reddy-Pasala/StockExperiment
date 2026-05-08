# APOLLOHOSP (APOLLOHOSP)

## Backtest Summary

- **Window:** 2025-08-14 09:15:00 → 2026-05-08 15:30:00 (1238 bars)
- **Last close:** 8097.00
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
| ALERT2_SKIP | 2 |
| ALERT3 | 2 |
| PENDING | 12 |
| PENDING_CANCEL | 6 |
| ENTRY1 | 0 |
| ENTRY2 | 6 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 0 / 4 / 2
- **Avg / median % per leg:** 1.71% / 2.26%
- **Sum % (uncompounded):** 10.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.10% | -4.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.10% | -4.2% |
| SELL (all) | 4 | 4 | 100.0% | 0 | 2 | 2 | 3.61% | 14.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 4 | 100.0% | 0 | 2 | 2 | 3.61% | 14.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 6 | 4 | 66.7% | 0 | 4 | 2 | 1.71% | 10.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 7449.00 | 7761.87 | 7763.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 13:15:00 | 7432.00 | 7718.44 | 7740.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 09:15:00 | 7259.50 | 7163.80 | 7313.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 12:15:00 | 7328.00 | 7168.06 | 7313.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 7328.00 | 7168.06 | 7313.31 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-01-09 11:15:00 | 7281.00 | 7205.44 | 7319.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 12:15:00 | 7269.50 | 7206.07 | 7318.91 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-14 10:15:00 | 7278.50 | 7215.45 | 7313.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-14 11:15:00 | 7303.50 | 7216.33 | 7313.49 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-14 13:15:00 | 7281.50 | 7217.91 | 7313.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:15:00 | 7275.00 | 7218.48 | 7313.13 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 14:15:00 | 6911.25 | 7202.53 | 7295.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 15:15:00 | 6906.02 | 7199.65 | 7293.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 10:15:00 | 7110.50 | 7049.30 | 7183.06 | SL hit (close>ema200) qty=0.50 sl=7049.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 10:15:00 | 7110.50 | 7049.30 | 7183.06 | SL hit (close>ema200) qty=0.50 sl=7049.30 alert=retest2 |

### Cycle 2 — BUY (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 15:15:00 | 7647.50 | 7252.68 | 7252.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 7653.50 | 7301.86 | 7277.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 7512.00 | 7544.11 | 7435.97 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-03-13 11:15:00 | 7598.00 | 7544.65 | 7437.31 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-13 12:15:00 | 7516.00 | 7544.37 | 7437.70 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-13 13:15:00 | 7572.50 | 7544.65 | 7438.38 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-13 14:15:00 | 7550.00 | 7544.70 | 7438.93 | ENTRY1 sustain failed after 60m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 10:15:00 | 7460.50 | 7543.95 | 7440.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 7460.50 | 7543.95 | 7440.13 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-03-18 09:15:00 | 7571.00 | 7538.62 | 7443.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-18 10:15:00 | 7548.00 | 7538.71 | 7444.40 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-18 11:15:00 | 7558.50 | 7538.91 | 7444.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-18 12:15:00 | 7498.00 | 7538.50 | 7445.23 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-25 10:15:00 | 7573.00 | 7481.19 | 7428.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 11:15:00 | 7588.50 | 7482.26 | 7429.14 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-27 10:15:00 | 7600.50 | 7488.03 | 7433.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 11:15:00 | 7585.00 | 7488.99 | 7434.40 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 7427.50 | 7490.91 | 7436.73 | SL hit (close<static) qty=1.00 sl=7430.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 7427.50 | 7490.91 | 7436.73 | SL hit (close<static) qty=1.00 sl=7430.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-10 11:15:00 | 7553.00 | 7445.71 | 7422.65 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-10 12:15:00 | 7535.00 | 7446.60 | 7423.21 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-15 09:15:00 | 7649.00 | 7455.12 | 7428.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 10:15:00 | 7656.00 | 7457.12 | 7429.95 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-16 14:15:00 | 7550.00 | 7472.20 | 7439.20 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 15:15:00 | 7555.00 | 7473.03 | 7439.77 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-01-09 12:15:00 | 7269.50 | 2026-01-20 14:15:00 | 6911.25 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2026-01-14 14:15:00 | 7275.00 | 2026-01-20 15:15:00 | 6906.02 | PARTIAL | 0.50 | 5.07% |
| SELL | retest2 | 2026-01-09 12:15:00 | 7269.50 | 2026-02-03 10:15:00 | 7110.50 | STOP_HIT | 0.50 | 2.19% |
| SELL | retest2 | 2026-01-14 14:15:00 | 7275.00 | 2026-02-03 10:15:00 | 7110.50 | STOP_HIT | 0.50 | 2.26% |
| BUY | retest2 | 2026-03-25 11:15:00 | 7588.50 | 2026-03-30 09:15:00 | 7427.50 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2026-03-27 11:15:00 | 7585.00 | 2026-03-30 09:15:00 | 7427.50 | STOP_HIT | 1.00 | -2.08% |
