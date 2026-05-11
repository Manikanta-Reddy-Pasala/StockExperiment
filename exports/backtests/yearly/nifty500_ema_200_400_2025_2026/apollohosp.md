# Apollo Hospitals Enterprise Ltd. (APOLLOHOSP)

## Backtest Summary

- **Window:** 2025-07-25 09:15:00 → 2026-05-08 15:15:00 (1346 bars)
- **Last close:** 8100.00
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
| ALERT2_SKIP | 0 |
| ALERT3 | 11 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 20 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 8 / 13
- **Target hits / Stop hits / Partials:** 0 / 17 / 4
- **Avg / median % per leg:** 0.24% / -0.67%
- **Sum % (uncompounded):** 5.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.90% | -17.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.90% | -17.1% |
| SELL (all) | 12 | 8 | 66.7% | 0 | 8 | 4 | 1.84% | 22.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 8 | 66.7% | 0 | 8 | 4 | 1.84% | 22.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 21 | 8 | 38.1% | 0 | 17 | 4 | 0.24% | 5.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 13:15:00 | 7499.50 | 7672.43 | 7673.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 7413.50 | 7666.29 | 7670.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 09:15:00 | 7259.50 | 7162.74 | 7299.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-06 10:00:00 | 7259.50 | 7162.74 | 7299.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 7310.00 | 7164.21 | 7299.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:30:00 | 7280.50 | 7164.21 | 7299.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 7286.00 | 7165.42 | 7299.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 12:45:00 | 7271.00 | 7205.24 | 7306.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 15:00:00 | 7265.00 | 7207.85 | 7303.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:30:00 | 7274.00 | 7210.79 | 7302.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 14:15:00 | 7313.50 | 7212.36 | 7302.26 | SL hit (close>static) qty=1.00 sl=7311.50 alert=retest2 |

### Cycle 2 — BUY (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 12:15:00 | 7660.00 | 7237.19 | 7235.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 10:15:00 | 7669.00 | 7302.20 | 7269.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 7513.50 | 7543.35 | 7430.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-13 10:00:00 | 7513.50 | 7543.35 | 7430.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 7460.50 | 7543.20 | 7434.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:45:00 | 7461.50 | 7543.20 | 7434.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 13:15:00 | 7425.50 | 7540.30 | 7434.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:00:00 | 7425.50 | 7540.30 | 7434.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 7486.00 | 7539.76 | 7434.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 10:00:00 | 7542.00 | 7539.29 | 7435.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 13:00:00 | 7525.00 | 7538.46 | 7436.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 13:30:00 | 7522.50 | 7538.47 | 7437.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 09:15:00 | 7586.50 | 7537.62 | 7437.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 7357.00 | 7534.19 | 7440.09 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-19 09:15:00 | 7357.00 | 7534.19 | 7440.09 | SL hit (close<static) qty=1.00 sl=7413.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-01-09 12:45:00 | 7271.00 | 2026-01-13 14:15:00 | 7313.50 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-01-12 15:00:00 | 7265.00 | 2026-01-13 14:15:00 | 7313.50 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2026-01-13 12:30:00 | 7274.00 | 2026-01-13 14:15:00 | 7313.50 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2026-01-14 09:30:00 | 7274.50 | 2026-01-20 14:15:00 | 6910.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 14:15:00 | 7261.50 | 2026-01-20 15:15:00 | 6898.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 15:15:00 | 7265.00 | 2026-01-20 15:15:00 | 6901.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 10:45:00 | 7258.50 | 2026-01-20 15:15:00 | 6895.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 09:30:00 | 7274.50 | 2026-02-03 10:15:00 | 7110.50 | STOP_HIT | 0.50 | 2.25% |
| SELL | retest2 | 2026-01-14 14:15:00 | 7261.50 | 2026-02-03 10:15:00 | 7110.50 | STOP_HIT | 0.50 | 2.08% |
| SELL | retest2 | 2026-01-14 15:15:00 | 7265.00 | 2026-02-03 10:15:00 | 7110.50 | STOP_HIT | 0.50 | 2.13% |
| SELL | retest2 | 2026-01-16 10:45:00 | 7258.50 | 2026-02-03 10:15:00 | 7110.50 | STOP_HIT | 0.50 | 2.04% |
| SELL | retest2 | 2026-02-09 13:45:00 | 7231.50 | 2026-02-11 09:15:00 | 7563.50 | STOP_HIT | 1.00 | -4.59% |
| BUY | retest2 | 2026-03-17 10:00:00 | 7542.00 | 2026-03-19 09:15:00 | 7357.00 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2026-03-17 13:00:00 | 7525.00 | 2026-03-19 09:15:00 | 7357.00 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2026-03-17 13:30:00 | 7522.50 | 2026-03-19 09:15:00 | 7357.00 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2026-03-18 09:15:00 | 7586.50 | 2026-03-19 09:15:00 | 7357.00 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2026-03-20 13:15:00 | 7442.00 | 2026-03-20 14:15:00 | 7365.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2026-03-24 12:45:00 | 7440.00 | 2026-03-30 10:15:00 | 7374.50 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2026-03-24 13:15:00 | 7441.00 | 2026-03-30 10:15:00 | 7374.50 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2026-03-25 09:15:00 | 7497.00 | 2026-03-30 10:15:00 | 7374.50 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2026-04-01 09:15:00 | 7514.00 | 2026-04-01 10:15:00 | 7306.50 | STOP_HIT | 1.00 | -2.76% |
