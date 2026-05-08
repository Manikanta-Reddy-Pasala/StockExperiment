# BAJAJ-AUTO (BAJAJ-AUTO)

## Backtest Summary

- **Window:** 2025-08-14 09:15:00 → 2026-05-08 15:15:00 (1237 bars)
- **Last close:** 10711.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 3 |
| ALERT3 | 4 |
| PENDING | 9 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 9 |
| PARTIAL | 0 |
| TARGET_HIT | 2 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 7
- **Target hits / Stop hits / Partials:** 2 / 7 / 0
- **Avg / median % per leg:** 1.29% / -0.88%
- **Sum % (uncompounded):** 11.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 2 | 22.2% | 2 | 7 | 0 | 1.29% | 11.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 2 | 22.2% | 2 | 7 | 0 | 1.29% | 11.6% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 9 | 2 | 22.2% | 2 | 7 | 0 | 1.29% | 11.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-11-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 12:15:00 | 9130.00 | 8924.36 | 8923.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 12:15:00 | 9140.00 | 8935.98 | 8929.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 10:15:00 | 8951.00 | 8951.50 | 8938.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 10:15:00 | 8951.00 | 8951.50 | 8938.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 8951.00 | 8951.50 | 8938.13 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-11-28 11:15:00 | 9003.50 | 8952.02 | 8938.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 12:15:00 | 9034.00 | 8952.84 | 8938.94 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-03 14:15:00 | 9003.00 | 8974.10 | 8951.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 15:15:00 | 9000.50 | 8974.36 | 8952.05 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-09 11:15:00 | 8986.50 | 8989.74 | 8962.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 12:15:00 | 8975.50 | 8989.60 | 8962.91 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-10 14:15:00 | 8993.00 | 8987.08 | 8962.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 15:15:00 | 8980.00 | 8987.01 | 8962.88 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 9031.00 | 8987.45 | 8963.22 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-12-11 10:15:00 | 9038.50 | 8987.96 | 8963.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 11:15:00 | 9080.00 | 8988.87 | 8964.18 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-15 09:15:00 | 8922.50 | 8992.15 | 8967.33 | SL hit (close<static) qty=1.00 sl=8962.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-15 10:15:00 | 8901.00 | 8991.24 | 8967.00 | SL hit (close<static) qty=1.00 sl=8922.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-15 10:15:00 | 8901.00 | 8991.24 | 8967.00 | SL hit (close<static) qty=1.00 sl=8922.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-15 10:15:00 | 8901.00 | 8991.24 | 8967.00 | SL hit (close<static) qty=1.00 sl=8922.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-15 10:15:00 | 8901.00 | 8991.24 | 8967.00 | SL hit (close<static) qty=1.00 sl=8922.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-22 09:15:00 | 9123.00 | 8971.82 | 8960.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 10:15:00 | 9130.00 | 8973.40 | 8961.12 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Target hit | 2026-02-19 09:15:00 | 10043.00 | 9586.16 | 9418.48 | Target hit (10%) qty=1.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-16 14:15:00 | 9079.50 | 9547.33 | 9490.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 15:15:00 | 9073.00 | 9542.61 | 9488.77 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-19 13:15:00 | 8879.00 | 9473.05 | 9457.31 | SL hit (close<static) qty=1.00 sl=8962.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-20 10:15:00 | 9064.00 | 9452.60 | 9447.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 11:15:00 | 9066.50 | 9448.76 | 9445.39 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-20 13:15:00 | 9045.50 | 9440.66 | 9441.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 13:15:00 | 9045.50 | 9440.66 | 9441.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 8836.50 | 9427.07 | 9434.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 9446.50 | 9187.25 | 9293.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 9446.50 | 9187.25 | 9293.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 9446.50 | 9187.25 | 9293.33 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 11:15:00 | 9771.50 | 9375.92 | 9375.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 14:15:00 | 9817.00 | 9399.20 | 9387.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 9374.00 | 9488.95 | 9441.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 9374.00 | 9488.95 | 9441.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 9374.00 | 9488.95 | 9441.56 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-04-30 10:15:00 | 9678.00 | 9490.83 | 9442.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 11:15:00 | 9757.00 | 9493.48 | 9444.30 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Target hit | 2026-05-07 09:15:00 | 10732.70 | 9644.39 | 9530.25 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-11-28 12:15:00 | 9034.00 | 2025-12-15 09:15:00 | 8922.50 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-12-03 15:15:00 | 9000.50 | 2025-12-15 10:15:00 | 8901.00 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-12-09 12:15:00 | 8975.50 | 2025-12-15 10:15:00 | 8901.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-12-10 15:15:00 | 8980.00 | 2025-12-15 10:15:00 | 8901.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-12-11 11:15:00 | 9080.00 | 2025-12-15 10:15:00 | 8901.00 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-12-22 10:15:00 | 9130.00 | 2026-02-19 09:15:00 | 10043.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-16 15:15:00 | 9073.00 | 2026-03-19 13:15:00 | 8879.00 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2026-03-20 11:15:00 | 9066.50 | 2026-03-20 13:15:00 | 9045.50 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2026-04-30 11:15:00 | 9757.00 | 2026-05-07 09:15:00 | 10732.70 | TARGET_HIT | 1.00 | 10.00% |
