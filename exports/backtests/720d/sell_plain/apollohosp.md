# APOLLOHOSP (APOLLOHOSP)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 7845.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 2 |
| ALERT3 | 3 |
| PENDING | 15 |
| PENDING_CANCEL | 7 |
| ENTRY1 | 0 |
| ENTRY2 | 8 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 8
- **Target hits / Stop hits / Partials:** 0 / 8 / 0
- **Avg / median % per leg:** -2.57% / -2.45%
- **Sum % (uncompounded):** -20.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.57% | -20.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.57% | -20.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.57% | -20.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 15:15:00 | 6767.00 | 7070.19 | 7070.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 14:15:00 | 6741.20 | 7026.36 | 7047.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 11:15:00 | 6954.25 | 6922.32 | 6981.55 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 11:15:00 | 6954.25 | 6922.32 | 6981.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 11:15:00 | 6954.25 | 6922.32 | 6981.55 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-02-06 09:15:00 | 6876.55 | 6923.63 | 6980.76 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 11:15:00 | 6863.10 | 6922.47 | 6979.61 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-04-15 13:15:00 | 6869.50 | 6576.27 | 6597.76 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-04-15 14:15:00 | 6905.00 | 6579.54 | 6599.29 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-04-16 12:15:00 | 7010.00 | 6598.18 | 6608.23 | SL hit (close>static) qty=1.00 sl=6998.30 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-08 13:15:00 | 6855.50 | 6862.17 | 6770.90 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 15:15:00 | 6815.00 | 6861.63 | 6771.54 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-05-15 10:15:00 | 6876.00 | 6867.55 | 6786.93 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-05-15 11:15:00 | 6886.50 | 6867.74 | 6787.43 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-05-15 13:15:00 | 7013.50 | 6869.90 | 6789.31 | SL hit (close>static) qty=1.00 sl=6998.30 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-30 14:15:00 | 6879.00 | 6938.11 | 6856.58 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-05-30 15:15:00 | 6886.00 | 6937.59 | 6856.73 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-03 09:15:00 | 6838.00 | 6938.33 | 6860.30 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 11:15:00 | 6826.00 | 6935.97 | 6859.89 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-06-05 12:15:00 | 6879.50 | 6922.64 | 6858.43 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 14:15:00 | 6870.00 | 6921.70 | 6858.59 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 6893.00 | 6921.41 | 6858.76 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-06 09:15:00 | 6841.00 | 6920.61 | 6858.67 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 11:15:00 | 6864.50 | 6919.49 | 6858.73 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-06-06 13:15:00 | 6930.00 | 6919.20 | 6859.19 | SL hit (close>static) qty=1.00 sl=6893.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-12 09:15:00 | 7038.50 | 6921.26 | 6867.05 | SL hit (close>static) qty=1.00 sl=6998.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-12 09:15:00 | 7038.50 | 6921.26 | 6867.05 | SL hit (close>static) qty=1.00 sl=6998.30 alert=retest2 |

### Cycle 2 — SELL (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 09:15:00 | 7455.50 | 7636.03 | 7636.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 14:15:00 | 7426.50 | 7627.42 | 7632.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 09:15:00 | 7259.50 | 7162.88 | 7293.82 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 10:15:00 | 7310.00 | 7164.35 | 7293.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 7310.00 | 7164.35 | 7293.90 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-01-09 13:15:00 | 7256.00 | 7205.86 | 7301.32 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 15:15:00 | 7255.00 | 7206.82 | 7300.85 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-01-13 09:15:00 | 7255.00 | 7209.00 | 7298.26 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-01-13 10:15:00 | 7274.50 | 7209.65 | 7298.14 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2026-01-13 14:15:00 | 7313.50 | 7212.46 | 7297.81 | SL hit (close>static) qty=1.00 sl=7310.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-16 10:15:00 | 7235.00 | 7218.96 | 7296.99 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:15:00 | 7226.00 | 7219.39 | 7296.43 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-02-11 09:15:00 | 7563.50 | 7081.45 | 7163.09 | SL hit (close>static) qty=1.00 sl=7310.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-19 14:15:00 | 7237.50 | 7522.54 | 7435.60 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-03-19 15:15:00 | 7292.00 | 7520.25 | 7434.88 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-23 09:15:00 | 7183.00 | 7507.31 | 7431.64 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 11:15:00 | 7118.50 | 7499.35 | 7428.40 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 7371.00 | 7483.66 | 7422.19 | SL hit (close>static) qty=1.00 sl=7310.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-02 09:15:00 | 7175.00 | 7472.85 | 7426.69 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-02 11:15:00 | 7265.00 | 7468.22 | 7424.82 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2026-04-06 11:15:00 | 7256.00 | 7456.31 | 7420.26 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-06 12:15:00 | 7325.00 | 7455.00 | 7419.79 | ENTRY2 sustain failed after 60m |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-02-06 11:15:00 | 6863.10 | 2025-04-16 12:15:00 | 7010.00 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-05-08 15:15:00 | 6815.00 | 2025-05-15 13:15:00 | 7013.50 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2025-06-03 11:15:00 | 6826.00 | 2025-06-06 13:15:00 | 6930.00 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-06-05 14:15:00 | 6870.00 | 2025-06-12 09:15:00 | 7038.50 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-06-06 11:15:00 | 6864.50 | 2025-06-12 09:15:00 | 7038.50 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2026-01-09 15:15:00 | 7255.00 | 2026-01-13 14:15:00 | 7313.50 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2026-01-16 12:15:00 | 7226.00 | 2026-02-11 09:15:00 | 7563.50 | STOP_HIT | 1.00 | -4.67% |
| SELL | retest2 | 2026-03-23 11:15:00 | 7118.50 | 2026-03-24 09:15:00 | 7371.00 | STOP_HIT | 1.00 | -3.55% |
