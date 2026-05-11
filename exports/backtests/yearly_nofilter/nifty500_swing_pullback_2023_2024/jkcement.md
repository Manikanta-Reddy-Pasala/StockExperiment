# J.K. Cement Ltd. (JKCEMENT)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 5461.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -1.47% / -1.87%
- **Sum % (uncompounded):** -5.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 4 | 0 | -1.47% | -5.9% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 4 | 0 | -1.47% | -5.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 4 | 0 | -1.47% | -5.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-09 00:00:00 | 3371.05 | 3031.78 | 3262.65 | Stage2 pullback-breakout RSI=61 vol=3.4x ATR=76.76 |
| Stop hit — per-position SL triggered | 2023-08-11 00:00:00 | 3255.91 | 3036.97 | 3267.99 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2023-10-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 00:00:00 | 3265.10 | 3095.25 | 3193.57 | Stage2 pullback-breakout RSI=56 vol=1.6x ATR=76.15 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 3150.87 | 3106.94 | 3230.66 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2023-11-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 00:00:00 | 3402.95 | 3109.55 | 3184.77 | Stage2 pullback-breakout RSI=64 vol=8.6x ATR=97.23 |
| Stop hit — per-position SL triggered | 2023-11-20 00:00:00 | 3502.25 | 3142.29 | 3360.51 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2024-04-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-01 00:00:00 | 4283.10 | 3708.18 | 4161.65 | Stage2 pullback-breakout RSI=58 vol=1.9x ATR=101.60 |
| Stop hit — per-position SL triggered | 2024-04-16 00:00:00 | 4203.15 | 3765.80 | 4250.50 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-09 00:00:00 | 3371.05 | 2023-08-11 00:00:00 | 3255.91 | STOP_HIT | 1.00 | -3.42% |
| BUY | retest1 | 2023-10-12 00:00:00 | 3265.10 | 2023-10-23 00:00:00 | 3150.87 | STOP_HIT | 1.00 | -3.50% |
| BUY | retest1 | 2023-11-06 00:00:00 | 3402.95 | 2023-11-20 00:00:00 | 3502.25 | STOP_HIT | 1.00 | 2.92% |
| BUY | retest1 | 2024-04-01 00:00:00 | 4283.10 | 2024-04-16 00:00:00 | 4203.15 | STOP_HIT | 1.00 | -1.87% |
