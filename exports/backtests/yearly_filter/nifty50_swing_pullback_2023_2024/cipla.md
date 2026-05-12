# CIPLA (CIPLA)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1334.90
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
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 3
- **Target hits / Stop hits / Partials:** 0 / 3 / 0
- **Avg / median % per leg:** -2.97% / -2.99%
- **Sum % (uncompounded):** -8.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.97% | -8.9% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.97% | -8.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.97% | -8.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-18 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-18 00:00:00 | 1210.60 | 1089.02 | 1182.18 | Stage2 pullback-breakout RSI=58 vol=4.3x ATR=24.16 |
| Stop hit — per-position SL triggered | 2023-10-25 00:00:00 | 1174.37 | 1093.02 | 1184.27 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2024-01-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-10 00:00:00 | 1329.20 | 1150.84 | 1263.31 | Stage2 pullback-breakout RSI=69 vol=2.0x ATR=25.17 |
| Stop hit — per-position SL triggered | 2024-01-17 00:00:00 | 1291.45 | 1158.53 | 1280.61 | SL hit (bars_held=5) |

### Cycle 3 — BUY (started 2024-03-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-11 00:00:00 | 1504.10 | 1243.62 | 1461.98 | Stage2 pullback-breakout RSI=66 vol=1.5x ATR=30.88 |
| Stop hit — per-position SL triggered | 2024-03-13 00:00:00 | 1457.78 | 1248.07 | 1463.19 | SL hit (bars_held=2) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-10-18 00:00:00 | 1210.60 | 2023-10-25 00:00:00 | 1174.37 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest1 | 2024-01-10 00:00:00 | 1329.20 | 2024-01-17 00:00:00 | 1291.45 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest1 | 2024-03-11 00:00:00 | 1504.10 | 2024-03-13 00:00:00 | 1457.78 | STOP_HIT | 1.00 | -3.08% |
