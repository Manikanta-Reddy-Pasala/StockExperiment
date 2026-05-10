# Aurobindo Pharma Ltd. (AUROPHARMA)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1487.30
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
- **Winners / losers:** 0 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -3.51% / -3.08%
- **Sum % (uncompounded):** -14.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.51% | -14.0% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.51% | -14.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.51% | -14.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-12-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 05:30:00 | 1225.10 | 1170.22 | 1195.73 | Stage2 pullback-breakout RSI=60 vol=1.9x ATR=25.19 |
| Stop hit — per-position SL triggered | 2025-12-30 05:30:00 | 1187.32 | 1172.39 | 1200.21 | SL hit (bars_held=6) |

### Cycle 2 — BUY (started 2026-01-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 05:30:00 | 1231.30 | 1174.04 | 1203.37 | Stage2 pullback-breakout RSI=60 vol=2.5x ATR=24.30 |
| Stop hit — per-position SL triggered | 2026-01-09 05:30:00 | 1194.85 | 1175.20 | 1205.66 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2026-01-30 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 05:30:00 | 1207.70 | 1172.86 | 1169.12 | Stage2 pullback-breakout RSI=58 vol=6.8x ATR=30.22 |
| Stop hit — per-position SL triggered | 2026-02-02 05:30:00 | 1162.37 | 1172.84 | 1169.56 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2026-02-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-03 05:30:00 | 1226.10 | 1173.37 | 1174.94 | Stage2 pullback-breakout RSI=60 vol=2.1x ATR=34.65 |
| Stop hit — per-position SL triggered | 2026-02-04 05:30:00 | 1174.13 | 1173.76 | 1178.54 | SL hit (bars_held=1) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-12-19 05:30:00 | 1225.10 | 2025-12-30 05:30:00 | 1187.32 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest1 | 2026-01-06 05:30:00 | 1231.30 | 2026-01-09 05:30:00 | 1194.85 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest1 | 2026-01-30 05:30:00 | 1207.70 | 2026-02-02 05:30:00 | 1162.37 | STOP_HIT | 1.00 | -3.75% |
| BUY | retest1 | 2026-02-03 05:30:00 | 1226.10 | 2026-02-04 05:30:00 | 1174.13 | STOP_HIT | 1.00 | -4.24% |
