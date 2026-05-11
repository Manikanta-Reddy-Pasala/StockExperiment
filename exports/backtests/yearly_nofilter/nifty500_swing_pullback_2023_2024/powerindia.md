# Hitachi Energy India Ltd. (POWERINDIA)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 33145.00
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
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 5
- **Target hits / Stop hits / Partials:** 0 / 7 / 2
- **Avg / median % per leg:** 0.60% / 0.00%
- **Sum % (uncompounded):** 5.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 4 | 44.4% | 0 | 7 | 2 | 0.60% | 5.4% |
| BUY @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 0 | 7 | 2 | 0.60% | 5.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 4 | 44.4% | 0 | 7 | 2 | 0.60% | 5.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-03 00:00:00 | 4275.85 | 3676.14 | 4108.87 | Stage2 pullback-breakout RSI=61 vol=4.0x ATR=129.58 |
| Stop hit — per-position SL triggered | 2023-08-18 00:00:00 | 4262.85 | 3736.19 | 4235.98 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-08-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-23 00:00:00 | 4635.25 | 3758.48 | 4303.97 | Stage2 pullback-breakout RSI=68 vol=4.2x ATR=164.99 |
| Stop hit — per-position SL triggered | 2023-08-31 00:00:00 | 4387.76 | 3803.46 | 4401.55 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2023-09-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 00:00:00 | 4567.90 | 3839.96 | 4418.31 | Stage2 pullback-breakout RSI=61 vol=1.7x ATR=152.06 |
| Stop hit — per-position SL triggered | 2023-09-12 00:00:00 | 4339.81 | 3852.23 | 4424.91 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2023-10-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 00:00:00 | 4317.10 | 3903.46 | 4148.53 | Stage2 pullback-breakout RSI=59 vol=1.7x ATR=134.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-17 00:00:00 | 4586.45 | 3921.84 | 4247.25 | T1 booked 50% @ 4586.45 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 4317.10 | 3945.25 | 4333.47 | SL hit (bars_held=7) |

### Cycle 5 — BUY (started 2023-11-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 00:00:00 | 4471.25 | 3964.73 | 4286.64 | Stage2 pullback-breakout RSI=58 vol=2.5x ATR=161.91 |
| Stop hit — per-position SL triggered | 2023-11-16 00:00:00 | 4577.00 | 4011.81 | 4394.05 | Time-stop (10d <3%) |

### Cycle 6 — BUY (started 2024-02-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-05 00:00:00 | 6190.65 | 4587.97 | 5779.02 | Stage2 pullback-breakout RSI=63 vol=1.6x ATR=264.35 |
| Stop hit — per-position SL triggered | 2024-02-19 00:00:00 | 6213.60 | 4729.12 | 5953.60 | Time-stop (10d <3%) |

### Cycle 7 — BUY (started 2024-03-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-05 00:00:00 | 6327.30 | 4872.94 | 6010.70 | Stage2 pullback-breakout RSI=64 vol=5.6x ATR=223.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 00:00:00 | 6773.87 | 4919.68 | 6127.54 | T1 booked 50% @ 6773.87 |
| Stop hit — per-position SL triggered | 2024-03-13 00:00:00 | 6327.30 | 4949.38 | 6180.15 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-03 00:00:00 | 4275.85 | 2023-08-18 00:00:00 | 4262.85 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-08-23 00:00:00 | 4635.25 | 2023-08-31 00:00:00 | 4387.76 | STOP_HIT | 1.00 | -5.34% |
| BUY | retest1 | 2023-09-08 00:00:00 | 4567.90 | 2023-09-12 00:00:00 | 4339.81 | STOP_HIT | 1.00 | -4.99% |
| BUY | retest1 | 2023-10-12 00:00:00 | 4317.10 | 2023-10-17 00:00:00 | 4586.45 | PARTIAL | 0.50 | 6.24% |
| BUY | retest1 | 2023-10-12 00:00:00 | 4317.10 | 2023-10-23 00:00:00 | 4317.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-02 00:00:00 | 4471.25 | 2023-11-16 00:00:00 | 4577.00 | STOP_HIT | 1.00 | 2.37% |
| BUY | retest1 | 2024-02-05 00:00:00 | 6190.65 | 2024-02-19 00:00:00 | 6213.60 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest1 | 2024-03-05 00:00:00 | 6327.30 | 2024-03-11 00:00:00 | 6773.87 | PARTIAL | 0.50 | 7.06% |
| BUY | retest1 | 2024-03-05 00:00:00 | 6327.30 | 2024-03-13 00:00:00 | 6327.30 | STOP_HIT | 0.50 | 0.00% |
