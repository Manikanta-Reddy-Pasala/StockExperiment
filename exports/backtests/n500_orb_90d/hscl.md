# Himadri Speciality Chemical Ltd. (HSCL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 631.60
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
| ENTRY1 | 8 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 8
- **Target hits / Stop hits / Partials:** 0 / 8 / 2
- **Avg / median % per leg:** -0.16% / -0.34%
- **Sum % (uncompounded):** -1.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 2 | 25.0% | 0 | 6 | 2 | -0.08% | -0.6% |
| BUY @ 2nd Alert (retest1) | 8 | 2 | 25.0% | 0 | 6 | 2 | -0.08% | -0.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.49% | -1.0% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.49% | -1.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 2 | 20.0% | 0 | 8 | 2 | -0.16% | -1.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:50:00 | 460.60 | 457.54 | 0.00 | ORB-long ORB[454.95,460.00] vol=4.6x ATR=1.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 14:20:00 | 463.54 | 459.26 | 0.00 | T1 1.5R @ 463.54 |
| Stop hit — per-position SL triggered | 2026-02-09 14:50:00 | 460.60 | 459.40 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:15:00 | 452.90 | 450.24 | 0.00 | ORB-long ORB[448.00,452.05] vol=4.0x ATR=1.53 |
| Stop hit — per-position SL triggered | 2026-02-16 11:00:00 | 451.37 | 451.16 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:30:00 | 476.35 | 472.76 | 0.00 | ORB-long ORB[467.50,473.75] vol=5.1x ATR=1.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:35:00 | 479.02 | 474.28 | 0.00 | T1 1.5R @ 479.02 |
| Stop hit — per-position SL triggered | 2026-02-18 09:40:00 | 476.35 | 474.12 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 10:00:00 | 492.50 | 482.14 | 0.00 | ORB-long ORB[470.10,477.35] vol=3.8x ATR=3.34 |
| Stop hit — per-position SL triggered | 2026-02-19 10:20:00 | 489.16 | 485.97 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:35:00 | 495.80 | 491.39 | 0.00 | ORB-long ORB[486.90,492.70] vol=2.6x ATR=2.34 |
| Stop hit — per-position SL triggered | 2026-02-24 09:40:00 | 493.46 | 491.83 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:10:00 | 496.85 | 493.03 | 0.00 | ORB-long ORB[489.05,494.00] vol=4.7x ATR=1.71 |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 495.14 | 493.27 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:40:00 | 489.15 | 492.74 | 0.00 | ORB-short ORB[490.45,496.00] vol=1.7x ATR=2.22 |
| Stop hit — per-position SL triggered | 2026-04-16 09:45:00 | 491.37 | 492.54 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-05-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:00:00 | 613.35 | 616.64 | 0.00 | ORB-short ORB[614.70,622.50] vol=2.9x ATR=3.20 |
| Stop hit — per-position SL triggered | 2026-05-08 10:05:00 | 616.55 | 616.49 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:50:00 | 460.60 | 2026-02-09 14:20:00 | 463.54 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2026-02-09 10:50:00 | 460.60 | 2026-02-09 14:50:00 | 460.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-16 10:15:00 | 452.90 | 2026-02-16 11:00:00 | 451.37 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-18 09:30:00 | 476.35 | 2026-02-18 09:35:00 | 479.02 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-02-18 09:30:00 | 476.35 | 2026-02-18 09:40:00 | 476.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-19 10:00:00 | 492.50 | 2026-02-19 10:20:00 | 489.16 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest1 | 2026-02-24 09:35:00 | 495.80 | 2026-02-24 09:40:00 | 493.46 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-02-25 10:10:00 | 496.85 | 2026-02-25 10:15:00 | 495.14 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-04-16 09:40:00 | 489.15 | 2026-04-16 09:45:00 | 491.37 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-05-08 10:00:00 | 613.35 | 2026-05-08 10:05:00 | 616.55 | STOP_HIT | 1.00 | -0.52% |
