# Adani Total Gas Ltd. (ATGL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 632.00
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
| ENTRY1 | 9 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 8
- **Target hits / Stop hits / Partials:** 1 / 8 / 2
- **Avg / median % per leg:** 0.03% / -0.18%
- **Sum % (uncompounded):** 0.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.25% | 1.5% |
| BUY @ 2nd Alert (retest1) | 6 | 3 | 50.0% | 1 | 3 | 2 | 0.25% | 1.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.24% | -1.2% |
| SELL @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.24% | -1.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 3 | 27.3% | 1 | 8 | 2 | 0.03% | 0.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:15:00 | 535.60 | 537.31 | 0.00 | ORB-short ORB[536.55,544.35] vol=1.6x ATR=0.96 |
| Stop hit — per-position SL triggered | 2026-02-12 12:10:00 | 536.56 | 537.12 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:45:00 | 521.70 | 520.17 | 0.00 | ORB-long ORB[515.10,521.00] vol=1.6x ATR=1.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 12:05:00 | 523.81 | 520.73 | 0.00 | T1 1.5R @ 523.81 |
| Target hit | 2026-02-16 15:20:00 | 528.20 | 525.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 11:15:00 | 532.80 | 531.03 | 0.00 | ORB-long ORB[525.00,529.50] vol=2.1x ATR=1.46 |
| Stop hit — per-position SL triggered | 2026-02-17 11:55:00 | 531.34 | 531.09 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:15:00 | 526.75 | 528.99 | 0.00 | ORB-short ORB[529.20,534.20] vol=2.4x ATR=0.93 |
| Stop hit — per-position SL triggered | 2026-02-19 12:05:00 | 527.68 | 528.57 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 518.60 | 520.73 | 0.00 | ORB-short ORB[519.20,523.70] vol=2.1x ATR=1.88 |
| Stop hit — per-position SL triggered | 2026-02-24 09:45:00 | 520.48 | 520.35 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:30:00 | 519.85 | 521.36 | 0.00 | ORB-short ORB[520.10,523.65] vol=1.6x ATR=1.37 |
| Stop hit — per-position SL triggered | 2026-02-25 10:10:00 | 521.22 | 520.68 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:05:00 | 480.05 | 482.76 | 0.00 | ORB-short ORB[481.50,486.50] vol=2.0x ATR=1.17 |
| Stop hit — per-position SL triggered | 2026-03-05 11:15:00 | 481.22 | 482.70 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:05:00 | 629.75 | 625.55 | 0.00 | ORB-long ORB[620.15,628.50] vol=1.5x ATR=3.25 |
| Stop hit — per-position SL triggered | 2026-04-21 10:20:00 | 626.50 | 625.71 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 648.50 | 642.88 | 0.00 | ORB-long ORB[638.10,647.00] vol=1.9x ATR=2.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 09:35:00 | 652.83 | 645.65 | 0.00 | T1 1.5R @ 652.83 |
| Stop hit — per-position SL triggered | 2026-05-05 09:50:00 | 648.50 | 646.62 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-12 11:15:00 | 535.60 | 2026-02-12 12:10:00 | 536.56 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2026-02-16 10:45:00 | 521.70 | 2026-02-16 12:05:00 | 523.81 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-02-16 10:45:00 | 521.70 | 2026-02-16 15:20:00 | 528.20 | TARGET_HIT | 0.50 | 1.25% |
| BUY | retest1 | 2026-02-17 11:15:00 | 532.80 | 2026-02-17 11:55:00 | 531.34 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-02-19 11:15:00 | 526.75 | 2026-02-19 12:05:00 | 527.68 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-02-24 09:30:00 | 518.60 | 2026-02-24 09:45:00 | 520.48 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-25 09:30:00 | 519.85 | 2026-02-25 10:10:00 | 521.22 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-03-05 11:05:00 | 480.05 | 2026-03-05 11:15:00 | 481.22 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-04-21 10:05:00 | 629.75 | 2026-04-21 10:20:00 | 626.50 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2026-05-05 09:30:00 | 648.50 | 2026-05-05 09:35:00 | 652.83 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2026-05-05 09:30:00 | 648.50 | 2026-05-05 09:50:00 | 648.50 | STOP_HIT | 0.50 | 0.00% |
