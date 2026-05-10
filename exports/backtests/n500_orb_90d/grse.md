# Garden Reach Shipbuilders & Engineers Ltd. (GRSE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 3043.00
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
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / Stop hits / Partials:** 0 / 5 / 1
- **Avg / median % per leg:** -0.19% / -0.38%
- **Sum % (uncompounded):** -1.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.09% | -0.4% |
| BUY @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.09% | -0.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.40% | -0.8% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.40% | -0.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.19% | -1.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 09:35:00 | 2494.80 | 2475.01 | 0.00 | ORB-long ORB[2455.30,2483.60] vol=3.6x ATR=9.55 |
| Stop hit — per-position SL triggered | 2026-02-12 09:45:00 | 2485.25 | 2480.15 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 2449.00 | 2464.81 | 0.00 | ORB-short ORB[2456.60,2488.00] vol=1.8x ATR=9.99 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 2458.99 | 2461.07 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 2407.90 | 2420.73 | 0.00 | ORB-short ORB[2410.10,2440.00] vol=1.9x ATR=9.37 |
| Stop hit — per-position SL triggered | 2026-02-24 10:00:00 | 2417.27 | 2415.65 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:30:00 | 2455.00 | 2436.49 | 0.00 | ORB-long ORB[2418.40,2438.90] vol=3.0x ATR=8.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 09:35:00 | 2467.02 | 2447.22 | 0.00 | T1 1.5R @ 2467.02 |
| Stop hit — per-position SL triggered | 2026-02-26 09:40:00 | 2455.00 | 2447.95 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 2392.20 | 2370.65 | 0.00 | ORB-long ORB[2350.00,2377.00] vol=2.5x ATR=11.35 |
| Stop hit — per-position SL triggered | 2026-03-18 09:55:00 | 2380.85 | 2376.52 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-12 09:35:00 | 2494.80 | 2026-02-12 09:45:00 | 2485.25 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-02-13 09:30:00 | 2449.00 | 2026-02-13 09:40:00 | 2458.99 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-02-24 09:30:00 | 2407.90 | 2026-02-24 10:00:00 | 2417.27 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-02-26 09:30:00 | 2455.00 | 2026-02-26 09:35:00 | 2467.02 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-02-26 09:30:00 | 2455.00 | 2026-02-26 09:40:00 | 2455.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-18 09:30:00 | 2392.20 | 2026-03-18 09:55:00 | 2380.85 | STOP_HIT | 1.00 | -0.47% |
