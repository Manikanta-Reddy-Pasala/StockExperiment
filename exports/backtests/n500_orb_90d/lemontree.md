# Lemon Tree Hotels Ltd. (LEMONTREE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 120.45
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
| ENTRY1 | 12 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 11
- **Target hits / Stop hits / Partials:** 1 / 11 / 4
- **Avg / median % per leg:** -0.03% / 0.00%
- **Sum % (uncompounded):** -0.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 2 | 22.2% | 0 | 7 | 2 | -0.14% | -1.3% |
| BUY @ 2nd Alert (retest1) | 9 | 2 | 22.2% | 0 | 7 | 2 | -0.14% | -1.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.11% | 0.8% |
| SELL @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.11% | 0.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 16 | 5 | 31.2% | 1 | 11 | 4 | -0.03% | -0.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 121.96 | 122.77 | 0.00 | ORB-short ORB[122.50,124.10] vol=2.9x ATR=0.30 |
| Stop hit — per-position SL triggered | 2026-02-19 09:45:00 | 122.26 | 122.54 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:15:00 | 117.33 | 115.57 | 0.00 | ORB-long ORB[114.00,115.20] vol=2.1x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:30:00 | 118.04 | 116.14 | 0.00 | T1 1.5R @ 118.04 |
| Stop hit — per-position SL triggered | 2026-02-26 11:05:00 | 117.33 | 116.62 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:55:00 | 114.88 | 116.05 | 0.00 | ORB-short ORB[115.99,117.59] vol=2.0x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:50:00 | 114.30 | 115.65 | 0.00 | T1 1.5R @ 114.30 |
| Target hit | 2026-02-27 15:20:00 | 114.11 | 114.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2026-03-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:20:00 | 107.91 | 104.89 | 0.00 | ORB-long ORB[101.55,103.13] vol=5.6x ATR=0.75 |
| Stop hit — per-position SL triggered | 2026-03-17 10:25:00 | 107.16 | 105.25 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-19 11:10:00 | 106.90 | 105.94 | 0.00 | ORB-long ORB[105.13,106.60] vol=2.2x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 12:05:00 | 107.46 | 106.20 | 0.00 | T1 1.5R @ 107.46 |
| Stop hit — per-position SL triggered | 2026-03-19 12:10:00 | 106.90 | 106.22 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:30:00 | 108.43 | 107.88 | 0.00 | ORB-long ORB[106.60,108.20] vol=1.8x ATR=0.52 |
| Stop hit — per-position SL triggered | 2026-03-25 09:40:00 | 107.91 | 107.95 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 114.89 | 114.30 | 0.00 | ORB-long ORB[113.30,114.55] vol=2.1x ATR=0.40 |
| Stop hit — per-position SL triggered | 2026-04-10 09:45:00 | 114.49 | 114.44 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:40:00 | 120.30 | 119.03 | 0.00 | ORB-long ORB[117.62,119.20] vol=3.5x ATR=0.55 |
| Stop hit — per-position SL triggered | 2026-04-27 09:45:00 | 119.75 | 119.24 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:15:00 | 116.74 | 117.29 | 0.00 | ORB-short ORB[116.80,118.50] vol=1.7x ATR=0.44 |
| Stop hit — per-position SL triggered | 2026-04-30 10:40:00 | 117.18 | 117.24 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-05-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:45:00 | 120.06 | 118.82 | 0.00 | ORB-long ORB[117.33,118.78] vol=7.4x ATR=0.54 |
| Stop hit — per-position SL triggered | 2026-05-04 10:20:00 | 119.52 | 119.36 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-05-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 09:50:00 | 121.44 | 122.01 | 0.00 | ORB-short ORB[121.67,122.90] vol=1.7x ATR=0.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:50:00 | 120.89 | 121.75 | 0.00 | T1 1.5R @ 120.89 |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 121.44 | 121.71 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 11:15:00 | 120.42 | 120.85 | 0.00 | ORB-short ORB[120.60,122.10] vol=1.9x ATR=0.29 |
| Stop hit — per-position SL triggered | 2026-05-08 14:55:00 | 120.71 | 120.69 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-19 09:30:00 | 121.96 | 2026-02-19 09:45:00 | 122.26 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-26 10:15:00 | 117.33 | 2026-02-26 10:30:00 | 118.04 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-02-26 10:15:00 | 117.33 | 2026-02-26 11:05:00 | 117.33 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 10:55:00 | 114.88 | 2026-02-27 11:50:00 | 114.30 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-02-27 10:55:00 | 114.88 | 2026-02-27 15:20:00 | 114.11 | TARGET_HIT | 0.50 | 0.67% |
| BUY | retest1 | 2026-03-17 10:20:00 | 107.91 | 2026-03-17 10:25:00 | 107.16 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest1 | 2026-03-19 11:10:00 | 106.90 | 2026-03-19 12:05:00 | 107.46 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-03-19 11:10:00 | 106.90 | 2026-03-19 12:10:00 | 106.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-25 09:30:00 | 108.43 | 2026-03-25 09:40:00 | 107.91 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-04-10 09:30:00 | 114.89 | 2026-04-10 09:45:00 | 114.49 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-27 09:40:00 | 120.30 | 2026-04-27 09:45:00 | 119.75 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-04-30 10:15:00 | 116.74 | 2026-04-30 10:40:00 | 117.18 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-05-04 09:45:00 | 120.06 | 2026-05-04 10:20:00 | 119.52 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-05-06 09:50:00 | 121.44 | 2026-05-06 10:50:00 | 120.89 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-05-06 09:50:00 | 121.44 | 2026-05-06 11:15:00 | 121.44 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-08 11:15:00 | 120.42 | 2026-05-08 14:55:00 | 120.71 | STOP_HIT | 1.00 | -0.24% |
