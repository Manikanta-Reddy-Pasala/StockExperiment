# Force Motors Ltd. (FORCEMOT)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 20851.00
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
| ENTRY1 | 67 |
| ENTRY2 | 0 |
| PARTIAL | 35 |
| TARGET_HIT | 15 |
| STOP_HIT | 52 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 102 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 50 / 52
- **Target hits / Stop hits / Partials:** 15 / 52 / 35
- **Avg / median % per leg:** 0.21% / 0.00%
- **Sum % (uncompounded):** 21.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 36 | 14 | 38.9% | 3 | 22 | 11 | 0.09% | 3.1% |
| BUY @ 2nd Alert (retest1) | 36 | 14 | 38.9% | 3 | 22 | 11 | 0.09% | 3.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 66 | 36 | 54.5% | 12 | 30 | 24 | 0.27% | 18.0% |
| SELL @ 2nd Alert (retest1) | 66 | 36 | 54.5% | 12 | 30 | 24 | 0.27% | 18.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 102 | 50 | 49.0% | 15 | 52 | 35 | 0.21% | 21.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 09:35:00 | 9139.95 | 9186.00 | 0.00 | ORB-short ORB[9160.00,9241.05] vol=1.7x ATR=26.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 09:40:00 | 9099.90 | 9173.36 | 0.00 | T1 1.5R @ 9099.90 |
| Target hit | 2024-05-16 15:00:00 | 9031.60 | 9029.54 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — SELL (started 2024-05-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 09:40:00 | 8860.00 | 8928.37 | 0.00 | ORB-short ORB[8905.00,9018.95] vol=1.8x ATR=32.32 |
| Stop hit — per-position SL triggered | 2024-05-17 09:45:00 | 8892.32 | 8926.38 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-18 09:35:00 | 8979.95 | 8928.50 | 0.00 | ORB-long ORB[8820.00,8919.00] vol=2.4x ATR=29.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-18 09:50:00 | 9024.27 | 8964.80 | 0.00 | T1 1.5R @ 9024.27 |
| Stop hit — per-position SL triggered | 2024-05-18 11:30:00 | 8979.95 | 8875.99 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:35:00 | 8589.00 | 8659.29 | 0.00 | ORB-short ORB[8626.05,8749.95] vol=1.8x ATR=37.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-22 09:40:00 | 8533.13 | 8628.31 | 0.00 | T1 1.5R @ 8533.13 |
| Stop hit — per-position SL triggered | 2024-05-22 10:30:00 | 8589.00 | 8595.87 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 09:30:00 | 8647.10 | 8549.42 | 0.00 | ORB-long ORB[8492.10,8577.00] vol=1.9x ATR=31.94 |
| Stop hit — per-position SL triggered | 2024-05-23 09:50:00 | 8615.16 | 8596.84 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 09:50:00 | 8450.00 | 8501.47 | 0.00 | ORB-short ORB[8517.00,8554.75] vol=2.1x ATR=24.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 10:00:00 | 8413.03 | 8474.61 | 0.00 | T1 1.5R @ 8413.03 |
| Target hit | 2024-05-27 13:50:00 | 8419.45 | 8413.17 | 0.00 | Trail-exit close>VWAP |

### Cycle 7 — SELL (started 2024-05-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 09:35:00 | 8303.40 | 8328.68 | 0.00 | ORB-short ORB[8315.20,8393.35] vol=2.3x ATR=26.24 |
| Stop hit — per-position SL triggered | 2024-05-28 09:40:00 | 8329.64 | 8339.82 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-05-31 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 10:00:00 | 8388.45 | 8491.15 | 0.00 | ORB-short ORB[8490.00,8600.05] vol=1.5x ATR=27.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 10:15:00 | 8346.83 | 8472.45 | 0.00 | T1 1.5R @ 8346.83 |
| Stop hit — per-position SL triggered | 2024-05-31 12:35:00 | 8388.45 | 8412.92 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 09:30:00 | 8574.45 | 8537.59 | 0.00 | ORB-long ORB[8449.35,8560.00] vol=3.5x ATR=22.34 |
| Stop hit — per-position SL triggered | 2024-06-07 09:55:00 | 8552.11 | 8545.38 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 11:15:00 | 8511.40 | 8582.32 | 0.00 | ORB-short ORB[8578.80,8675.00] vol=1.9x ATR=28.76 |
| Stop hit — per-position SL triggered | 2024-06-10 11:40:00 | 8540.16 | 8577.72 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-11 09:35:00 | 8469.50 | 8503.58 | 0.00 | ORB-short ORB[8476.20,8549.00] vol=3.4x ATR=25.65 |
| Stop hit — per-position SL triggered | 2024-06-11 09:50:00 | 8495.15 | 8496.50 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-06-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 09:30:00 | 9034.25 | 9093.14 | 0.00 | ORB-short ORB[9054.90,9145.50] vol=1.6x ATR=28.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 09:40:00 | 8992.14 | 9054.15 | 0.00 | T1 1.5R @ 8992.14 |
| Target hit | 2024-06-13 12:30:00 | 8962.10 | 8949.86 | 0.00 | Trail-exit close>VWAP |

### Cycle 13 — SELL (started 2024-06-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 10:25:00 | 9000.00 | 9050.76 | 0.00 | ORB-short ORB[9050.05,9118.95] vol=1.7x ATR=33.12 |
| Stop hit — per-position SL triggered | 2024-06-19 11:00:00 | 9033.12 | 9048.15 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-06-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 09:35:00 | 9044.90 | 8931.77 | 0.00 | ORB-long ORB[8850.25,8926.30] vol=2.7x ATR=36.09 |
| Stop hit — per-position SL triggered | 2024-06-20 09:45:00 | 9008.81 | 8960.99 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 09:30:00 | 8709.05 | 8734.20 | 0.00 | ORB-short ORB[8717.00,8792.50] vol=2.9x ATR=23.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 09:50:00 | 8673.54 | 8720.84 | 0.00 | T1 1.5R @ 8673.54 |
| Target hit | 2024-07-05 15:20:00 | 8500.00 | 8589.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2024-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 11:15:00 | 8370.00 | 8425.08 | 0.00 | ORB-short ORB[8402.25,8474.90] vol=1.6x ATR=19.19 |
| Stop hit — per-position SL triggered | 2024-07-09 12:20:00 | 8389.19 | 8418.36 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:05:00 | 8439.95 | 8486.77 | 0.00 | ORB-short ORB[8485.05,8542.00] vol=3.2x ATR=31.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:20:00 | 8393.05 | 8475.30 | 0.00 | T1 1.5R @ 8393.05 |
| Stop hit — per-position SL triggered | 2024-07-10 11:30:00 | 8439.95 | 8450.80 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-15 11:10:00 | 8366.90 | 8412.05 | 0.00 | ORB-short ORB[8395.00,8477.85] vol=2.4x ATR=25.68 |
| Stop hit — per-position SL triggered | 2024-07-15 11:20:00 | 8392.58 | 8411.03 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 10:20:00 | 8180.00 | 8209.89 | 0.00 | ORB-short ORB[8201.00,8265.05] vol=3.1x ATR=18.33 |
| Stop hit — per-position SL triggered | 2024-07-19 10:25:00 | 8198.33 | 8207.91 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 10:15:00 | 8195.00 | 8134.24 | 0.00 | ORB-long ORB[8057.60,8150.00] vol=2.3x ATR=30.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 10:30:00 | 8240.75 | 8149.54 | 0.00 | T1 1.5R @ 8240.75 |
| Stop hit — per-position SL triggered | 2024-07-22 14:25:00 | 8195.00 | 8195.83 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-07-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 10:30:00 | 8320.00 | 8239.59 | 0.00 | ORB-long ORB[8173.80,8251.90] vol=4.5x ATR=33.07 |
| Stop hit — per-position SL triggered | 2024-07-23 10:35:00 | 8286.93 | 8249.23 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-07-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-25 10:00:00 | 8190.45 | 8217.15 | 0.00 | ORB-short ORB[8199.95,8249.00] vol=1.5x ATR=21.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 10:05:00 | 8157.53 | 8209.82 | 0.00 | T1 1.5R @ 8157.53 |
| Stop hit — per-position SL triggered | 2024-07-25 10:15:00 | 8190.45 | 8208.38 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-07-26 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:25:00 | 8440.25 | 8357.17 | 0.00 | ORB-long ORB[8244.70,8328.00] vol=5.6x ATR=32.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 10:45:00 | 8489.53 | 8398.92 | 0.00 | T1 1.5R @ 8489.53 |
| Stop hit — per-position SL triggered | 2024-07-26 11:05:00 | 8440.25 | 8404.83 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-07-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 09:35:00 | 8839.00 | 8802.76 | 0.00 | ORB-long ORB[8733.90,8834.00] vol=2.7x ATR=25.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 09:45:00 | 8876.68 | 8818.04 | 0.00 | T1 1.5R @ 8876.68 |
| Stop hit — per-position SL triggered | 2024-07-30 09:50:00 | 8839.00 | 8820.50 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-08 10:55:00 | 8305.00 | 8355.25 | 0.00 | ORB-short ORB[8349.95,8448.80] vol=1.6x ATR=25.88 |
| Stop hit — per-position SL triggered | 2024-08-08 11:20:00 | 8330.88 | 8349.98 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-08-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 09:30:00 | 8291.55 | 8318.98 | 0.00 | ORB-short ORB[8301.00,8392.70] vol=2.1x ATR=26.23 |
| Stop hit — per-position SL triggered | 2024-08-13 09:35:00 | 8317.78 | 8317.72 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-08-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 09:45:00 | 8105.00 | 8162.97 | 0.00 | ORB-short ORB[8177.10,8230.00] vol=1.8x ATR=30.77 |
| Stop hit — per-position SL triggered | 2024-08-14 09:50:00 | 8135.77 | 8158.27 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 10:30:00 | 8545.00 | 8474.12 | 0.00 | ORB-long ORB[8434.65,8494.95] vol=2.8x ATR=26.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 10:35:00 | 8584.61 | 8571.76 | 0.00 | T1 1.5R @ 8584.61 |
| Stop hit — per-position SL triggered | 2024-08-21 10:50:00 | 8545.00 | 8571.52 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-08-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-23 10:05:00 | 8750.05 | 8690.50 | 0.00 | ORB-long ORB[8620.00,8684.45] vol=4.6x ATR=27.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 10:20:00 | 8791.77 | 8715.26 | 0.00 | T1 1.5R @ 8791.77 |
| Stop hit — per-position SL triggered | 2024-08-23 10:25:00 | 8750.05 | 8720.16 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 8300.00 | 8332.06 | 0.00 | ORB-short ORB[8319.95,8420.00] vol=2.3x ATR=26.42 |
| Stop hit — per-position SL triggered | 2024-08-28 09:50:00 | 8326.42 | 8325.81 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-08-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 09:35:00 | 8221.00 | 8247.44 | 0.00 | ORB-short ORB[8225.00,8299.75] vol=1.7x ATR=20.85 |
| Stop hit — per-position SL triggered | 2024-08-29 09:45:00 | 8241.85 | 8246.95 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-08-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 10:50:00 | 8205.30 | 8183.24 | 0.00 | ORB-long ORB[8105.10,8179.40] vol=1.7x ATR=23.91 |
| Stop hit — per-position SL triggered | 2024-08-30 10:55:00 | 8181.39 | 8183.58 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 10:55:00 | 8332.55 | 8300.55 | 0.00 | ORB-long ORB[8250.00,8317.85] vol=2.2x ATR=21.01 |
| Stop hit — per-position SL triggered | 2024-09-03 11:05:00 | 8311.54 | 8303.35 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-09-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 10:50:00 | 7471.10 | 7531.51 | 0.00 | ORB-short ORB[7491.70,7599.00] vol=3.5x ATR=27.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 11:30:00 | 7430.08 | 7505.42 | 0.00 | T1 1.5R @ 7430.08 |
| Target hit | 2024-09-10 15:20:00 | 7342.20 | 7368.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — BUY (started 2024-09-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 09:50:00 | 7430.00 | 7378.03 | 0.00 | ORB-long ORB[7274.75,7383.90] vol=1.9x ATR=38.78 |
| Stop hit — per-position SL triggered | 2024-09-11 09:55:00 | 7391.22 | 7381.64 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-09-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 09:30:00 | 7058.70 | 7080.00 | 0.00 | ORB-short ORB[7064.00,7158.55] vol=1.5x ATR=20.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 09:35:00 | 7028.14 | 7066.44 | 0.00 | T1 1.5R @ 7028.14 |
| Stop hit — per-position SL triggered | 2024-09-17 10:25:00 | 7058.70 | 7061.25 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-09-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 09:45:00 | 7044.70 | 7061.40 | 0.00 | ORB-short ORB[7045.00,7095.70] vol=1.6x ATR=20.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 10:05:00 | 7013.90 | 7055.46 | 0.00 | T1 1.5R @ 7013.90 |
| Target hit | 2024-09-18 15:20:00 | 6915.65 | 6986.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — BUY (started 2024-09-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 11:10:00 | 7330.10 | 7137.18 | 0.00 | ORB-long ORB[6740.05,6850.00] vol=2.6x ATR=66.56 |
| Stop hit — per-position SL triggered | 2024-09-20 11:15:00 | 7263.54 | 7147.54 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-09-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 09:35:00 | 7152.50 | 7172.98 | 0.00 | ORB-short ORB[7155.60,7215.15] vol=1.9x ATR=30.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 09:50:00 | 7106.54 | 7154.95 | 0.00 | T1 1.5R @ 7106.54 |
| Target hit | 2024-09-24 13:35:00 | 7110.00 | 7106.77 | 0.00 | Trail-exit close>VWAP |

### Cycle 40 — SELL (started 2024-09-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 09:40:00 | 7075.00 | 7111.69 | 0.00 | ORB-short ORB[7087.60,7150.00] vol=1.5x ATR=26.80 |
| Stop hit — per-position SL triggered | 2024-09-25 09:50:00 | 7101.80 | 7109.51 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-10-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 09:35:00 | 7580.00 | 7511.31 | 0.00 | ORB-long ORB[7450.00,7560.00] vol=2.5x ATR=33.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-01 09:40:00 | 7630.55 | 7561.07 | 0.00 | T1 1.5R @ 7630.55 |
| Stop hit — per-position SL triggered | 2024-10-01 10:00:00 | 7580.00 | 7611.07 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-10-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-10 10:50:00 | 7232.00 | 7308.37 | 0.00 | ORB-short ORB[7271.00,7344.30] vol=1.6x ATR=23.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-10 11:25:00 | 7196.22 | 7283.68 | 0.00 | T1 1.5R @ 7196.22 |
| Stop hit — per-position SL triggered | 2024-10-10 12:35:00 | 7232.00 | 7263.53 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-10-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 10:20:00 | 6910.00 | 6979.88 | 0.00 | ORB-short ORB[6985.05,7072.00] vol=1.9x ATR=32.33 |
| Stop hit — per-position SL triggered | 2024-10-17 10:45:00 | 6942.33 | 6971.65 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-10-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-23 09:30:00 | 6410.00 | 6472.49 | 0.00 | ORB-short ORB[6450.00,6540.05] vol=1.8x ATR=26.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:40:00 | 6370.33 | 6446.95 | 0.00 | T1 1.5R @ 6370.33 |
| Stop hit — per-position SL triggered | 2024-10-23 09:50:00 | 6410.00 | 6436.81 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-11-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:30:00 | 7010.90 | 7056.98 | 0.00 | ORB-short ORB[7016.45,7105.10] vol=1.6x ATR=19.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 10:40:00 | 6982.00 | 7044.25 | 0.00 | T1 1.5R @ 6982.00 |
| Target hit | 2024-11-28 15:20:00 | 6968.50 | 7007.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — SELL (started 2024-11-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-29 09:35:00 | 6923.95 | 6944.16 | 0.00 | ORB-short ORB[6935.55,7012.20] vol=2.9x ATR=21.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 10:00:00 | 6891.81 | 6926.33 | 0.00 | T1 1.5R @ 6891.81 |
| Target hit | 2024-11-29 10:25:00 | 6898.15 | 6887.10 | 0.00 | Trail-exit close>VWAP |

### Cycle 47 — SELL (started 2024-12-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 09:30:00 | 6866.25 | 6905.52 | 0.00 | ORB-short ORB[6888.50,6951.00] vol=1.7x ATR=22.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 10:05:00 | 6833.21 | 6880.95 | 0.00 | T1 1.5R @ 6833.21 |
| Stop hit — per-position SL triggered | 2024-12-06 10:20:00 | 6866.25 | 6877.68 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-12-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-11 11:00:00 | 6659.50 | 6674.51 | 0.00 | ORB-short ORB[6663.00,6723.95] vol=8.8x ATR=15.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 15:15:00 | 6635.85 | 6663.72 | 0.00 | T1 1.5R @ 6635.85 |
| Target hit | 2024-12-11 15:20:00 | 6625.00 | 6654.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2024-12-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 10:20:00 | 6573.60 | 6607.08 | 0.00 | ORB-short ORB[6610.00,6674.00] vol=1.6x ATR=18.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 12:25:00 | 6545.60 | 6586.83 | 0.00 | T1 1.5R @ 6545.60 |
| Stop hit — per-position SL triggered | 2024-12-12 15:05:00 | 6573.60 | 6579.26 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-12-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-16 09:45:00 | 6687.85 | 6661.36 | 0.00 | ORB-long ORB[6621.25,6684.95] vol=2.3x ATR=23.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 09:50:00 | 6723.02 | 6686.21 | 0.00 | T1 1.5R @ 6723.02 |
| Target hit | 2024-12-16 14:30:00 | 6773.05 | 6777.95 | 0.00 | Trail-exit close<VWAP |

### Cycle 51 — BUY (started 2024-12-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 10:00:00 | 6830.00 | 6796.85 | 0.00 | ORB-long ORB[6750.00,6798.95] vol=2.6x ATR=30.67 |
| Stop hit — per-position SL triggered | 2024-12-17 10:15:00 | 6799.33 | 6798.33 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-12-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-18 10:55:00 | 6710.20 | 6744.66 | 0.00 | ORB-short ORB[6740.00,6836.20] vol=1.6x ATR=18.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 11:05:00 | 6682.95 | 6740.89 | 0.00 | T1 1.5R @ 6682.95 |
| Stop hit — per-position SL triggered | 2024-12-18 11:10:00 | 6710.20 | 6739.32 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-12-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 09:30:00 | 6680.00 | 6655.32 | 0.00 | ORB-long ORB[6609.00,6675.50] vol=2.3x ATR=19.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 09:35:00 | 6709.11 | 6672.95 | 0.00 | T1 1.5R @ 6709.11 |
| Target hit | 2024-12-20 10:00:00 | 6709.25 | 6727.43 | 0.00 | Trail-exit close<VWAP |

### Cycle 54 — SELL (started 2024-12-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-30 11:00:00 | 6503.00 | 6554.12 | 0.00 | ORB-short ORB[6551.00,6598.95] vol=1.6x ATR=14.33 |
| Stop hit — per-position SL triggered | 2024-12-30 11:20:00 | 6517.33 | 6552.87 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-01-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 09:30:00 | 6798.60 | 6870.81 | 0.00 | ORB-short ORB[6836.50,6917.15] vol=1.5x ATR=28.47 |
| Stop hit — per-position SL triggered | 2025-01-09 09:35:00 | 6827.07 | 6855.89 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-01-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-17 10:55:00 | 6513.35 | 6554.44 | 0.00 | ORB-short ORB[6534.05,6630.00] vol=1.7x ATR=21.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 11:10:00 | 6481.23 | 6538.72 | 0.00 | T1 1.5R @ 6481.23 |
| Target hit | 2025-01-17 13:35:00 | 6496.65 | 6485.57 | 0.00 | Trail-exit close>VWAP |

### Cycle 57 — BUY (started 2025-01-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 10:30:00 | 6545.50 | 6515.49 | 0.00 | ORB-long ORB[6471.05,6532.00] vol=1.6x ATR=20.51 |
| Stop hit — per-position SL triggered | 2025-01-20 10:35:00 | 6524.99 | 6516.98 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-01-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 09:35:00 | 6580.00 | 6537.37 | 0.00 | ORB-long ORB[6500.00,6550.05] vol=1.6x ATR=23.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 09:45:00 | 6615.69 | 6554.31 | 0.00 | T1 1.5R @ 6615.69 |
| Target hit | 2025-01-23 11:15:00 | 6687.00 | 6687.30 | 0.00 | Trail-exit close<VWAP |

### Cycle 59 — BUY (started 2025-01-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:45:00 | 6500.80 | 6458.37 | 0.00 | ORB-long ORB[6421.70,6476.95] vol=2.3x ATR=25.58 |
| Stop hit — per-position SL triggered | 2025-01-30 10:10:00 | 6475.22 | 6470.71 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-01-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-31 09:30:00 | 6291.30 | 6329.45 | 0.00 | ORB-short ORB[6330.00,6376.80] vol=2.9x ATR=21.03 |
| Stop hit — per-position SL triggered | 2025-01-31 09:35:00 | 6312.33 | 6324.53 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-02-01 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-01 10:25:00 | 6340.85 | 6372.33 | 0.00 | ORB-short ORB[6345.00,6421.45] vol=1.6x ATR=21.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 10:30:00 | 6308.47 | 6366.60 | 0.00 | T1 1.5R @ 6308.47 |
| Stop hit — per-position SL triggered | 2025-02-01 11:10:00 | 6340.85 | 6357.70 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-02-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 11:05:00 | 6687.40 | 6752.64 | 0.00 | ORB-short ORB[6758.05,6855.00] vol=1.8x ATR=24.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 11:15:00 | 6650.20 | 6744.57 | 0.00 | T1 1.5R @ 6650.20 |
| Target hit | 2025-02-06 15:20:00 | 6591.50 | 6658.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — BUY (started 2025-02-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 10:45:00 | 6890.00 | 6810.64 | 0.00 | ORB-long ORB[6752.50,6839.95] vol=1.8x ATR=30.69 |
| Stop hit — per-position SL triggered | 2025-02-25 10:55:00 | 6859.31 | 6813.88 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:30:00 | 7248.90 | 7165.23 | 0.00 | ORB-long ORB[7101.00,7192.65] vol=2.0x ATR=38.89 |
| Stop hit — per-position SL triggered | 2025-03-18 09:35:00 | 7210.01 | 7175.13 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-04-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 10:00:00 | 8876.50 | 8810.22 | 0.00 | ORB-long ORB[8725.00,8850.00] vol=2.9x ATR=37.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-15 10:25:00 | 8933.13 | 8847.05 | 0.00 | T1 1.5R @ 8933.13 |
| Stop hit — per-position SL triggered | 2025-04-15 10:45:00 | 8876.50 | 8852.49 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-04-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 09:40:00 | 9249.50 | 9144.64 | 0.00 | ORB-long ORB[9059.00,9165.00] vol=2.8x ATR=43.05 |
| Stop hit — per-position SL triggered | 2025-04-24 09:45:00 | 9206.45 | 9151.84 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-04-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 11:05:00 | 9053.50 | 9129.46 | 0.00 | ORB-short ORB[9070.50,9189.00] vol=1.5x ATR=36.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 12:20:00 | 8998.80 | 9091.21 | 0.00 | T1 1.5R @ 8998.80 |
| Stop hit — per-position SL triggered | 2025-04-29 14:45:00 | 9053.50 | 9072.71 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-16 09:35:00 | 9139.95 | 2024-05-16 09:40:00 | 9099.90 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-05-16 09:35:00 | 9139.95 | 2024-05-16 15:00:00 | 9031.60 | TARGET_HIT | 0.50 | 1.19% |
| SELL | retest1 | 2024-05-17 09:40:00 | 8860.00 | 2024-05-17 09:45:00 | 8892.32 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-05-18 09:35:00 | 8979.95 | 2024-05-18 09:50:00 | 9024.27 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-05-18 09:35:00 | 8979.95 | 2024-05-18 11:30:00 | 8979.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-22 09:35:00 | 8589.00 | 2024-05-22 09:40:00 | 8533.13 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2024-05-22 09:35:00 | 8589.00 | 2024-05-22 10:30:00 | 8589.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-23 09:30:00 | 8647.10 | 2024-05-23 09:50:00 | 8615.16 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-05-27 09:50:00 | 8450.00 | 2024-05-27 10:00:00 | 8413.03 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-05-27 09:50:00 | 8450.00 | 2024-05-27 13:50:00 | 8419.45 | TARGET_HIT | 0.50 | 0.36% |
| SELL | retest1 | 2024-05-28 09:35:00 | 8303.40 | 2024-05-28 09:40:00 | 8329.64 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-05-31 10:00:00 | 8388.45 | 2024-05-31 10:15:00 | 8346.83 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-05-31 10:00:00 | 8388.45 | 2024-05-31 12:35:00 | 8388.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-07 09:30:00 | 8574.45 | 2024-06-07 09:55:00 | 8552.11 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-06-10 11:15:00 | 8511.40 | 2024-06-10 11:40:00 | 8540.16 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-06-11 09:35:00 | 8469.50 | 2024-06-11 09:50:00 | 8495.15 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-06-13 09:30:00 | 9034.25 | 2024-06-13 09:40:00 | 8992.14 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-06-13 09:30:00 | 9034.25 | 2024-06-13 12:30:00 | 8962.10 | TARGET_HIT | 0.50 | 0.80% |
| SELL | retest1 | 2024-06-19 10:25:00 | 9000.00 | 2024-06-19 11:00:00 | 9033.12 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-06-20 09:35:00 | 9044.90 | 2024-06-20 09:45:00 | 9008.81 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-07-05 09:30:00 | 8709.05 | 2024-07-05 09:50:00 | 8673.54 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-07-05 09:30:00 | 8709.05 | 2024-07-05 15:20:00 | 8500.00 | TARGET_HIT | 0.50 | 2.40% |
| SELL | retest1 | 2024-07-09 11:15:00 | 8370.00 | 2024-07-09 12:20:00 | 8389.19 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-07-10 10:05:00 | 8439.95 | 2024-07-10 10:20:00 | 8393.05 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-07-10 10:05:00 | 8439.95 | 2024-07-10 11:30:00 | 8439.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-15 11:10:00 | 8366.90 | 2024-07-15 11:20:00 | 8392.58 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-07-19 10:20:00 | 8180.00 | 2024-07-19 10:25:00 | 8198.33 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-07-22 10:15:00 | 8195.00 | 2024-07-22 10:30:00 | 8240.75 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-07-22 10:15:00 | 8195.00 | 2024-07-22 14:25:00 | 8195.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-23 10:30:00 | 8320.00 | 2024-07-23 10:35:00 | 8286.93 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-07-25 10:00:00 | 8190.45 | 2024-07-25 10:05:00 | 8157.53 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-07-25 10:00:00 | 8190.45 | 2024-07-25 10:15:00 | 8190.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-26 10:25:00 | 8440.25 | 2024-07-26 10:45:00 | 8489.53 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-07-26 10:25:00 | 8440.25 | 2024-07-26 11:05:00 | 8440.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-30 09:35:00 | 8839.00 | 2024-07-30 09:45:00 | 8876.68 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-07-30 09:35:00 | 8839.00 | 2024-07-30 09:50:00 | 8839.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-08 10:55:00 | 8305.00 | 2024-08-08 11:20:00 | 8330.88 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-08-13 09:30:00 | 8291.55 | 2024-08-13 09:35:00 | 8317.78 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-08-14 09:45:00 | 8105.00 | 2024-08-14 09:50:00 | 8135.77 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-08-21 10:30:00 | 8545.00 | 2024-08-21 10:35:00 | 8584.61 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-08-21 10:30:00 | 8545.00 | 2024-08-21 10:50:00 | 8545.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-23 10:05:00 | 8750.05 | 2024-08-23 10:20:00 | 8791.77 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-08-23 10:05:00 | 8750.05 | 2024-08-23 10:25:00 | 8750.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-28 09:30:00 | 8300.00 | 2024-08-28 09:50:00 | 8326.42 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-08-29 09:35:00 | 8221.00 | 2024-08-29 09:45:00 | 8241.85 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-08-30 10:50:00 | 8205.30 | 2024-08-30 10:55:00 | 8181.39 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-09-03 10:55:00 | 8332.55 | 2024-09-03 11:05:00 | 8311.54 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-09-10 10:50:00 | 7471.10 | 2024-09-10 11:30:00 | 7430.08 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-09-10 10:50:00 | 7471.10 | 2024-09-10 15:20:00 | 7342.20 | TARGET_HIT | 0.50 | 1.73% |
| BUY | retest1 | 2024-09-11 09:50:00 | 7430.00 | 2024-09-11 09:55:00 | 7391.22 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2024-09-17 09:30:00 | 7058.70 | 2024-09-17 09:35:00 | 7028.14 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-09-17 09:30:00 | 7058.70 | 2024-09-17 10:25:00 | 7058.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-18 09:45:00 | 7044.70 | 2024-09-18 10:05:00 | 7013.90 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-09-18 09:45:00 | 7044.70 | 2024-09-18 15:20:00 | 6915.65 | TARGET_HIT | 0.50 | 1.83% |
| BUY | retest1 | 2024-09-20 11:10:00 | 7330.10 | 2024-09-20 11:15:00 | 7263.54 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest1 | 2024-09-24 09:35:00 | 7152.50 | 2024-09-24 09:50:00 | 7106.54 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-09-24 09:35:00 | 7152.50 | 2024-09-24 13:35:00 | 7110.00 | TARGET_HIT | 0.50 | 0.59% |
| SELL | retest1 | 2024-09-25 09:40:00 | 7075.00 | 2024-09-25 09:50:00 | 7101.80 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-10-01 09:35:00 | 7580.00 | 2024-10-01 09:40:00 | 7630.55 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2024-10-01 09:35:00 | 7580.00 | 2024-10-01 10:00:00 | 7580.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-10 10:50:00 | 7232.00 | 2024-10-10 11:25:00 | 7196.22 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-10-10 10:50:00 | 7232.00 | 2024-10-10 12:35:00 | 7232.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-17 10:20:00 | 6910.00 | 2024-10-17 10:45:00 | 6942.33 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-10-23 09:30:00 | 6410.00 | 2024-10-23 09:40:00 | 6370.33 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-10-23 09:30:00 | 6410.00 | 2024-10-23 09:50:00 | 6410.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-28 10:30:00 | 7010.90 | 2024-11-28 10:40:00 | 6982.00 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-11-28 10:30:00 | 7010.90 | 2024-11-28 15:20:00 | 6968.50 | TARGET_HIT | 0.50 | 0.60% |
| SELL | retest1 | 2024-11-29 09:35:00 | 6923.95 | 2024-11-29 10:00:00 | 6891.81 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-11-29 09:35:00 | 6923.95 | 2024-11-29 10:25:00 | 6898.15 | TARGET_HIT | 0.50 | 0.37% |
| SELL | retest1 | 2024-12-06 09:30:00 | 6866.25 | 2024-12-06 10:05:00 | 6833.21 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-12-06 09:30:00 | 6866.25 | 2024-12-06 10:20:00 | 6866.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-11 11:00:00 | 6659.50 | 2024-12-11 15:15:00 | 6635.85 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-12-11 11:00:00 | 6659.50 | 2024-12-11 15:20:00 | 6625.00 | TARGET_HIT | 0.50 | 0.52% |
| SELL | retest1 | 2024-12-12 10:20:00 | 6573.60 | 2024-12-12 12:25:00 | 6545.60 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-12-12 10:20:00 | 6573.60 | 2024-12-12 15:05:00 | 6573.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-16 09:45:00 | 6687.85 | 2024-12-16 09:50:00 | 6723.02 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-12-16 09:45:00 | 6687.85 | 2024-12-16 14:30:00 | 6773.05 | TARGET_HIT | 0.50 | 1.27% |
| BUY | retest1 | 2024-12-17 10:00:00 | 6830.00 | 2024-12-17 10:15:00 | 6799.33 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-12-18 10:55:00 | 6710.20 | 2024-12-18 11:05:00 | 6682.95 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-12-18 10:55:00 | 6710.20 | 2024-12-18 11:10:00 | 6710.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-20 09:30:00 | 6680.00 | 2024-12-20 09:35:00 | 6709.11 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-12-20 09:30:00 | 6680.00 | 2024-12-20 10:00:00 | 6709.25 | TARGET_HIT | 0.50 | 0.44% |
| SELL | retest1 | 2024-12-30 11:00:00 | 6503.00 | 2024-12-30 11:20:00 | 6517.33 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-01-09 09:30:00 | 6798.60 | 2025-01-09 09:35:00 | 6827.07 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-01-17 10:55:00 | 6513.35 | 2025-01-17 11:10:00 | 6481.23 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-01-17 10:55:00 | 6513.35 | 2025-01-17 13:35:00 | 6496.65 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2025-01-20 10:30:00 | 6545.50 | 2025-01-20 10:35:00 | 6524.99 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-01-23 09:35:00 | 6580.00 | 2025-01-23 09:45:00 | 6615.69 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-01-23 09:35:00 | 6580.00 | 2025-01-23 11:15:00 | 6687.00 | TARGET_HIT | 0.50 | 1.63% |
| BUY | retest1 | 2025-01-30 09:45:00 | 6500.80 | 2025-01-30 10:10:00 | 6475.22 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-01-31 09:30:00 | 6291.30 | 2025-01-31 09:35:00 | 6312.33 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-02-01 10:25:00 | 6340.85 | 2025-02-01 10:30:00 | 6308.47 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-02-01 10:25:00 | 6340.85 | 2025-02-01 11:10:00 | 6340.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-06 11:05:00 | 6687.40 | 2025-02-06 11:15:00 | 6650.20 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-02-06 11:05:00 | 6687.40 | 2025-02-06 15:20:00 | 6591.50 | TARGET_HIT | 0.50 | 1.43% |
| BUY | retest1 | 2025-02-25 10:45:00 | 6890.00 | 2025-02-25 10:55:00 | 6859.31 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-03-18 09:30:00 | 7248.90 | 2025-03-18 09:35:00 | 7210.01 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2025-04-15 10:00:00 | 8876.50 | 2025-04-15 10:25:00 | 8933.13 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-04-15 10:00:00 | 8876.50 | 2025-04-15 10:45:00 | 8876.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-24 09:40:00 | 9249.50 | 2025-04-24 09:45:00 | 9206.45 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2025-04-29 11:05:00 | 9053.50 | 2025-04-29 12:20:00 | 8998.80 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2025-04-29 11:05:00 | 9053.50 | 2025-04-29 14:45:00 | 9053.50 | STOP_HIT | 0.50 | 0.00% |
