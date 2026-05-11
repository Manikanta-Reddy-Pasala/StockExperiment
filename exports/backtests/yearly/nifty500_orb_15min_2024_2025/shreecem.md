# Shree Cement Ltd. (SHREECEM)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 25445.00
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
| ENTRY1 | 102 |
| ENTRY2 | 0 |
| PARTIAL | 42 |
| TARGET_HIT | 19 |
| STOP_HIT | 83 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 144 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 61 / 83
- **Target hits / Stop hits / Partials:** 19 / 83 / 42
- **Avg / median % per leg:** 0.13% / 0.00%
- **Sum % (uncompounded):** 18.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 71 | 30 | 42.3% | 11 | 41 | 19 | 0.15% | 10.9% |
| BUY @ 2nd Alert (retest1) | 71 | 30 | 42.3% | 11 | 41 | 19 | 0.15% | 10.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 73 | 31 | 42.5% | 8 | 42 | 23 | 0.10% | 7.6% |
| SELL @ 2nd Alert (retest1) | 73 | 31 | 42.5% | 8 | 42 | 23 | 0.10% | 7.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 144 | 61 | 42.4% | 19 | 83 | 42 | 0.13% | 18.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-14 10:25:00 | 25931.05 | 26219.11 | 0.00 | ORB-short ORB[26049.95,26431.75] vol=2.5x ATR=123.81 |
| Stop hit — per-position SL triggered | 2024-05-14 10:30:00 | 26054.86 | 26214.07 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 10:50:00 | 25574.70 | 25610.43 | 0.00 | ORB-short ORB[25673.10,25850.00] vol=3.1x ATR=65.04 |
| Stop hit — per-position SL triggered | 2024-05-17 10:55:00 | 25639.74 | 25612.38 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-21 09:30:00 | 25833.30 | 26031.77 | 0.00 | ORB-short ORB[25921.00,26300.00] vol=1.9x ATR=103.61 |
| Stop hit — per-position SL triggered | 2024-05-21 10:10:00 | 25936.91 | 25975.79 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-22 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 10:25:00 | 25600.00 | 25634.55 | 0.00 | ORB-short ORB[25795.90,25950.00] vol=7.6x ATR=71.17 |
| Stop hit — per-position SL triggered | 2024-05-22 10:35:00 | 25671.17 | 25634.79 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 10:35:00 | 25486.30 | 25583.08 | 0.00 | ORB-short ORB[25535.00,25851.00] vol=1.7x ATR=54.56 |
| Stop hit — per-position SL triggered | 2024-05-23 12:40:00 | 25540.86 | 25530.77 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 10:00:00 | 25327.15 | 25345.79 | 0.00 | ORB-short ORB[25331.00,25500.00] vol=3.6x ATR=55.51 |
| Stop hit — per-position SL triggered | 2024-05-27 10:05:00 | 25382.66 | 25349.02 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-03 10:50:00 | 25460.00 | 25383.18 | 0.00 | ORB-long ORB[25088.10,25298.00] vol=1.6x ATR=102.39 |
| Stop hit — per-position SL triggered | 2024-06-03 12:25:00 | 25357.61 | 25405.47 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 11:00:00 | 25794.20 | 25642.93 | 0.00 | ORB-long ORB[25500.00,25689.90] vol=1.6x ATR=71.11 |
| Stop hit — per-position SL triggered | 2024-06-06 11:10:00 | 25723.09 | 25649.54 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 10:30:00 | 26080.80 | 25964.54 | 0.00 | ORB-long ORB[25511.20,25825.00] vol=12.5x ATR=95.16 |
| Stop hit — per-position SL triggered | 2024-06-07 10:45:00 | 25985.64 | 25972.92 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 10:00:00 | 26842.55 | 26498.74 | 0.00 | ORB-long ORB[26078.00,26366.00] vol=2.5x ATR=115.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 10:05:00 | 27015.29 | 26672.48 | 0.00 | T1 1.5R @ 27015.29 |
| Target hit | 2024-06-10 15:20:00 | 27227.15 | 27135.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2024-06-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 10:35:00 | 27389.75 | 27208.14 | 0.00 | ORB-long ORB[26890.10,27261.80] vol=1.9x ATR=82.98 |
| Stop hit — per-position SL triggered | 2024-06-12 10:45:00 | 27306.77 | 27212.90 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-06-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 11:00:00 | 27241.70 | 27454.16 | 0.00 | ORB-short ORB[27471.60,27625.00] vol=3.7x ATR=65.25 |
| Stop hit — per-position SL triggered | 2024-06-13 12:00:00 | 27306.95 | 27401.00 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-06-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:30:00 | 27981.45 | 27806.55 | 0.00 | ORB-long ORB[27491.00,27897.15] vol=2.3x ATR=91.55 |
| Stop hit — per-position SL triggered | 2024-06-14 09:35:00 | 27889.90 | 27807.68 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-06-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-18 10:35:00 | 27343.60 | 27421.26 | 0.00 | ORB-short ORB[27400.00,27699.90] vol=2.4x ATR=79.89 |
| Stop hit — per-position SL triggered | 2024-06-18 12:30:00 | 27423.49 | 27387.69 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 10:15:00 | 27477.55 | 27433.32 | 0.00 | ORB-long ORB[27286.90,27464.75] vol=1.8x ATR=76.04 |
| Stop hit — per-position SL triggered | 2024-06-25 11:15:00 | 27401.51 | 27457.53 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-06-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 09:45:00 | 27680.50 | 27398.15 | 0.00 | ORB-long ORB[27086.25,27351.95] vol=1.7x ATR=101.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 09:55:00 | 27832.86 | 27561.97 | 0.00 | T1 1.5R @ 27832.86 |
| Target hit | 2024-06-26 10:50:00 | 27769.25 | 27802.47 | 0.00 | Trail-exit close<VWAP |

### Cycle 17 — BUY (started 2024-06-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 09:50:00 | 28106.80 | 27989.84 | 0.00 | ORB-long ORB[27711.10,27941.65] vol=8.5x ATR=117.78 |
| Stop hit — per-position SL triggered | 2024-06-28 10:00:00 | 27989.02 | 27991.54 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-01 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 10:05:00 | 28321.65 | 28130.82 | 0.00 | ORB-long ORB[27916.90,28018.90] vol=2.9x ATR=72.20 |
| Stop hit — per-position SL triggered | 2024-07-01 10:10:00 | 28249.45 | 28150.97 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 10:55:00 | 28189.60 | 28309.22 | 0.00 | ORB-short ORB[28213.50,28488.00] vol=2.0x ATR=77.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 11:40:00 | 28072.97 | 28266.77 | 0.00 | T1 1.5R @ 28072.97 |
| Target hit | 2024-07-02 15:20:00 | 27654.50 | 28008.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2024-07-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-03 10:40:00 | 27501.00 | 27544.04 | 0.00 | ORB-short ORB[27705.05,27899.00] vol=15.9x ATR=72.26 |
| Stop hit — per-position SL triggered | 2024-07-03 10:55:00 | 27573.26 | 27544.14 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-07-04 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 10:20:00 | 27335.65 | 27441.22 | 0.00 | ORB-short ORB[27430.05,27554.90] vol=2.1x ATR=66.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 10:25:00 | 27236.39 | 27420.66 | 0.00 | T1 1.5R @ 27236.39 |
| Stop hit — per-position SL triggered | 2024-07-04 12:10:00 | 27335.65 | 27334.33 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-07-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 10:35:00 | 27087.65 | 27217.63 | 0.00 | ORB-short ORB[27280.00,27549.60] vol=2.8x ATR=61.60 |
| Stop hit — per-position SL triggered | 2024-07-08 10:40:00 | 27149.25 | 27208.52 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-07-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:05:00 | 27641.00 | 27792.11 | 0.00 | ORB-short ORB[27764.35,27950.00] vol=1.8x ATR=85.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:20:00 | 27512.59 | 27732.64 | 0.00 | T1 1.5R @ 27512.59 |
| Stop hit — per-position SL triggered | 2024-07-10 10:55:00 | 27641.00 | 27670.96 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-07-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 11:05:00 | 27586.35 | 27637.98 | 0.00 | ORB-short ORB[27656.10,27885.00] vol=1.9x ATR=41.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 11:15:00 | 27524.51 | 27624.95 | 0.00 | T1 1.5R @ 27524.51 |
| Stop hit — per-position SL triggered | 2024-07-12 11:45:00 | 27586.35 | 27570.66 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-07-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:55:00 | 27741.20 | 27561.89 | 0.00 | ORB-long ORB[27320.45,27679.90] vol=1.7x ATR=72.03 |
| Stop hit — per-position SL triggered | 2024-07-26 11:15:00 | 27669.17 | 27600.58 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-07-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-29 11:05:00 | 27555.10 | 27563.32 | 0.00 | ORB-short ORB[27601.40,27800.00] vol=1.6x ATR=54.81 |
| Stop hit — per-position SL triggered | 2024-07-29 11:30:00 | 27609.91 | 27566.72 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-07-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 11:10:00 | 27577.85 | 27484.57 | 0.00 | ORB-long ORB[27312.30,27476.70] vol=3.0x ATR=59.82 |
| Stop hit — per-position SL triggered | 2024-07-31 11:30:00 | 27518.03 | 27519.41 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-02 11:10:00 | 27730.80 | 27613.18 | 0.00 | ORB-long ORB[27475.00,27635.20] vol=4.0x ATR=65.54 |
| Stop hit — per-position SL triggered | 2024-08-02 12:00:00 | 27665.26 | 27630.10 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-08-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 10:00:00 | 24295.95 | 24327.39 | 0.00 | ORB-short ORB[24300.00,24509.25] vol=2.8x ATR=52.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 10:15:00 | 24216.67 | 24311.35 | 0.00 | T1 1.5R @ 24216.67 |
| Stop hit — per-position SL triggered | 2024-08-13 10:40:00 | 24295.95 | 24295.88 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-08-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-14 10:55:00 | 24350.00 | 24202.49 | 0.00 | ORB-long ORB[24011.10,24244.95] vol=4.5x ATR=46.97 |
| Stop hit — per-position SL triggered | 2024-08-14 11:00:00 | 24303.03 | 24232.30 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-08-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-16 10:20:00 | 24326.15 | 24349.23 | 0.00 | ORB-short ORB[24375.00,24499.90] vol=3.7x ATR=48.20 |
| Stop hit — per-position SL triggered | 2024-08-16 10:25:00 | 24374.35 | 24349.40 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-08-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 10:45:00 | 24730.00 | 24836.84 | 0.00 | ORB-short ORB[24804.55,24960.00] vol=2.6x ATR=44.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 11:30:00 | 24662.85 | 24800.65 | 0.00 | T1 1.5R @ 24662.85 |
| Stop hit — per-position SL triggered | 2024-08-20 12:00:00 | 24730.00 | 24764.02 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-08-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 09:35:00 | 25019.95 | 24966.06 | 0.00 | ORB-long ORB[24800.45,25000.20] vol=2.5x ATR=40.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 10:25:00 | 25081.19 | 24993.72 | 0.00 | T1 1.5R @ 25081.19 |
| Target hit | 2024-08-22 11:15:00 | 25046.10 | 25047.85 | 0.00 | Trail-exit close<VWAP |

### Cycle 34 — SELL (started 2024-08-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 09:30:00 | 24722.50 | 24868.19 | 0.00 | ORB-short ORB[24821.35,25100.00] vol=2.0x ATR=59.30 |
| Stop hit — per-position SL triggered | 2024-08-23 12:05:00 | 24781.80 | 24728.30 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-08-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 10:35:00 | 24872.90 | 24739.82 | 0.00 | ORB-long ORB[24602.20,24751.00] vol=2.9x ATR=40.53 |
| Stop hit — per-position SL triggered | 2024-08-26 10:50:00 | 24832.37 | 24752.09 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-08-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 09:40:00 | 24827.10 | 24859.78 | 0.00 | ORB-short ORB[24830.75,25049.75] vol=1.7x ATR=44.57 |
| Stop hit — per-position SL triggered | 2024-08-27 10:20:00 | 24871.67 | 24853.33 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 24713.95 | 24797.29 | 0.00 | ORB-short ORB[24770.05,24897.30] vol=2.5x ATR=42.65 |
| Stop hit — per-position SL triggered | 2024-08-28 09:35:00 | 24756.60 | 24782.35 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-08-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 10:45:00 | 25104.50 | 24964.88 | 0.00 | ORB-long ORB[24810.00,24962.90] vol=3.6x ATR=58.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-30 11:00:00 | 25191.79 | 25019.05 | 0.00 | T1 1.5R @ 25191.79 |
| Target hit | 2024-08-30 15:20:00 | 25481.10 | 25382.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — SELL (started 2024-09-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:30:00 | 25448.45 | 25507.69 | 0.00 | ORB-short ORB[25561.25,25888.00] vol=1.6x ATR=71.30 |
| Stop hit — per-position SL triggered | 2024-09-06 10:45:00 | 25519.75 | 25495.75 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-09-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 11:10:00 | 25800.00 | 25710.24 | 0.00 | ORB-long ORB[25651.00,25766.75] vol=1.8x ATR=33.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 11:20:00 | 25849.92 | 25723.84 | 0.00 | T1 1.5R @ 25849.92 |
| Target hit | 2024-09-12 15:20:00 | 26008.00 | 25925.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — SELL (started 2024-09-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 11:00:00 | 25777.50 | 25926.27 | 0.00 | ORB-short ORB[25902.05,26050.00] vol=4.1x ATR=55.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-16 11:55:00 | 25694.71 | 25858.27 | 0.00 | T1 1.5R @ 25694.71 |
| Target hit | 2024-09-16 15:20:00 | 25627.00 | 25694.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — SELL (started 2024-09-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 10:35:00 | 25350.00 | 25408.89 | 0.00 | ORB-short ORB[25359.20,25496.65] vol=2.4x ATR=38.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 10:40:00 | 25292.14 | 25371.94 | 0.00 | T1 1.5R @ 25292.14 |
| Stop hit — per-position SL triggered | 2024-09-18 11:05:00 | 25350.00 | 25357.35 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-09-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 11:05:00 | 24994.75 | 25155.37 | 0.00 | ORB-short ORB[25111.00,25287.95] vol=3.5x ATR=61.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 11:20:00 | 24902.75 | 25129.65 | 0.00 | T1 1.5R @ 24902.75 |
| Target hit | 2024-09-19 15:20:00 | 24968.95 | 24950.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — SELL (started 2024-09-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 10:55:00 | 25802.30 | 25849.74 | 0.00 | ORB-short ORB[25807.60,25927.25] vol=6.4x ATR=33.49 |
| Stop hit — per-position SL triggered | 2024-09-26 11:15:00 | 25835.79 | 25845.83 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-09-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 09:40:00 | 26439.05 | 26312.63 | 0.00 | ORB-long ORB[26040.10,26269.95] vol=2.8x ATR=61.05 |
| Stop hit — per-position SL triggered | 2024-09-27 09:50:00 | 26378.00 | 26342.81 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-10-01 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 10:05:00 | 26278.05 | 26301.95 | 0.00 | ORB-short ORB[26280.00,26580.80] vol=4.6x ATR=62.73 |
| Stop hit — per-position SL triggered | 2024-10-01 10:15:00 | 26340.78 | 26307.78 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-10-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-03 09:50:00 | 26626.10 | 26459.83 | 0.00 | ORB-long ORB[26208.30,26420.00] vol=2.5x ATR=67.62 |
| Stop hit — per-position SL triggered | 2024-10-03 10:10:00 | 26558.48 | 26531.14 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-10-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:40:00 | 25760.00 | 25890.46 | 0.00 | ORB-short ORB[25989.80,26168.00] vol=2.1x ATR=67.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:45:00 | 25659.12 | 25869.68 | 0.00 | T1 1.5R @ 25659.12 |
| Stop hit — per-position SL triggered | 2024-10-07 11:05:00 | 25760.00 | 25818.47 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-10-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 10:40:00 | 25562.00 | 25395.40 | 0.00 | ORB-long ORB[25188.45,25376.45] vol=1.9x ATR=66.69 |
| Stop hit — per-position SL triggered | 2024-10-08 10:50:00 | 25495.31 | 25403.02 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-10-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-10 11:00:00 | 24850.00 | 24963.60 | 0.00 | ORB-short ORB[24903.60,25196.90] vol=2.8x ATR=58.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-10 12:15:00 | 24762.90 | 24917.37 | 0.00 | T1 1.5R @ 24762.90 |
| Stop hit — per-position SL triggered | 2024-10-10 13:10:00 | 24850.00 | 24840.92 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 11:15:00 | 24317.75 | 24491.83 | 0.00 | ORB-short ORB[24503.00,24657.65] vol=3.5x ATR=45.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 11:30:00 | 24248.77 | 24447.45 | 0.00 | T1 1.5R @ 24248.77 |
| Stop hit — per-position SL triggered | 2024-10-11 11:45:00 | 24317.75 | 24438.33 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 10:15:00 | 24170.00 | 24244.44 | 0.00 | ORB-short ORB[24285.00,24478.60] vol=1.7x ATR=57.22 |
| Stop hit — per-position SL triggered | 2024-10-14 10:20:00 | 24227.22 | 24242.49 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-10-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 10:45:00 | 24802.00 | 24710.17 | 0.00 | ORB-long ORB[24581.00,24727.25] vol=1.6x ATR=47.19 |
| Stop hit — per-position SL triggered | 2024-10-15 11:05:00 | 24754.81 | 24719.90 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 11:15:00 | 24377.60 | 24482.48 | 0.00 | ORB-short ORB[24464.05,24731.00] vol=2.8x ATR=40.45 |
| Stop hit — per-position SL triggered | 2024-10-16 11:20:00 | 24418.05 | 24480.83 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-10-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:30:00 | 24106.30 | 24256.16 | 0.00 | ORB-short ORB[24186.75,24361.40] vol=1.6x ATR=48.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 10:15:00 | 24033.91 | 24144.49 | 0.00 | T1 1.5R @ 24033.91 |
| Stop hit — per-position SL triggered | 2024-10-21 10:20:00 | 24106.30 | 24141.11 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-10-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:40:00 | 24924.95 | 25011.68 | 0.00 | ORB-short ORB[25025.10,25170.00] vol=2.2x ATR=83.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:50:00 | 24800.33 | 24990.67 | 0.00 | T1 1.5R @ 24800.33 |
| Stop hit — per-position SL triggered | 2024-10-25 11:15:00 | 24924.95 | 24964.29 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-10-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 10:40:00 | 25012.95 | 25070.99 | 0.00 | ORB-short ORB[25025.00,25198.75] vol=1.6x ATR=58.55 |
| Stop hit — per-position SL triggered | 2024-10-29 11:35:00 | 25071.50 | 25057.35 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2024-10-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 11:10:00 | 25341.85 | 25287.60 | 0.00 | ORB-long ORB[25020.05,25293.15] vol=1.5x ATR=75.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 12:00:00 | 25455.78 | 25305.54 | 0.00 | T1 1.5R @ 25455.78 |
| Stop hit — per-position SL triggered | 2024-10-30 13:10:00 | 25341.85 | 25350.89 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-11-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-04 10:05:00 | 25061.10 | 25114.41 | 0.00 | ORB-short ORB[25062.50,25400.00] vol=3.6x ATR=71.68 |
| Stop hit — per-position SL triggered | 2024-11-04 11:25:00 | 25132.78 | 25079.81 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-11-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-05 11:10:00 | 24760.00 | 24877.45 | 0.00 | ORB-short ORB[24836.75,25000.00] vol=2.4x ATR=44.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-05 11:25:00 | 24692.83 | 24866.02 | 0.00 | T1 1.5R @ 24692.83 |
| Stop hit — per-position SL triggered | 2024-11-05 12:05:00 | 24760.00 | 24809.02 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2024-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 11:15:00 | 24740.00 | 24873.48 | 0.00 | ORB-short ORB[25010.05,25198.00] vol=4.1x ATR=44.44 |
| Stop hit — per-position SL triggered | 2024-11-07 11:35:00 | 24784.44 | 24860.98 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-11-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-08 11:00:00 | 24535.15 | 24653.19 | 0.00 | ORB-short ORB[24615.05,24849.05] vol=2.5x ATR=51.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-08 11:30:00 | 24458.36 | 24615.02 | 0.00 | T1 1.5R @ 24458.36 |
| Target hit | 2024-11-08 13:10:00 | 24500.00 | 24458.57 | 0.00 | Trail-exit close>VWAP |

### Cycle 63 — BUY (started 2024-11-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-21 09:30:00 | 24137.40 | 24057.33 | 0.00 | ORB-long ORB[23821.10,24095.65] vol=1.5x ATR=61.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-21 09:40:00 | 24229.62 | 24097.14 | 0.00 | T1 1.5R @ 24229.62 |
| Target hit | 2024-11-21 14:10:00 | 24220.30 | 24238.95 | 0.00 | Trail-exit close<VWAP |

### Cycle 64 — BUY (started 2024-11-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 10:55:00 | 24523.90 | 24363.19 | 0.00 | ORB-long ORB[24043.65,24398.00] vol=1.6x ATR=58.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-22 11:25:00 | 24611.40 | 24413.98 | 0.00 | T1 1.5R @ 24611.40 |
| Target hit | 2024-11-22 15:20:00 | 24815.85 | 24575.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 65 — BUY (started 2024-11-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 09:35:00 | 26000.00 | 25813.54 | 0.00 | ORB-long ORB[25563.35,25880.00] vol=2.5x ATR=110.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 09:40:00 | 26166.15 | 25943.18 | 0.00 | T1 1.5R @ 26166.15 |
| Target hit | 2024-11-29 13:05:00 | 26090.00 | 26134.69 | 0.00 | Trail-exit close<VWAP |

### Cycle 66 — BUY (started 2024-12-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 11:00:00 | 26850.00 | 26549.30 | 0.00 | ORB-long ORB[26000.00,26294.45] vol=1.7x ATR=103.28 |
| Stop hit — per-position SL triggered | 2024-12-02 11:15:00 | 26746.72 | 26574.21 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2024-12-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 09:40:00 | 27136.15 | 27030.84 | 0.00 | ORB-long ORB[26784.00,27089.95] vol=2.1x ATR=69.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 11:10:00 | 27240.73 | 27095.58 | 0.00 | T1 1.5R @ 27240.73 |
| Stop hit — per-position SL triggered | 2024-12-03 14:10:00 | 27136.15 | 27129.70 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 27048.00 | 27208.76 | 0.00 | ORB-short ORB[27205.00,27510.00] vol=6.3x ATR=77.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-05 11:05:00 | 26931.65 | 27175.88 | 0.00 | T1 1.5R @ 26931.65 |
| Target hit | 2024-12-05 15:20:00 | 26601.70 | 26890.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 69 — BUY (started 2024-12-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 09:40:00 | 26870.00 | 26718.58 | 0.00 | ORB-long ORB[26515.15,26748.40] vol=2.1x ATR=83.71 |
| Stop hit — per-position SL triggered | 2024-12-06 09:45:00 | 26786.29 | 26721.52 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2024-12-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 10:50:00 | 27349.35 | 27249.23 | 0.00 | ORB-long ORB[27180.00,27300.00] vol=1.6x ATR=45.40 |
| Stop hit — per-position SL triggered | 2024-12-12 10:55:00 | 27303.95 | 27252.40 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2024-12-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-16 09:30:00 | 27749.00 | 27688.13 | 0.00 | ORB-long ORB[27502.00,27744.00] vol=2.1x ATR=63.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 10:05:00 | 27843.86 | 27754.43 | 0.00 | T1 1.5R @ 27843.86 |
| Stop hit — per-position SL triggered | 2024-12-16 10:40:00 | 27749.00 | 27756.80 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2024-12-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 10:30:00 | 28200.30 | 28137.95 | 0.00 | ORB-long ORB[27960.60,28152.50] vol=6.0x ATR=50.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 10:40:00 | 28275.45 | 28146.32 | 0.00 | T1 1.5R @ 28275.45 |
| Stop hit — per-position SL triggered | 2024-12-17 11:05:00 | 28200.30 | 28158.33 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2024-12-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-18 11:10:00 | 28062.95 | 28250.26 | 0.00 | ORB-short ORB[28139.95,28350.00] vol=1.6x ATR=54.08 |
| Stop hit — per-position SL triggered | 2024-12-18 11:25:00 | 28117.03 | 28236.64 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2024-12-20 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 10:20:00 | 27496.05 | 27634.21 | 0.00 | ORB-short ORB[27580.35,27786.00] vol=2.4x ATR=66.68 |
| Stop hit — per-position SL triggered | 2024-12-20 10:30:00 | 27562.73 | 27622.76 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2025-01-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 10:05:00 | 25811.25 | 25742.22 | 0.00 | ORB-long ORB[25587.70,25802.65] vol=2.1x ATR=56.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 10:30:00 | 25895.75 | 25783.60 | 0.00 | T1 1.5R @ 25895.75 |
| Target hit | 2025-01-02 15:20:00 | 26689.50 | 26393.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 76 — BUY (started 2025-01-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-14 11:00:00 | 25128.45 | 25020.01 | 0.00 | ORB-long ORB[24870.05,25108.65] vol=1.8x ATR=53.61 |
| Stop hit — per-position SL triggered | 2025-01-14 11:15:00 | 25074.84 | 25030.90 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2025-01-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-15 10:35:00 | 25320.45 | 25191.82 | 0.00 | ORB-long ORB[25107.65,25301.00] vol=1.7x ATR=62.07 |
| Stop hit — per-position SL triggered | 2025-01-15 11:00:00 | 25258.38 | 25214.94 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2025-01-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-22 10:30:00 | 25300.00 | 25223.19 | 0.00 | ORB-long ORB[25025.40,25297.05] vol=1.9x ATR=75.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 11:20:00 | 25413.92 | 25262.08 | 0.00 | T1 1.5R @ 25413.92 |
| Stop hit — per-position SL triggered | 2025-01-22 11:35:00 | 25300.00 | 25272.03 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2025-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-28 11:15:00 | 25438.05 | 25308.11 | 0.00 | ORB-long ORB[25069.95,25312.45] vol=1.6x ATR=77.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 11:30:00 | 25554.59 | 25335.63 | 0.00 | T1 1.5R @ 25554.59 |
| Target hit | 2025-01-28 15:20:00 | 26041.65 | 25798.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 80 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 10:15:00 | 26560.00 | 26257.86 | 0.00 | ORB-long ORB[25896.00,26240.30] vol=1.8x ATR=108.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 11:35:00 | 26723.05 | 26425.15 | 0.00 | T1 1.5R @ 26723.05 |
| Target hit | 2025-01-29 15:10:00 | 26654.00 | 26688.44 | 0.00 | Trail-exit close<VWAP |

### Cycle 81 — BUY (started 2025-02-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-04 10:55:00 | 27832.25 | 27591.24 | 0.00 | ORB-long ORB[27349.75,27717.85] vol=2.2x ATR=94.45 |
| Stop hit — per-position SL triggered | 2025-02-04 11:50:00 | 27737.80 | 27671.26 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2025-02-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 09:40:00 | 28150.35 | 27929.31 | 0.00 | ORB-long ORB[27735.20,28052.25] vol=1.7x ATR=92.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-05 09:50:00 | 28289.01 | 27992.44 | 0.00 | T1 1.5R @ 28289.01 |
| Stop hit — per-position SL triggered | 2025-02-05 09:55:00 | 28150.35 | 28013.79 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2025-02-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-11 09:35:00 | 28303.60 | 28219.90 | 0.00 | ORB-long ORB[28020.65,28288.00] vol=2.0x ATR=83.38 |
| Stop hit — per-position SL triggered | 2025-02-11 09:45:00 | 28220.22 | 28222.23 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2025-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-13 09:30:00 | 28520.55 | 28363.84 | 0.00 | ORB-long ORB[28113.25,28451.00] vol=1.9x ATR=100.07 |
| Stop hit — per-position SL triggered | 2025-02-13 09:45:00 | 28420.48 | 28393.73 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2025-02-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 10:40:00 | 28164.70 | 28326.83 | 0.00 | ORB-short ORB[28189.05,28525.00] vol=1.5x ATR=98.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 11:05:00 | 28017.03 | 28286.54 | 0.00 | T1 1.5R @ 28017.03 |
| Stop hit — per-position SL triggered | 2025-02-14 11:35:00 | 28164.70 | 28243.49 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2025-02-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-18 10:55:00 | 28413.90 | 28237.65 | 0.00 | ORB-long ORB[28007.65,28400.00] vol=1.8x ATR=73.71 |
| Stop hit — per-position SL triggered | 2025-02-18 11:10:00 | 28340.19 | 28246.09 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2025-02-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 11:05:00 | 28265.45 | 28387.59 | 0.00 | ORB-short ORB[28407.55,28592.80] vol=1.6x ATR=67.71 |
| Stop hit — per-position SL triggered | 2025-02-21 11:25:00 | 28333.16 | 28376.77 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2025-02-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 09:45:00 | 28549.85 | 28408.79 | 0.00 | ORB-long ORB[28207.60,28394.60] vol=2.2x ATR=77.64 |
| Stop hit — per-position SL triggered | 2025-02-25 09:50:00 | 28472.21 | 28414.05 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2025-03-04 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-04 10:20:00 | 27569.80 | 27481.30 | 0.00 | ORB-long ORB[27212.35,27553.95] vol=7.2x ATR=82.30 |
| Stop hit — per-position SL triggered | 2025-03-04 11:30:00 | 27487.50 | 27493.09 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2025-03-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-06 11:05:00 | 28506.00 | 28310.90 | 0.00 | ORB-long ORB[28120.00,28413.20] vol=2.3x ATR=76.75 |
| Stop hit — per-position SL triggered | 2025-03-06 11:25:00 | 28429.25 | 28348.05 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2025-03-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 10:30:00 | 28328.55 | 28212.96 | 0.00 | ORB-long ORB[27936.50,28241.00] vol=1.7x ATR=77.06 |
| Stop hit — per-position SL triggered | 2025-03-10 10:55:00 | 28251.49 | 28230.82 | 0.00 | SL hit |

### Cycle 92 — SELL (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 11:15:00 | 27495.90 | 27730.09 | 0.00 | ORB-short ORB[27950.00,28200.00] vol=6.2x ATR=85.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 11:35:00 | 27367.06 | 27666.92 | 0.00 | T1 1.5R @ 27367.06 |
| Stop hit — per-position SL triggered | 2025-03-12 13:50:00 | 27495.90 | 27576.45 | 0.00 | SL hit |

### Cycle 93 — SELL (started 2025-03-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-18 11:10:00 | 27648.30 | 27850.17 | 0.00 | ORB-short ORB[27740.00,27949.90] vol=1.9x ATR=55.01 |
| Stop hit — per-position SL triggered | 2025-03-18 11:35:00 | 27703.31 | 27806.14 | 0.00 | SL hit |

### Cycle 94 — BUY (started 2025-03-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:30:00 | 29050.00 | 28895.38 | 0.00 | ORB-long ORB[28642.10,28980.00] vol=1.9x ATR=90.26 |
| Stop hit — per-position SL triggered | 2025-03-21 09:35:00 | 28959.74 | 28913.00 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2025-03-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-25 10:05:00 | 30009.20 | 29851.03 | 0.00 | ORB-long ORB[29603.35,29952.90] vol=1.8x ATR=107.09 |
| Stop hit — per-position SL triggered | 2025-03-25 10:10:00 | 29902.11 | 29854.25 | 0.00 | SL hit |

### Cycle 96 — BUY (started 2025-03-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 11:00:00 | 30179.95 | 30093.85 | 0.00 | ORB-long ORB[29731.10,30047.50] vol=5.7x ATR=63.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-27 11:05:00 | 30274.98 | 30108.77 | 0.00 | T1 1.5R @ 30274.98 |
| Stop hit — per-position SL triggered | 2025-03-27 11:20:00 | 30179.95 | 30145.04 | 0.00 | SL hit |

### Cycle 97 — SELL (started 2025-04-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-11 09:40:00 | 30293.95 | 30410.07 | 0.00 | ORB-short ORB[30323.90,30709.95] vol=1.8x ATR=92.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-11 10:00:00 | 30155.33 | 30371.96 | 0.00 | T1 1.5R @ 30155.33 |
| Target hit | 2025-04-11 11:30:00 | 30167.05 | 30101.77 | 0.00 | Trail-exit close>VWAP |

### Cycle 98 — SELL (started 2025-04-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-16 11:10:00 | 30930.00 | 31045.94 | 0.00 | ORB-short ORB[31000.00,31290.00] vol=5.6x ATR=70.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 11:25:00 | 30824.67 | 31009.19 | 0.00 | T1 1.5R @ 30824.67 |
| Target hit | 2025-04-16 15:20:00 | 30835.00 | 30864.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 99 — BUY (started 2025-04-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 10:35:00 | 31395.00 | 31242.24 | 0.00 | ORB-long ORB[30950.00,31165.00] vol=1.6x ATR=71.75 |
| Stop hit — per-position SL triggered | 2025-04-21 10:40:00 | 31323.25 | 31243.55 | 0.00 | SL hit |

### Cycle 100 — SELL (started 2025-04-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 10:55:00 | 29935.00 | 30240.82 | 0.00 | ORB-short ORB[30420.00,30780.00] vol=1.6x ATR=68.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 11:05:00 | 29832.62 | 30149.79 | 0.00 | T1 1.5R @ 29832.62 |
| Stop hit — per-position SL triggered | 2025-04-29 11:50:00 | 29935.00 | 30033.70 | 0.00 | SL hit |

### Cycle 101 — SELL (started 2025-05-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-02 11:00:00 | 29550.00 | 29614.26 | 0.00 | ORB-short ORB[29620.00,29825.00] vol=1.7x ATR=53.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-02 11:35:00 | 29470.05 | 29531.14 | 0.00 | T1 1.5R @ 29470.05 |
| Target hit | 2025-05-02 14:45:00 | 29330.00 | 29328.25 | 0.00 | Trail-exit close>VWAP |

### Cycle 102 — BUY (started 2025-05-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-06 09:35:00 | 29785.00 | 29648.78 | 0.00 | ORB-long ORB[29405.00,29695.00] vol=2.4x ATR=84.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 10:35:00 | 29912.12 | 29773.58 | 0.00 | T1 1.5R @ 29912.12 |
| Stop hit — per-position SL triggered | 2025-05-06 10:40:00 | 29785.00 | 29776.23 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-14 10:25:00 | 25931.05 | 2024-05-14 10:30:00 | 26054.86 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-05-17 10:50:00 | 25574.70 | 2024-05-17 10:55:00 | 25639.74 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-05-21 09:30:00 | 25833.30 | 2024-05-21 10:10:00 | 25936.91 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-05-22 10:25:00 | 25600.00 | 2024-05-22 10:35:00 | 25671.17 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-05-23 10:35:00 | 25486.30 | 2024-05-23 12:40:00 | 25540.86 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-05-27 10:00:00 | 25327.15 | 2024-05-27 10:05:00 | 25382.66 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-06-03 10:50:00 | 25460.00 | 2024-06-03 12:25:00 | 25357.61 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-06-06 11:00:00 | 25794.20 | 2024-06-06 11:10:00 | 25723.09 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-06-07 10:30:00 | 26080.80 | 2024-06-07 10:45:00 | 25985.64 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-06-10 10:00:00 | 26842.55 | 2024-06-10 10:05:00 | 27015.29 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-06-10 10:00:00 | 26842.55 | 2024-06-10 15:20:00 | 27227.15 | TARGET_HIT | 0.50 | 1.43% |
| BUY | retest1 | 2024-06-12 10:35:00 | 27389.75 | 2024-06-12 10:45:00 | 27306.77 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-06-13 11:00:00 | 27241.70 | 2024-06-13 12:00:00 | 27306.95 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-06-14 09:30:00 | 27981.45 | 2024-06-14 09:35:00 | 27889.90 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-06-18 10:35:00 | 27343.60 | 2024-06-18 12:30:00 | 27423.49 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-06-25 10:15:00 | 27477.55 | 2024-06-25 11:15:00 | 27401.51 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-06-26 09:45:00 | 27680.50 | 2024-06-26 09:55:00 | 27832.86 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-06-26 09:45:00 | 27680.50 | 2024-06-26 10:50:00 | 27769.25 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2024-06-28 09:50:00 | 28106.80 | 2024-06-28 10:00:00 | 27989.02 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-07-01 10:05:00 | 28321.65 | 2024-07-01 10:10:00 | 28249.45 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-07-02 10:55:00 | 28189.60 | 2024-07-02 11:40:00 | 28072.97 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-07-02 10:55:00 | 28189.60 | 2024-07-02 15:20:00 | 27654.50 | TARGET_HIT | 0.50 | 1.90% |
| SELL | retest1 | 2024-07-03 10:40:00 | 27501.00 | 2024-07-03 10:55:00 | 27573.26 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-07-04 10:20:00 | 27335.65 | 2024-07-04 10:25:00 | 27236.39 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-07-04 10:20:00 | 27335.65 | 2024-07-04 12:10:00 | 27335.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-08 10:35:00 | 27087.65 | 2024-07-08 10:40:00 | 27149.25 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-07-10 10:05:00 | 27641.00 | 2024-07-10 10:20:00 | 27512.59 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-07-10 10:05:00 | 27641.00 | 2024-07-10 10:55:00 | 27641.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-12 11:05:00 | 27586.35 | 2024-07-12 11:15:00 | 27524.51 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2024-07-12 11:05:00 | 27586.35 | 2024-07-12 11:45:00 | 27586.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-26 10:55:00 | 27741.20 | 2024-07-26 11:15:00 | 27669.17 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-07-29 11:05:00 | 27555.10 | 2024-07-29 11:30:00 | 27609.91 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-07-31 11:10:00 | 27577.85 | 2024-07-31 11:30:00 | 27518.03 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-08-02 11:10:00 | 27730.80 | 2024-08-02 12:00:00 | 27665.26 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-08-13 10:00:00 | 24295.95 | 2024-08-13 10:15:00 | 24216.67 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-08-13 10:00:00 | 24295.95 | 2024-08-13 10:40:00 | 24295.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-14 10:55:00 | 24350.00 | 2024-08-14 11:00:00 | 24303.03 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-08-16 10:20:00 | 24326.15 | 2024-08-16 10:25:00 | 24374.35 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-08-20 10:45:00 | 24730.00 | 2024-08-20 11:30:00 | 24662.85 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2024-08-20 10:45:00 | 24730.00 | 2024-08-20 12:00:00 | 24730.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-22 09:35:00 | 25019.95 | 2024-08-22 10:25:00 | 25081.19 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2024-08-22 09:35:00 | 25019.95 | 2024-08-22 11:15:00 | 25046.10 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2024-08-23 09:30:00 | 24722.50 | 2024-08-23 12:05:00 | 24781.80 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-08-26 10:35:00 | 24872.90 | 2024-08-26 10:50:00 | 24832.37 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2024-08-27 09:40:00 | 24827.10 | 2024-08-27 10:20:00 | 24871.67 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-08-28 09:30:00 | 24713.95 | 2024-08-28 09:35:00 | 24756.60 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2024-08-30 10:45:00 | 25104.50 | 2024-08-30 11:00:00 | 25191.79 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-08-30 10:45:00 | 25104.50 | 2024-08-30 15:20:00 | 25481.10 | TARGET_HIT | 0.50 | 1.50% |
| SELL | retest1 | 2024-09-06 10:30:00 | 25448.45 | 2024-09-06 10:45:00 | 25519.75 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-09-12 11:10:00 | 25800.00 | 2024-09-12 11:20:00 | 25849.92 | PARTIAL | 0.50 | 0.19% |
| BUY | retest1 | 2024-09-12 11:10:00 | 25800.00 | 2024-09-12 15:20:00 | 26008.00 | TARGET_HIT | 0.50 | 0.81% |
| SELL | retest1 | 2024-09-16 11:00:00 | 25777.50 | 2024-09-16 11:55:00 | 25694.71 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-09-16 11:00:00 | 25777.50 | 2024-09-16 15:20:00 | 25627.00 | TARGET_HIT | 0.50 | 0.58% |
| SELL | retest1 | 2024-09-18 10:35:00 | 25350.00 | 2024-09-18 10:40:00 | 25292.14 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2024-09-18 10:35:00 | 25350.00 | 2024-09-18 11:05:00 | 25350.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-19 11:05:00 | 24994.75 | 2024-09-19 11:20:00 | 24902.75 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-09-19 11:05:00 | 24994.75 | 2024-09-19 15:20:00 | 24968.95 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2024-09-26 10:55:00 | 25802.30 | 2024-09-26 11:15:00 | 25835.79 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2024-09-27 09:40:00 | 26439.05 | 2024-09-27 09:50:00 | 26378.00 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-10-01 10:05:00 | 26278.05 | 2024-10-01 10:15:00 | 26340.78 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-10-03 09:50:00 | 26626.10 | 2024-10-03 10:10:00 | 26558.48 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-10-07 10:40:00 | 25760.00 | 2024-10-07 10:45:00 | 25659.12 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-10-07 10:40:00 | 25760.00 | 2024-10-07 11:05:00 | 25760.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-08 10:40:00 | 25562.00 | 2024-10-08 10:50:00 | 25495.31 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-10-10 11:00:00 | 24850.00 | 2024-10-10 12:15:00 | 24762.90 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-10-10 11:00:00 | 24850.00 | 2024-10-10 13:10:00 | 24850.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-11 11:15:00 | 24317.75 | 2024-10-11 11:30:00 | 24248.77 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2024-10-11 11:15:00 | 24317.75 | 2024-10-11 11:45:00 | 24317.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-14 10:15:00 | 24170.00 | 2024-10-14 10:20:00 | 24227.22 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-10-15 10:45:00 | 24802.00 | 2024-10-15 11:05:00 | 24754.81 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-10-16 11:15:00 | 24377.60 | 2024-10-16 11:20:00 | 24418.05 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-10-21 09:30:00 | 24106.30 | 2024-10-21 10:15:00 | 24033.91 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-10-21 09:30:00 | 24106.30 | 2024-10-21 10:20:00 | 24106.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-25 10:40:00 | 24924.95 | 2024-10-25 10:50:00 | 24800.33 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-10-25 10:40:00 | 24924.95 | 2024-10-25 11:15:00 | 24924.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-29 10:40:00 | 25012.95 | 2024-10-29 11:35:00 | 25071.50 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-10-30 11:10:00 | 25341.85 | 2024-10-30 12:00:00 | 25455.78 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-10-30 11:10:00 | 25341.85 | 2024-10-30 13:10:00 | 25341.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-04 10:05:00 | 25061.10 | 2024-11-04 11:25:00 | 25132.78 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-11-05 11:10:00 | 24760.00 | 2024-11-05 11:25:00 | 24692.83 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2024-11-05 11:10:00 | 24760.00 | 2024-11-05 12:05:00 | 24760.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-07 11:15:00 | 24740.00 | 2024-11-07 11:35:00 | 24784.44 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-11-08 11:00:00 | 24535.15 | 2024-11-08 11:30:00 | 24458.36 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-11-08 11:00:00 | 24535.15 | 2024-11-08 13:10:00 | 24500.00 | TARGET_HIT | 0.50 | 0.14% |
| BUY | retest1 | 2024-11-21 09:30:00 | 24137.40 | 2024-11-21 09:40:00 | 24229.62 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-11-21 09:30:00 | 24137.40 | 2024-11-21 14:10:00 | 24220.30 | TARGET_HIT | 0.50 | 0.34% |
| BUY | retest1 | 2024-11-22 10:55:00 | 24523.90 | 2024-11-22 11:25:00 | 24611.40 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-11-22 10:55:00 | 24523.90 | 2024-11-22 15:20:00 | 24815.85 | TARGET_HIT | 0.50 | 1.19% |
| BUY | retest1 | 2024-11-29 09:35:00 | 26000.00 | 2024-11-29 09:40:00 | 26166.15 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-11-29 09:35:00 | 26000.00 | 2024-11-29 13:05:00 | 26090.00 | TARGET_HIT | 0.50 | 0.35% |
| BUY | retest1 | 2024-12-02 11:00:00 | 26850.00 | 2024-12-02 11:15:00 | 26746.72 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-12-03 09:40:00 | 27136.15 | 2024-12-03 11:10:00 | 27240.73 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-12-03 09:40:00 | 27136.15 | 2024-12-03 14:10:00 | 27136.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-05 10:55:00 | 27048.00 | 2024-12-05 11:05:00 | 26931.65 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-12-05 10:55:00 | 27048.00 | 2024-12-05 15:20:00 | 26601.70 | TARGET_HIT | 0.50 | 1.65% |
| BUY | retest1 | 2024-12-06 09:40:00 | 26870.00 | 2024-12-06 09:45:00 | 26786.29 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-12-12 10:50:00 | 27349.35 | 2024-12-12 10:55:00 | 27303.95 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2024-12-16 09:30:00 | 27749.00 | 2024-12-16 10:05:00 | 27843.86 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-12-16 09:30:00 | 27749.00 | 2024-12-16 10:40:00 | 27749.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-17 10:30:00 | 28200.30 | 2024-12-17 10:40:00 | 28275.45 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2024-12-17 10:30:00 | 28200.30 | 2024-12-17 11:05:00 | 28200.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-18 11:10:00 | 28062.95 | 2024-12-18 11:25:00 | 28117.03 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-12-20 10:20:00 | 27496.05 | 2024-12-20 10:30:00 | 27562.73 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-01-02 10:05:00 | 25811.25 | 2025-01-02 10:30:00 | 25895.75 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-01-02 10:05:00 | 25811.25 | 2025-01-02 15:20:00 | 26689.50 | TARGET_HIT | 0.50 | 3.40% |
| BUY | retest1 | 2025-01-14 11:00:00 | 25128.45 | 2025-01-14 11:15:00 | 25074.84 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-01-15 10:35:00 | 25320.45 | 2025-01-15 11:00:00 | 25258.38 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-01-22 10:30:00 | 25300.00 | 2025-01-22 11:20:00 | 25413.92 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-01-22 10:30:00 | 25300.00 | 2025-01-22 11:35:00 | 25300.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-28 11:15:00 | 25438.05 | 2025-01-28 11:30:00 | 25554.59 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-01-28 11:15:00 | 25438.05 | 2025-01-28 15:20:00 | 26041.65 | TARGET_HIT | 0.50 | 2.37% |
| BUY | retest1 | 2025-01-29 10:15:00 | 26560.00 | 2025-01-29 11:35:00 | 26723.05 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2025-01-29 10:15:00 | 26560.00 | 2025-01-29 15:10:00 | 26654.00 | TARGET_HIT | 0.50 | 0.35% |
| BUY | retest1 | 2025-02-04 10:55:00 | 27832.25 | 2025-02-04 11:50:00 | 27737.80 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-02-05 09:40:00 | 28150.35 | 2025-02-05 09:50:00 | 28289.01 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-02-05 09:40:00 | 28150.35 | 2025-02-05 09:55:00 | 28150.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-11 09:35:00 | 28303.60 | 2025-02-11 09:45:00 | 28220.22 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-02-13 09:30:00 | 28520.55 | 2025-02-13 09:45:00 | 28420.48 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-02-14 10:40:00 | 28164.70 | 2025-02-14 11:05:00 | 28017.03 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-02-14 10:40:00 | 28164.70 | 2025-02-14 11:35:00 | 28164.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-18 10:55:00 | 28413.90 | 2025-02-18 11:10:00 | 28340.19 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-02-21 11:05:00 | 28265.45 | 2025-02-21 11:25:00 | 28333.16 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-02-25 09:45:00 | 28549.85 | 2025-02-25 09:50:00 | 28472.21 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-03-04 10:20:00 | 27569.80 | 2025-03-04 11:30:00 | 27487.50 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-03-06 11:05:00 | 28506.00 | 2025-03-06 11:25:00 | 28429.25 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-03-10 10:30:00 | 28328.55 | 2025-03-10 10:55:00 | 28251.49 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-03-12 11:15:00 | 27495.90 | 2025-03-12 11:35:00 | 27367.06 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-03-12 11:15:00 | 27495.90 | 2025-03-12 13:50:00 | 27495.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-18 11:10:00 | 27648.30 | 2025-03-18 11:35:00 | 27703.31 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-03-21 09:30:00 | 29050.00 | 2025-03-21 09:35:00 | 28959.74 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-03-25 10:05:00 | 30009.20 | 2025-03-25 10:10:00 | 29902.11 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-03-27 11:00:00 | 30179.95 | 2025-03-27 11:05:00 | 30274.98 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-03-27 11:00:00 | 30179.95 | 2025-03-27 11:20:00 | 30179.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-11 09:40:00 | 30293.95 | 2025-04-11 10:00:00 | 30155.33 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-04-11 09:40:00 | 30293.95 | 2025-04-11 11:30:00 | 30167.05 | TARGET_HIT | 0.50 | 0.42% |
| SELL | retest1 | 2025-04-16 11:10:00 | 30930.00 | 2025-04-16 11:25:00 | 30824.67 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-04-16 11:10:00 | 30930.00 | 2025-04-16 15:20:00 | 30835.00 | TARGET_HIT | 0.50 | 0.31% |
| BUY | retest1 | 2025-04-21 10:35:00 | 31395.00 | 2025-04-21 10:40:00 | 31323.25 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-04-29 10:55:00 | 29935.00 | 2025-04-29 11:05:00 | 29832.62 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-04-29 10:55:00 | 29935.00 | 2025-04-29 11:50:00 | 29935.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-02 11:00:00 | 29550.00 | 2025-05-02 11:35:00 | 29470.05 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-05-02 11:00:00 | 29550.00 | 2025-05-02 14:45:00 | 29330.00 | TARGET_HIT | 0.50 | 0.74% |
| BUY | retest1 | 2025-05-06 09:35:00 | 29785.00 | 2025-05-06 10:35:00 | 29912.12 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-05-06 09:35:00 | 29785.00 | 2025-05-06 10:40:00 | 29785.00 | STOP_HIT | 0.50 | 0.00% |
