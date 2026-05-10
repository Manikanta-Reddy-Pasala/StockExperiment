# APOLLOHOSP (APOLLOHOSP)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 8100.00
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
| ENTRY1 | 18 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 4 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 14
- **Target hits / Stop hits / Partials:** 4 / 14 / 7
- **Avg / median % per leg:** 0.09% / 0.00%
- **Sum % (uncompounded):** 2.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 9 | 60.0% | 4 | 6 | 5 | 0.20% | 3.0% |
| BUY @ 2nd Alert (retest1) | 15 | 9 | 60.0% | 4 | 6 | 5 | 0.20% | 3.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 2 | 20.0% | 0 | 8 | 2 | -0.09% | -0.9% |
| SELL @ 2nd Alert (retest1) | 10 | 2 | 20.0% | 0 | 8 | 2 | -0.09% | -0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 25 | 11 | 44.0% | 4 | 14 | 7 | 0.09% | 2.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-09 10:55:00 | 7173.50 | 7198.75 | 0.00 | ORB-short ORB[7184.00,7250.00] vol=3.0x ATR=26.89 |
| Stop hit — per-position SL triggered | 2026-02-09 12:10:00 | 7200.39 | 7194.81 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 09:40:00 | 7208.50 | 7245.05 | 0.00 | ORB-short ORB[7218.00,7272.00] vol=1.6x ATR=23.31 |
| Stop hit — per-position SL triggered | 2026-02-10 11:00:00 | 7231.81 | 7225.57 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 09:45:00 | 7557.50 | 7530.69 | 0.00 | ORB-long ORB[7485.00,7537.00] vol=2.1x ATR=18.31 |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 7539.19 | 7545.01 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:10:00 | 7544.50 | 7520.86 | 0.00 | ORB-long ORB[7478.00,7532.50] vol=1.8x ATR=18.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 12:15:00 | 7572.55 | 7541.26 | 0.00 | T1 1.5R @ 7572.55 |
| Target hit | 2026-02-16 15:20:00 | 7606.00 | 7575.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2026-02-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:35:00 | 7651.50 | 7637.37 | 0.00 | ORB-long ORB[7590.00,7638.50] vol=8.0x ATR=16.09 |
| Stop hit — per-position SL triggered | 2026-02-18 09:55:00 | 7635.41 | 7639.80 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:00:00 | 7670.50 | 7718.67 | 0.00 | ORB-short ORB[7680.00,7746.00] vol=1.7x ATR=19.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:55:00 | 7641.98 | 7706.74 | 0.00 | T1 1.5R @ 7641.98 |
| Stop hit — per-position SL triggered | 2026-03-05 12:25:00 | 7670.50 | 7695.11 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 7723.00 | 7747.97 | 0.00 | ORB-short ORB[7732.50,7781.00] vol=1.9x ATR=15.85 |
| Stop hit — per-position SL triggered | 2026-03-06 11:00:00 | 7738.85 | 7746.29 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 11:15:00 | 7785.00 | 7816.81 | 0.00 | ORB-short ORB[7801.50,7870.00] vol=4.5x ATR=15.85 |
| Stop hit — per-position SL triggered | 2026-03-10 11:20:00 | 7800.85 | 7816.42 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 11:10:00 | 7712.50 | 7740.97 | 0.00 | ORB-short ORB[7732.00,7798.50] vol=3.4x ATR=14.94 |
| Stop hit — per-position SL triggered | 2026-03-11 12:10:00 | 7727.44 | 7735.14 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 11:15:00 | 7558.50 | 7535.50 | 0.00 | ORB-long ORB[7437.50,7525.00] vol=2.6x ATR=16.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 12:00:00 | 7583.14 | 7544.20 | 0.00 | T1 1.5R @ 7583.14 |
| Target hit | 2026-03-25 15:20:00 | 7586.00 | 7561.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2026-04-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 11:05:00 | 7554.50 | 7517.68 | 0.00 | ORB-long ORB[7461.50,7515.00] vol=2.9x ATR=14.74 |
| Stop hit — per-position SL triggered | 2026-04-10 11:30:00 | 7539.76 | 7528.41 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:45:00 | 7497.00 | 7467.19 | 0.00 | ORB-long ORB[7403.50,7485.00] vol=2.0x ATR=17.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 10:50:00 | 7523.93 | 7472.80 | 0.00 | T1 1.5R @ 7523.93 |
| Target hit | 2026-04-13 15:20:00 | 7510.50 | 7506.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2026-04-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:05:00 | 7627.00 | 7580.70 | 0.00 | ORB-long ORB[7515.00,7591.00] vol=2.8x ATR=20.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 11:35:00 | 7657.68 | 7595.71 | 0.00 | T1 1.5R @ 7657.68 |
| Target hit | 2026-04-17 15:20:00 | 7703.50 | 7655.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2026-04-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:50:00 | 7715.00 | 7692.23 | 0.00 | ORB-long ORB[7667.50,7710.00] vol=1.6x ATR=14.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:05:00 | 7736.09 | 7705.33 | 0.00 | T1 1.5R @ 7736.09 |
| Stop hit — per-position SL triggered | 2026-04-21 11:35:00 | 7715.00 | 7717.18 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:30:00 | 7757.00 | 7734.54 | 0.00 | ORB-long ORB[7697.00,7752.00] vol=2.8x ATR=15.23 |
| Stop hit — per-position SL triggered | 2026-04-22 10:35:00 | 7741.77 | 7735.28 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:15:00 | 7718.00 | 7765.45 | 0.00 | ORB-short ORB[7751.00,7834.50] vol=7.8x ATR=18.30 |
| Stop hit — per-position SL triggered | 2026-04-24 11:20:00 | 7736.30 | 7765.03 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 11:15:00 | 7792.50 | 7772.87 | 0.00 | ORB-long ORB[7735.00,7786.50] vol=2.1x ATR=16.77 |
| Stop hit — per-position SL triggered | 2026-04-27 11:30:00 | 7775.73 | 7773.82 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-04-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 10:55:00 | 7771.50 | 7807.69 | 0.00 | ORB-short ORB[7794.00,7878.50] vol=3.3x ATR=14.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:55:00 | 7749.75 | 7797.52 | 0.00 | T1 1.5R @ 7749.75 |
| Stop hit — per-position SL triggered | 2026-04-28 14:30:00 | 7771.50 | 7771.30 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-09 10:55:00 | 7173.50 | 2026-02-09 12:10:00 | 7200.39 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-10 09:40:00 | 7208.50 | 2026-02-10 11:00:00 | 7231.81 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-13 09:45:00 | 7557.50 | 2026-02-13 10:15:00 | 7539.19 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-16 10:10:00 | 7544.50 | 2026-02-16 12:15:00 | 7572.55 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-02-16 10:10:00 | 7544.50 | 2026-02-16 15:20:00 | 7606.00 | TARGET_HIT | 0.50 | 0.82% |
| BUY | retest1 | 2026-02-18 09:35:00 | 7651.50 | 2026-02-18 09:55:00 | 7635.41 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-03-05 11:00:00 | 7670.50 | 2026-03-05 11:55:00 | 7641.98 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-03-05 11:00:00 | 7670.50 | 2026-03-05 12:25:00 | 7670.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-06 10:45:00 | 7723.00 | 2026-03-06 11:00:00 | 7738.85 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-03-10 11:15:00 | 7785.00 | 2026-03-10 11:20:00 | 7800.85 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-03-11 11:10:00 | 7712.50 | 2026-03-11 12:10:00 | 7727.44 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-03-25 11:15:00 | 7558.50 | 2026-03-25 12:00:00 | 7583.14 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-03-25 11:15:00 | 7558.50 | 2026-03-25 15:20:00 | 7586.00 | TARGET_HIT | 0.50 | 0.36% |
| BUY | retest1 | 2026-04-10 11:05:00 | 7554.50 | 2026-04-10 11:30:00 | 7539.76 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-04-13 10:45:00 | 7497.00 | 2026-04-13 10:50:00 | 7523.93 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-04-13 10:45:00 | 7497.00 | 2026-04-13 15:20:00 | 7510.50 | TARGET_HIT | 0.50 | 0.18% |
| BUY | retest1 | 2026-04-17 10:05:00 | 7627.00 | 2026-04-17 11:35:00 | 7657.68 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-04-17 10:05:00 | 7627.00 | 2026-04-17 15:20:00 | 7703.50 | TARGET_HIT | 0.50 | 1.00% |
| BUY | retest1 | 2026-04-21 09:50:00 | 7715.00 | 2026-04-21 10:05:00 | 7736.09 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2026-04-21 09:50:00 | 7715.00 | 2026-04-21 11:35:00 | 7715.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 10:30:00 | 7757.00 | 2026-04-22 10:35:00 | 7741.77 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-04-24 11:15:00 | 7718.00 | 2026-04-24 11:20:00 | 7736.30 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-04-27 11:15:00 | 7792.50 | 2026-04-27 11:30:00 | 7775.73 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-04-28 10:55:00 | 7771.50 | 2026-04-28 11:55:00 | 7749.75 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2026-04-28 10:55:00 | 7771.50 | 2026-04-28 14:30:00 | 7771.50 | STOP_HIT | 0.50 | 0.00% |
