# DIVISLAB (DIVISLAB)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 6705.00
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
| PARTIAL | 8 |
| TARGET_HIT | 3 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 15
- **Target hits / Stop hits / Partials:** 3 / 15 / 8
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 2.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 9 | 45.0% | 3 | 11 | 6 | 0.09% | 1.8% |
| BUY @ 2nd Alert (retest1) | 20 | 9 | 45.0% | 3 | 11 | 6 | 0.09% | 1.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.06% | 0.3% |
| SELL @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 4 | 2 | 0.06% | 0.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 26 | 11 | 42.3% | 3 | 15 | 8 | 0.08% | 2.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:55:00 | 6102.50 | 6069.75 | 0.00 | ORB-long ORB[6044.00,6085.00] vol=3.2x ATR=19.00 |
| Stop hit — per-position SL triggered | 2026-02-09 11:45:00 | 6083.50 | 6078.83 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:35:00 | 6217.50 | 6188.27 | 0.00 | ORB-long ORB[6151.00,6209.50] vol=1.7x ATR=16.05 |
| Stop hit — per-position SL triggered | 2026-02-10 09:45:00 | 6201.45 | 6192.60 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:25:00 | 6211.50 | 6189.52 | 0.00 | ORB-long ORB[6126.00,6206.00] vol=1.7x ATR=16.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 10:30:00 | 6236.92 | 6193.89 | 0.00 | T1 1.5R @ 6236.92 |
| Stop hit — per-position SL triggered | 2026-02-11 10:40:00 | 6211.50 | 6197.41 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:55:00 | 6152.00 | 6134.27 | 0.00 | ORB-long ORB[6102.50,6128.50] vol=2.5x ATR=14.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:20:00 | 6174.13 | 6142.04 | 0.00 | T1 1.5R @ 6174.13 |
| Target hit | 2026-02-17 14:10:00 | 6181.50 | 6189.81 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — BUY (started 2026-02-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:45:00 | 6224.50 | 6209.48 | 0.00 | ORB-long ORB[6165.00,6223.00] vol=3.0x ATR=15.33 |
| Stop hit — per-position SL triggered | 2026-02-18 09:55:00 | 6209.17 | 6212.98 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 6333.00 | 6310.43 | 0.00 | ORB-long ORB[6251.00,6327.50] vol=1.6x ATR=15.79 |
| Stop hit — per-position SL triggered | 2026-02-19 09:40:00 | 6317.21 | 6313.03 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:30:00 | 6308.00 | 6277.06 | 0.00 | ORB-long ORB[6250.00,6297.50] vol=1.6x ATR=13.08 |
| Stop hit — per-position SL triggered | 2026-02-25 09:35:00 | 6294.92 | 6279.48 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-02-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:35:00 | 6492.50 | 6458.28 | 0.00 | ORB-long ORB[6420.00,6468.00] vol=1.6x ATR=20.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 09:45:00 | 6522.98 | 6476.72 | 0.00 | T1 1.5R @ 6522.98 |
| Target hit | 2026-02-26 10:30:00 | 6504.00 | 6512.07 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2026-03-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 10:45:00 | 6149.00 | 6129.46 | 0.00 | ORB-long ORB[6086.50,6144.00] vol=2.8x ATR=15.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 13:15:00 | 6172.43 | 6140.10 | 0.00 | T1 1.5R @ 6172.43 |
| Stop hit — per-position SL triggered | 2026-03-18 14:35:00 | 6149.00 | 6144.85 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:35:00 | 5987.50 | 6028.94 | 0.00 | ORB-short ORB[6048.00,6092.00] vol=3.3x ATR=19.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-24 10:55:00 | 5958.45 | 6020.62 | 0.00 | T1 1.5R @ 5958.45 |
| Stop hit — per-position SL triggered | 2026-03-24 11:30:00 | 5987.50 | 6013.35 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 10:55:00 | 6023.50 | 5987.61 | 0.00 | ORB-long ORB[5935.00,6017.00] vol=2.8x ATR=16.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 11:05:00 | 6047.71 | 5991.11 | 0.00 | T1 1.5R @ 6047.71 |
| Stop hit — per-position SL triggered | 2026-03-27 11:15:00 | 6023.50 | 5993.94 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:45:00 | 5999.50 | 5976.39 | 0.00 | ORB-long ORB[5931.00,5986.00] vol=1.7x ATR=15.94 |
| Stop hit — per-position SL triggered | 2026-04-10 10:05:00 | 5983.56 | 5980.61 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 10:10:00 | 6265.50 | 6274.82 | 0.00 | ORB-short ORB[6266.00,6305.00] vol=2.1x ATR=14.08 |
| Stop hit — per-position SL triggered | 2026-04-21 10:55:00 | 6279.58 | 6272.20 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:30:00 | 6336.50 | 6311.85 | 0.00 | ORB-long ORB[6250.50,6334.50] vol=1.9x ATR=16.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 09:40:00 | 6361.60 | 6347.89 | 0.00 | T1 1.5R @ 6361.60 |
| Target hit | 2026-04-23 10:40:00 | 6381.50 | 6405.78 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — SELL (started 2026-04-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:10:00 | 6371.00 | 6397.31 | 0.00 | ORB-short ORB[6380.50,6442.50] vol=1.7x ATR=15.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 12:25:00 | 6347.05 | 6389.25 | 0.00 | T1 1.5R @ 6347.05 |
| Stop hit — per-position SL triggered | 2026-04-24 12:35:00 | 6371.00 | 6388.48 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:15:00 | 6598.00 | 6569.70 | 0.00 | ORB-long ORB[6483.50,6557.00] vol=4.1x ATR=18.11 |
| Stop hit — per-position SL triggered | 2026-05-04 11:30:00 | 6579.89 | 6570.43 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-05-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:40:00 | 6644.50 | 6628.82 | 0.00 | ORB-long ORB[6601.50,6635.00] vol=1.6x ATR=14.70 |
| Stop hit — per-position SL triggered | 2026-05-05 09:50:00 | 6629.80 | 6630.37 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-05-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 09:45:00 | 6696.00 | 6721.63 | 0.00 | ORB-short ORB[6703.00,6760.00] vol=2.4x ATR=19.80 |
| Stop hit — per-position SL triggered | 2026-05-07 10:05:00 | 6715.80 | 6719.15 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:55:00 | 6102.50 | 2026-02-09 11:45:00 | 6083.50 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-10 09:35:00 | 6217.50 | 2026-02-10 09:45:00 | 6201.45 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-11 10:25:00 | 6211.50 | 2026-02-11 10:30:00 | 6236.92 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-02-11 10:25:00 | 6211.50 | 2026-02-11 10:40:00 | 6211.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 09:55:00 | 6152.00 | 2026-02-17 10:20:00 | 6174.13 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-02-17 09:55:00 | 6152.00 | 2026-02-17 14:10:00 | 6181.50 | TARGET_HIT | 0.50 | 0.48% |
| BUY | retest1 | 2026-02-18 09:45:00 | 6224.50 | 2026-02-18 09:55:00 | 6209.17 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-19 09:30:00 | 6333.00 | 2026-02-19 09:40:00 | 6317.21 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-25 09:30:00 | 6308.00 | 2026-02-25 09:35:00 | 6294.92 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-02-26 09:35:00 | 6492.50 | 2026-02-26 09:45:00 | 6522.98 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-02-26 09:35:00 | 6492.50 | 2026-02-26 10:30:00 | 6504.00 | TARGET_HIT | 0.50 | 0.18% |
| BUY | retest1 | 2026-03-18 10:45:00 | 6149.00 | 2026-03-18 13:15:00 | 6172.43 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-03-18 10:45:00 | 6149.00 | 2026-03-18 14:35:00 | 6149.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-24 10:35:00 | 5987.50 | 2026-03-24 10:55:00 | 5958.45 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-03-24 10:35:00 | 5987.50 | 2026-03-24 11:30:00 | 5987.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-27 10:55:00 | 6023.50 | 2026-03-27 11:05:00 | 6047.71 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-03-27 10:55:00 | 6023.50 | 2026-03-27 11:15:00 | 6023.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-10 09:45:00 | 5999.50 | 2026-04-10 10:05:00 | 5983.56 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-04-21 10:10:00 | 6265.50 | 2026-04-21 10:55:00 | 6279.58 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-04-23 09:30:00 | 6336.50 | 2026-04-23 09:40:00 | 6361.60 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-04-23 09:30:00 | 6336.50 | 2026-04-23 10:40:00 | 6381.50 | TARGET_HIT | 0.50 | 0.71% |
| SELL | retest1 | 2026-04-24 11:10:00 | 6371.00 | 2026-04-24 12:25:00 | 6347.05 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-04-24 11:10:00 | 6371.00 | 2026-04-24 12:35:00 | 6371.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-04 11:15:00 | 6598.00 | 2026-05-04 11:30:00 | 6579.89 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-05-05 09:40:00 | 6644.50 | 2026-05-05 09:50:00 | 6629.80 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-05-07 09:45:00 | 6696.00 | 2026-05-07 10:05:00 | 6715.80 | STOP_HIT | 1.00 | -0.30% |
