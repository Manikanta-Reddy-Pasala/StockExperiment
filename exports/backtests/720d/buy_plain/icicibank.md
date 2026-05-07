# ICICIBANK (ICICIBANK)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1279.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 3 |
| ALERT3 | 3 |
| PENDING | 7 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 0 |
| ENTRY2 | 5 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 0 / 5 / 2
- **Avg / median % per leg:** 7.43% / 12.03%
- **Sum % (uncompounded):** 52.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 0 | 5 | 2 | 7.43% | 52.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 4 | 57.1% | 0 | 5 | 2 | 7.43% | 52.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 7 | 4 | 57.1% | 0 | 5 | 2 | 7.43% | 52.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-03-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 11:15:00 | 1341.20 | 1254.85 | 1254.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 09:15:00 | 1358.20 | 1259.18 | 1256.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1288.00 | 1295.39 | 1278.52 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 10:15:00 | 1276.05 | 1295.20 | 1278.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 10:15:00 | 1276.05 | 1295.20 | 1278.51 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-04-07 15:15:00 | 1294.10 | 1294.50 | 1278.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 09:15:00 | 1295.85 | 1294.51 | 1278.65 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2025-04-09 11:15:00 | 1291.20 | 1294.67 | 1279.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 12:15:00 | 1296.75 | 1294.69 | 1279.52 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-07-25 09:15:00 | 1490.23 | 1439.79 | 1423.70 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-07-25 09:15:00 | 1491.26 | 1439.79 | 1423.70 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 09:15:00 | 1452.70 | 1454.61 | 1435.59 | SL hit (close<ema200) qty=0.50 sl=1454.61 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 09:15:00 | 1452.70 | 1454.61 | 1435.59 | SL hit (close<ema200) qty=0.50 sl=1454.61 alert=retest2 |

### Cycle 2 — BUY (started 2026-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 13:15:00 | 1416.60 | 1377.36 | 1377.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 09:15:00 | 1422.70 | 1378.55 | 1377.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 1362.10 | 1386.14 | 1381.98 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 09:15:00 | 1362.10 | 1386.14 | 1381.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1362.10 | 1386.14 | 1381.98 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-02-04 09:15:00 | 1402.90 | 1372.79 | 1375.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 10:15:00 | 1406.20 | 1373.12 | 1375.37 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-06 09:15:00 | 1403.50 | 1376.96 | 1377.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-06 10:15:00 | 1398.50 | 1377.18 | 1377.33 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2026-02-06 12:15:00 | 1401.90 | 1377.60 | 1377.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 12:15:00 | 1401.90 | 1377.60 | 1377.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 13:15:00 | 1403.70 | 1377.86 | 1377.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 1386.80 | 1391.26 | 1385.53 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 15:15:00 | 1388.70 | 1391.24 | 1385.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 1388.70 | 1391.24 | 1385.54 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-02-20 09:15:00 | 1398.10 | 1391.31 | 1385.61 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:15:00 | 1398.70 | 1391.38 | 1385.67 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-24 14:15:00 | 1384.70 | 1392.13 | 1386.56 | SL hit (close<static) qty=1.00 sl=1384.80 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-25 09:15:00 | 1399.30 | 1392.14 | 1386.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 10:15:00 | 1394.90 | 1392.16 | 1386.66 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-27 12:15:00 | 1383.70 | 1393.09 | 1387.57 | SL hit (close<static) qty=1.00 sl=1384.80 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-21 14:15:00 | 1389.70 | 1308.51 | 1323.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-21 15:15:00 | 1385.60 | 1309.28 | 1323.57 | ENTRY2 sustain failed after 60m |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-04-08 09:15:00 | 1295.85 | 2025-07-25 09:15:00 | 1490.23 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-04-09 12:15:00 | 1296.75 | 2025-07-25 09:15:00 | 1491.26 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-04-08 09:15:00 | 1295.85 | 2025-08-05 09:15:00 | 1452.70 | STOP_HIT | 0.50 | 12.10% |
| BUY | retest2 | 2025-04-09 12:15:00 | 1296.75 | 2025-08-05 09:15:00 | 1452.70 | STOP_HIT | 0.50 | 12.03% |
| BUY | retest2 | 2026-02-04 10:15:00 | 1406.20 | 2026-02-06 12:15:00 | 1401.90 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2026-02-20 10:15:00 | 1398.70 | 2026-02-24 14:15:00 | 1384.70 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2026-02-25 10:15:00 | 1394.90 | 2026-02-27 12:15:00 | 1383.70 | STOP_HIT | 1.00 | -0.80% |
