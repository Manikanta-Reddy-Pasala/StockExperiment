# LG Electronics India Ltd. (LGEINDIA)

## Backtest Summary

- **Window:** 2025-10-14 09:15:00 → 2026-05-08 15:15:00 (968 bars)
- **Last close:** 1508.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 7 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 6 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / Stop hits / Partials:** 1 / 5 / 1
- **Avg / median % per leg:** -0.21% / -3.17%
- **Sum % (uncompounded):** -1.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.08% | -6.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.08% | -6.2% |
| SELL (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.93% | 4.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.93% | 4.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 7 | 2 | 28.6% | 1 | 5 | 1 | -0.21% | -1.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 09:15:00 | 1556.60 | 1531.60 | 1531.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 10:15:00 | 1579.30 | 1532.07 | 1531.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 12:15:00 | 1538.60 | 1540.08 | 1536.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-13 13:00:00 | 1538.60 | 1540.08 | 1536.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 1529.80 | 1540.97 | 1536.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 09:15:00 | 1547.10 | 1540.97 | 1536.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:30:00 | 1544.20 | 1545.53 | 1539.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 1498.00 | 1545.78 | 1539.98 | SL hit (close<static) qty=1.00 sl=1520.10 alert=retest2 |

### Cycle 2 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 1500.70 | 1534.67 | 1534.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 13:15:00 | 1495.00 | 1534.27 | 1534.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 14:15:00 | 1538.80 | 1534.32 | 1534.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 14:15:00 | 1538.80 | 1534.32 | 1534.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 1538.80 | 1534.32 | 1534.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 1538.80 | 1534.32 | 1534.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 1512.00 | 1534.09 | 1534.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 09:15:00 | 1457.90 | 1534.09 | 1534.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-01 14:15:00 | 1385.01 | 1522.49 | 1528.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-04-02 12:15:00 | 1312.11 | 1513.61 | 1523.80 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 1592.10 | 1519.42 | 1519.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 10:15:00 | 1624.30 | 1520.46 | 1519.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 11:15:00 | 1544.20 | 1546.91 | 1535.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 12:00:00 | 1544.20 | 1546.91 | 1535.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1528.00 | 1546.55 | 1535.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 1529.00 | 1546.55 | 1535.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 1525.10 | 1546.33 | 1535.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:00:00 | 1525.10 | 1546.33 | 1535.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2026-03-17 09:15:00 | 1547.10 | 2026-03-23 09:15:00 | 1498.00 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2026-03-20 09:30:00 | 1544.20 | 2026-03-23 09:15:00 | 1498.00 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2026-03-30 09:15:00 | 1457.90 | 2026-04-01 14:15:00 | 1385.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-30 09:15:00 | 1457.90 | 2026-04-02 12:15:00 | 1312.11 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-13 12:45:00 | 1499.50 | 2026-04-16 14:15:00 | 1551.00 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2026-04-13 14:15:00 | 1498.50 | 2026-04-16 14:15:00 | 1551.00 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2026-04-13 14:45:00 | 1500.00 | 2026-04-16 14:15:00 | 1551.00 | STOP_HIT | 1.00 | -3.40% |
