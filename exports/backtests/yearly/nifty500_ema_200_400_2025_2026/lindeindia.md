# Linde India Ltd. (LINDEINDIA)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 7765.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 8 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 13 |
| PARTIAL | 9 |
| TARGET_HIT | 9 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 4
- **Target hits / Stop hits / Partials:** 9 / 4 / 9
- **Avg / median % per leg:** 5.81% / 5.00%
- **Sum % (uncompounded):** 127.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 22 | 18 | 81.8% | 9 | 4 | 9 | 5.81% | 127.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 22 | 18 | 81.8% | 9 | 4 | 9 | 5.81% | 127.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 22 | 18 | 81.8% | 9 | 4 | 9 | 5.81% | 127.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 6434.00 | 6673.30 | 6673.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 6410.50 | 6659.83 | 6666.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 11:15:00 | 6559.00 | 6466.96 | 6546.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 11:15:00 | 6559.00 | 6466.96 | 6546.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 6559.00 | 6466.96 | 6546.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 12:00:00 | 6559.00 | 6466.96 | 6546.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 6525.00 | 6467.53 | 6546.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 13:45:00 | 6490.50 | 6467.98 | 6546.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 14:45:00 | 6474.00 | 6467.33 | 6545.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:30:00 | 6480.00 | 6443.33 | 6511.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:00:00 | 6491.50 | 6446.72 | 6508.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 6493.00 | 6446.75 | 6506.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 13:30:00 | 6452.00 | 6447.35 | 6505.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 14:00:00 | 6451.00 | 6447.35 | 6505.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 10:30:00 | 6436.00 | 6447.22 | 6504.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 11:15:00 | 6441.50 | 6447.22 | 6504.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 6528.00 | 6447.87 | 6504.34 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-15 12:15:00 | 6528.00 | 6447.87 | 6504.34 | SL hit (close>static) qty=1.00 sl=6510.50 alert=retest2 |

### Cycle 2 — BUY (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 12:15:00 | 6606.00 | 6004.07 | 6002.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 13:15:00 | 6909.00 | 6105.42 | 6055.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 10:15:00 | 6597.50 | 6769.09 | 6532.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-23 11:00:00 | 6597.50 | 6769.09 | 6532.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-08-26 13:45:00 | 6490.50 | 2025-09-15 12:15:00 | 6528.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-08-26 14:45:00 | 6474.00 | 2025-09-15 12:15:00 | 6528.00 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-09-09 11:30:00 | 6480.00 | 2025-09-15 12:15:00 | 6528.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-09-11 11:00:00 | 6491.50 | 2025-09-15 12:15:00 | 6528.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-09-12 13:30:00 | 6452.00 | 2025-10-06 10:15:00 | 6165.97 | PARTIAL | 0.50 | 4.43% |
| SELL | retest2 | 2025-09-12 14:00:00 | 6451.00 | 2025-10-06 10:15:00 | 6150.30 | PARTIAL | 0.50 | 4.66% |
| SELL | retest2 | 2025-09-15 10:30:00 | 6436.00 | 2025-10-06 10:15:00 | 6156.00 | PARTIAL | 0.50 | 4.35% |
| SELL | retest2 | 2025-09-15 11:15:00 | 6441.50 | 2025-10-06 10:15:00 | 6166.92 | PARTIAL | 0.50 | 4.26% |
| SELL | retest2 | 2025-09-17 09:45:00 | 6484.50 | 2025-10-06 10:15:00 | 6160.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 10:30:00 | 6491.00 | 2025-10-06 10:15:00 | 6166.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 12:30:00 | 6495.50 | 2025-10-06 10:15:00 | 6170.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 13:45:00 | 6487.00 | 2025-10-06 10:15:00 | 6162.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 10:45:00 | 6400.00 | 2025-10-24 09:15:00 | 6080.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-12 13:30:00 | 6452.00 | 2025-11-06 14:15:00 | 5841.45 | TARGET_HIT | 0.50 | 9.46% |
| SELL | retest2 | 2025-09-12 14:00:00 | 6451.00 | 2025-11-06 14:15:00 | 5826.60 | TARGET_HIT | 0.50 | 9.68% |
| SELL | retest2 | 2025-09-15 10:30:00 | 6436.00 | 2025-11-06 14:15:00 | 5832.00 | TARGET_HIT | 0.50 | 9.38% |
| SELL | retest2 | 2025-09-15 11:15:00 | 6441.50 | 2025-11-06 14:15:00 | 5842.35 | TARGET_HIT | 0.50 | 9.30% |
| SELL | retest2 | 2025-09-17 09:45:00 | 6484.50 | 2025-11-06 14:15:00 | 5836.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-17 10:30:00 | 6491.00 | 2025-11-06 14:15:00 | 5841.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-17 12:30:00 | 6495.50 | 2025-11-06 14:15:00 | 5845.95 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-23 13:45:00 | 6487.00 | 2025-11-06 14:15:00 | 5838.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-24 10:45:00 | 6400.00 | 2025-11-11 10:15:00 | 5760.00 | TARGET_HIT | 0.50 | 10.00% |
