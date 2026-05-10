# Jindal Saw Ltd. (JINDALSAW)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 243.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 10 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 8 |
| PARTIAL | 0 |
| TARGET_HIT | 3 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / Stop hits / Partials:** 3 / 5 / 0
- **Avg / median % per leg:** 2.80% / -1.06%
- **Sum % (uncompounded):** 22.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 3 | 37.5% | 3 | 5 | 0 | 2.80% | 22.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 3 | 37.5% | 3 | 5 | 0 | 2.80% | 22.4% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 8 | 3 | 37.5% | 3 | 5 | 0 | 2.80% | 22.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 13:15:00 | 189.53 | 174.46 | 174.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 191.20 | 177.11 | 175.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 180.15 | 182.13 | 178.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 12:15:00 | 180.56 | 182.08 | 179.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 180.56 | 182.08 | 179.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 13:15:00 | 180.74 | 182.08 | 179.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 14:15:00 | 178.26 | 182.02 | 179.02 | SL hit (close<static) qty=1.00 sl=179.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:45:00 | 180.86 | 181.93 | 179.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 11:30:00 | 181.13 | 181.92 | 179.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 12:15:00 | 181.45 | 181.92 | 179.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 178.95 | 181.83 | 179.07 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-23 10:15:00 | 178.95 | 181.83 | 179.07 | SL hit (close<static) qty=1.00 sl=179.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 10:15:00 | 178.95 | 181.83 | 179.07 | SL hit (close<static) qty=1.00 sl=179.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 10:15:00 | 178.95 | 181.83 | 179.07 | SL hit (close<static) qty=1.00 sl=179.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 178.95 | 181.83 | 179.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 178.98 | 181.81 | 179.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:45:00 | 178.49 | 181.81 | 179.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 179.37 | 181.78 | 179.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:30:00 | 178.91 | 181.78 | 179.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 179.12 | 181.74 | 179.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:30:00 | 178.61 | 181.74 | 179.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 180.00 | 181.72 | 179.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:15:00 | 177.51 | 181.72 | 179.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 177.17 | 181.67 | 179.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 176.71 | 181.67 | 179.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 181.70 | 182.25 | 179.70 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 13:15:00 | 165.30 | 177.69 | 177.69 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 11:15:00 | 188.06 | 177.77 | 177.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 13:15:00 | 190.20 | 178.01 | 177.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 183.78 | 184.87 | 181.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-23 10:00:00 | 183.78 | 184.87 | 181.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 182.50 | 184.85 | 181.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 09:15:00 | 184.95 | 184.57 | 181.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 180.17 | 184.52 | 181.71 | SL hit (close<static) qty=1.00 sl=181.62 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 09:15:00 | 185.96 | 184.37 | 181.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 12:15:00 | 184.65 | 184.58 | 181.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 13:00:00 | 185.43 | 184.59 | 181.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 189.28 | 184.71 | 182.09 | EMA400 retest candle locked (from upside) |
| Target hit | 2026-04-09 09:15:00 | 204.56 | 187.50 | 184.08 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-09 09:15:00 | 203.12 | 187.50 | 184.08 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-09 09:15:00 | 203.97 | 187.50 | 184.08 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2026-02-19 13:15:00 | 180.74 | 2026-02-19 14:15:00 | 178.26 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-02-20 10:45:00 | 180.86 | 2026-02-23 10:15:00 | 178.95 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2026-02-20 11:30:00 | 181.13 | 2026-02-23 10:15:00 | 178.95 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-02-20 12:15:00 | 181.45 | 2026-02-23 10:15:00 | 178.95 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-03-24 09:15:00 | 184.95 | 2026-03-24 09:15:00 | 180.17 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2026-03-25 09:15:00 | 185.96 | 2026-04-09 09:15:00 | 204.56 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-27 12:15:00 | 184.65 | 2026-04-09 09:15:00 | 203.12 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-27 13:00:00 | 185.43 | 2026-04-09 09:15:00 | 203.97 | TARGET_HIT | 1.00 | 10.00% |
