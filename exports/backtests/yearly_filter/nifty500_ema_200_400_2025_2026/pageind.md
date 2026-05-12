# Page Industries Ltd. (PAGEIND)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 37365.00
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
| ENTRY2 | 10 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 7
- **Target hits / Stop hits / Partials:** 2 / 8 / 3
- **Avg / median % per leg:** 1.32% / -1.14%
- **Sum % (uncompounded):** 17.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.03% | -16.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.03% | -16.1% |
| SELL (all) | 9 | 6 | 66.7% | 2 | 4 | 3 | 3.70% | 33.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 9 | 6 | 66.7% | 2 | 4 | 3 | 3.70% | 33.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 13 | 6 | 46.2% | 2 | 8 | 3 | 1.32% | 17.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 10:15:00 | 43840.00 | 46509.07 | 46516.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 11:15:00 | 43630.00 | 46480.42 | 46502.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 46215.00 | 45977.12 | 46221.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 46215.00 | 45977.12 | 46221.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 46215.00 | 45977.12 | 46221.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:45:00 | 46405.00 | 45977.12 | 46221.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 46190.00 | 45979.24 | 46220.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:45:00 | 46285.00 | 45979.24 | 46220.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 46190.00 | 45981.34 | 46220.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 12:15:00 | 46015.00 | 45981.34 | 46220.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 15:00:00 | 45955.00 | 45985.31 | 46219.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 10:45:00 | 45950.00 | 45980.33 | 46213.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 09:15:00 | 46540.00 | 45970.21 | 46201.11 | SL hit (close>static) qty=1.00 sl=46285.00 alert=retest2 |

### Cycle 2 — BUY (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 10:15:00 | 38120.00 | 33873.65 | 33859.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 11:15:00 | 38230.00 | 33917.00 | 33881.54 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-24 09:15:00 | 46510.00 | 2025-08-08 09:15:00 | 44110.00 | STOP_HIT | 1.00 | -5.16% |
| BUY | retest2 | 2025-08-05 15:00:00 | 45820.00 | 2025-08-08 09:15:00 | 44110.00 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest2 | 2025-08-07 13:30:00 | 45695.00 | 2025-08-08 09:15:00 | 44110.00 | STOP_HIT | 1.00 | -3.47% |
| BUY | retest2 | 2025-08-07 15:00:00 | 45830.00 | 2025-08-08 09:15:00 | 44110.00 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest2 | 2025-08-21 12:15:00 | 46015.00 | 2025-08-25 09:15:00 | 46540.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-08-21 15:00:00 | 45955.00 | 2025-08-25 09:15:00 | 46540.00 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-08-22 10:45:00 | 45950.00 | 2025-08-25 09:15:00 | 46540.00 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-08-26 11:30:00 | 46005.00 | 2025-08-29 09:15:00 | 43704.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 11:30:00 | 46005.00 | 2025-09-16 09:15:00 | 45100.00 | STOP_HIT | 0.50 | 1.97% |
| SELL | retest2 | 2025-08-26 13:15:00 | 45905.00 | 2025-09-19 12:15:00 | 43609.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 13:45:00 | 45905.00 | 2025-09-19 12:15:00 | 43609.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 13:15:00 | 45905.00 | 2025-09-26 14:15:00 | 41314.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-26 13:45:00 | 45905.00 | 2025-09-26 14:15:00 | 41314.50 | TARGET_HIT | 0.50 | 10.00% |
