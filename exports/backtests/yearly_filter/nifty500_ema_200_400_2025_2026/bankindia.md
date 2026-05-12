# Bank of India (BANKINDIA)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 139.85
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
| ALERT2_SKIP | 0 |
| ALERT3 | 20 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 18 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 2 / 17
- **Target hits / Stop hits / Partials:** 1 / 17 / 1
- **Avg / median % per leg:** -1.26% / -1.35%
- **Sum % (uncompounded):** -23.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 1 | 10.0% | 1 | 9 | 0 | -1.12% | -11.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 1 | 10.0% | 1 | 9 | 0 | -1.12% | -11.2% |
| SELL (all) | 9 | 1 | 11.1% | 0 | 8 | 1 | -1.41% | -12.7% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.48% | -3.5% |
| SELL @ 3rd Alert (retest2) | 8 | 1 | 12.5% | 0 | 7 | 1 | -1.15% | -9.2% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.48% | -3.5% |
| retest2 (combined) | 18 | 2 | 11.1% | 1 | 16 | 1 | -1.13% | -20.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 13:15:00 | 112.08 | 115.70 | 115.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 111.60 | 115.55 | 115.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 11:15:00 | 113.88 | 113.75 | 114.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-12 12:00:00 | 113.88 | 113.75 | 114.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 114.25 | 113.75 | 114.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 11:15:00 | 113.40 | 113.75 | 114.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 14:15:00 | 113.45 | 113.74 | 114.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 15:00:00 | 113.50 | 113.73 | 114.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 09:15:00 | 113.39 | 113.73 | 114.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 114.75 | 113.71 | 114.48 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 114.75 | 113.71 | 114.48 | SL hit (close>static) qty=1.00 sl=114.63 alert=retest2 |

### Cycle 2 — BUY (started 2025-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 11:15:00 | 117.40 | 114.59 | 114.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 10:15:00 | 118.32 | 114.76 | 114.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 14:15:00 | 116.43 | 116.98 | 115.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-26 15:00:00 | 116.43 | 116.98 | 115.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 139.88 | 141.41 | 138.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:30:00 | 138.65 | 141.41 | 138.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 151.81 | 153.94 | 147.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 156.65 | 153.73 | 147.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-18 09:15:00 | 172.32 | 159.72 | 153.30 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 144.97 | 155.47 | 155.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 141.73 | 155.33 | 155.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-16 09:15:00 | 150.11 | 149.42 | 151.86 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 10:45:00 | 147.59 | 149.38 | 151.74 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 152.72 | 149.42 | 151.52 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-22 10:15:00 | 152.72 | 149.42 | 151.52 | SL hit (close>ema400) qty=1.00 sl=151.52 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-26 12:00:00 | 116.61 | 2025-07-09 14:15:00 | 116.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-06-26 14:30:00 | 116.89 | 2025-07-10 09:15:00 | 115.31 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-07-08 11:15:00 | 116.65 | 2025-07-10 09:15:00 | 115.31 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-07-08 12:15:00 | 116.58 | 2025-07-10 09:15:00 | 115.31 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-07-08 14:45:00 | 116.98 | 2025-07-10 09:15:00 | 115.31 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-07-16 13:00:00 | 117.74 | 2025-07-17 15:15:00 | 115.92 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-08-13 11:15:00 | 113.40 | 2025-08-18 09:15:00 | 114.75 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-08-13 14:15:00 | 113.45 | 2025-08-18 09:15:00 | 114.75 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-08-13 15:00:00 | 113.50 | 2025-08-18 09:15:00 | 114.75 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-08-14 09:15:00 | 113.39 | 2025-08-18 09:15:00 | 114.75 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-09-04 12:15:00 | 113.00 | 2025-09-10 10:15:00 | 116.55 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2025-09-08 10:45:00 | 113.03 | 2025-09-10 10:15:00 | 116.55 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2025-09-09 09:15:00 | 112.80 | 2025-09-10 10:15:00 | 116.55 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2026-02-03 09:15:00 | 156.65 | 2026-02-18 09:15:00 | 172.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-10 09:15:00 | 153.71 | 2026-03-19 13:15:00 | 145.85 | STOP_HIT | 1.00 | -5.11% |
| BUY | retest2 | 2026-03-10 10:00:00 | 152.90 | 2026-03-19 13:15:00 | 145.85 | STOP_HIT | 1.00 | -4.61% |
| BUY | retest2 | 2026-03-12 10:15:00 | 152.47 | 2026-03-19 13:15:00 | 145.85 | STOP_HIT | 1.00 | -4.34% |
| SELL | retest1 | 2026-04-17 10:45:00 | 147.59 | 2026-04-22 10:15:00 | 152.72 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2026-04-24 09:15:00 | 148.93 | 2026-04-30 09:15:00 | 141.48 | PARTIAL | 0.50 | 5.00% |
