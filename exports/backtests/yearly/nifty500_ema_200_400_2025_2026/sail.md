# Steel Authority of India Ltd. (SAIL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 184.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 19 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 20 |
| PARTIAL | 0 |
| TARGET_HIT | 10 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 14
- **Target hits / Stop hits / Partials:** 10 / 14 / 0
- **Avg / median % per leg:** 2.72% / -1.30%
- **Sum % (uncompounded):** 65.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 10 | 50.0% | 10 | 10 | 0 | 4.00% | 80.0% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.63% | -6.5% |
| BUY @ 3rd Alert (retest2) | 16 | 10 | 62.5% | 10 | 6 | 0 | 5.41% | 86.6% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.69% | -14.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.69% | -14.8% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.63% | -6.5% |
| retest2 (combined) | 20 | 10 | 50.0% | 10 | 10 | 0 | 3.59% | 71.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 121.35 | 126.97 | 127.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 11:15:00 | 121.17 | 126.92 | 126.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 11:15:00 | 124.39 | 124.04 | 125.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-02 12:00:00 | 124.39 | 124.04 | 125.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 125.00 | 124.01 | 125.23 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 13:15:00 | 131.75 | 126.17 | 126.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 14:15:00 | 131.80 | 126.23 | 126.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 15:15:00 | 130.60 | 130.76 | 128.87 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 09:15:00 | 132.50 | 130.76 | 128.87 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 10:45:00 | 131.31 | 131.75 | 129.80 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 11:45:00 | 131.35 | 131.75 | 129.81 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 15:15:00 | 132.00 | 131.73 | 129.83 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 131.25 | 132.02 | 130.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 11:45:00 | 131.67 | 132.01 | 130.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:30:00 | 131.60 | 132.00 | 130.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 14:15:00 | 131.70 | 131.99 | 130.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 11:15:00 | 129.64 | 131.97 | 130.18 | SL hit (close<ema400) qty=1.00 sl=130.18 alert=retest1 |

### Cycle 3 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 129.83 | 133.45 | 133.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 13:15:00 | 127.59 | 132.96 | 133.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 10:15:00 | 132.25 | 132.18 | 132.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 10:15:00 | 132.25 | 132.18 | 132.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 132.25 | 132.18 | 132.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:00:00 | 132.25 | 132.18 | 132.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 133.68 | 132.20 | 132.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:00:00 | 133.68 | 132.20 | 132.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 132.68 | 132.20 | 132.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 14:30:00 | 132.03 | 132.21 | 132.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 11:15:00 | 132.38 | 132.23 | 132.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 14:15:00 | 132.28 | 132.25 | 132.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 15:00:00 | 132.20 | 132.25 | 132.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 132.07 | 132.23 | 132.75 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 137.10 | 132.28 | 132.77 | SL hit (close>static) qty=1.00 sl=133.74 alert=retest2 |

### Cycle 4 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 147.34 | 133.22 | 133.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 13:15:00 | 149.20 | 136.21 | 134.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 10:15:00 | 143.94 | 146.51 | 142.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-02 11:00:00 | 143.94 | 146.51 | 142.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 141.97 | 146.46 | 142.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:00:00 | 141.97 | 146.46 | 142.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 144.53 | 146.44 | 142.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 13:30:00 | 146.43 | 146.44 | 142.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-06 14:15:00 | 161.07 | 148.86 | 144.00 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 09:15:00 | 113.84 | 2025-05-20 09:15:00 | 125.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2025-09-29 09:15:00 | 132.50 | 2025-10-14 11:15:00 | 129.64 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest1 | 2025-10-08 10:45:00 | 131.31 | 2025-10-14 11:15:00 | 129.64 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest1 | 2025-10-08 11:45:00 | 131.35 | 2025-10-14 11:15:00 | 129.64 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest1 | 2025-10-08 15:15:00 | 132.00 | 2025-10-14 11:15:00 | 129.64 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-10-13 11:45:00 | 131.67 | 2025-10-14 12:15:00 | 128.62 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-10-13 12:30:00 | 131.60 | 2025-10-14 12:15:00 | 128.62 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-10-13 14:15:00 | 131.70 | 2025-10-14 12:15:00 | 128.62 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2025-10-16 10:45:00 | 131.75 | 2025-10-17 14:15:00 | 128.79 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-10-24 09:45:00 | 131.38 | 2025-11-10 12:15:00 | 144.52 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-28 10:30:00 | 131.67 | 2025-11-10 12:15:00 | 144.84 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-03 11:45:00 | 131.63 | 2025-12-08 13:15:00 | 129.02 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-12-03 15:00:00 | 132.07 | 2025-12-08 13:15:00 | 129.02 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-12-23 14:30:00 | 132.03 | 2025-12-29 09:15:00 | 137.10 | STOP_HIT | 1.00 | -3.84% |
| SELL | retest2 | 2025-12-24 11:15:00 | 132.38 | 2025-12-29 09:15:00 | 137.10 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest2 | 2025-12-24 14:15:00 | 132.28 | 2025-12-29 09:15:00 | 137.10 | STOP_HIT | 1.00 | -3.64% |
| SELL | retest2 | 2025-12-24 15:00:00 | 132.20 | 2025-12-29 09:15:00 | 137.10 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2026-02-02 13:30:00 | 146.43 | 2026-02-06 14:15:00 | 161.07 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-16 14:00:00 | 145.30 | 2026-04-06 12:15:00 | 159.83 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-17 09:15:00 | 145.14 | 2026-04-06 12:15:00 | 159.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-24 09:15:00 | 145.21 | 2026-04-06 12:15:00 | 159.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-30 11:30:00 | 154.43 | 2026-04-08 11:15:00 | 168.61 | TARGET_HIT | 1.00 | 9.18% |
| BUY | retest2 | 2026-04-01 09:15:00 | 154.85 | 2026-04-13 11:15:00 | 169.87 | TARGET_HIT | 1.00 | 9.70% |
| BUY | retest2 | 2026-04-02 12:30:00 | 153.28 | 2026-04-13 11:15:00 | 170.34 | TARGET_HIT | 1.00 | 11.13% |
