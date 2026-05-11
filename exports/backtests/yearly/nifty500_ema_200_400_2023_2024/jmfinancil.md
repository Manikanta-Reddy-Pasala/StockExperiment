# JM Financial Ltd. (JMFINANCIL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 145.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 0 |
| ALERT3 | 43 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 43 |
| PARTIAL | 7 |
| TARGET_HIT | 10 |
| STOP_HIT | 33 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 50 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 24 / 26
- **Target hits / Stop hits / Partials:** 10 / 33 / 7
- **Avg / median % per leg:** 1.65% / -0.81%
- **Sum % (uncompounded):** 82.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 9 | 34.6% | 9 | 17 | 0 | 1.88% | 48.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 26 | 9 | 34.6% | 9 | 17 | 0 | 1.88% | 48.8% |
| SELL (all) | 24 | 15 | 62.5% | 1 | 16 | 7 | 1.40% | 33.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 24 | 15 | 62.5% | 1 | 16 | 7 | 1.40% | 33.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 50 | 24 | 48.0% | 10 | 33 | 7 | 1.65% | 82.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 11:15:00 | 79.35 | 98.90 | 98.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 09:15:00 | 78.70 | 97.95 | 98.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-12 10:15:00 | 83.30 | 83.23 | 88.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-12 10:30:00 | 83.40 | 83.23 | 88.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 09:15:00 | 85.85 | 82.42 | 86.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 10:00:00 | 85.85 | 82.42 | 86.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 10:15:00 | 88.80 | 82.48 | 86.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 11:00:00 | 88.80 | 82.48 | 86.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 11:15:00 | 88.55 | 82.54 | 86.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-30 14:15:00 | 87.85 | 82.67 | 86.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 09:15:00 | 87.35 | 82.80 | 86.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 09:15:00 | 83.46 | 83.09 | 86.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-06 09:15:00 | 83.10 | 83.09 | 86.05 | SL hit (close>static) qty=0.50 sl=83.09 alert=retest2 |

### Cycle 2 — BUY (started 2024-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 15:15:00 | 91.00 | 83.46 | 83.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 91.57 | 84.02 | 83.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 09:15:00 | 95.73 | 96.61 | 92.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-12 10:00:00 | 95.73 | 96.61 | 92.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 91.81 | 96.43 | 92.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 91.65 | 96.43 | 92.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 91.83 | 96.39 | 92.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:15:00 | 91.55 | 96.39 | 92.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 12:15:00 | 93.07 | 95.72 | 92.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 13:00:00 | 93.07 | 95.72 | 92.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 13:15:00 | 92.35 | 95.69 | 92.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 14:00:00 | 92.35 | 95.69 | 92.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 14:15:00 | 92.90 | 95.66 | 92.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 15:00:00 | 92.90 | 95.66 | 92.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 15:15:00 | 93.00 | 95.64 | 92.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 09:15:00 | 93.90 | 95.64 | 92.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 14:00:00 | 93.04 | 95.44 | 92.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 09:15:00 | 93.45 | 95.38 | 92.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 12:15:00 | 93.05 | 95.32 | 92.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 12:15:00 | 92.70 | 95.29 | 92.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 13:00:00 | 92.70 | 95.29 | 92.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 13:15:00 | 93.10 | 95.27 | 92.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:30:00 | 93.86 | 95.00 | 92.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-26 12:15:00 | 103.29 | 95.13 | 92.87 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 123.00 | 133.33 | 133.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 121.23 | 133.10 | 133.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 10:15:00 | 117.09 | 115.64 | 121.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-07 11:00:00 | 117.09 | 115.64 | 121.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 11:15:00 | 100.33 | 95.79 | 101.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-16 12:30:00 | 99.75 | 95.83 | 101.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-17 10:00:00 | 99.95 | 95.98 | 101.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-17 12:00:00 | 99.90 | 96.06 | 101.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-17 12:45:00 | 100.00 | 96.10 | 101.00 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 102.39 | 96.29 | 101.00 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-21 09:15:00 | 102.39 | 96.29 | 101.00 | SL hit (close>static) qty=1.00 sl=101.26 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 12:15:00 | 115.95 | 102.79 | 102.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 118.04 | 103.31 | 103.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-29 09:15:00 | 161.15 | 162.16 | 148.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-29 10:00:00 | 161.15 | 162.16 | 148.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 155.14 | 161.14 | 150.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 12:30:00 | 156.50 | 160.97 | 150.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 10:30:00 | 156.06 | 160.74 | 150.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 09:15:00 | 155.80 | 160.40 | 150.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 12:15:00 | 155.90 | 160.21 | 150.50 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-13 09:15:00 | 172.15 | 159.60 | 151.62 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 10:15:00 | 150.01 | 169.54 | 169.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 12:15:00 | 148.66 | 169.14 | 169.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 156.93 | 154.25 | 160.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 09:45:00 | 156.14 | 154.25 | 160.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 137.82 | 132.61 | 139.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:00:00 | 137.82 | 132.61 | 139.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 140.60 | 132.69 | 139.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:00:00 | 140.60 | 132.69 | 139.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 140.37 | 132.77 | 139.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 12:15:00 | 140.82 | 132.77 | 139.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 141.70 | 132.86 | 139.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 15:15:00 | 139.86 | 133.02 | 139.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 137.35 | 133.86 | 139.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 132.87 | 134.12 | 139.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 135.50 | 134.12 | 139.42 | SL hit (close>static) qty=0.50 sl=134.12 alert=retest2 |

### Cycle 6 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 138.78 | 131.28 | 131.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 141.50 | 131.96 | 131.61 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-30 14:15:00 | 87.85 | 2024-05-06 09:15:00 | 83.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-30 14:15:00 | 87.85 | 2024-05-06 09:15:00 | 83.10 | STOP_HIT | 0.50 | 5.41% |
| SELL | retest2 | 2024-05-02 09:15:00 | 87.35 | 2024-05-06 09:15:00 | 82.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-02 09:15:00 | 87.35 | 2024-05-06 09:15:00 | 83.10 | STOP_HIT | 0.50 | 4.87% |
| SELL | retest2 | 2024-06-28 12:15:00 | 88.15 | 2024-07-02 09:15:00 | 91.25 | STOP_HIT | 1.00 | -3.52% |
| SELL | retest2 | 2024-06-28 13:15:00 | 87.90 | 2024-07-02 09:15:00 | 91.25 | STOP_HIT | 1.00 | -3.81% |
| BUY | retest2 | 2024-08-20 09:15:00 | 93.90 | 2024-08-26 12:15:00 | 103.29 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-21 14:00:00 | 93.04 | 2024-08-26 12:15:00 | 102.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-22 09:15:00 | 93.45 | 2024-08-26 12:15:00 | 102.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-22 12:15:00 | 93.05 | 2024-08-26 12:15:00 | 102.36 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-26 09:30:00 | 93.86 | 2024-08-26 12:15:00 | 103.25 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-16 12:30:00 | 99.75 | 2025-04-21 09:15:00 | 102.39 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-04-17 10:00:00 | 99.95 | 2025-04-21 09:15:00 | 102.39 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-04-17 12:00:00 | 99.90 | 2025-04-21 09:15:00 | 102.39 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2025-04-17 12:45:00 | 100.00 | 2025-04-21 09:15:00 | 102.39 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2025-04-30 12:00:00 | 101.59 | 2025-05-07 10:15:00 | 96.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 15:15:00 | 100.48 | 2025-05-07 10:15:00 | 96.53 | PARTIAL | 0.50 | 3.93% |
| SELL | retest2 | 2025-04-30 12:00:00 | 101.59 | 2025-05-07 14:15:00 | 100.38 | STOP_HIT | 0.50 | 1.19% |
| SELL | retest2 | 2025-04-30 15:15:00 | 100.48 | 2025-05-07 14:15:00 | 100.38 | STOP_HIT | 0.50 | 0.10% |
| SELL | retest2 | 2025-05-05 10:00:00 | 101.61 | 2025-05-12 09:15:00 | 104.45 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2025-05-08 10:00:00 | 101.50 | 2025-05-12 09:15:00 | 104.45 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2025-05-08 13:30:00 | 100.60 | 2025-05-12 09:15:00 | 104.45 | STOP_HIT | 1.00 | -3.83% |
| BUY | retest2 | 2025-08-04 12:30:00 | 156.50 | 2025-08-13 09:15:00 | 172.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-05 10:30:00 | 156.06 | 2025-08-13 09:15:00 | 171.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-06 09:15:00 | 155.80 | 2025-08-13 09:15:00 | 171.38 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-06 12:15:00 | 155.90 | 2025-08-13 09:15:00 | 171.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-22 10:30:00 | 174.00 | 2025-09-25 13:15:00 | 167.50 | STOP_HIT | 1.00 | -3.74% |
| BUY | retest2 | 2025-09-23 09:15:00 | 174.66 | 2025-09-25 13:15:00 | 167.50 | STOP_HIT | 1.00 | -4.10% |
| BUY | retest2 | 2025-09-23 13:45:00 | 173.99 | 2025-09-25 13:15:00 | 167.50 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest2 | 2025-09-23 14:15:00 | 174.39 | 2025-09-25 13:15:00 | 167.50 | STOP_HIT | 1.00 | -3.95% |
| BUY | retest2 | 2025-10-13 10:30:00 | 173.75 | 2025-10-27 12:15:00 | 169.70 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-10-14 13:30:00 | 173.51 | 2025-10-27 12:15:00 | 169.70 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-10-14 14:30:00 | 173.60 | 2025-10-27 12:15:00 | 169.70 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-10-15 09:30:00 | 173.58 | 2025-10-27 12:15:00 | 169.70 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-10-20 12:00:00 | 173.54 | 2025-10-27 12:15:00 | 169.70 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-10-23 09:15:00 | 174.30 | 2025-10-27 12:15:00 | 169.70 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2025-10-23 11:30:00 | 172.73 | 2025-10-27 12:15:00 | 169.70 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-10-23 14:30:00 | 172.77 | 2025-10-28 09:15:00 | 169.66 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-10-24 09:15:00 | 172.19 | 2025-10-28 11:15:00 | 168.64 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-10-24 15:15:00 | 171.78 | 2025-10-28 11:15:00 | 168.64 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-10-27 10:45:00 | 171.60 | 2025-10-28 11:15:00 | 168.64 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-10-28 09:15:00 | 171.75 | 2025-10-28 11:15:00 | 168.64 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-10-29 14:30:00 | 171.01 | 2025-10-30 09:15:00 | 169.63 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2026-02-10 15:15:00 | 139.86 | 2026-02-16 09:15:00 | 132.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 15:15:00 | 139.86 | 2026-02-16 09:15:00 | 135.50 | STOP_HIT | 0.50 | 3.12% |
| SELL | retest2 | 2026-02-13 09:15:00 | 137.35 | 2026-02-16 09:15:00 | 130.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-13 09:15:00 | 137.35 | 2026-02-16 09:15:00 | 135.50 | STOP_HIT | 0.50 | 1.35% |
| SELL | retest2 | 2026-02-19 11:15:00 | 140.15 | 2026-02-24 09:15:00 | 133.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 11:15:00 | 140.15 | 2026-03-02 09:15:00 | 126.14 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-24 09:45:00 | 139.60 | 2026-04-29 11:15:00 | 138.78 | STOP_HIT | 1.00 | 0.59% |
