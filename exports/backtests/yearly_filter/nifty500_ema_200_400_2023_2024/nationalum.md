# National Aluminium Co. Ltd. (NATIONALUM)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 401.75
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
| ALERT2 | 6 |
| ALERT2_SKIP | 1 |
| ALERT3 | 30 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 18 |
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 15
- **Target hits / Stop hits / Partials:** 3 / 15 / 3
- **Avg / median % per leg:** 1.03% / -1.10%
- **Sum % (uncompounded):** 21.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 3 | 21.4% | 3 | 11 | 0 | 0.67% | 9.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 3 | 21.4% | 3 | 11 | 0 | 0.67% | 9.4% |
| SELL (all) | 7 | 3 | 42.9% | 0 | 4 | 3 | 1.76% | 12.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 7 | 3 | 42.9% | 0 | 4 | 3 | 1.76% | 12.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 21 | 6 | 28.6% | 3 | 15 | 3 | 1.03% | 21.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 13:15:00 | 91.15 | 93.67 | 93.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-23 09:15:00 | 90.35 | 93.59 | 93.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-01 10:15:00 | 93.05 | 93.05 | 93.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-01 10:15:00 | 93.05 | 93.05 | 93.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 10:15:00 | 93.05 | 93.05 | 93.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-01 10:45:00 | 93.15 | 93.05 | 93.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 13:15:00 | 93.90 | 93.05 | 93.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-01 14:00:00 | 93.90 | 93.05 | 93.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 14:15:00 | 93.70 | 93.06 | 93.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-01 14:45:00 | 93.80 | 93.06 | 93.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2023-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 11:15:00 | 97.20 | 93.60 | 93.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-06 13:15:00 | 99.90 | 93.69 | 93.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-13 11:15:00 | 140.10 | 140.97 | 128.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-13 12:00:00 | 140.10 | 140.97 | 128.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 145.55 | 154.92 | 144.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 10:00:00 | 145.55 | 154.92 | 144.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 10:15:00 | 142.85 | 154.80 | 144.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 11:00:00 | 142.85 | 154.80 | 144.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 11:15:00 | 139.45 | 154.65 | 144.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 11:45:00 | 140.30 | 154.65 | 144.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 176.60 | 187.19 | 177.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 176.60 | 187.19 | 177.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 158.05 | 186.90 | 177.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 158.05 | 186.90 | 177.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 175.50 | 184.85 | 176.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-06 09:15:00 | 180.80 | 184.85 | 176.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-05 11:15:00 | 198.88 | 188.42 | 183.05 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 11:15:00 | 173.11 | 185.19 | 185.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 12:15:00 | 172.33 | 185.07 | 185.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 12:15:00 | 178.78 | 178.74 | 181.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-26 12:45:00 | 178.76 | 178.74 | 181.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 14:15:00 | 181.93 | 178.79 | 181.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 14:45:00 | 181.74 | 178.79 | 181.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 15:15:00 | 181.77 | 178.82 | 181.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:15:00 | 185.01 | 178.82 | 181.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 181.02 | 179.65 | 181.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 11:45:00 | 179.52 | 179.97 | 181.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 09:30:00 | 179.35 | 179.91 | 181.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 10:00:00 | 179.40 | 179.91 | 181.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 170.54 | 178.89 | 180.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 170.38 | 178.89 | 180.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 170.43 | 178.89 | 180.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-12 14:15:00 | 179.65 | 177.70 | 179.99 | SL hit (close>ema200) qty=0.50 sl=177.70 alert=retest2 |

### Cycle 4 — BUY (started 2024-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 14:15:00 | 194.95 | 181.59 | 181.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 11:15:00 | 198.47 | 182.14 | 181.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 09:15:00 | 221.49 | 223.09 | 211.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-13 09:45:00 | 223.05 | 223.09 | 211.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 226.99 | 239.75 | 228.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:45:00 | 226.45 | 239.75 | 228.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 227.10 | 239.63 | 228.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:45:00 | 224.51 | 239.63 | 228.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 227.73 | 239.39 | 228.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 12:30:00 | 226.41 | 239.39 | 228.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 227.29 | 239.27 | 228.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:00:00 | 227.29 | 239.27 | 228.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 227.30 | 236.79 | 227.98 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2025-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 09:15:00 | 206.25 | 222.87 | 222.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 09:15:00 | 200.88 | 220.45 | 221.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-21 09:15:00 | 196.23 | 195.15 | 202.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 196.23 | 195.15 | 202.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 171.85 | 163.16 | 171.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-14 09:45:00 | 171.30 | 163.16 | 171.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 10:15:00 | 170.32 | 163.23 | 171.85 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 11:15:00 | 185.45 | 176.39 | 176.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 12:15:00 | 188.00 | 176.51 | 176.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 15:15:00 | 181.50 | 181.64 | 179.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-20 09:15:00 | 182.42 | 181.64 | 179.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 188.93 | 190.12 | 186.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 14:45:00 | 189.76 | 190.07 | 186.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 185.83 | 189.86 | 186.78 | SL hit (close<static) qty=1.00 sl=186.02 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-06 09:15:00 | 180.80 | 2024-07-05 11:15:00 | 198.88 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-05 09:30:00 | 178.40 | 2024-08-05 10:15:00 | 172.98 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2024-08-06 09:15:00 | 178.20 | 2024-08-06 13:15:00 | 173.69 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2024-08-06 12:00:00 | 177.03 | 2024-08-06 13:15:00 | 173.69 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2024-09-02 11:45:00 | 179.52 | 2024-09-09 09:15:00 | 170.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-03 09:30:00 | 179.35 | 2024-09-09 09:15:00 | 170.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-03 10:00:00 | 179.40 | 2024-09-09 09:15:00 | 170.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-02 11:45:00 | 179.52 | 2024-09-12 14:15:00 | 179.65 | STOP_HIT | 0.50 | -0.07% |
| SELL | retest2 | 2024-09-03 09:30:00 | 179.35 | 2024-09-12 14:15:00 | 179.65 | STOP_HIT | 0.50 | -0.17% |
| SELL | retest2 | 2024-09-03 10:00:00 | 179.40 | 2024-09-12 14:15:00 | 179.65 | STOP_HIT | 0.50 | -0.14% |
| SELL | retest2 | 2024-09-19 10:45:00 | 179.37 | 2024-09-19 15:15:00 | 183.50 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-07-29 14:45:00 | 189.76 | 2025-07-31 09:15:00 | 185.83 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-08-08 10:15:00 | 189.84 | 2025-08-26 14:15:00 | 185.87 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-08-13 09:15:00 | 190.31 | 2025-08-26 14:15:00 | 185.87 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-08-13 12:15:00 | 189.63 | 2025-08-26 14:15:00 | 185.87 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-08-14 15:00:00 | 187.93 | 2025-08-26 14:15:00 | 185.87 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-08-18 09:30:00 | 188.03 | 2025-08-26 14:15:00 | 185.87 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-08-28 11:30:00 | 187.62 | 2025-08-28 12:15:00 | 184.81 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-08-29 11:00:00 | 188.00 | 2025-08-29 14:15:00 | 186.17 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-09-01 09:15:00 | 188.57 | 2025-09-03 12:15:00 | 207.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-01 10:30:00 | 188.65 | 2025-09-03 12:15:00 | 207.52 | TARGET_HIT | 1.00 | 10.00% |
