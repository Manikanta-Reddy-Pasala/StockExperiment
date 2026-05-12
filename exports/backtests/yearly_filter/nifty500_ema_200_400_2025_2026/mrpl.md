# Mangalore Refinery & Petrochemicals Ltd. (MRPL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 167.97
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 27 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 25 |
| PARTIAL | 1 |
| TARGET_HIT | 6 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 19
- **Target hits / Stop hits / Partials:** 6 / 19 / 1
- **Avg / median % per leg:** -0.60% / -1.51%
- **Sum % (uncompounded):** -15.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 6 | 46.2% | 6 | 7 | 0 | 0.36% | 4.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 6 | 46.2% | 6 | 7 | 0 | 0.36% | 4.7% |
| SELL (all) | 13 | 1 | 7.7% | 0 | 12 | 1 | -1.57% | -20.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 13 | 1 | 7.7% | 0 | 12 | 1 | -1.57% | -20.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 26 | 7 | 26.9% | 6 | 19 | 1 | -0.60% | -15.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 12:15:00 | 125.39 | 139.68 | 139.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 124.46 | 138.30 | 139.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 128.85 | 127.85 | 131.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 128.85 | 127.85 | 131.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 128.85 | 127.85 | 131.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:30:00 | 129.52 | 127.85 | 131.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 130.05 | 127.89 | 130.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 131.24 | 127.89 | 130.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 130.80 | 127.92 | 130.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 10:45:00 | 130.60 | 127.94 | 130.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 11:30:00 | 130.62 | 127.98 | 130.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 10:45:00 | 130.61 | 128.16 | 130.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 11:30:00 | 130.68 | 128.19 | 130.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 130.72 | 128.21 | 130.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:45:00 | 130.62 | 128.21 | 130.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 130.55 | 128.23 | 130.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:45:00 | 130.59 | 128.23 | 130.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 130.80 | 128.29 | 130.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:45:00 | 131.14 | 128.29 | 130.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 130.50 | 128.31 | 130.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:30:00 | 130.92 | 128.31 | 130.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 131.35 | 128.34 | 130.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:00:00 | 131.35 | 128.34 | 130.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 131.27 | 128.37 | 130.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:45:00 | 131.77 | 128.37 | 130.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 129.50 | 128.43 | 130.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 10:45:00 | 129.00 | 128.44 | 130.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 14:15:00 | 129.22 | 128.46 | 130.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 15:00:00 | 129.13 | 128.47 | 130.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:15:00 | 129.23 | 128.49 | 130.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 128.98 | 128.46 | 130.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:30:00 | 129.70 | 128.46 | 130.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 129.45 | 128.50 | 130.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:30:00 | 129.55 | 128.50 | 130.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 132.57 | 128.53 | 130.45 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 132.57 | 128.53 | 130.45 | SL hit (close>static) qty=1.00 sl=132.20 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 13:15:00 | 148.61 | 131.94 | 131.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 14:15:00 | 149.59 | 132.11 | 132.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 09:15:00 | 160.23 | 163.85 | 154.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-26 09:45:00 | 159.91 | 163.85 | 154.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 155.02 | 162.13 | 155.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:45:00 | 154.62 | 162.13 | 155.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 155.42 | 162.07 | 155.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:30:00 | 155.26 | 162.07 | 155.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 154.26 | 161.99 | 155.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:00:00 | 154.26 | 161.99 | 155.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 154.51 | 161.91 | 155.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-08 10:30:00 | 155.45 | 161.54 | 155.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 11:15:00 | 151.80 | 161.44 | 155.24 | SL hit (close<static) qty=1.00 sl=154.20 alert=retest2 |

### Cycle 3 — SELL (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 09:15:00 | 146.81 | 152.05 | 152.08 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 157.65 | 152.11 | 152.10 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 15:15:00 | 150.60 | 152.09 | 152.10 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 09:15:00 | 153.87 | 152.11 | 152.11 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 15:15:00 | 151.40 | 152.10 | 152.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 09:15:00 | 150.32 | 152.08 | 152.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 11:15:00 | 152.48 | 149.02 | 150.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 11:15:00 | 152.48 | 149.02 | 150.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 152.48 | 149.02 | 150.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:00:00 | 152.48 | 149.02 | 150.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 151.52 | 149.05 | 150.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:15:00 | 151.16 | 149.05 | 150.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 14:15:00 | 156.66 | 149.15 | 150.44 | SL hit (close>static) qty=1.00 sl=152.64 alert=retest2 |

### Cycle 8 — BUY (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 09:15:00 | 170.09 | 151.12 | 151.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 10:15:00 | 174.43 | 151.35 | 151.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 15:15:00 | 186.86 | 186.93 | 176.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-12 09:15:00 | 178.35 | 186.93 | 176.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 178.47 | 186.74 | 177.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 15:00:00 | 178.47 | 186.74 | 177.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 187.23 | 186.67 | 177.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 11:15:00 | 193.15 | 186.69 | 177.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 10:30:00 | 189.91 | 188.59 | 179.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 196.51 | 188.42 | 179.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 169.81 | 183.75 | 180.10 | SL hit (close<static) qty=1.00 sl=173.91 alert=retest2 |

### Cycle 9 — SELL (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 11:15:00 | 156.94 | 178.38 | 178.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 14:15:00 | 155.60 | 177.73 | 178.15 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 11:15:00 | 133.50 | 2025-05-28 12:15:00 | 146.85 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-14 13:15:00 | 133.71 | 2025-05-28 12:15:00 | 147.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-19 12:00:00 | 133.64 | 2025-06-23 13:15:00 | 146.85 | TARGET_HIT | 1.00 | 9.88% |
| BUY | retest2 | 2025-06-20 10:30:00 | 133.50 | 2025-07-01 09:15:00 | 147.00 | TARGET_HIT | 1.00 | 10.12% |
| BUY | retest2 | 2025-07-22 09:15:00 | 145.21 | 2025-07-24 09:15:00 | 157.07 | TARGET_HIT | 1.00 | 8.17% |
| BUY | retest2 | 2025-07-22 11:15:00 | 142.79 | 2025-07-24 09:15:00 | 157.60 | TARGET_HIT | 1.00 | 10.37% |
| BUY | retest2 | 2025-07-22 12:45:00 | 143.27 | 2025-07-28 12:15:00 | 136.39 | STOP_HIT | 1.00 | -4.80% |
| SELL | retest2 | 2025-09-18 10:45:00 | 130.60 | 2025-09-29 09:15:00 | 132.57 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-09-18 11:30:00 | 130.62 | 2025-09-29 09:15:00 | 132.57 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-09-19 10:45:00 | 130.61 | 2025-09-29 09:15:00 | 132.57 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-09-19 11:30:00 | 130.68 | 2025-09-29 09:15:00 | 132.57 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-09-23 10:45:00 | 129.00 | 2025-09-29 09:15:00 | 132.57 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2025-09-23 14:15:00 | 129.22 | 2025-09-29 09:15:00 | 132.57 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-09-23 15:00:00 | 129.13 | 2025-09-29 09:15:00 | 132.57 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-09-24 10:15:00 | 129.23 | 2025-09-29 09:15:00 | 132.57 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2025-12-08 10:30:00 | 155.45 | 2025-12-08 11:15:00 | 151.80 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-12-10 09:30:00 | 155.09 | 2025-12-10 13:15:00 | 153.75 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2026-01-14 13:15:00 | 151.16 | 2026-01-14 14:15:00 | 156.66 | STOP_HIT | 1.00 | -3.64% |
| SELL | retest2 | 2026-01-16 15:15:00 | 149.96 | 2026-01-20 09:15:00 | 142.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 15:15:00 | 149.96 | 2026-01-21 12:15:00 | 153.07 | STOP_HIT | 0.50 | -2.07% |
| SELL | retest2 | 2026-01-21 13:30:00 | 151.00 | 2026-01-21 15:15:00 | 153.20 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-01-21 14:00:00 | 150.70 | 2026-01-21 15:15:00 | 153.20 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2026-03-16 11:15:00 | 193.15 | 2026-04-13 09:15:00 | 169.81 | STOP_HIT | 1.00 | -12.08% |
| BUY | retest2 | 2026-03-19 10:30:00 | 189.91 | 2026-04-13 09:15:00 | 169.81 | STOP_HIT | 1.00 | -10.58% |
| BUY | retest2 | 2026-03-20 09:15:00 | 196.51 | 2026-04-13 09:15:00 | 169.81 | STOP_HIT | 1.00 | -13.59% |
| BUY | retest2 | 2026-04-23 09:30:00 | 191.67 | 2026-04-27 09:15:00 | 173.27 | STOP_HIT | 1.00 | -9.60% |
