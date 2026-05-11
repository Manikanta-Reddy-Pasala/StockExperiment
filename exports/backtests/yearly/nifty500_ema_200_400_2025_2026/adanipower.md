# Adani Power Ltd. (ADANIPOWER)

## Backtest Summary

- **Window:** 2024-07-10 09:15:00 → 2026-05-08 15:15:00 (3161 bars)
- **Last close:** 225.02
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 3 |
| ALERT3 | 20 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 9 |
| PARTIAL | 0 |
| TARGET_HIT | 4 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 5
- **Target hits / Stop hits / Partials:** 4 / 5 / 0
- **Avg / median % per leg:** 3.41% / -1.01%
- **Sum % (uncompounded):** 30.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 4 | 44.4% | 4 | 5 | 0 | 3.41% | 30.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 4 | 44.4% | 4 | 5 | 0 | 3.41% | 30.7% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 9 | 4 | 44.4% | 4 | 5 | 0 | 3.41% | 30.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 140.23 | 145.13 | 145.14 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 11:15:00 | 149.48 | 145.16 | 145.14 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2026-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 10:15:00 | 140.26 | 145.14 | 145.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 10:15:00 | 140.18 | 144.26 | 144.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 142.20 | 139.91 | 142.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 142.20 | 139.91 | 142.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 142.20 | 139.91 | 142.06 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 12:15:00 | 149.50 | 143.83 | 143.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 14:15:00 | 150.95 | 143.96 | 143.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 143.38 | 144.39 | 144.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 143.38 | 144.39 | 144.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 143.38 | 144.39 | 144.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 143.65 | 144.39 | 144.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 143.30 | 144.38 | 144.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 09:45:00 | 144.12 | 144.08 | 143.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 12:30:00 | 144.38 | 144.07 | 143.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 13:00:00 | 144.00 | 144.07 | 143.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 14:30:00 | 144.00 | 144.07 | 143.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 142.90 | 144.06 | 143.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:00:00 | 142.90 | 144.06 | 143.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 142.55 | 144.04 | 143.94 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-18 10:15:00 | 142.55 | 144.04 | 143.94 | SL hit (close<static) qty=1.00 sl=142.61 alert=retest2 |

### Cycle 5 — SELL (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 10:15:00 | 141.02 | 143.82 | 143.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 13:15:00 | 140.39 | 143.71 | 143.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 09:15:00 | 143.27 | 141.58 | 142.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 143.27 | 141.58 | 142.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 143.27 | 141.58 | 142.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 143.67 | 141.58 | 142.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 143.90 | 141.60 | 142.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:00:00 | 143.90 | 141.60 | 142.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 155.64 | 143.37 | 143.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 12:15:00 | 155.76 | 143.49 | 143.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 11:15:00 | 145.25 | 145.60 | 144.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-23 12:00:00 | 145.25 | 145.60 | 144.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-29 09:15:00 | 110.86 | 2025-06-10 12:15:00 | 121.95 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-29 12:45:00 | 111.31 | 2025-06-10 12:15:00 | 121.95 | TARGET_HIT | 1.00 | 9.56% |
| BUY | retest2 | 2025-05-30 09:15:00 | 110.86 | 2025-06-10 12:15:00 | 121.68 | TARGET_HIT | 1.00 | 9.76% |
| BUY | retest2 | 2025-05-30 13:15:00 | 110.62 | 2025-06-20 14:15:00 | 105.94 | STOP_HIT | 1.00 | -4.23% |
| BUY | retest2 | 2025-06-25 09:15:00 | 110.25 | 2025-06-27 10:15:00 | 121.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-17 09:45:00 | 144.12 | 2026-02-18 10:15:00 | 142.55 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2026-02-17 12:30:00 | 144.38 | 2026-02-18 10:15:00 | 142.55 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2026-02-17 13:00:00 | 144.00 | 2026-02-18 10:15:00 | 142.55 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2026-02-17 14:30:00 | 144.00 | 2026-02-18 10:15:00 | 142.55 | STOP_HIT | 1.00 | -1.01% |
