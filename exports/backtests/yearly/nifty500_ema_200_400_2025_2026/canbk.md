# Canara Bank (CANBK)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3164 bars)
- **Last close:** 134.13
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 9 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 9
- **Target hits / Stop hits / Partials:** 0 / 9 / 0
- **Avg / median % per leg:** -3.01% / -3.32%
- **Sum % (uncompounded):** -27.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.49% | -19.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.49% | -19.9% |
| SELL (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -7.17% | -7.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -7.17% | -7.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 9 | 0 | 0.0% | 0 | 9 | 0 | -3.01% | -27.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 13:15:00 | 107.30 | 101.32 | 101.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 14:15:00 | 107.88 | 101.39 | 101.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 109.90 | 110.34 | 106.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:45:00 | 109.82 | 110.34 | 106.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 106.40 | 110.21 | 107.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:00:00 | 106.40 | 110.21 | 107.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 105.90 | 110.16 | 107.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:00:00 | 105.90 | 110.16 | 107.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 106.95 | 109.80 | 107.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:00:00 | 106.95 | 109.80 | 107.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 106.67 | 109.77 | 107.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:00:00 | 106.67 | 109.77 | 107.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 107.25 | 109.72 | 107.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:15:00 | 106.89 | 109.72 | 107.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 107.23 | 109.69 | 107.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:15:00 | 107.56 | 109.67 | 107.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 11:00:00 | 107.67 | 112.45 | 110.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 14:30:00 | 107.41 | 112.24 | 110.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:00:00 | 107.46 | 111.55 | 110.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 14:15:00 | 105.90 | 111.09 | 110.11 | SL hit (close<static) qty=1.00 sl=105.98 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 09:15:00 | 103.82 | 109.57 | 109.58 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 12:15:00 | 113.34 | 109.34 | 109.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 10:15:00 | 115.14 | 109.55 | 109.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 143.26 | 143.63 | 135.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 12:45:00 | 142.98 | 143.63 | 135.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 148.10 | 152.17 | 147.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 09:30:00 | 150.95 | 148.43 | 147.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 10:45:00 | 150.80 | 148.46 | 147.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 11:15:00 | 151.07 | 148.46 | 147.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:30:00 | 151.35 | 148.77 | 147.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 148.36 | 151.65 | 149.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:00:00 | 148.36 | 151.65 | 149.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 147.20 | 151.61 | 149.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:45:00 | 146.55 | 151.61 | 149.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-04 11:15:00 | 145.80 | 151.55 | 149.22 | SL hit (close<static) qty=1.00 sl=146.62 alert=retest2 |

### Cycle 4 — SELL (started 2026-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 15:15:00 | 134.70 | 147.46 | 147.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 134.26 | 147.33 | 147.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 137.19 | 137.14 | 141.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 141.55 | 137.58 | 140.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 141.55 | 137.58 | 140.93 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 09:15:00 | 100.12 | 2025-05-21 13:15:00 | 107.30 | STOP_HIT | 1.00 | -7.17% |
| BUY | retest2 | 2025-06-23 11:15:00 | 107.56 | 2025-08-01 14:15:00 | 105.90 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-07-23 11:00:00 | 107.67 | 2025-08-01 14:15:00 | 105.90 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-07-23 14:30:00 | 107.41 | 2025-08-01 14:15:00 | 105.90 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-07-31 11:00:00 | 107.46 | 2025-08-01 14:15:00 | 105.90 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2026-02-18 09:30:00 | 150.95 | 2026-03-04 11:15:00 | 145.80 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest2 | 2026-02-18 10:45:00 | 150.80 | 2026-03-04 11:15:00 | 145.80 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2026-02-18 11:15:00 | 151.07 | 2026-03-04 11:15:00 | 145.80 | STOP_HIT | 1.00 | -3.49% |
| BUY | retest2 | 2026-02-20 09:30:00 | 151.35 | 2026-03-04 11:15:00 | 145.80 | STOP_HIT | 1.00 | -3.67% |
