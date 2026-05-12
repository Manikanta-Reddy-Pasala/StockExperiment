# Afcons Infrastructure Ltd. (AFCONS)

## Backtest Summary

- **Window:** 2024-11-04 09:15:00 → 2026-05-11 15:15:00 (2613 bars)
- **Last close:** 326.30
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
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 17 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 19 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 18
- **Target hits / Stop hits / Partials:** 0 / 19 / 1
- **Avg / median % per leg:** -1.66% / -2.13%
- **Sum % (uncompounded):** -33.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.72% | -15.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.72% | -15.5% |
| SELL (all) | 11 | 2 | 18.2% | 0 | 10 | 1 | -1.61% | -17.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 11 | 2 | 18.2% | 0 | 10 | 1 | -1.61% | -17.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 20 | 2 | 10.0% | 0 | 19 | 1 | -1.66% | -33.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 13:15:00 | 438.95 | 427.25 | 427.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 446.25 | 427.65 | 427.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 440.20 | 441.89 | 436.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-26 09:30:00 | 440.40 | 441.89 | 436.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 438.90 | 441.82 | 436.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:00:00 | 438.90 | 441.82 | 436.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 445.05 | 442.17 | 436.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 10:15:00 | 446.00 | 442.18 | 436.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 12:30:00 | 446.95 | 442.33 | 436.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 12:30:00 | 446.85 | 450.00 | 443.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 11:30:00 | 446.50 | 449.78 | 443.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 443.40 | 449.49 | 443.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:45:00 | 443.35 | 449.49 | 443.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 444.20 | 449.44 | 443.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:30:00 | 443.70 | 449.44 | 443.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 444.55 | 449.39 | 443.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:15:00 | 444.00 | 449.39 | 443.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 444.00 | 449.34 | 443.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:30:00 | 443.30 | 449.34 | 443.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 443.60 | 449.28 | 443.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:45:00 | 443.45 | 449.28 | 443.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 443.50 | 449.22 | 443.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 446.75 | 449.22 | 443.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 12:15:00 | 441.65 | 449.00 | 443.69 | SL hit (close<static) qty=1.00 sl=443.10 alert=retest2 |

### Cycle 2 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 408.00 | 441.44 | 441.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 12:15:00 | 404.35 | 438.20 | 439.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 10:15:00 | 291.60 | 290.68 | 312.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-07 10:45:00 | 291.75 | 290.68 | 312.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 12:15:00 | 311.35 | 292.03 | 311.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 14:15:00 | 310.70 | 292.23 | 311.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 15:15:00 | 310.10 | 292.42 | 311.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 318.00 | 292.85 | 311.66 | SL hit (close>static) qty=1.00 sl=317.80 alert=retest2 |

### Cycle 3 — BUY (started 2026-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 14:15:00 | 336.55 | 321.22 | 321.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 09:15:00 | 343.55 | 321.60 | 321.38 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-15 10:45:00 | 458.75 | 2025-05-19 09:15:00 | 471.00 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-05-21 11:45:00 | 459.75 | 2025-05-26 09:15:00 | 436.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-21 11:45:00 | 459.75 | 2025-06-09 09:15:00 | 448.20 | STOP_HIT | 0.50 | 2.51% |
| BUY | retest2 | 2025-09-30 10:15:00 | 446.00 | 2025-10-27 12:15:00 | 441.65 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-09-30 12:30:00 | 446.95 | 2025-11-06 09:15:00 | 440.50 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-10-20 12:30:00 | 446.85 | 2025-11-06 09:15:00 | 440.50 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-10-23 11:30:00 | 446.50 | 2025-11-07 09:15:00 | 437.00 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-10-27 09:15:00 | 446.75 | 2025-11-10 12:15:00 | 437.80 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-10-28 09:15:00 | 446.45 | 2025-11-10 15:15:00 | 436.00 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2025-11-04 12:00:00 | 444.50 | 2025-11-10 15:15:00 | 436.00 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-11-06 10:45:00 | 444.60 | 2025-11-10 15:15:00 | 436.00 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-11-10 11:30:00 | 441.90 | 2025-11-10 15:15:00 | 436.00 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-04-09 14:15:00 | 310.70 | 2026-04-10 09:15:00 | 318.00 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2026-04-09 15:15:00 | 310.10 | 2026-04-10 09:15:00 | 318.00 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2026-04-13 11:15:00 | 310.85 | 2026-04-15 09:15:00 | 323.20 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2026-04-13 12:15:00 | 310.10 | 2026-04-15 09:15:00 | 323.20 | STOP_HIT | 1.00 | -4.22% |
| SELL | retest2 | 2026-04-24 11:45:00 | 315.00 | 2026-04-27 09:15:00 | 322.20 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2026-04-24 12:15:00 | 315.00 | 2026-04-27 09:15:00 | 322.20 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2026-04-24 13:00:00 | 314.85 | 2026-04-27 09:15:00 | 322.20 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2026-04-24 13:45:00 | 314.05 | 2026-04-27 09:15:00 | 322.20 | STOP_HIT | 1.00 | -2.60% |
