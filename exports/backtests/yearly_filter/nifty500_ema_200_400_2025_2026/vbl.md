# Varun Beverages Ltd. (VBL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 508.35
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 22 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 12
- **Target hits / Stop hits / Partials:** 0 / 12 / 0
- **Avg / median % per leg:** -3.25% / -2.49%
- **Sum % (uncompounded):** -39.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -3.08% | -21.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -3.08% | -21.5% |
| SELL (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -3.49% | -17.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -3.49% | -17.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 12 | 0 | 0.0% | 0 | 12 | 0 | -3.25% | -39.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 505.50 | 486.35 | 486.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 516.60 | 491.12 | 488.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-28 12:15:00 | 501.05 | 501.41 | 495.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-28 13:00:00 | 501.05 | 501.41 | 495.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 496.90 | 501.32 | 495.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:15:00 | 497.85 | 501.32 | 495.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 486.20 | 501.17 | 495.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:45:00 | 488.30 | 501.17 | 495.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 487.90 | 501.04 | 495.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 11:30:00 | 492.60 | 500.97 | 495.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 15:15:00 | 485.95 | 500.48 | 495.40 | SL hit (close<static) qty=1.00 sl=486.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 09:15:00 | 473.60 | 492.22 | 492.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 10:15:00 | 470.40 | 492.00 | 492.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 11:15:00 | 458.75 | 456.68 | 468.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-16 12:00:00 | 458.75 | 456.68 | 468.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 467.40 | 457.42 | 467.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:30:00 | 469.80 | 457.42 | 467.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 469.60 | 457.54 | 467.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:30:00 | 470.40 | 457.54 | 467.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 469.20 | 457.66 | 467.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:00:00 | 469.20 | 457.66 | 467.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 465.00 | 457.94 | 467.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:45:00 | 468.30 | 457.94 | 467.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 467.65 | 458.35 | 467.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:00:00 | 467.65 | 458.35 | 467.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 464.15 | 458.40 | 467.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 13:45:00 | 461.05 | 458.44 | 467.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 11:15:00 | 479.95 | 458.57 | 467.07 | SL hit (close>static) qty=1.00 sl=468.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 475.80 | 467.90 | 467.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 14:15:00 | 478.55 | 468.05 | 467.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 13:15:00 | 469.75 | 470.20 | 469.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 13:15:00 | 469.75 | 470.20 | 469.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 469.75 | 470.20 | 469.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:00:00 | 469.75 | 470.20 | 469.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 471.10 | 470.21 | 469.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:45:00 | 469.45 | 470.21 | 469.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 469.90 | 470.21 | 469.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:15:00 | 465.50 | 470.21 | 469.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 463.65 | 470.14 | 469.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:00:00 | 463.65 | 470.14 | 469.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 466.50 | 470.10 | 469.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 11:15:00 | 469.20 | 470.10 | 469.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 12:00:00 | 468.95 | 470.20 | 469.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 14:30:00 | 470.85 | 485.85 | 480.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 15:15:00 | 469.10 | 484.52 | 480.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 09:15:00 | 457.40 | 484.10 | 480.10 | SL hit (close<static) qty=1.00 sl=463.10 alert=retest2 |

### Cycle 4 — SELL (started 2026-02-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 15:15:00 | 444.10 | 476.68 | 476.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 09:15:00 | 441.95 | 476.33 | 476.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 15:15:00 | 461.80 | 461.78 | 467.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-27 09:15:00 | 460.75 | 461.78 | 467.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 13:15:00 | 433.00 | 418.27 | 433.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:00:00 | 433.00 | 418.27 | 433.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 442.30 | 419.57 | 433.74 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 495.90 | 444.13 | 444.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 501.75 | 445.68 | 444.79 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-08-29 11:30:00 | 492.60 | 2025-08-29 15:15:00 | 485.95 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-09-01 14:00:00 | 492.80 | 2025-09-05 09:15:00 | 482.45 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-10-27 13:45:00 | 461.05 | 2025-10-29 11:15:00 | 479.95 | STOP_HIT | 1.00 | -4.10% |
| SELL | retest2 | 2025-11-11 11:30:00 | 462.50 | 2025-11-11 14:15:00 | 470.55 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-11-12 10:15:00 | 462.00 | 2025-11-28 09:15:00 | 484.00 | STOP_HIT | 1.00 | -4.76% |
| SELL | retest2 | 2025-11-12 10:45:00 | 462.05 | 2025-11-28 09:15:00 | 484.00 | STOP_HIT | 1.00 | -4.75% |
| SELL | retest2 | 2025-12-09 09:15:00 | 462.50 | 2025-12-09 11:15:00 | 472.25 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-12-18 11:15:00 | 469.20 | 2026-01-29 09:15:00 | 457.40 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2025-12-19 12:00:00 | 468.95 | 2026-01-29 09:15:00 | 457.40 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2026-01-27 14:30:00 | 470.85 | 2026-01-29 09:15:00 | 457.40 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2026-01-28 15:15:00 | 469.10 | 2026-01-29 09:15:00 | 457.40 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2026-02-03 12:00:00 | 487.85 | 2026-02-03 12:15:00 | 450.00 | STOP_HIT | 1.00 | -7.76% |
