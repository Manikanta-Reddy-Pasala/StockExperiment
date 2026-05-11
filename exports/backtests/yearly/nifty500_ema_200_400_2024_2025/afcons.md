# Afcons Infrastructure Ltd. (AFCONS)

## Backtest Summary

- **Window:** 2024-11-04 09:15:00 → 2026-05-08 15:15:00 (2606 bars)
- **Last close:** 340.40
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
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 43 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 41 |
| PARTIAL | 13 |
| TARGET_HIT | 2 |
| STOP_HIT | 39 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 54 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 28 / 26
- **Target hits / Stop hits / Partials:** 2 / 39 / 13
- **Avg / median % per leg:** 0.82% / 0.20%
- **Sum % (uncompounded):** 44.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.72% | -15.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.72% | -15.5% |
| SELL (all) | 45 | 28 | 62.2% | 2 | 30 | 13 | 1.33% | 59.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 45 | 28 | 62.2% | 2 | 30 | 13 | 1.33% | 59.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 54 | 28 | 51.9% | 2 | 39 | 13 | 0.82% | 44.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 13:15:00 | 440.80 | 509.31 | 509.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 439.05 | 507.40 | 508.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-07 14:15:00 | 450.45 | 447.96 | 465.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-07 14:45:00 | 451.90 | 447.96 | 465.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 463.35 | 448.45 | 465.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-10 13:45:00 | 464.85 | 448.45 | 465.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 459.75 | 448.92 | 465.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:00:00 | 459.75 | 448.92 | 465.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 462.25 | 449.47 | 465.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:30:00 | 462.00 | 449.47 | 465.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 457.75 | 450.35 | 463.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 09:45:00 | 461.95 | 450.35 | 463.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 462.80 | 450.78 | 463.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 11:45:00 | 457.85 | 463.25 | 467.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-07 09:15:00 | 412.07 | 462.46 | 466.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-09-09 13:15:00)

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

### Cycle 3 — SELL (started 2025-11-14 09:15:00)

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

### Cycle 4 — BUY (started 2026-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 14:15:00 | 336.55 | 321.22 | 321.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 09:15:00 | 343.55 | 321.60 | 321.38 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-04-04 11:45:00 | 457.85 | 2025-04-07 09:15:00 | 412.07 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-13 15:15:00 | 456.00 | 2025-05-26 09:15:00 | 433.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-15 11:00:00 | 455.70 | 2025-05-26 09:15:00 | 432.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-15 12:45:00 | 457.05 | 2025-05-26 09:15:00 | 434.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-22 10:30:00 | 449.10 | 2025-05-28 09:15:00 | 426.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-23 09:15:00 | 449.00 | 2025-05-28 09:15:00 | 426.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-13 15:15:00 | 456.00 | 2025-06-09 09:15:00 | 448.20 | STOP_HIT | 0.50 | 1.71% |
| SELL | retest2 | 2025-05-15 11:00:00 | 455.70 | 2025-06-09 09:15:00 | 448.20 | STOP_HIT | 0.50 | 1.65% |
| SELL | retest2 | 2025-05-15 12:45:00 | 457.05 | 2025-06-09 09:15:00 | 448.20 | STOP_HIT | 0.50 | 1.94% |
| SELL | retest2 | 2025-05-22 10:30:00 | 449.10 | 2025-06-09 09:15:00 | 448.20 | STOP_HIT | 0.50 | 0.20% |
| SELL | retest2 | 2025-05-23 09:15:00 | 449.00 | 2025-06-09 09:15:00 | 448.20 | STOP_HIT | 0.50 | 0.18% |
| SELL | retest2 | 2025-06-09 12:45:00 | 449.00 | 2025-06-17 11:15:00 | 447.40 | STOP_HIT | 1.00 | 0.36% |
| SELL | retest2 | 2025-06-12 13:30:00 | 445.60 | 2025-06-19 12:15:00 | 426.55 | PARTIAL | 0.50 | 4.28% |
| SELL | retest2 | 2025-06-17 09:15:00 | 441.50 | 2025-06-19 13:15:00 | 423.32 | PARTIAL | 0.50 | 4.12% |
| SELL | retest2 | 2025-06-18 10:45:00 | 442.25 | 2025-06-19 14:15:00 | 420.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-12 13:30:00 | 445.60 | 2025-06-20 14:15:00 | 438.70 | STOP_HIT | 0.50 | 1.55% |
| SELL | retest2 | 2025-06-17 09:15:00 | 441.50 | 2025-06-20 14:15:00 | 438.70 | STOP_HIT | 0.50 | 0.63% |
| SELL | retest2 | 2025-06-18 10:45:00 | 442.25 | 2025-06-20 14:15:00 | 438.70 | STOP_HIT | 0.50 | 0.80% |
| SELL | retest2 | 2025-06-18 13:00:00 | 439.70 | 2025-06-20 15:15:00 | 455.75 | STOP_HIT | 1.00 | -3.65% |
| SELL | retest2 | 2025-06-23 09:15:00 | 435.90 | 2025-07-25 12:15:00 | 414.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-02 09:15:00 | 431.00 | 2025-07-28 09:15:00 | 409.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-04 10:00:00 | 431.00 | 2025-07-28 09:15:00 | 409.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 13:45:00 | 430.80 | 2025-07-28 09:15:00 | 409.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 14:15:00 | 429.10 | 2025-07-28 09:15:00 | 407.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-23 09:15:00 | 435.90 | 2025-07-29 10:15:00 | 392.31 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-02 09:15:00 | 431.00 | 2025-08-11 09:15:00 | 420.80 | STOP_HIT | 0.50 | 2.37% |
| SELL | retest2 | 2025-07-04 10:00:00 | 431.00 | 2025-08-11 09:15:00 | 420.80 | STOP_HIT | 0.50 | 2.37% |
| SELL | retest2 | 2025-07-22 13:45:00 | 430.80 | 2025-08-11 09:15:00 | 420.80 | STOP_HIT | 0.50 | 2.32% |
| SELL | retest2 | 2025-07-22 14:15:00 | 429.10 | 2025-08-11 09:15:00 | 420.80 | STOP_HIT | 0.50 | 1.93% |
| SELL | retest2 | 2025-08-11 14:00:00 | 421.80 | 2025-08-18 09:15:00 | 427.70 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-08-12 10:15:00 | 422.00 | 2025-08-18 09:15:00 | 427.70 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-08-12 10:45:00 | 419.70 | 2025-08-18 10:15:00 | 429.95 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-08-12 15:15:00 | 421.50 | 2025-08-18 10:15:00 | 429.95 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-08-14 09:45:00 | 418.60 | 2025-08-18 10:15:00 | 429.95 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-08-14 15:15:00 | 420.20 | 2025-08-18 10:15:00 | 429.95 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-08-29 09:30:00 | 419.70 | 2025-09-01 14:15:00 | 426.65 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-08-29 11:30:00 | 420.70 | 2025-09-01 14:15:00 | 426.65 | STOP_HIT | 1.00 | -1.41% |
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
