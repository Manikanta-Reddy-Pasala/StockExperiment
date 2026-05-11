# Sumitomo Chemical India Ltd. (SUMICHEM)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 485.90
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
| ALERT2_SKIP | 2 |
| ALERT3 | 9 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 7 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 5
- **Target hits / Stop hits / Partials:** 0 / 7 / 2
- **Avg / median % per leg:** -0.15% / -1.43%
- **Sum % (uncompounded):** -1.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.49% | -10.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.49% | -10.5% |
| SELL (all) | 6 | 4 | 66.7% | 0 | 4 | 2 | 1.52% | 9.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 4 | 66.7% | 0 | 4 | 2 | 1.52% | 9.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 9 | 4 | 44.4% | 0 | 7 | 2 | -0.15% | -1.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 11:15:00 | 504.05 | 521.14 | 521.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 502.60 | 520.96 | 521.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 14:15:00 | 519.05 | 518.49 | 519.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 14:15:00 | 519.05 | 518.49 | 519.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 519.05 | 518.49 | 519.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 14:45:00 | 517.40 | 518.49 | 519.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 518.90 | 518.49 | 519.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:15:00 | 521.85 | 518.49 | 519.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 521.30 | 518.52 | 519.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 10:45:00 | 517.50 | 518.49 | 519.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 15:15:00 | 517.95 | 518.43 | 519.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 491.62 | 515.97 | 518.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 492.05 | 515.97 | 518.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 12:15:00 | 512.00 | 509.56 | 514.21 | SL hit (close>ema200) qty=0.50 sl=509.56 alert=retest2 |

### Cycle 2 — BUY (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 12:15:00 | 553.90 | 517.03 | 517.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 09:15:00 | 561.55 | 518.52 | 517.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 09:15:00 | 583.55 | 585.63 | 561.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-08 10:00:00 | 583.55 | 585.63 | 561.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 572.65 | 587.48 | 570.79 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 11:15:00 | 531.25 | 566.71 | 566.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 13:15:00 | 530.50 | 566.04 | 566.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 13:15:00 | 462.60 | 462.14 | 481.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 11:15:00 | 478.15 | 463.91 | 478.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 478.15 | 463.91 | 478.87 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 447.15 | 415.90 | 415.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 14:15:00 | 449.65 | 420.79 | 418.44 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 09:30:00 | 526.25 | 2025-05-27 09:15:00 | 506.95 | STOP_HIT | 1.00 | -3.67% |
| BUY | retest2 | 2025-05-16 09:15:00 | 525.75 | 2025-05-27 09:15:00 | 506.95 | STOP_HIT | 1.00 | -3.58% |
| BUY | retest2 | 2025-05-16 11:30:00 | 523.90 | 2025-05-27 09:15:00 | 506.95 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest2 | 2025-06-09 10:45:00 | 517.50 | 2025-06-13 09:15:00 | 491.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-09 15:15:00 | 517.95 | 2025-06-13 09:15:00 | 492.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-09 10:45:00 | 517.50 | 2025-06-23 12:15:00 | 512.00 | STOP_HIT | 0.50 | 1.06% |
| SELL | retest2 | 2025-06-09 15:15:00 | 517.95 | 2025-06-23 12:15:00 | 512.00 | STOP_HIT | 0.50 | 1.15% |
| SELL | retest2 | 2025-06-25 09:15:00 | 516.85 | 2025-06-27 10:15:00 | 525.40 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-07-01 11:30:00 | 518.25 | 2025-07-02 11:15:00 | 525.65 | STOP_HIT | 1.00 | -1.43% |
