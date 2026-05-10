# Indegene Ltd. (INDGN)

## Backtest Summary

- **Window:** 2025-04-21 09:15:00 → 2026-05-08 15:15:00 (1822 bars)
- **Last close:** 530.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 2 |
| ALERT2 | 1 |
| ALERT2_SKIP | 1 |
| ALERT3 | 9 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 6 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 4
- **Target hits / Stop hits / Partials:** 2 / 4 / 2
- **Avg / median % per leg:** 2.62% / 4.40%
- **Sum % (uncompounded):** 20.95%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 4 | 50.0% | 2 | 4 | 2 | 2.62% | 21.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 4 | 50.0% | 2 | 4 | 2 | 2.62% | 21.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 8 | 4 | 50.0% | 2 | 4 | 2 | 2.62% | 21.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 11:15:00 | 573.40 | 569.99 | 569.98 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 12:15:00 | 567.15 | 569.97 | 569.98 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-26 14:15:00 | 576.55 | 570.04 | 570.01 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-09-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 13:15:00 | 561.50 | 569.92 | 569.95 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-09-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 15:15:00 | 574.90 | 570.02 | 570.00 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 10:15:00 | 566.30 | 569.95 | 569.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 11:15:00 | 563.45 | 569.88 | 569.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 11:15:00 | 569.85 | 569.21 | 569.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 11:15:00 | 569.85 | 569.21 | 569.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 569.85 | 569.21 | 569.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 12:00:00 | 569.85 | 569.21 | 569.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 568.85 | 569.20 | 569.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 13:15:00 | 571.55 | 569.20 | 569.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 13:15:00 | 581.10 | 569.32 | 569.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 13:45:00 | 583.00 | 569.32 | 569.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 579.85 | 569.43 | 569.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 14:30:00 | 581.85 | 569.43 | 569.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 571.95 | 569.67 | 569.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:30:00 | 571.70 | 569.67 | 569.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 536.85 | 527.95 | 537.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 536.85 | 527.95 | 537.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 532.95 | 528.44 | 536.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 14:30:00 | 525.70 | 528.62 | 536.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:30:00 | 527.05 | 528.33 | 535.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 10:30:00 | 528.45 | 528.31 | 535.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 538.10 | 528.44 | 535.25 | SL hit (close>static) qty=1.00 sl=537.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 538.10 | 528.44 | 535.25 | SL hit (close>static) qty=1.00 sl=537.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 538.10 | 528.44 | 535.25 | SL hit (close>static) qty=1.00 sl=537.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:00:00 | 529.20 | 528.46 | 535.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 535.75 | 528.70 | 534.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 15:00:00 | 535.75 | 528.70 | 534.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 543.80 | 528.85 | 535.00 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-22 15:15:00 | 543.80 | 528.85 | 535.00 | SL hit (close>static) qty=1.00 sl=537.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 09:15:00 | 531.30 | 528.85 | 535.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 10:15:00 | 534.65 | 528.92 | 535.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 15:15:00 | 507.92 | 524.16 | 530.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 504.73 | 524.00 | 529.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-19 11:15:00 | 481.19 | 518.85 | 526.35 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-19 14:15:00 | 478.17 | 517.68 | 525.65 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 7 — BUY (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 09:15:00 | 499.15 | 480.90 | 480.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 525.30 | 482.50 | 481.65 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-12-12 14:30:00 | 525.70 | 2025-12-18 15:15:00 | 538.10 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-12-18 09:30:00 | 527.05 | 2025-12-18 15:15:00 | 538.10 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-12-18 10:30:00 | 528.45 | 2025-12-18 15:15:00 | 538.10 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-12-19 11:00:00 | 529.20 | 2025-12-22 15:15:00 | 543.80 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2025-12-23 09:15:00 | 531.30 | 2026-01-09 15:15:00 | 507.92 | PARTIAL | 0.50 | 4.40% |
| SELL | retest2 | 2025-12-23 10:15:00 | 534.65 | 2026-01-12 09:15:00 | 504.73 | PARTIAL | 0.50 | 5.60% |
| SELL | retest2 | 2025-12-23 09:15:00 | 531.30 | 2026-01-19 11:15:00 | 481.19 | TARGET_HIT | 0.50 | 9.43% |
| SELL | retest2 | 2025-12-23 10:15:00 | 534.65 | 2026-01-19 14:15:00 | 478.17 | TARGET_HIT | 0.50 | 10.56% |
