# Graphite India Ltd. (GRAPHITE)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 752.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 0 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 15 |
| PARTIAL | 0 |
| TARGET_HIT | 6 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 9
- **Target hits / Stop hits / Partials:** 6 / 9 / 0
- **Avg / median % per leg:** 2.67% / -0.93%
- **Sum % (uncompounded):** 39.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 6 | 46.2% | 6 | 7 | 0 | 3.42% | 44.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 6 | 46.2% | 6 | 7 | 0 | 3.42% | 44.5% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.24% | -4.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.24% | -4.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 15 | 6 | 40.0% | 6 | 9 | 0 | 2.67% | 40.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 10:15:00 | 556.20 | 468.36 | 468.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 11:15:00 | 566.45 | 469.34 | 468.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 525.00 | 527.16 | 506.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 09:30:00 | 522.70 | 527.16 | 506.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 545.25 | 563.12 | 545.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:00:00 | 545.25 | 563.12 | 545.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 544.40 | 562.93 | 545.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 09:15:00 | 547.50 | 562.31 | 544.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 12:15:00 | 543.00 | 561.76 | 545.04 | SL hit (close<static) qty=1.00 sl=543.10 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 13:15:00 | 510.60 | 539.93 | 540.07 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 13:15:00 | 576.00 | 537.64 | 537.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 09:15:00 | 578.20 | 545.20 | 541.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 11:15:00 | 545.25 | 547.58 | 543.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-29 12:00:00 | 545.25 | 547.58 | 543.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 543.45 | 547.84 | 543.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 12:00:00 | 543.45 | 547.84 | 543.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 542.45 | 547.78 | 543.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 13:15:00 | 547.65 | 547.78 | 543.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-08 12:15:00 | 602.42 | 552.38 | 546.52 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 14:15:00 | 537.65 | 561.50 | 561.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 15:15:00 | 536.20 | 561.25 | 561.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 550.15 | 549.67 | 554.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 15:00:00 | 550.15 | 549.67 | 554.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 565.70 | 549.84 | 554.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:00:00 | 565.70 | 549.84 | 554.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 562.00 | 549.96 | 554.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 12:00:00 | 559.40 | 550.05 | 554.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 13:45:00 | 561.85 | 550.26 | 554.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 10:15:00 | 573.20 | 550.90 | 555.00 | SL hit (close>static) qty=1.00 sl=566.20 alert=retest2 |

### Cycle 5 — BUY (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 15:15:00 | 598.00 | 558.62 | 558.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 09:15:00 | 609.55 | 561.25 | 559.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 12:15:00 | 600.15 | 601.83 | 584.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-13 13:00:00 | 600.15 | 601.83 | 584.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 597.00 | 619.94 | 600.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 618.20 | 616.94 | 600.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 10:30:00 | 612.10 | 618.98 | 603.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 13:30:00 | 614.80 | 618.99 | 603.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-10 12:15:00 | 673.31 | 621.85 | 606.03 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-28 09:15:00 | 547.50 | 2025-07-28 12:15:00 | 543.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-07-29 13:45:00 | 549.05 | 2025-08-01 14:15:00 | 527.10 | STOP_HIT | 1.00 | -4.00% |
| BUY | retest2 | 2025-08-18 14:15:00 | 550.25 | 2025-08-20 15:15:00 | 542.75 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-08-21 09:15:00 | 547.85 | 2025-08-21 12:15:00 | 542.75 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-10-03 13:15:00 | 547.65 | 2025-10-08 12:15:00 | 602.42 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-11 09:30:00 | 546.80 | 2025-11-18 13:15:00 | 601.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-03 10:00:00 | 545.00 | 2025-12-04 12:15:00 | 538.95 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-12-04 09:45:00 | 544.55 | 2025-12-04 12:15:00 | 538.95 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-12-22 12:00:00 | 559.40 | 2025-12-23 10:15:00 | 573.20 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-12-22 13:45:00 | 561.85 | 2025-12-23 10:15:00 | 573.20 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2026-02-03 09:15:00 | 618.20 | 2026-02-10 12:15:00 | 673.31 | TARGET_HIT | 1.00 | 8.91% |
| BUY | retest2 | 2026-02-06 10:30:00 | 612.10 | 2026-02-11 09:15:00 | 676.28 | TARGET_HIT | 1.00 | 10.49% |
| BUY | retest2 | 2026-02-06 13:30:00 | 614.80 | 2026-02-17 09:15:00 | 680.02 | TARGET_HIT | 1.00 | 10.61% |
| BUY | retest2 | 2026-03-16 09:30:00 | 614.05 | 2026-03-23 09:15:00 | 575.50 | STOP_HIT | 1.00 | -6.28% |
| BUY | retest2 | 2026-04-15 10:00:00 | 663.40 | 2026-04-20 10:15:00 | 729.74 | TARGET_HIT | 1.00 | 10.00% |
