# Ather Energy Ltd. (ATHERENERG)

## Backtest Summary

- **Window:** 2025-05-06 09:15:00 → 2026-05-08 15:15:00 (1752 bars)
- **Last close:** 916.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 7 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 8 |
| PARTIAL | 0 |
| TARGET_HIT | 5 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 3
- **Target hits / Stop hits / Partials:** 5 / 3 / 0
- **Avg / median % per leg:** 5.53% / 10.00%
- **Sum % (uncompounded):** 44.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 5 | 62.5% | 5 | 3 | 0 | 5.53% | 44.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 5 | 62.5% | 5 | 3 | 0 | 5.53% | 44.2% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 8 | 5 | 62.5% | 5 | 3 | 0 | 5.53% | 44.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 13:15:00 | 596.40 | 651.00 | 651.17 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2026-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 12:15:00 | 696.35 | 650.88 | 650.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-05 14:15:00 | 713.90 | 652.01 | 651.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 12:15:00 | 686.70 | 690.55 | 675.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-24 13:00:00 | 686.70 | 690.55 | 675.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 12:15:00 | 679.15 | 692.54 | 678.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 12:45:00 | 679.00 | 692.54 | 678.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 13:15:00 | 680.00 | 692.41 | 678.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 13:45:00 | 676.30 | 692.41 | 678.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 680.10 | 692.48 | 678.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:45:00 | 676.55 | 692.48 | 678.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 677.00 | 692.32 | 678.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 11:30:00 | 677.00 | 692.32 | 678.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 12:15:00 | 677.95 | 692.18 | 678.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 12:45:00 | 676.50 | 692.18 | 678.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 13:15:00 | 682.30 | 692.08 | 678.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 14:45:00 | 698.95 | 692.18 | 678.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 682.45 | 691.95 | 679.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 09:45:00 | 683.75 | 691.89 | 679.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 14:15:00 | 675.00 | 691.43 | 679.17 | SL hit (close<static) qty=1.00 sl=676.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2026-03-04 14:45:00 | 698.95 | 2026-03-06 14:15:00 | 675.00 | STOP_HIT | 1.00 | -3.43% |
| BUY | retest2 | 2026-03-06 09:15:00 | 682.45 | 2026-03-06 14:15:00 | 675.00 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2026-03-06 09:45:00 | 683.75 | 2026-03-06 14:15:00 | 675.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2026-03-09 15:00:00 | 684.65 | 2026-03-18 09:15:00 | 753.12 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-10 11:15:00 | 697.20 | 2026-03-18 09:15:00 | 766.92 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-10 14:45:00 | 696.95 | 2026-03-18 09:15:00 | 766.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-10 15:15:00 | 697.00 | 2026-03-18 09:15:00 | 766.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-11 14:30:00 | 699.85 | 2026-03-18 09:15:00 | 769.84 | TARGET_HIT | 1.00 | 10.00% |
