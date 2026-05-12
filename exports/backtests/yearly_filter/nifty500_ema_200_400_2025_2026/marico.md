# Marico Ltd. (MARICO)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 830.50
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
| ALERT2_SKIP | 2 |
| ALERT3 | 57 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 39 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 39 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 39 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 35
- **Target hits / Stop hits / Partials:** 0 / 39 / 0
- **Avg / median % per leg:** -1.44% / -1.46%
- **Sum % (uncompounded):** -56.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 37 | 4 | 10.8% | 0 | 37 | 0 | -1.46% | -54.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 37 | 4 | 10.8% | 0 | 37 | 0 | -1.46% | -54.0% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.14% | -2.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.14% | -2.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 39 | 4 | 10.3% | 0 | 39 | 0 | -1.44% | -56.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 10:15:00 | 693.70 | 716.49 | 716.58 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 10:15:00 | 731.10 | 716.12 | 716.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 735.10 | 716.99 | 716.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 13:15:00 | 717.75 | 718.49 | 717.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 13:15:00 | 717.75 | 718.49 | 717.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 717.75 | 718.49 | 717.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:45:00 | 718.00 | 718.49 | 717.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 720.00 | 718.50 | 717.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:30:00 | 716.35 | 718.50 | 717.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 719.70 | 718.72 | 717.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 719.70 | 718.72 | 717.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 719.75 | 719.15 | 717.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 13:00:00 | 725.40 | 718.11 | 717.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 13:00:00 | 728.00 | 718.33 | 717.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 14:00:00 | 727.80 | 718.43 | 717.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 12:00:00 | 725.00 | 728.89 | 724.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 721.35 | 728.82 | 724.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:00:00 | 721.35 | 728.82 | 724.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 724.20 | 728.77 | 724.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 15:00:00 | 727.95 | 728.76 | 724.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 13:15:00 | 718.00 | 728.41 | 724.14 | SL hit (close<static) qty=1.00 sl=720.55 alert=retest2 |

### Cycle 3 — SELL (started 2026-04-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 14:15:00 | 753.15 | 755.64 | 755.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-08 11:15:00 | 748.00 | 755.52 | 755.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 09:15:00 | 756.35 | 754.93 | 755.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 09:15:00 | 756.35 | 754.93 | 755.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 756.35 | 754.93 | 755.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 10:00:00 | 756.35 | 754.93 | 755.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 755.90 | 754.94 | 755.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 15:00:00 | 753.10 | 755.41 | 755.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 09:30:00 | 748.25 | 755.35 | 755.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 09:15:00 | 759.20 | 754.88 | 755.21 | SL hit (close>static) qty=1.00 sl=758.00 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 15:15:00 | 762.80 | 755.53 | 755.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 10:15:00 | 767.60 | 755.73 | 755.63 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-24 09:15:00 | 700.00 | 2025-08-08 13:15:00 | 706.00 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest2 | 2025-07-28 09:15:00 | 698.75 | 2025-08-08 14:15:00 | 703.55 | STOP_HIT | 1.00 | 0.69% |
| BUY | retest2 | 2025-07-28 14:15:00 | 695.35 | 2025-08-08 14:15:00 | 703.55 | STOP_HIT | 1.00 | 1.18% |
| BUY | retest2 | 2025-07-29 12:45:00 | 695.20 | 2025-08-08 14:15:00 | 703.55 | STOP_HIT | 1.00 | 1.20% |
| BUY | retest2 | 2025-07-31 11:15:00 | 712.20 | 2025-08-08 14:15:00 | 703.55 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-07-31 13:15:00 | 712.10 | 2025-08-29 09:15:00 | 711.80 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2025-08-01 09:15:00 | 721.50 | 2025-09-17 11:15:00 | 711.55 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-08-04 10:00:00 | 712.60 | 2025-09-17 11:15:00 | 711.55 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2025-08-07 15:00:00 | 718.75 | 2025-09-17 11:15:00 | 711.55 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-08-18 09:15:00 | 719.50 | 2025-09-22 13:15:00 | 716.30 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-08-18 12:30:00 | 719.25 | 2025-09-23 13:15:00 | 706.15 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-08-19 11:30:00 | 720.40 | 2025-09-23 13:15:00 | 706.15 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-08-29 09:15:00 | 720.00 | 2025-09-23 13:15:00 | 706.15 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-08-29 11:00:00 | 717.75 | 2025-10-01 09:15:00 | 692.15 | STOP_HIT | 1.00 | -3.57% |
| BUY | retest2 | 2025-09-17 10:15:00 | 717.10 | 2025-10-01 09:15:00 | 692.15 | STOP_HIT | 1.00 | -3.48% |
| BUY | retest2 | 2025-09-17 10:45:00 | 718.50 | 2025-10-01 09:15:00 | 692.15 | STOP_HIT | 1.00 | -3.67% |
| BUY | retest2 | 2025-09-22 10:15:00 | 723.60 | 2025-10-01 09:15:00 | 692.15 | STOP_HIT | 1.00 | -4.35% |
| BUY | retest2 | 2025-11-13 13:00:00 | 725.40 | 2025-11-28 13:15:00 | 718.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-11-14 13:00:00 | 728.00 | 2025-12-01 09:15:00 | 714.15 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-11-14 14:00:00 | 727.80 | 2025-12-01 09:15:00 | 714.15 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-11-27 12:00:00 | 725.00 | 2025-12-01 09:15:00 | 714.15 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-11-27 15:00:00 | 727.95 | 2025-12-01 09:15:00 | 714.15 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-12-05 11:30:00 | 725.90 | 2026-01-30 12:15:00 | 721.65 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-12-09 10:00:00 | 726.85 | 2026-01-30 12:15:00 | 721.65 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-12-09 11:00:00 | 727.35 | 2026-01-30 12:15:00 | 721.65 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-12-11 10:30:00 | 727.00 | 2026-01-30 12:15:00 | 721.65 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-12-11 15:15:00 | 727.10 | 2026-02-01 14:15:00 | 719.10 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-12-12 11:15:00 | 726.25 | 2026-02-01 14:15:00 | 719.10 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-12-12 12:00:00 | 727.00 | 2026-02-01 14:15:00 | 719.10 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-12-12 14:30:00 | 726.65 | 2026-02-02 10:15:00 | 710.55 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-12-12 15:00:00 | 727.25 | 2026-02-02 10:15:00 | 710.55 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-12-15 09:15:00 | 727.05 | 2026-02-02 10:15:00 | 710.55 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2026-01-28 10:45:00 | 727.35 | 2026-02-02 10:15:00 | 710.55 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2026-03-12 12:00:00 | 761.65 | 2026-03-16 09:15:00 | 744.15 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2026-03-12 12:45:00 | 762.35 | 2026-03-16 09:15:00 | 744.15 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2026-03-12 13:15:00 | 762.30 | 2026-03-16 09:15:00 | 744.15 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2026-03-13 09:15:00 | 764.00 | 2026-03-16 09:15:00 | 744.15 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2026-04-13 15:00:00 | 753.10 | 2026-04-20 09:15:00 | 759.20 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2026-04-16 09:30:00 | 748.25 | 2026-04-20 09:15:00 | 759.20 | STOP_HIT | 1.00 | -1.46% |
