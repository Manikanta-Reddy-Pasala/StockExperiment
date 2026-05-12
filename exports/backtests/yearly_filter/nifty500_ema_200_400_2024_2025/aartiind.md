# Aarti Industries Ltd. (AARTIIND)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 486.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 0 |
| ALERT3 | 65 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 52 |
| PARTIAL | 13 |
| TARGET_HIT | 11 |
| STOP_HIT | 42 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 65 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 35
- **Target hits / Stop hits / Partials:** 11 / 41 / 13
- **Avg / median % per leg:** 0.88% / -1.08%
- **Sum % (uncompounded):** 57.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 27 | 5 | 18.5% | 5 | 22 | 0 | -0.51% | -13.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 27 | 5 | 18.5% | 5 | 22 | 0 | -0.51% | -13.8% |
| SELL (all) | 38 | 25 | 65.8% | 6 | 19 | 13 | 1.87% | 70.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 38 | 25 | 65.8% | 6 | 19 | 13 | 1.87% | 70.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 65 | 30 | 46.2% | 11 | 41 | 13 | 0.88% | 57.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 10:15:00 | 639.95 | 676.56 | 676.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 11:15:00 | 636.65 | 676.16 | 676.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-10 12:15:00 | 654.60 | 648.33 | 659.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-10 13:00:00 | 654.60 | 648.33 | 659.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 13:15:00 | 667.50 | 648.52 | 659.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:00:00 | 667.50 | 648.52 | 659.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 664.00 | 648.68 | 659.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-11 09:30:00 | 660.10 | 648.96 | 659.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-11 10:30:00 | 661.05 | 649.12 | 659.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-11 12:45:00 | 661.90 | 649.41 | 659.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-11 14:30:00 | 661.85 | 649.64 | 659.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 660.15 | 649.83 | 659.77 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-06-13 12:15:00 | 675.00 | 651.46 | 660.12 | SL hit (close>static) qty=1.00 sl=672.80 alert=retest2 |

### Cycle 2 — BUY (started 2024-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 14:15:00 | 705.35 | 666.87 | 666.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 09:15:00 | 718.00 | 675.40 | 671.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 688.10 | 688.16 | 679.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-10 10:00:00 | 688.10 | 688.16 | 679.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 685.10 | 693.18 | 683.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:00:00 | 685.10 | 693.18 | 683.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 11:15:00 | 684.60 | 693.09 | 683.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:45:00 | 680.25 | 693.09 | 683.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 12:15:00 | 680.00 | 692.96 | 683.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 13:00:00 | 680.00 | 692.96 | 683.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 13:15:00 | 678.05 | 692.81 | 683.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 14:00:00 | 678.05 | 692.81 | 683.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 14:15:00 | 672.45 | 692.61 | 683.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 15:00:00 | 672.45 | 692.61 | 683.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 683.95 | 692.07 | 683.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 12:45:00 | 689.45 | 692.05 | 683.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 10:15:00 | 687.45 | 692.04 | 683.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 12:15:00 | 677.30 | 691.94 | 683.86 | SL hit (close<static) qty=1.00 sl=681.55 alert=retest2 |

### Cycle 3 — SELL (started 2024-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 13:15:00 | 622.30 | 688.44 | 688.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 15:15:00 | 620.10 | 687.09 | 688.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 12:15:00 | 425.80 | 425.31 | 456.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-10 12:30:00 | 423.10 | 425.31 | 456.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 453.70 | 427.67 | 450.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 09:30:00 | 462.50 | 427.67 | 450.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 454.15 | 429.35 | 451.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 441.45 | 429.35 | 451.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 448.35 | 430.11 | 450.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:45:00 | 450.35 | 430.11 | 450.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 448.80 | 430.77 | 450.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 14:45:00 | 450.05 | 430.77 | 450.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 438.55 | 431.01 | 450.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 14:30:00 | 437.15 | 431.48 | 450.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 09:15:00 | 429.25 | 431.56 | 450.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 415.29 | 430.69 | 448.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 10:15:00 | 407.79 | 430.49 | 448.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-30 09:15:00 | 437.10 | 429.43 | 447.01 | SL hit (close>ema200) qty=0.50 sl=429.43 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 09:15:00 | 440.85 | 414.91 | 414.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 10:15:00 | 448.20 | 415.24 | 414.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 10:15:00 | 464.65 | 466.01 | 450.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 11:15:00 | 465.60 | 466.01 | 450.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 453.60 | 465.66 | 451.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:45:00 | 450.65 | 465.66 | 451.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 453.95 | 464.70 | 451.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:45:00 | 452.90 | 464.70 | 451.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 453.55 | 464.59 | 451.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:30:00 | 453.40 | 464.59 | 451.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 451.95 | 464.47 | 451.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 13:00:00 | 451.95 | 464.47 | 451.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 450.75 | 464.33 | 451.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 14:00:00 | 450.75 | 464.33 | 451.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 450.30 | 464.19 | 451.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 09:30:00 | 452.95 | 463.89 | 451.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 10:15:00 | 445.80 | 463.71 | 451.71 | SL hit (close<static) qty=1.00 sl=448.15 alert=retest2 |

### Cycle 5 — SELL (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 15:15:00 | 433.35 | 452.93 | 452.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 429.80 | 452.69 | 452.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 13:15:00 | 450.00 | 449.29 | 450.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-30 13:30:00 | 449.50 | 449.29 | 450.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 388.20 | 381.25 | 389.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:45:00 | 388.05 | 381.25 | 389.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 389.00 | 381.33 | 389.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 14:45:00 | 387.25 | 381.62 | 389.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 10:00:00 | 386.70 | 381.74 | 389.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 416.10 | 382.58 | 389.21 | SL hit (close>static) qty=1.00 sl=392.60 alert=retest2 |

### Cycle 6 — BUY (started 2026-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 12:15:00 | 447.85 | 374.88 | 374.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-05 14:15:00 | 454.90 | 376.39 | 375.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 417.15 | 429.09 | 410.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-04 10:00:00 | 417.15 | 429.09 | 410.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 15:15:00 | 408.15 | 428.33 | 410.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 09:30:00 | 416.25 | 428.17 | 410.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 14:45:00 | 422.45 | 427.45 | 410.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 13:45:00 | 415.95 | 426.89 | 410.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 09:45:00 | 415.05 | 425.38 | 410.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-12 12:15:00 | 456.56 | 426.31 | 412.42 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-13 10:45:00 | 654.75 | 2024-05-15 10:15:00 | 626.20 | STOP_HIT | 1.00 | -4.36% |
| BUY | retest2 | 2024-05-14 12:45:00 | 655.00 | 2024-05-15 10:15:00 | 626.20 | STOP_HIT | 1.00 | -4.40% |
| BUY | retest2 | 2024-05-24 09:15:00 | 654.15 | 2024-05-24 10:15:00 | 639.95 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2024-06-11 09:30:00 | 660.10 | 2024-06-13 12:15:00 | 675.00 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2024-06-11 10:30:00 | 661.05 | 2024-06-13 12:15:00 | 675.00 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2024-06-11 12:45:00 | 661.90 | 2024-06-13 12:15:00 | 675.00 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2024-06-11 14:30:00 | 661.85 | 2024-06-13 12:15:00 | 675.00 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-07-22 12:45:00 | 689.45 | 2024-07-23 12:15:00 | 677.30 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-07-23 10:15:00 | 687.45 | 2024-07-23 12:15:00 | 677.30 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2024-07-23 15:00:00 | 686.90 | 2024-08-01 09:15:00 | 755.59 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-24 09:15:00 | 694.20 | 2024-08-08 09:15:00 | 763.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-26 09:15:00 | 702.45 | 2024-08-13 09:15:00 | 641.60 | STOP_HIT | 1.00 | -8.66% |
| SELL | retest2 | 2025-01-24 14:30:00 | 437.15 | 2025-01-28 09:15:00 | 415.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-27 09:15:00 | 429.25 | 2025-01-28 10:15:00 | 407.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 14:30:00 | 437.15 | 2025-01-30 09:15:00 | 437.10 | STOP_HIT | 0.50 | 0.01% |
| SELL | retest2 | 2025-01-27 09:15:00 | 429.25 | 2025-01-30 09:15:00 | 437.10 | STOP_HIT | 0.50 | -1.83% |
| SELL | retest2 | 2025-01-30 10:00:00 | 437.10 | 2025-02-01 13:15:00 | 461.55 | STOP_HIT | 1.00 | -5.59% |
| SELL | retest2 | 2025-01-30 10:30:00 | 435.30 | 2025-02-01 13:15:00 | 461.55 | STOP_HIT | 1.00 | -6.03% |
| SELL | retest2 | 2025-02-12 09:15:00 | 433.65 | 2025-02-14 10:15:00 | 415.25 | PARTIAL | 0.50 | 4.24% |
| SELL | retest2 | 2025-02-13 10:15:00 | 437.10 | 2025-02-14 10:15:00 | 415.48 | PARTIAL | 0.50 | 4.95% |
| SELL | retest2 | 2025-02-13 13:15:00 | 437.35 | 2025-02-14 12:15:00 | 411.97 | PARTIAL | 0.50 | 5.80% |
| SELL | retest2 | 2025-02-12 09:15:00 | 433.65 | 2025-02-27 12:15:00 | 393.62 | TARGET_HIT | 0.50 | 9.23% |
| SELL | retest2 | 2025-02-13 10:15:00 | 437.10 | 2025-02-27 13:15:00 | 393.39 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-13 13:15:00 | 437.35 | 2025-02-28 09:15:00 | 390.28 | TARGET_HIT | 0.50 | 10.76% |
| SELL | retest2 | 2025-04-22 11:00:00 | 438.45 | 2025-05-05 15:15:00 | 451.90 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2025-06-19 09:30:00 | 452.95 | 2025-06-19 10:15:00 | 445.80 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-06-24 09:15:00 | 461.10 | 2025-07-11 10:15:00 | 445.50 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest2 | 2025-07-10 12:15:00 | 452.20 | 2025-07-11 10:15:00 | 445.50 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-07-15 13:00:00 | 452.15 | 2025-07-18 09:15:00 | 448.20 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-07-16 12:15:00 | 457.00 | 2025-07-18 09:15:00 | 448.20 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-07-16 13:15:00 | 457.10 | 2025-07-18 09:15:00 | 448.20 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-07-17 09:15:00 | 457.50 | 2025-07-18 09:15:00 | 448.20 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-07-17 11:00:00 | 456.85 | 2025-07-18 10:15:00 | 446.25 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-11-03 14:45:00 | 387.25 | 2025-11-07 09:15:00 | 416.10 | STOP_HIT | 1.00 | -7.45% |
| SELL | retest2 | 2025-11-04 10:00:00 | 386.70 | 2025-11-07 09:15:00 | 416.10 | STOP_HIT | 1.00 | -7.60% |
| SELL | retest2 | 2025-11-07 15:15:00 | 385.05 | 2025-11-10 09:15:00 | 396.15 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-11-11 09:15:00 | 386.75 | 2025-11-12 09:15:00 | 395.20 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-11-20 09:30:00 | 389.20 | 2025-12-03 13:15:00 | 370.02 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2025-11-20 11:00:00 | 389.50 | 2025-12-03 14:15:00 | 369.74 | PARTIAL | 0.50 | 5.07% |
| SELL | retest2 | 2025-11-20 11:30:00 | 389.05 | 2025-12-03 15:15:00 | 369.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 09:30:00 | 389.20 | 2025-12-08 15:15:00 | 350.28 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-20 11:00:00 | 389.50 | 2025-12-08 15:15:00 | 350.55 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-20 11:30:00 | 389.05 | 2025-12-08 15:15:00 | 350.15 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-23 09:45:00 | 389.40 | 2025-12-30 11:15:00 | 369.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-23 09:45:00 | 389.40 | 2025-12-31 11:15:00 | 374.55 | STOP_HIT | 0.50 | 3.81% |
| SELL | retest2 | 2026-01-05 10:45:00 | 374.35 | 2026-01-12 09:15:00 | 355.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 12:15:00 | 374.45 | 2026-01-12 09:15:00 | 355.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 09:30:00 | 374.20 | 2026-01-12 09:15:00 | 355.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 10:15:00 | 373.70 | 2026-01-12 09:15:00 | 355.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 10:45:00 | 374.35 | 2026-01-30 10:15:00 | 372.45 | STOP_HIT | 0.50 | 0.51% |
| SELL | retest2 | 2026-01-05 12:15:00 | 374.45 | 2026-01-30 10:15:00 | 372.45 | STOP_HIT | 0.50 | 0.53% |
| SELL | retest2 | 2026-01-07 09:30:00 | 374.20 | 2026-01-30 10:15:00 | 372.45 | STOP_HIT | 0.50 | 0.47% |
| SELL | retest2 | 2026-01-07 10:15:00 | 373.70 | 2026-01-30 10:15:00 | 372.45 | STOP_HIT | 0.50 | 0.33% |
| SELL | retest2 | 2026-02-02 09:15:00 | 361.80 | 2026-02-03 09:15:00 | 415.35 | STOP_HIT | 1.00 | -14.80% |
| BUY | retest2 | 2026-03-05 09:30:00 | 416.25 | 2026-03-12 12:15:00 | 456.56 | TARGET_HIT | 1.00 | 9.68% |
| BUY | retest2 | 2026-03-05 14:45:00 | 422.45 | 2026-03-19 14:15:00 | 411.45 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2026-03-06 13:45:00 | 415.95 | 2026-03-19 14:15:00 | 411.45 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2026-03-10 09:45:00 | 415.05 | 2026-03-19 14:15:00 | 411.45 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2026-03-16 11:15:00 | 419.00 | 2026-03-23 09:15:00 | 410.65 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2026-03-16 11:45:00 | 424.55 | 2026-03-30 14:15:00 | 399.50 | STOP_HIT | 1.00 | -5.90% |
| BUY | retest2 | 2026-03-17 13:00:00 | 417.75 | 2026-03-30 14:15:00 | 399.50 | STOP_HIT | 1.00 | -4.37% |
| BUY | retest2 | 2026-03-20 09:15:00 | 418.15 | 2026-03-30 14:15:00 | 399.50 | STOP_HIT | 1.00 | -4.46% |
| BUY | retest2 | 2026-04-01 12:00:00 | 414.65 | 2026-04-02 09:15:00 | 398.85 | STOP_HIT | 1.00 | -3.81% |
| BUY | retest2 | 2026-04-08 09:15:00 | 416.50 | 2026-04-22 09:15:00 | 455.90 | TARGET_HIT | 1.00 | 9.46% |
| BUY | retest2 | 2026-04-13 09:30:00 | 414.45 | 2026-04-22 13:15:00 | 458.15 | TARGET_HIT | 1.00 | 10.54% |
