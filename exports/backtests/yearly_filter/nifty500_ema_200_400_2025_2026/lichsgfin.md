# LIC Housing Finance Ltd. (LICHSGFIN)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 581.85
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 22 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 25 |
| PARTIAL | 11 |
| TARGET_HIT | 3 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 14
- **Target hits / Stop hits / Partials:** 3 / 22 / 11
- **Avg / median % per leg:** 2.92% / 4.75%
- **Sum % (uncompounded):** 104.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.48% | -5.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.48% | -5.9% |
| SELL (all) | 32 | 22 | 68.8% | 3 | 18 | 11 | 3.46% | 110.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 32 | 22 | 68.8% | 3 | 18 | 11 | 3.46% | 110.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 36 | 22 | 61.1% | 3 | 22 | 11 | 2.92% | 105.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 574.40 | 600.84 | 600.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 11:15:00 | 573.50 | 600.56 | 600.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 573.30 | 569.89 | 579.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 10:00:00 | 573.30 | 569.89 | 579.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 585.50 | 570.39 | 578.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:00:00 | 585.50 | 570.39 | 578.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 588.90 | 570.58 | 578.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:00:00 | 588.90 | 570.58 | 578.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 580.50 | 574.08 | 579.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:15:00 | 581.15 | 574.08 | 579.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 582.25 | 574.16 | 579.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:30:00 | 582.55 | 574.16 | 579.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 581.70 | 574.57 | 580.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:30:00 | 580.70 | 574.91 | 580.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 15:15:00 | 580.00 | 574.91 | 580.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 09:30:00 | 580.00 | 573.50 | 578.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 09:15:00 | 585.45 | 571.95 | 575.58 | SL hit (close>static) qty=1.00 sl=584.50 alert=retest2 |

### Cycle 2 — BUY (started 2026-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 15:15:00 | 564.80 | 520.52 | 520.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 578.15 | 535.47 | 528.95 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 09:15:00 | 593.65 | 2025-08-07 10:15:00 | 574.40 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2025-08-04 09:15:00 | 585.25 | 2025-08-07 10:15:00 | 574.40 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-08-06 10:15:00 | 577.25 | 2025-08-07 10:15:00 | 574.40 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-08-07 09:15:00 | 576.20 | 2025-08-07 10:15:00 | 574.40 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-09-24 14:30:00 | 580.70 | 2025-10-24 09:15:00 | 585.45 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-09-24 15:15:00 | 580.00 | 2025-10-24 09:15:00 | 585.45 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-10-03 09:30:00 | 580.00 | 2025-10-24 09:15:00 | 585.45 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-10-24 13:15:00 | 580.00 | 2025-10-27 09:15:00 | 584.90 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-10-30 11:30:00 | 571.75 | 2025-12-05 10:15:00 | 543.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 12:00:00 | 570.00 | 2025-12-05 10:15:00 | 542.92 | PARTIAL | 0.50 | 4.75% |
| SELL | retest2 | 2025-10-31 09:30:00 | 571.50 | 2025-12-05 10:15:00 | 542.54 | PARTIAL | 0.50 | 5.07% |
| SELL | retest2 | 2025-10-31 10:15:00 | 571.10 | 2025-12-05 10:15:00 | 542.02 | PARTIAL | 0.50 | 5.09% |
| SELL | retest2 | 2025-11-06 09:15:00 | 568.50 | 2025-12-05 11:15:00 | 541.50 | PARTIAL | 0.50 | 4.75% |
| SELL | retest2 | 2025-11-10 13:00:00 | 569.70 | 2025-12-05 14:15:00 | 540.07 | PARTIAL | 0.50 | 5.20% |
| SELL | retest2 | 2025-11-11 09:15:00 | 568.65 | 2025-12-05 14:15:00 | 541.22 | PARTIAL | 0.50 | 4.82% |
| SELL | retest2 | 2025-11-13 09:45:00 | 570.55 | 2025-12-05 14:15:00 | 540.22 | PARTIAL | 0.50 | 5.32% |
| SELL | retest2 | 2025-10-30 11:30:00 | 571.75 | 2026-01-02 09:15:00 | 543.45 | STOP_HIT | 0.50 | 4.95% |
| SELL | retest2 | 2025-10-30 12:00:00 | 570.00 | 2026-01-02 09:15:00 | 543.45 | STOP_HIT | 0.50 | 4.66% |
| SELL | retest2 | 2025-10-31 09:30:00 | 571.50 | 2026-01-02 09:15:00 | 543.45 | STOP_HIT | 0.50 | 4.91% |
| SELL | retest2 | 2025-10-31 10:15:00 | 571.10 | 2026-01-02 09:15:00 | 543.45 | STOP_HIT | 0.50 | 4.84% |
| SELL | retest2 | 2025-11-06 09:15:00 | 568.50 | 2026-01-02 09:15:00 | 543.45 | STOP_HIT | 0.50 | 4.41% |
| SELL | retest2 | 2025-11-10 13:00:00 | 569.70 | 2026-01-02 09:15:00 | 543.45 | STOP_HIT | 0.50 | 4.61% |
| SELL | retest2 | 2025-11-11 09:15:00 | 568.65 | 2026-01-02 09:15:00 | 543.45 | STOP_HIT | 0.50 | 4.43% |
| SELL | retest2 | 2025-11-13 09:45:00 | 570.55 | 2026-01-02 09:15:00 | 543.45 | STOP_HIT | 0.50 | 4.75% |
| SELL | retest2 | 2026-02-24 13:30:00 | 527.15 | 2026-02-24 14:15:00 | 531.85 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2026-03-02 12:30:00 | 525.15 | 2026-03-09 09:15:00 | 498.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 14:15:00 | 526.75 | 2026-03-09 09:15:00 | 500.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 14:45:00 | 527.05 | 2026-03-09 09:15:00 | 500.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 12:30:00 | 525.15 | 2026-03-23 09:15:00 | 472.63 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-02 14:15:00 | 526.75 | 2026-03-23 09:15:00 | 474.07 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-02 14:45:00 | 527.05 | 2026-03-23 09:15:00 | 474.34 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-01 13:30:00 | 514.10 | 2026-04-06 13:15:00 | 518.55 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2026-04-01 14:30:00 | 513.80 | 2026-04-06 13:15:00 | 518.55 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-04-02 09:15:00 | 504.95 | 2026-04-06 13:15:00 | 518.55 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2026-04-02 14:30:00 | 513.65 | 2026-04-06 13:15:00 | 518.55 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-04-07 09:15:00 | 515.00 | 2026-04-08 09:15:00 | 524.35 | STOP_HIT | 1.00 | -1.82% |
