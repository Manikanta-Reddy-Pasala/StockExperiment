# Aadhar Housing Finance Ltd. (AADHARHFC)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 502.75
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
| ALERT2_SKIP | 2 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 21 |
| PARTIAL | 6 |
| TARGET_HIT | 4 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 9
- **Target hits / Stop hits / Partials:** 4 / 17 / 6
- **Avg / median % per leg:** 2.46% / 2.81%
- **Sum % (uncompounded):** 66.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 6 | 42.9% | 4 | 10 | 0 | 1.68% | 23.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 6 | 42.9% | 4 | 10 | 0 | 1.68% | 23.6% |
| SELL (all) | 13 | 12 | 92.3% | 0 | 7 | 6 | 3.29% | 42.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 13 | 12 | 92.3% | 0 | 7 | 6 | 3.29% | 42.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 27 | 18 | 66.7% | 4 | 17 | 6 | 2.46% | 66.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 15:15:00 | 465.00 | 446.83 | 446.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 09:15:00 | 467.80 | 448.14 | 447.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 12:15:00 | 448.40 | 449.33 | 448.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 12:15:00 | 448.40 | 449.33 | 448.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 448.40 | 449.33 | 448.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:00:00 | 448.40 | 449.33 | 448.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 448.40 | 449.32 | 448.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:30:00 | 448.55 | 449.32 | 448.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 451.20 | 449.34 | 448.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 15:15:00 | 452.55 | 449.34 | 448.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 09:15:00 | 452.45 | 449.36 | 448.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 14:15:00 | 452.85 | 449.39 | 448.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 15:00:00 | 452.60 | 449.43 | 448.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-16 11:15:00 | 497.81 | 452.88 | 450.19 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 10:15:00 | 493.15 | 506.21 | 506.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 491.75 | 504.84 | 505.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 09:15:00 | 492.95 | 490.61 | 495.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 492.95 | 490.61 | 495.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 492.95 | 490.61 | 495.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:00:00 | 492.95 | 490.61 | 495.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 13:15:00 | 496.60 | 490.78 | 495.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 14:00:00 | 496.60 | 490.78 | 495.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 497.20 | 490.84 | 495.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 15:00:00 | 497.20 | 490.84 | 495.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 499.80 | 486.63 | 491.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:45:00 | 499.95 | 486.63 | 491.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 498.85 | 486.75 | 491.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 13:30:00 | 497.55 | 487.14 | 491.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 14:00:00 | 497.95 | 488.12 | 492.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 10:45:00 | 497.80 | 488.61 | 492.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 09:15:00 | 493.55 | 489.01 | 492.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 493.75 | 489.10 | 492.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 493.75 | 489.10 | 492.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 491.25 | 489.12 | 492.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:00:00 | 490.05 | 489.42 | 492.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 472.67 | 488.65 | 491.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 473.05 | 488.65 | 491.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 472.91 | 488.65 | 491.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 14:15:00 | 468.87 | 485.39 | 489.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-30 10:15:00 | 483.55 | 479.92 | 485.54 | SL hit (close>ema200) qty=0.50 sl=479.92 alert=retest2 |

### Cycle 3 — BUY (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 09:15:00 | 494.40 | 471.20 | 471.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 15:15:00 | 499.00 | 479.55 | 475.87 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-08 15:15:00 | 452.55 | 2025-07-16 11:15:00 | 497.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-10 09:15:00 | 452.45 | 2025-07-16 11:15:00 | 497.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-10 14:15:00 | 452.85 | 2025-07-16 11:15:00 | 498.14 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-10 15:00:00 | 452.60 | 2025-07-16 11:15:00 | 497.86 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-03 14:45:00 | 506.95 | 2025-11-04 09:15:00 | 508.85 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2025-10-17 14:30:00 | 507.25 | 2025-11-04 09:15:00 | 508.85 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2025-10-20 09:30:00 | 508.50 | 2025-11-06 09:15:00 | 501.50 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-10-23 14:30:00 | 507.05 | 2025-11-06 09:15:00 | 501.50 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-10-24 12:00:00 | 516.50 | 2025-11-06 09:15:00 | 501.50 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2025-10-27 13:45:00 | 514.35 | 2025-11-06 09:15:00 | 501.50 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2025-10-27 14:30:00 | 513.70 | 2025-11-06 09:15:00 | 501.50 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2025-10-28 09:15:00 | 514.55 | 2025-11-06 09:15:00 | 501.50 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-11-03 09:15:00 | 515.30 | 2025-11-06 09:15:00 | 501.50 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2025-11-03 13:45:00 | 510.00 | 2025-11-06 09:15:00 | 501.50 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2026-01-02 13:30:00 | 497.55 | 2026-01-12 09:15:00 | 472.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 14:00:00 | 497.95 | 2026-01-12 09:15:00 | 473.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 10:45:00 | 497.80 | 2026-01-12 09:15:00 | 472.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 09:15:00 | 493.55 | 2026-01-19 14:15:00 | 468.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-02 13:30:00 | 497.55 | 2026-01-30 10:15:00 | 483.55 | STOP_HIT | 0.50 | 2.81% |
| SELL | retest2 | 2026-01-05 14:00:00 | 497.95 | 2026-01-30 10:15:00 | 483.55 | STOP_HIT | 0.50 | 2.89% |
| SELL | retest2 | 2026-01-06 10:45:00 | 497.80 | 2026-01-30 10:15:00 | 483.55 | STOP_HIT | 0.50 | 2.86% |
| SELL | retest2 | 2026-01-07 09:15:00 | 493.55 | 2026-01-30 10:15:00 | 483.55 | STOP_HIT | 0.50 | 2.03% |
| SELL | retest2 | 2026-01-08 11:00:00 | 490.05 | 2026-02-13 09:15:00 | 465.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-01 09:15:00 | 479.30 | 2026-02-18 09:15:00 | 455.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:00:00 | 490.05 | 2026-02-19 09:15:00 | 474.70 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2026-02-01 09:15:00 | 479.30 | 2026-02-19 09:15:00 | 474.70 | STOP_HIT | 0.50 | 0.96% |
| SELL | retest2 | 2026-04-20 09:15:00 | 488.35 | 2026-04-21 09:15:00 | 497.50 | STOP_HIT | 1.00 | -1.87% |
