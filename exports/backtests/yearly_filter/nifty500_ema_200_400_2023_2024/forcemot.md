# Force Motors Ltd. (FORCEMOT)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (6521 bars)
- **Last close:** 20851.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 2 |
| ALERT3 | 45 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 30 |
| PARTIAL | 9 |
| TARGET_HIT | 9 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 39 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 27 / 12
- **Target hits / Stop hits / Partials:** 9 / 21 / 9
- **Avg / median % per leg:** 3.20% / 3.63%
- **Sum % (uncompounded):** 124.95%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 9 | 47.4% | 9 | 10 | 0 | 3.46% | 65.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 19 | 9 | 47.4% | 9 | 10 | 0 | 3.46% | 65.8% |
| SELL (all) | 20 | 18 | 90.0% | 0 | 11 | 9 | 2.96% | 59.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 20 | 18 | 90.0% | 0 | 11 | 9 | 2.96% | 59.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 39 | 27 | 69.2% | 9 | 21 | 9 | 3.20% | 124.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 10:15:00 | 8169.10 | 8491.02 | 8491.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 14:15:00 | 8127.45 | 8442.86 | 8465.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 10:15:00 | 7604.00 | 7577.91 | 7915.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-27 10:45:00 | 7630.10 | 7577.91 | 7915.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 7661.90 | 6964.41 | 7358.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 10:00:00 | 7661.90 | 6964.41 | 7358.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 10:15:00 | 7661.90 | 6971.35 | 7359.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 10:30:00 | 7661.90 | 6971.35 | 7359.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 15:15:00 | 7361.00 | 7103.28 | 7390.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:15:00 | 7305.90 | 7103.28 | 7390.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 7366.20 | 7105.90 | 7390.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 09:15:00 | 7213.85 | 7192.48 | 7400.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 6853.16 | 7167.56 | 7372.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-28 09:15:00 | 7036.30 | 7012.26 | 7225.70 | SL hit (close>ema200) qty=0.50 sl=7012.26 alert=retest2 |

### Cycle 2 — BUY (started 2025-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 15:15:00 | 7426.00 | 6723.76 | 6722.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 7473.50 | 6731.22 | 6726.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 11:15:00 | 18770.00 | 18853.41 | 17174.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 11:45:00 | 18745.00 | 18853.41 | 17174.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 16975.00 | 18710.85 | 17230.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:00:00 | 16975.00 | 18710.85 | 17230.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 17019.00 | 18694.02 | 17229.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:45:00 | 16860.00 | 18694.02 | 17229.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 17005.00 | 18677.21 | 17228.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:15:00 | 17624.00 | 18677.21 | 17228.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 17735.00 | 18605.08 | 17714.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:00:00 | 17735.00 | 18605.08 | 17714.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 17758.00 | 18596.65 | 17714.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:15:00 | 17753.00 | 18596.65 | 17714.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 17789.00 | 18588.62 | 17714.78 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 14:15:00 | 16638.00 | 17198.00 | 17199.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-16 15:15:00 | 16530.00 | 17191.35 | 17195.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 09:15:00 | 17418.00 | 17193.61 | 17196.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 17418.00 | 17193.61 | 17196.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 17418.00 | 17193.61 | 17196.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:00:00 | 17418.00 | 17193.61 | 17196.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 17759.00 | 17199.23 | 17199.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:45:00 | 17759.00 | 17199.23 | 17199.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 11:15:00 | 17648.00 | 17203.70 | 17201.98 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-10-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 13:15:00 | 16339.00 | 17196.13 | 17199.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 13:15:00 | 16282.00 | 17140.94 | 17171.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 14:15:00 | 17121.00 | 17029.71 | 17109.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 14:15:00 | 17121.00 | 17029.71 | 17109.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 17121.00 | 17029.71 | 17109.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 15:00:00 | 17121.00 | 17029.71 | 17109.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 17269.00 | 17032.09 | 17109.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:15:00 | 17615.00 | 17032.09 | 17109.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 10:15:00 | 18332.00 | 17186.32 | 17183.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-06 09:15:00 | 18613.00 | 17249.31 | 17215.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 10:15:00 | 17408.00 | 17437.48 | 17325.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 10:30:00 | 17350.00 | 17437.48 | 17325.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 17307.00 | 17436.18 | 17325.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:30:00 | 17262.00 | 17436.18 | 17325.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 17275.00 | 17434.58 | 17325.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:30:00 | 17240.00 | 17434.58 | 17325.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 17300.00 | 17441.81 | 17338.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 17300.00 | 17441.81 | 17338.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 17427.00 | 17441.67 | 17338.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 12:00:00 | 17461.00 | 17441.86 | 17339.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 15:15:00 | 17452.00 | 17442.07 | 17341.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 17289.00 | 17453.94 | 17355.14 | SL hit (close<static) qty=1.00 sl=17301.00 alert=retest2 |

### Cycle 7 — SELL (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 12:15:00 | 19166.00 | 21237.66 | 21241.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 10:15:00 | 18956.00 | 21139.10 | 21191.04 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-29 12:00:00 | 8296.15 | 2024-06-04 11:15:00 | 7892.95 | STOP_HIT | 1.00 | -4.86% |
| BUY | retest2 | 2024-05-29 15:15:00 | 8367.50 | 2024-06-04 11:15:00 | 7892.95 | STOP_HIT | 1.00 | -5.67% |
| BUY | retest2 | 2024-06-04 11:30:00 | 8350.20 | 2024-06-11 13:15:00 | 9117.46 | TARGET_HIT | 1.00 | 9.19% |
| BUY | retest2 | 2024-06-04 13:15:00 | 8288.60 | 2024-06-12 09:15:00 | 9185.22 | TARGET_HIT | 1.00 | 10.82% |
| BUY | retest2 | 2024-06-05 10:30:00 | 8363.10 | 2024-06-12 09:15:00 | 9199.41 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-05 12:30:00 | 8364.00 | 2024-06-12 09:15:00 | 9200.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-11 09:15:00 | 8418.20 | 2024-07-30 14:15:00 | 9260.02 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-12 12:45:00 | 8355.45 | 2024-07-30 14:15:00 | 9191.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-29 09:15:00 | 8723.85 | 2024-07-30 14:15:00 | 9596.24 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-29 10:30:00 | 8615.30 | 2024-07-30 14:15:00 | 9476.83 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-05 11:45:00 | 8635.60 | 2024-08-06 13:15:00 | 8391.65 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2024-08-06 09:15:00 | 8628.90 | 2024-08-06 13:15:00 | 8391.65 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2024-11-11 09:15:00 | 7213.85 | 2024-11-13 09:15:00 | 6853.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 09:15:00 | 7213.85 | 2024-11-28 09:15:00 | 7036.30 | STOP_HIT | 0.50 | 2.46% |
| SELL | retest2 | 2025-01-02 14:15:00 | 7237.00 | 2025-01-03 09:15:00 | 7482.25 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2025-01-02 15:15:00 | 7181.00 | 2025-01-03 09:15:00 | 7482.25 | STOP_HIT | 1.00 | -4.20% |
| SELL | retest2 | 2025-01-06 09:15:00 | 7207.50 | 2025-01-06 14:15:00 | 6847.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-06 09:15:00 | 7207.50 | 2025-01-06 14:15:00 | 6850.15 | STOP_HIT | 0.50 | 4.96% |
| SELL | retest2 | 2025-01-07 15:00:00 | 6862.00 | 2025-01-10 09:15:00 | 6518.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 10:30:00 | 6824.55 | 2025-01-10 09:15:00 | 6483.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 09:30:00 | 6788.15 | 2025-01-10 09:15:00 | 6448.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-07 15:00:00 | 6862.00 | 2025-01-23 09:15:00 | 6744.95 | STOP_HIT | 0.50 | 1.71% |
| SELL | retest2 | 2025-01-08 10:30:00 | 6824.55 | 2025-01-23 09:15:00 | 6744.95 | STOP_HIT | 0.50 | 1.17% |
| SELL | retest2 | 2025-01-09 09:30:00 | 6788.15 | 2025-01-23 09:15:00 | 6744.95 | STOP_HIT | 0.50 | 0.64% |
| SELL | retest2 | 2025-02-05 11:45:00 | 6785.05 | 2025-02-10 09:15:00 | 6445.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-05 11:45:00 | 6785.05 | 2025-02-12 11:15:00 | 6707.70 | STOP_HIT | 0.50 | 1.14% |
| SELL | retest2 | 2025-02-13 12:30:00 | 6845.00 | 2025-02-14 10:15:00 | 6502.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 12:30:00 | 6845.00 | 2025-02-14 10:15:00 | 6596.70 | STOP_HIT | 0.50 | 3.63% |
| SELL | retest2 | 2025-02-24 09:15:00 | 6830.75 | 2025-02-28 09:15:00 | 6489.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-24 10:00:00 | 6881.20 | 2025-02-28 09:15:00 | 6537.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-24 09:15:00 | 6830.75 | 2025-02-28 11:15:00 | 6649.70 | STOP_HIT | 0.50 | 2.65% |
| SELL | retest2 | 2025-02-24 10:00:00 | 6881.20 | 2025-02-28 11:15:00 | 6649.70 | STOP_HIT | 0.50 | 3.36% |
| BUY | retest2 | 2025-11-18 12:00:00 | 17461.00 | 2025-11-21 09:15:00 | 17289.00 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-11-18 15:15:00 | 17452.00 | 2025-11-21 09:15:00 | 17289.00 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-11-28 09:15:00 | 17537.00 | 2025-12-05 09:15:00 | 17168.00 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-12-12 12:00:00 | 17515.00 | 2025-12-15 10:15:00 | 17300.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-12-15 10:15:00 | 17450.00 | 2025-12-15 12:15:00 | 17199.00 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-12-16 12:00:00 | 17423.00 | 2025-12-17 10:15:00 | 17181.00 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-12-19 09:15:00 | 17940.00 | 2025-12-30 14:15:00 | 19734.00 | TARGET_HIT | 1.00 | 10.00% |
