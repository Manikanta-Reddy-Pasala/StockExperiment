# Gillette India Ltd. (GILLETTE)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 8188.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 0 |
| ALERT3 | 32 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 28 |
| PARTIAL | 11 |
| TARGET_HIT | 8 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 39 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 24 / 15
- **Target hits / Stop hits / Partials:** 8 / 20 / 11
- **Avg / median % per leg:** 2.87% / 4.05%
- **Sum % (uncompounded):** 111.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 2 | 25.0% | 2 | 6 | 0 | 0.87% | 7.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 2 | 25.0% | 2 | 6 | 0 | 0.87% | 7.0% |
| SELL (all) | 31 | 22 | 71.0% | 6 | 14 | 11 | 3.38% | 104.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 31 | 22 | 71.0% | 6 | 14 | 11 | 3.38% | 104.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 39 | 24 | 61.5% | 8 | 20 | 11 | 2.87% | 111.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 09:15:00 | 6390.00 | 6528.15 | 6528.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-24 12:15:00 | 6354.20 | 6523.70 | 6526.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-30 09:15:00 | 6487.70 | 6468.30 | 6496.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-30 10:00:00 | 6487.70 | 6468.30 | 6496.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 10:15:00 | 6550.00 | 6469.11 | 6496.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 10:30:00 | 6554.90 | 6469.11 | 6496.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 11:15:00 | 6636.00 | 6470.77 | 6497.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 11:45:00 | 6649.65 | 6470.77 | 6497.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-07 10:15:00 | 6800.00 | 6523.57 | 6522.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-08 13:15:00 | 6872.75 | 6552.11 | 6537.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 11:15:00 | 6842.50 | 6860.49 | 6737.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-04 12:00:00 | 6842.50 | 6860.49 | 6737.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 7146.55 | 6863.34 | 6739.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 09:15:00 | 7195.00 | 6870.70 | 6744.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-11 11:15:00 | 7914.50 | 7055.29 | 6861.81 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 11:15:00 | 8677.20 | 9425.62 | 9426.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 15:15:00 | 8653.00 | 9351.43 | 9387.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-19 10:15:00 | 8763.40 | 8705.12 | 8988.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-19 10:45:00 | 8780.00 | 8705.12 | 8988.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 11:15:00 | 8727.90 | 8705.35 | 8987.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 13:30:00 | 8450.05 | 8704.71 | 8984.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 11:30:00 | 8515.60 | 8694.01 | 8971.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 09:30:00 | 8605.20 | 8655.38 | 8935.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 10:15:00 | 8580.05 | 8655.38 | 8935.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 15:15:00 | 8174.94 | 8615.82 | 8897.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 15:15:00 | 8151.05 | 8615.82 | 8897.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 11:15:00 | 8089.82 | 8603.66 | 8887.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 8027.55 | 8580.06 | 8868.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-03-03 11:15:00 | 7744.68 | 8517.37 | 8823.75 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 15:15:00 | 8750.00 | 8271.88 | 8270.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 8885.50 | 8277.99 | 8273.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 11:15:00 | 10605.00 | 10658.42 | 10129.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-28 12:00:00 | 10605.00 | 10658.42 | 10129.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 10250.00 | 10631.95 | 10262.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 10250.00 | 10631.95 | 10262.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 10250.00 | 10628.15 | 10262.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 09:15:00 | 10318.00 | 10628.15 | 10262.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 14:15:00 | 10000.00 | 10498.86 | 10302.75 | SL hit (close<static) qty=1.00 sl=10054.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 15:15:00 | 10000.00 | 10216.87 | 10217.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 9960.00 | 10214.31 | 10216.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 09:15:00 | 8834.00 | 8761.03 | 9152.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-27 09:30:00 | 8945.00 | 8761.03 | 9152.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 8223.00 | 8080.84 | 8332.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:00:00 | 8223.00 | 8080.84 | 8332.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 8300.00 | 8083.02 | 8332.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:45:00 | 8309.00 | 8083.02 | 8332.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 8275.00 | 8084.93 | 8332.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:15:00 | 8499.00 | 8084.93 | 8332.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 8610.00 | 8090.15 | 8333.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:45:00 | 8637.50 | 8090.15 | 8333.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 8594.00 | 8095.16 | 8334.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 09:15:00 | 8504.50 | 8293.44 | 8400.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 10:15:00 | 8652.00 | 8300.02 | 8402.91 | SL hit (close>static) qty=1.00 sl=8620.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-05 09:15:00 | 7195.00 | 2024-06-11 11:15:00 | 7914.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-05 11:30:00 | 7202.00 | 2024-07-18 10:15:00 | 7922.20 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-19 13:30:00 | 8450.05 | 2025-02-25 15:15:00 | 8174.94 | PARTIAL | 0.50 | 3.26% |
| SELL | retest2 | 2025-02-20 11:30:00 | 8515.60 | 2025-02-25 15:15:00 | 8151.05 | PARTIAL | 0.50 | 4.28% |
| SELL | retest2 | 2025-02-24 09:30:00 | 8605.20 | 2025-02-27 11:15:00 | 8089.82 | PARTIAL | 0.50 | 5.99% |
| SELL | retest2 | 2025-02-24 10:15:00 | 8580.05 | 2025-02-28 09:15:00 | 8027.55 | PARTIAL | 0.50 | 6.44% |
| SELL | retest2 | 2025-02-19 13:30:00 | 8450.05 | 2025-03-03 11:15:00 | 7744.68 | TARGET_HIT | 0.50 | 8.35% |
| SELL | retest2 | 2025-02-20 11:30:00 | 8515.60 | 2025-03-04 09:15:00 | 7722.04 | TARGET_HIT | 0.50 | 9.32% |
| SELL | retest2 | 2025-02-24 09:30:00 | 8605.20 | 2025-03-18 14:15:00 | 8232.95 | STOP_HIT | 0.50 | 4.33% |
| SELL | retest2 | 2025-02-24 10:15:00 | 8580.05 | 2025-03-18 14:15:00 | 8232.95 | STOP_HIT | 0.50 | 4.05% |
| SELL | retest2 | 2025-04-22 14:45:00 | 8086.00 | 2025-05-09 09:15:00 | 7709.72 | PARTIAL | 0.50 | 4.65% |
| SELL | retest2 | 2025-04-23 10:00:00 | 8081.00 | 2025-05-09 09:15:00 | 7699.75 | PARTIAL | 0.50 | 4.72% |
| SELL | retest2 | 2025-04-23 10:30:00 | 8115.50 | 2025-05-09 11:15:00 | 7681.70 | PARTIAL | 0.50 | 5.35% |
| SELL | retest2 | 2025-04-22 14:45:00 | 8086.00 | 2025-05-13 09:15:00 | 8026.50 | STOP_HIT | 0.50 | 0.74% |
| SELL | retest2 | 2025-04-23 10:00:00 | 8081.00 | 2025-05-13 09:15:00 | 8026.50 | STOP_HIT | 0.50 | 0.67% |
| SELL | retest2 | 2025-04-23 10:30:00 | 8115.50 | 2025-05-13 09:15:00 | 8026.50 | STOP_HIT | 0.50 | 1.10% |
| SELL | retest2 | 2025-04-23 13:30:00 | 8105.00 | 2025-05-16 10:15:00 | 8466.50 | STOP_HIT | 1.00 | -4.46% |
| BUY | retest2 | 2025-08-11 09:15:00 | 10318.00 | 2025-08-26 14:15:00 | 10000.00 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2025-09-01 09:30:00 | 10272.00 | 2025-09-01 14:15:00 | 10031.00 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-09-04 09:15:00 | 10285.00 | 2025-09-10 10:15:00 | 10208.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-09-09 09:45:00 | 10261.00 | 2025-09-11 15:15:00 | 10025.00 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-09-10 09:15:00 | 10363.00 | 2025-09-11 15:15:00 | 10025.00 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2025-09-19 15:00:00 | 10384.00 | 2025-09-19 15:15:00 | 10251.00 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2026-02-06 09:15:00 | 8504.50 | 2026-02-06 10:15:00 | 8652.00 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2026-02-13 09:45:00 | 8549.00 | 2026-02-13 11:15:00 | 8634.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2026-02-13 15:15:00 | 8528.00 | 2026-02-17 11:15:00 | 8674.00 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2026-02-19 12:45:00 | 8547.50 | 2026-02-23 12:15:00 | 8667.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-02-24 13:30:00 | 8417.50 | 2026-03-04 09:15:00 | 7996.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 14:15:00 | 8429.50 | 2026-03-04 09:15:00 | 8008.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 13:00:00 | 8427.00 | 2026-03-04 09:15:00 | 8005.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 10:15:00 | 8429.50 | 2026-03-04 09:15:00 | 8008.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 13:30:00 | 8417.50 | 2026-03-27 09:15:00 | 7575.75 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-24 14:15:00 | 8429.50 | 2026-03-27 09:15:00 | 7586.55 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-25 13:00:00 | 8427.00 | 2026-03-27 09:15:00 | 7584.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-26 10:15:00 | 8429.50 | 2026-03-27 09:15:00 | 7586.55 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-22 14:00:00 | 7921.50 | 2026-04-23 14:15:00 | 8082.00 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2026-05-06 11:00:00 | 7907.00 | 2026-05-08 11:15:00 | 8065.50 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2026-05-07 09:45:00 | 7898.00 | 2026-05-08 11:15:00 | 8065.50 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2026-05-07 11:15:00 | 7916.00 | 2026-05-08 11:15:00 | 8065.50 | STOP_HIT | 1.00 | -1.89% |
