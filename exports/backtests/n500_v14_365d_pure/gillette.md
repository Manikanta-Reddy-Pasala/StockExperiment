# Gillette India Ltd. (GILLETTE)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 8188.00
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
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 18 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 21 |
| PARTIAL | 4 |
| TARGET_HIT | 4 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 17
- **Target hits / Stop hits / Partials:** 4 / 17 / 4
- **Avg / median % per leg:** 1.13% / -1.41%
- **Sum % (uncompounded):** 28.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.17% | -13.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.17% | -13.0% |
| SELL (all) | 19 | 8 | 42.1% | 4 | 11 | 4 | 2.17% | 41.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 19 | 8 | 42.1% | 4 | 11 | 4 | 2.17% | 41.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 25 | 8 | 32.0% | 4 | 17 | 4 | 1.13% | 28.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 15:15:00 | 8750.00 | 8271.88 | 8270.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 8885.50 | 8277.99 | 8273.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 11:15:00 | 10605.00 | 10658.42 | 10129.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-28 12:00:00 | 10605.00 | 10658.42 | 10129.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 10250.00 | 10631.95 | 10262.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 10250.00 | 10631.95 | 10262.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 10250.00 | 10628.15 | 10262.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 09:15:00 | 10318.00 | 10628.15 | 10262.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 14:15:00 | 10000.00 | 10498.86 | 10302.74 | SL hit (close<static) qty=1.00 sl=10054.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 09:30:00 | 10272.00 | 10429.22 | 10281.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 14:15:00 | 10031.00 | 10412.85 | 10276.86 | SL hit (close<static) qty=1.00 sl=10054.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:15:00 | 10285.00 | 10377.61 | 10268.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 09:45:00 | 10261.00 | 10388.92 | 10286.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 10236.00 | 10387.40 | 10285.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:00:00 | 10236.00 | 10387.40 | 10285.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 10243.00 | 10385.96 | 10285.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:45:00 | 10231.00 | 10385.96 | 10285.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 10286.00 | 10382.47 | 10285.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:30:00 | 10309.00 | 10382.47 | 10285.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 10305.00 | 10381.70 | 10285.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 10363.00 | 10381.70 | 10285.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 10:15:00 | 10208.00 | 10378.98 | 10285.05 | SL hit (close<static) qty=1.00 sl=10273.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-11 15:15:00 | 10025.00 | 10348.54 | 10274.85 | SL hit (close<static) qty=1.00 sl=10054.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-11 15:15:00 | 10025.00 | 10348.54 | 10274.85 | SL hit (close<static) qty=1.00 sl=10054.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 15:00:00 | 10384.00 | 10234.52 | 10225.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 15:15:00 | 10251.00 | 10234.68 | 10225.87 | SL hit (close<static) qty=1.00 sl=10273.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-09-23 15:15:00)

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
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:45:00 | 8549.00 | 8425.01 | 8454.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 11:15:00 | 8634.00 | 8428.94 | 8456.39 | SL hit (close>static) qty=1.00 sl=8620.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 15:15:00 | 8528.00 | 8434.14 | 8458.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 11:15:00 | 8674.00 | 8443.93 | 8462.29 | SL hit (close>static) qty=1.00 sl=8620.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 12:45:00 | 8547.50 | 8467.75 | 8473.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 8477.50 | 8468.84 | 8473.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:30:00 | 8470.00 | 8468.84 | 8473.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 8470.00 | 8468.85 | 8473.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:30:00 | 8474.50 | 8468.85 | 8473.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 8458.00 | 8468.74 | 8473.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:30:00 | 8472.00 | 8468.74 | 8473.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 8444.00 | 8468.35 | 8473.43 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-23 12:15:00 | 8667.00 | 8472.27 | 8475.27 | SL hit (close>static) qty=1.00 sl=8620.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 13:30:00 | 8417.50 | 8476.49 | 8477.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 14:15:00 | 8429.50 | 8476.49 | 8477.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 13:00:00 | 8427.00 | 8474.74 | 8476.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 10:15:00 | 8429.50 | 8472.49 | 8475.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 7996.62 | 8440.05 | 8458.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 8008.02 | 8440.05 | 8458.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 8005.65 | 8440.05 | 8458.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 8008.02 | 8440.05 | 8458.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-27 09:15:00 | 7575.75 | 8101.68 | 8241.65 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-27 09:15:00 | 7586.55 | 8101.68 | 8241.65 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-27 09:15:00 | 7584.30 | 8101.68 | 8241.65 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-27 09:15:00 | 7586.55 | 8101.68 | 8241.65 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 7952.00 | 7856.87 | 8019.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 14:00:00 | 7921.50 | 7860.13 | 8018.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 14:15:00 | 8082.00 | 7869.88 | 8017.02 | SL hit (close>static) qty=1.00 sl=8055.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 11:00:00 | 7907.00 | 7947.60 | 8028.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 09:45:00 | 7898.00 | 7944.88 | 8024.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 11:15:00 | 7916.00 | 7944.87 | 8024.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 8065.50 | 7945.82 | 8021.67 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-08 11:15:00 | 8065.50 | 7945.82 | 8021.67 | SL hit (close>static) qty=1.00 sl=8055.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-08 11:15:00 | 8065.50 | 7945.82 | 8021.67 | SL hit (close>static) qty=1.00 sl=8055.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-08 11:15:00 | 8065.50 | 7945.82 | 8021.67 | SL hit (close>static) qty=1.00 sl=8055.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-05-08 12:00:00 | 8065.50 | 7945.82 | 8021.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 8051.50 | 7946.87 | 8021.82 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-16 14:00:00 | 8615.00 | 2025-05-23 15:15:00 | 8750.00 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-05-16 15:00:00 | 8585.00 | 2025-05-23 15:15:00 | 8750.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-05-19 11:00:00 | 8628.00 | 2025-05-23 15:15:00 | 8750.00 | STOP_HIT | 1.00 | -1.41% |
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
