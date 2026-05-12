# Force Motors Ltd. (FORCEMOT)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 20851.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 145 |
| ALERT1 | 101 |
| ALERT2 | 98 |
| ALERT2_SKIP | 46 |
| ALERT3 | 237 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 88 |
| PARTIAL | 13 |
| TARGET_HIT | 9 |
| STOP_HIT | 84 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 106 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 42 / 64
- **Target hits / Stop hits / Partials:** 9 / 84 / 13
- **Avg / median % per leg:** 0.59% / -0.75%
- **Sum % (uncompounded):** 62.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 49 | 14 | 28.6% | 8 | 40 | 1 | 0.24% | 11.7% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 3.05% | 12.2% |
| BUY @ 3rd Alert (retest2) | 45 | 12 | 26.7% | 7 | 38 | 0 | -0.01% | -0.5% |
| SELL (all) | 57 | 28 | 49.1% | 1 | 44 | 12 | 0.90% | 51.2% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 2 | 2 | 2.82% | 11.3% |
| SELL @ 3rd Alert (retest2) | 53 | 24 | 45.3% | 1 | 42 | 10 | 0.75% | 39.9% |
| retest1 (combined) | 8 | 6 | 75.0% | 1 | 4 | 3 | 2.93% | 23.5% |
| retest2 (combined) | 98 | 36 | 36.7% | 8 | 80 | 10 | 0.40% | 39.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 15:15:00 | 9200.00 | 9056.28 | 9040.24 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 11:15:00 | 8976.55 | 9092.59 | 9092.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 13:15:00 | 8939.05 | 9043.93 | 9069.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-18 09:15:00 | 9020.05 | 8937.52 | 8980.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-18 09:15:00 | 9020.05 | 8937.52 | 8980.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 09:15:00 | 9020.05 | 8937.52 | 8980.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-18 09:45:00 | 9020.05 | 8937.52 | 8980.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 8708.90 | 8879.74 | 8942.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-21 13:30:00 | 8625.70 | 8771.24 | 8865.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-21 15:15:00 | 8669.60 | 8756.00 | 8850.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-22 09:30:00 | 8518.05 | 8701.37 | 8808.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 12:30:00 | 8623.70 | 8661.64 | 8702.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 11:15:00 | 8236.12 | 8343.16 | 8444.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 12:15:00 | 8194.42 | 8321.07 | 8425.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 12:15:00 | 8192.51 | 8321.07 | 8425.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 15:15:00 | 8092.15 | 8232.91 | 8355.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-29 09:15:00 | 8269.95 | 8240.31 | 8347.95 | SL hit (close>ema200) qty=0.50 sl=8240.31 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 11:15:00 | 8464.95 | 8362.43 | 8355.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-30 12:15:00 | 8485.70 | 8387.09 | 8367.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-31 09:15:00 | 8367.05 | 8422.79 | 8394.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 09:15:00 | 8367.05 | 8422.79 | 8394.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 8367.05 | 8422.79 | 8394.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-31 10:00:00 | 8367.05 | 8422.79 | 8394.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 8309.45 | 8400.12 | 8387.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 12:45:00 | 8388.40 | 8389.70 | 8384.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 15:00:00 | 8789.30 | 8481.93 | 8427.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 11:15:00 | 7892.95 | 8541.32 | 8583.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 7892.95 | 8541.32 | 8583.63 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 09:15:00 | 8559.90 | 8466.22 | 8455.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 14:15:00 | 8625.40 | 8512.25 | 8482.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 10:15:00 | 8514.05 | 8525.02 | 8496.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 10:30:00 | 8527.40 | 8525.02 | 8496.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 11:15:00 | 8573.40 | 8534.69 | 8503.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 11:30:00 | 8541.45 | 8534.69 | 8503.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 13:15:00 | 8536.45 | 8533.19 | 8508.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 13:30:00 | 8504.00 | 8533.19 | 8508.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 8518.00 | 8530.15 | 8509.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:30:00 | 8501.20 | 8530.15 | 8509.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 8495.00 | 8523.12 | 8507.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:15:00 | 8544.15 | 8523.12 | 8507.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 8520.60 | 8522.62 | 8509.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:30:00 | 8479.10 | 8522.62 | 8509.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 10:15:00 | 8500.05 | 8518.10 | 8508.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 11:15:00 | 8533.30 | 8518.10 | 8508.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 13:15:00 | 8968.50 | 9031.72 | 9034.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 13:15:00 | 8968.50 | 9031.72 | 9034.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 15:15:00 | 8948.00 | 9003.50 | 9020.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 10:15:00 | 9018.50 | 8998.33 | 9014.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 10:15:00 | 9018.50 | 8998.33 | 9014.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 9018.50 | 8998.33 | 9014.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:45:00 | 9040.00 | 8998.33 | 9014.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 11:15:00 | 9063.00 | 9011.27 | 9018.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 11:45:00 | 9061.50 | 9011.27 | 9018.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 12:15:00 | 9051.55 | 9019.32 | 9021.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 12:30:00 | 9054.80 | 9019.32 | 9021.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 13:15:00 | 9067.90 | 9029.04 | 9025.94 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 14:15:00 | 8982.60 | 9019.75 | 9022.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 15:15:00 | 8915.05 | 8998.81 | 9012.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 9046.95 | 9008.44 | 9015.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 09:15:00 | 9046.95 | 9008.44 | 9015.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 9046.95 | 9008.44 | 9015.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 13:00:00 | 8926.70 | 8992.59 | 9006.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 09:30:00 | 8969.90 | 8965.08 | 8985.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 12:15:00 | 9078.85 | 9003.50 | 8999.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2024-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 12:15:00 | 9078.85 | 9003.50 | 8999.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 13:15:00 | 9100.00 | 9022.80 | 9008.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 14:15:00 | 9123.15 | 9207.53 | 9145.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 14:15:00 | 9123.15 | 9207.53 | 9145.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 9123.15 | 9207.53 | 9145.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 14:30:00 | 9148.00 | 9207.53 | 9145.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 15:15:00 | 9185.00 | 9203.02 | 9148.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 09:15:00 | 9215.85 | 9203.02 | 9148.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 11:15:00 | 9223.30 | 9184.75 | 9149.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 13:15:00 | 9050.00 | 9148.42 | 9140.82 | SL hit (close<static) qty=1.00 sl=9115.70 alert=retest2 |

### Cycle 10 — SELL (started 2024-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 14:15:00 | 9072.20 | 9133.17 | 9134.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 09:15:00 | 9033.60 | 9103.96 | 9120.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 10:15:00 | 9009.00 | 8996.88 | 9042.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-28 10:45:00 | 9004.00 | 8996.88 | 9042.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 13:15:00 | 9025.00 | 8999.52 | 9031.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 13:30:00 | 9065.75 | 8999.52 | 9031.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 8970.05 | 8993.63 | 9026.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 14:30:00 | 9036.95 | 8993.63 | 9026.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 9039.00 | 8998.76 | 9022.70 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2024-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 10:15:00 | 9188.95 | 9052.86 | 9034.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 13:15:00 | 9215.70 | 9117.88 | 9071.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 09:15:00 | 9131.35 | 9139.78 | 9094.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 09:15:00 | 9131.35 | 9139.78 | 9094.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 9131.35 | 9139.78 | 9094.82 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2024-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 13:15:00 | 8905.00 | 9072.40 | 9076.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 11:15:00 | 8859.10 | 8972.72 | 9020.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 12:15:00 | 8529.40 | 8445.48 | 8545.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 12:15:00 | 8529.40 | 8445.48 | 8545.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 12:15:00 | 8529.40 | 8445.48 | 8545.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 12:45:00 | 8543.05 | 8445.48 | 8545.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 8530.10 | 8462.40 | 8544.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 15:15:00 | 8411.00 | 8472.15 | 8541.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 09:30:00 | 8477.40 | 8455.07 | 8520.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-11 15:15:00 | 8531.00 | 8487.48 | 8483.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 15:15:00 | 8531.00 | 8487.48 | 8483.41 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 11:15:00 | 8358.25 | 8466.55 | 8475.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 10:15:00 | 8264.35 | 8318.48 | 8354.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 12:15:00 | 8315.00 | 8311.11 | 8344.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-18 13:00:00 | 8315.00 | 8311.11 | 8344.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 13:15:00 | 8371.15 | 8270.46 | 8298.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 14:00:00 | 8371.15 | 8270.46 | 8298.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 14:15:00 | 8210.95 | 8258.56 | 8290.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 15:15:00 | 8192.00 | 8258.56 | 8290.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 09:15:00 | 8173.80 | 8225.98 | 8253.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 8075.65 | 8251.46 | 8258.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 14:15:00 | 8301.00 | 8261.43 | 8260.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 14:15:00 | 8301.00 | 8261.43 | 8260.93 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 09:15:00 | 8187.75 | 8256.44 | 8262.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 11:15:00 | 8148.00 | 8225.24 | 8246.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-25 14:15:00 | 8240.00 | 8205.35 | 8230.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 14:15:00 | 8240.00 | 8205.35 | 8230.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 8240.00 | 8205.35 | 8230.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 14:30:00 | 8195.35 | 8205.35 | 8230.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 8200.00 | 8204.28 | 8227.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:15:00 | 8295.10 | 8204.28 | 8227.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 8338.00 | 8231.02 | 8237.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:45:00 | 8338.10 | 8231.02 | 8237.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2024-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 10:15:00 | 8439.10 | 8272.64 | 8255.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 11:15:00 | 8515.00 | 8321.11 | 8279.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 10:15:00 | 9260.00 | 9266.99 | 9095.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-01 10:45:00 | 9278.95 | 9266.99 | 9095.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 8989.95 | 9168.97 | 9118.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:45:00 | 8986.00 | 9168.97 | 9118.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 9021.30 | 9139.43 | 9109.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 11:30:00 | 9097.90 | 9126.11 | 9106.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 12:45:00 | 9048.10 | 9110.49 | 9101.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 13:30:00 | 9115.80 | 9100.31 | 9097.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 15:15:00 | 9030.00 | 9082.97 | 9089.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-08-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 15:15:00 | 9030.00 | 9082.97 | 9089.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 8797.90 | 9025.96 | 9063.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-08 11:15:00 | 8443.65 | 8375.81 | 8472.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-08 12:00:00 | 8443.65 | 8375.81 | 8472.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 12:15:00 | 8341.30 | 8368.91 | 8460.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 12:30:00 | 8386.60 | 8368.91 | 8460.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 13:15:00 | 8416.10 | 8357.98 | 8399.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 14:00:00 | 8416.10 | 8357.98 | 8399.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 8392.20 | 8364.82 | 8398.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 14:45:00 | 8460.00 | 8364.82 | 8398.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 8400.00 | 8371.86 | 8398.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:15:00 | 8368.20 | 8371.86 | 8398.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:45:00 | 8320.00 | 8368.97 | 8394.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 11:15:00 | 8370.00 | 8370.50 | 8393.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 13:00:00 | 8362.90 | 8368.33 | 8388.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 13:15:00 | 8379.70 | 8370.61 | 8387.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 13:30:00 | 8408.00 | 8370.61 | 8387.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 8321.35 | 8346.97 | 8371.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 14:30:00 | 8252.15 | 8306.31 | 8343.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 10:45:00 | 8265.15 | 8225.37 | 8257.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 11:30:00 | 8283.35 | 8233.44 | 8258.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 14:30:00 | 8273.95 | 8262.79 | 8267.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 8250.00 | 8264.59 | 8267.52 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-19 11:15:00 | 8299.40 | 8272.42 | 8270.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 11:15:00 | 8299.40 | 8272.42 | 8270.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 12:15:00 | 8330.85 | 8284.10 | 8276.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 14:15:00 | 8265.95 | 8286.32 | 8278.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-19 14:15:00 | 8265.95 | 8286.32 | 8278.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 14:15:00 | 8265.95 | 8286.32 | 8278.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 15:00:00 | 8265.95 | 8286.32 | 8278.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 15:15:00 | 8276.10 | 8284.28 | 8278.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 09:15:00 | 8517.95 | 8284.28 | 8278.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-26 15:15:00 | 8572.00 | 8651.34 | 8653.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 15:15:00 | 8572.00 | 8651.34 | 8653.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 10:15:00 | 8519.35 | 8613.11 | 8635.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 8202.65 | 8180.40 | 8275.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-30 09:45:00 | 8211.75 | 8180.40 | 8275.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 13:15:00 | 8243.65 | 8197.86 | 8253.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 14:00:00 | 8243.65 | 8197.86 | 8253.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 8360.25 | 8230.34 | 8263.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 15:00:00 | 8360.25 | 8230.34 | 8263.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 15:15:00 | 8289.85 | 8242.24 | 8265.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:15:00 | 8253.95 | 8242.24 | 8265.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2024-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 10:15:00 | 8360.70 | 8280.91 | 8279.95 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 13:15:00 | 8276.90 | 8289.85 | 8290.43 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 14:15:00 | 8294.95 | 8290.87 | 8290.84 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 15:15:00 | 8280.45 | 8288.78 | 8289.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 09:15:00 | 8260.65 | 8283.16 | 8287.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-11 09:15:00 | 7420.05 | 7415.99 | 7557.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-11 10:00:00 | 7420.05 | 7415.99 | 7557.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 7186.65 | 7142.47 | 7260.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:45:00 | 7246.60 | 7142.47 | 7260.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 13:15:00 | 7265.00 | 7198.73 | 7252.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 13:30:00 | 7292.85 | 7198.73 | 7252.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 14:15:00 | 7205.00 | 7199.98 | 7247.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 15:15:00 | 7189.00 | 7199.98 | 7247.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 09:15:00 | 6829.55 | 6942.91 | 7018.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-20 09:15:00 | 6863.90 | 6806.85 | 6891.96 | SL hit (close>ema200) qty=0.50 sl=6806.85 alert=retest2 |

### Cycle 25 — BUY (started 2024-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 11:15:00 | 7270.00 | 6983.20 | 6962.54 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 11:15:00 | 7038.15 | 7089.36 | 7094.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 09:15:00 | 6930.00 | 7039.20 | 7067.48 | Break + close below crossover candle low |

### Cycle 27 — BUY (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 09:15:00 | 7460.00 | 7073.67 | 7058.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 10:15:00 | 7604.00 | 7179.74 | 7107.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 14:15:00 | 7472.00 | 7482.27 | 7373.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-30 15:00:00 | 7472.00 | 7482.27 | 7373.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 7499.35 | 7564.53 | 7493.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 10:00:00 | 7499.35 | 7564.53 | 7493.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 10:15:00 | 7575.15 | 7566.66 | 7501.20 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 15:15:00 | 7399.00 | 7466.72 | 7473.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 14:15:00 | 7163.00 | 7370.89 | 7421.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 7058.85 | 7027.14 | 7165.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 10:00:00 | 7058.85 | 7027.14 | 7165.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 7151.00 | 7065.17 | 7159.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:45:00 | 7170.05 | 7065.17 | 7159.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 7169.05 | 7085.95 | 7160.43 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 7289.75 | 7204.12 | 7195.19 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 14:15:00 | 7023.60 | 7190.38 | 7209.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 11:15:00 | 7009.95 | 7075.48 | 7123.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 10:15:00 | 7003.25 | 6996.19 | 7055.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 10:15:00 | 7003.25 | 6996.19 | 7055.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 7003.25 | 6996.19 | 7055.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:30:00 | 7134.90 | 6996.19 | 7055.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 7075.45 | 7012.04 | 7057.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 12:00:00 | 7075.45 | 7012.04 | 7057.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 6962.95 | 7002.22 | 7048.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 10:00:00 | 6958.00 | 6993.53 | 7016.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:15:00 | 6610.10 | 6760.60 | 6833.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-23 13:15:00 | 6262.20 | 6434.87 | 6585.81 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 13:15:00 | 6479.55 | 6378.66 | 6373.94 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 10:15:00 | 6317.30 | 6372.49 | 6373.78 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 12:15:00 | 6379.00 | 6375.00 | 6374.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 13:15:00 | 6399.95 | 6379.99 | 6377.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 7539.25 | 7757.15 | 7501.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 10:00:00 | 7539.25 | 7757.15 | 7501.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 7491.95 | 7704.11 | 7500.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 7491.95 | 7704.11 | 7500.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 7501.15 | 7663.52 | 7500.91 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2024-11-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 13:15:00 | 7422.55 | 7447.21 | 7449.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 14:15:00 | 7384.05 | 7434.58 | 7443.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 10:15:00 | 7430.50 | 7420.28 | 7433.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 10:15:00 | 7430.50 | 7420.28 | 7433.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 7430.50 | 7420.28 | 7433.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 11:00:00 | 7430.50 | 7420.28 | 7433.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 11:15:00 | 7412.60 | 7418.75 | 7431.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 11:45:00 | 7431.65 | 7418.75 | 7431.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2024-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 13:15:00 | 7510.00 | 7438.80 | 7438.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 09:15:00 | 7556.45 | 7475.49 | 7456.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 10:15:00 | 7522.80 | 7533.45 | 7505.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 10:15:00 | 7522.80 | 7533.45 | 7505.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 7522.80 | 7533.45 | 7505.57 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2024-11-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 14:15:00 | 7393.90 | 7477.72 | 7486.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 7058.00 | 7371.73 | 7435.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 6846.65 | 6808.44 | 6936.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 12:15:00 | 6946.95 | 6828.61 | 6913.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 6946.95 | 6828.61 | 6913.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:00:00 | 6946.95 | 6828.61 | 6913.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 6912.10 | 6845.31 | 6912.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:30:00 | 6929.40 | 6845.31 | 6912.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 14:15:00 | 6925.15 | 6861.27 | 6914.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 15:00:00 | 6925.15 | 6861.27 | 6914.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 15:15:00 | 6920.00 | 6873.02 | 6914.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 09:15:00 | 6829.00 | 6873.02 | 6914.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 6807.85 | 6859.99 | 6904.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 10:15:00 | 6715.00 | 6859.99 | 6904.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 12:45:00 | 6731.80 | 6793.02 | 6859.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 14:00:00 | 6724.20 | 6779.26 | 6846.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:15:00 | 6732.30 | 6788.33 | 6816.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 6692.10 | 6661.28 | 6709.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 09:30:00 | 6684.85 | 6661.28 | 6709.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 6695.90 | 6668.21 | 6708.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 13:30:00 | 6676.20 | 6683.96 | 6706.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 6947.80 | 6741.31 | 6727.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 6947.80 | 6741.31 | 6727.29 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 10:15:00 | 6858.25 | 6934.47 | 6941.35 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 14:15:00 | 6965.95 | 6925.54 | 6924.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 09:15:00 | 7020.25 | 6961.60 | 6946.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 15:15:00 | 6980.00 | 6984.52 | 6967.18 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 09:15:00 | 7040.00 | 6984.52 | 6967.18 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 6900.00 | 6966.57 | 6961.97 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-05 10:15:00 | 6900.00 | 6966.57 | 6961.97 | SL hit (close<ema400) qty=1.00 sl=6961.97 alert=retest1 |

### Cycle 40 — SELL (started 2024-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 11:15:00 | 6899.60 | 6953.18 | 6956.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 09:15:00 | 6845.00 | 6897.44 | 6924.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 11:15:00 | 6575.90 | 6572.72 | 6614.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 12:00:00 | 6575.90 | 6572.72 | 6614.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 6613.70 | 6578.73 | 6606.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 6613.70 | 6578.73 | 6606.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 6615.00 | 6585.99 | 6607.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 6637.75 | 6585.99 | 6607.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 6734.30 | 6615.65 | 6619.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 6788.00 | 6615.65 | 6619.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2024-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 10:15:00 | 6831.80 | 6658.88 | 6638.55 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 13:15:00 | 6685.20 | 6721.43 | 6724.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 6579.65 | 6686.47 | 6707.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 09:15:00 | 6701.50 | 6639.41 | 6664.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 09:15:00 | 6701.50 | 6639.41 | 6664.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 6701.50 | 6639.41 | 6664.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:45:00 | 6732.00 | 6639.41 | 6664.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 6789.95 | 6669.52 | 6675.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 10:30:00 | 6797.95 | 6669.52 | 6675.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2024-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 11:15:00 | 6751.00 | 6685.82 | 6682.38 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 09:15:00 | 6639.55 | 6684.31 | 6684.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 10:15:00 | 6505.10 | 6570.23 | 6598.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 13:15:00 | 6460.05 | 6458.19 | 6506.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-31 13:30:00 | 6442.60 | 6458.19 | 6506.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 6507.00 | 6467.95 | 6506.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 15:00:00 | 6507.00 | 6467.95 | 6506.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 15:15:00 | 6528.00 | 6479.96 | 6508.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:15:00 | 6536.00 | 6479.96 | 6508.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 6543.00 | 6492.57 | 6511.68 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 13:15:00 | 6561.85 | 6529.38 | 6525.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 14:15:00 | 6630.05 | 6549.51 | 6534.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 7219.70 | 7273.64 | 7095.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 10:15:00 | 7038.15 | 7226.54 | 7090.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 7038.15 | 7226.54 | 7090.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 7038.15 | 7226.54 | 7090.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 7018.35 | 7184.90 | 7083.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 6995.05 | 7184.90 | 7083.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 12:15:00 | 6951.45 | 7138.21 | 7071.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 13:00:00 | 6951.45 | 7138.21 | 7071.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 15:15:00 | 6860.00 | 7003.43 | 7019.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 11:15:00 | 6765.40 | 6868.73 | 6927.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 14:15:00 | 6847.70 | 6844.88 | 6899.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 15:00:00 | 6847.70 | 6844.88 | 6899.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 6803.50 | 6840.62 | 6888.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 10:45:00 | 6737.40 | 6820.37 | 6874.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 11:15:00 | 6400.53 | 6502.98 | 6604.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 09:15:00 | 6418.50 | 6411.09 | 6511.87 | SL hit (close>ema200) qty=0.50 sl=6411.09 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 11:15:00 | 6578.75 | 6520.34 | 6518.55 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 11:15:00 | 6456.00 | 6537.51 | 6544.80 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 13:15:00 | 6569.25 | 6535.45 | 6533.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-21 11:15:00 | 6578.05 | 6550.22 | 6542.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-22 09:15:00 | 6465.00 | 6547.59 | 6546.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 09:15:00 | 6465.00 | 6547.59 | 6546.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 6465.00 | 6547.59 | 6546.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:45:00 | 6476.30 | 6547.59 | 6546.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 6436.85 | 6525.44 | 6536.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 6400.00 | 6500.35 | 6524.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 6529.00 | 6485.83 | 6509.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 6529.00 | 6485.83 | 6509.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 6529.00 | 6485.83 | 6509.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 6529.00 | 6485.83 | 6509.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 6500.10 | 6488.68 | 6508.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:15:00 | 6527.75 | 6488.68 | 6508.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 09:15:00 | 6744.95 | 6539.94 | 6530.13 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 14:15:00 | 6522.40 | 6560.57 | 6564.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 6345.80 | 6511.13 | 6540.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 14:15:00 | 6378.80 | 6343.80 | 6401.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 14:15:00 | 6378.80 | 6343.80 | 6401.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 6378.80 | 6343.80 | 6401.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:30:00 | 6417.50 | 6343.80 | 6401.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 15:15:00 | 6400.00 | 6355.04 | 6400.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:15:00 | 6435.00 | 6355.04 | 6400.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 6378.60 | 6359.75 | 6398.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:30:00 | 6462.10 | 6359.75 | 6398.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 6452.70 | 6378.34 | 6403.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:45:00 | 6433.70 | 6378.34 | 6403.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 6413.00 | 6385.28 | 6404.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 13:00:00 | 6384.35 | 6385.09 | 6402.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 09:15:00 | 6470.35 | 6418.04 | 6414.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 6470.35 | 6418.04 | 6414.19 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 13:15:00 | 6347.30 | 6400.53 | 6407.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-31 09:15:00 | 6277.15 | 6355.87 | 6383.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 14:15:00 | 6322.00 | 6318.66 | 6351.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-31 15:00:00 | 6322.00 | 6318.66 | 6351.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 6340.00 | 6321.71 | 6346.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 10:30:00 | 6327.95 | 6332.57 | 6349.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 11:45:00 | 6307.85 | 6333.06 | 6348.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 12:15:00 | 6325.00 | 6333.06 | 6348.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 13:00:00 | 6317.65 | 6329.98 | 6345.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 6390.30 | 6342.04 | 6349.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 14:00:00 | 6390.30 | 6342.04 | 6349.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-01 14:15:00 | 6490.30 | 6371.69 | 6362.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 14:15:00 | 6490.30 | 6371.69 | 6362.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 6712.95 | 6463.97 | 6423.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 10:15:00 | 6668.95 | 6699.58 | 6605.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 11:00:00 | 6668.95 | 6699.58 | 6605.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 6602.00 | 6680.06 | 6605.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:45:00 | 6614.95 | 6680.06 | 6605.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 6572.50 | 6658.55 | 6602.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 6572.50 | 6658.55 | 6602.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 6590.00 | 6644.84 | 6601.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:45:00 | 6564.85 | 6644.84 | 6601.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 6594.00 | 6626.54 | 6599.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:15:00 | 6495.70 | 6626.54 | 6599.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 6580.05 | 6617.24 | 6598.15 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 13:15:00 | 6530.00 | 6590.00 | 6590.60 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-02-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 15:15:00 | 6599.45 | 6591.89 | 6591.35 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 6437.45 | 6561.00 | 6577.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 6388.95 | 6504.35 | 6547.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 15:15:00 | 6305.00 | 6301.98 | 6378.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 09:15:00 | 6300.00 | 6301.98 | 6378.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 6456.95 | 6333.65 | 6379.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:00:00 | 6456.95 | 6333.65 | 6379.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 6707.70 | 6408.46 | 6409.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:45:00 | 6703.70 | 6408.46 | 6409.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 12:15:00 | 6712.30 | 6469.23 | 6436.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-12 13:15:00 | 6738.45 | 6523.07 | 6464.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-13 13:15:00 | 6720.00 | 6735.73 | 6621.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-13 14:00:00 | 6720.00 | 6735.73 | 6621.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 10:15:00 | 6596.70 | 6716.54 | 6649.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 11:00:00 | 6596.70 | 6716.54 | 6649.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 11:15:00 | 6479.10 | 6669.05 | 6634.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 12:00:00 | 6479.10 | 6669.05 | 6634.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2025-02-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 13:15:00 | 6413.55 | 6585.86 | 6600.44 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 15:15:00 | 6612.95 | 6531.69 | 6531.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 09:15:00 | 6720.95 | 6569.54 | 6548.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 6751.00 | 6752.31 | 6691.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:15:00 | 6767.00 | 6752.31 | 6691.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 6881.20 | 6898.71 | 6810.97 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 6687.30 | 6795.55 | 6803.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 11:15:00 | 6613.20 | 6737.33 | 6774.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 11:15:00 | 6649.70 | 6633.66 | 6691.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-28 11:30:00 | 6658.05 | 6633.66 | 6691.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 6698.65 | 6642.21 | 6680.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 14:45:00 | 6712.45 | 6642.21 | 6680.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 6694.05 | 6652.58 | 6681.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 09:15:00 | 6740.65 | 6652.58 | 6681.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 6666.25 | 6655.31 | 6680.07 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2025-03-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 12:15:00 | 6858.00 | 6711.39 | 6700.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-03 14:15:00 | 6938.20 | 6783.66 | 6737.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 14:15:00 | 7555.55 | 7648.02 | 7548.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 15:00:00 | 7555.55 | 7648.02 | 7548.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 7540.00 | 7626.42 | 7548.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 7405.25 | 7626.42 | 7548.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 7647.80 | 7630.69 | 7557.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:30:00 | 7422.60 | 7630.69 | 7557.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 7452.00 | 7594.96 | 7547.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 10:30:00 | 7429.10 | 7594.96 | 7547.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 7391.70 | 7554.30 | 7533.38 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2025-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 12:15:00 | 7336.55 | 7510.75 | 7515.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 13:15:00 | 7285.45 | 7465.69 | 7494.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 7290.00 | 7244.43 | 7326.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-13 10:00:00 | 7290.00 | 7244.43 | 7326.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 11:15:00 | 7265.00 | 7244.24 | 7311.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 12:00:00 | 7265.00 | 7244.24 | 7311.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 7227.00 | 7244.31 | 7287.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 11:00:00 | 7179.15 | 7231.28 | 7277.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 09:45:00 | 7131.55 | 7140.08 | 7200.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 11:15:00 | 7324.40 | 7198.92 | 7217.94 | SL hit (close>static) qty=1.00 sl=7308.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 12:15:00 | 7385.95 | 7236.33 | 7233.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 13:15:00 | 7444.80 | 7278.02 | 7252.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 14:15:00 | 8726.50 | 8789.25 | 8604.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 15:00:00 | 8726.50 | 8789.25 | 8604.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 8834.50 | 8796.33 | 8714.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 09:15:00 | 9195.05 | 8807.66 | 8757.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 12:15:00 | 8883.00 | 8922.75 | 8877.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 15:00:00 | 8885.30 | 8896.69 | 8875.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 10:30:00 | 9035.00 | 8907.78 | 8884.45 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 11:15:00 | 9063.40 | 9084.67 | 9011.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 14:00:00 | 9116.80 | 9085.55 | 9024.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 14:45:00 | 9135.30 | 9093.41 | 9033.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 09:15:00 | 8855.55 | 9048.49 | 9023.43 | SL hit (close<static) qty=1.00 sl=9006.35 alert=retest2 |

### Cycle 66 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 8862.90 | 8992.13 | 9000.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 8850.00 | 8963.71 | 8987.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 8385.00 | 8373.89 | 8555.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 11:15:00 | 8435.00 | 8373.89 | 8555.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 09:15:00 | 8400.50 | 8420.40 | 8505.50 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 8712.40 | 8520.19 | 8518.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 8895.00 | 8709.38 | 8631.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 09:15:00 | 8599.50 | 8784.03 | 8726.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-16 09:15:00 | 8599.50 | 8784.03 | 8726.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 09:15:00 | 8599.50 | 8784.03 | 8726.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 10:00:00 | 8599.50 | 8784.03 | 8726.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 10:15:00 | 8707.50 | 8768.72 | 8724.33 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2025-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-16 15:15:00 | 8635.00 | 8693.28 | 8700.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-17 09:15:00 | 8593.50 | 8673.33 | 8690.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-17 11:15:00 | 8700.00 | 8674.93 | 8688.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 11:15:00 | 8700.00 | 8674.93 | 8688.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 11:15:00 | 8700.00 | 8674.93 | 8688.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-17 12:00:00 | 8700.00 | 8674.93 | 8688.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 12:15:00 | 8727.50 | 8685.44 | 8691.59 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2025-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 15:15:00 | 8721.50 | 8696.29 | 8695.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 09:15:00 | 9124.00 | 8781.83 | 8734.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 12:15:00 | 9167.00 | 9181.68 | 9038.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 13:00:00 | 9167.00 | 9181.68 | 9038.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 9121.50 | 9159.72 | 9063.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 9214.00 | 9159.72 | 9063.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 8972.50 | 9122.28 | 9054.89 | SL hit (close<static) qty=1.00 sl=9055.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 8940.00 | 9071.79 | 9087.80 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-04-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 15:15:00 | 9121.00 | 9096.79 | 9095.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 09:15:00 | 9290.00 | 9135.43 | 9112.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-28 13:15:00 | 9091.00 | 9152.67 | 9131.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 13:15:00 | 9091.00 | 9152.67 | 9131.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 13:15:00 | 9091.00 | 9152.67 | 9131.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-28 14:00:00 | 9091.00 | 9152.67 | 9131.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 9005.00 | 9123.13 | 9120.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-28 14:30:00 | 9040.00 | 9123.13 | 9120.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2025-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-28 15:15:00 | 9001.00 | 9098.71 | 9109.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 12:15:00 | 8980.50 | 9051.49 | 9082.34 | Break + close below crossover candle low |

### Cycle 73 — BUY (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 09:15:00 | 9776.00 | 9131.12 | 9084.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 10:15:00 | 9888.00 | 9282.50 | 9157.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 10025.00 | 10098.23 | 9851.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 14:15:00 | 10005.00 | 10070.92 | 9932.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 10005.00 | 10070.92 | 9932.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:30:00 | 10011.50 | 10070.92 | 9932.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 9960.00 | 10048.73 | 9934.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:15:00 | 10016.50 | 10048.73 | 9934.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 9926.00 | 10024.19 | 9934.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 11:30:00 | 10190.00 | 10052.62 | 9963.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 13:15:00 | 10128.00 | 10062.50 | 9975.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 14:15:00 | 9747.00 | 10008.49 | 10019.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 14:15:00 | 9747.00 | 10008.49 | 10019.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 9645.00 | 9935.79 | 9985.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 10390.00 | 9863.08 | 9885.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 10390.00 | 9863.08 | 9885.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 10390.00 | 9863.08 | 9885.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 10390.00 | 9863.08 | 9885.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 10730.00 | 10036.46 | 9962.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 10754.00 | 10411.38 | 10184.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 09:15:00 | 10425.00 | 10469.88 | 10254.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 09:15:00 | 10425.00 | 10469.88 | 10254.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 10425.00 | 10469.88 | 10254.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 09:45:00 | 10310.00 | 10469.88 | 10254.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 10:15:00 | 10544.50 | 10564.34 | 10424.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:00:00 | 10544.50 | 10564.34 | 10424.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 10492.00 | 10549.87 | 10430.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:45:00 | 10490.00 | 10549.87 | 10430.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 10411.00 | 10516.34 | 10444.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 15:00:00 | 10411.00 | 10516.34 | 10444.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 15:15:00 | 10420.50 | 10497.17 | 10442.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 09:15:00 | 10549.00 | 10497.17 | 10442.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 13:15:00 | 10605.50 | 10767.08 | 10772.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 13:15:00 | 10605.50 | 10767.08 | 10772.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 12:15:00 | 10595.00 | 10684.93 | 10721.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 13:15:00 | 10510.00 | 10507.82 | 10592.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 14:00:00 | 10510.00 | 10507.82 | 10592.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 10599.00 | 10526.06 | 10592.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 14:30:00 | 10620.00 | 10526.06 | 10592.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 15:15:00 | 10561.00 | 10533.05 | 10590.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:15:00 | 10550.00 | 10533.05 | 10590.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 10527.50 | 10531.94 | 10584.32 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 10840.00 | 10617.88 | 10597.16 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 14:15:00 | 10203.50 | 10526.85 | 10564.66 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 09:15:00 | 10933.50 | 10611.96 | 10569.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 14:15:00 | 11030.00 | 10827.40 | 10701.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 09:15:00 | 12163.00 | 12446.37 | 12142.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-02 10:00:00 | 12163.00 | 12446.37 | 12142.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 11960.00 | 12349.10 | 12125.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:00:00 | 11960.00 | 12349.10 | 12125.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 11870.00 | 12253.28 | 12102.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 12:00:00 | 11870.00 | 12253.28 | 12102.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2025-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 14:15:00 | 11652.00 | 12011.15 | 12018.54 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 13:15:00 | 12100.00 | 11990.02 | 11989.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 10:15:00 | 12278.00 | 12061.24 | 12023.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 15:15:00 | 12244.00 | 12256.36 | 12151.46 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 09:15:00 | 12498.00 | 12256.36 | 12151.46 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 12397.00 | 12579.39 | 12420.05 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-06 09:15:00 | 12397.00 | 12579.39 | 12420.05 | SL hit (close<ema400) qty=1.00 sl=12420.05 alert=retest1 |

### Cycle 82 — SELL (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 11:15:00 | 12234.00 | 12405.57 | 12421.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-09 12:15:00 | 12169.00 | 12358.26 | 12398.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 09:15:00 | 12524.00 | 12321.10 | 12360.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 09:15:00 | 12524.00 | 12321.10 | 12360.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 12524.00 | 12321.10 | 12360.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:45:00 | 12562.00 | 12321.10 | 12360.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 12390.00 | 12334.88 | 12362.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 11:30:00 | 12328.00 | 12306.90 | 12347.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 11:00:00 | 12328.00 | 12213.87 | 12238.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 11:15:00 | 12687.00 | 12308.50 | 12279.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 11:15:00 | 12687.00 | 12308.50 | 12279.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-12 12:15:00 | 13120.00 | 12470.80 | 12355.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 14:15:00 | 14000.00 | 14063.88 | 13698.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 15:00:00 | 14000.00 | 14063.88 | 13698.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 13696.00 | 13981.52 | 13723.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 11:30:00 | 14180.00 | 14062.30 | 13806.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 15:15:00 | 13659.00 | 13789.65 | 13801.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 15:15:00 | 13659.00 | 13789.65 | 13801.66 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 10:15:00 | 14000.00 | 13843.93 | 13825.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 11:15:00 | 14185.00 | 13912.15 | 13857.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 09:15:00 | 14125.00 | 14228.10 | 14130.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 14125.00 | 14228.10 | 14130.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 14125.00 | 14228.10 | 14130.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:00:00 | 14125.00 | 14228.10 | 14130.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 14221.00 | 14226.68 | 14138.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:15:00 | 14149.00 | 14226.68 | 14138.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 14120.00 | 14205.34 | 14136.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 12:00:00 | 14120.00 | 14205.34 | 14136.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 12:15:00 | 14137.00 | 14191.68 | 14136.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:00:00 | 14137.00 | 14191.68 | 14136.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2025-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 13:15:00 | 13650.00 | 14083.34 | 14092.71 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 11:15:00 | 14260.00 | 13981.52 | 13968.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 11:15:00 | 14546.00 | 14218.45 | 14102.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 12:15:00 | 15335.00 | 15535.13 | 15145.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 13:00:00 | 15335.00 | 15535.13 | 15145.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 15071.00 | 15442.31 | 15139.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:45:00 | 15048.00 | 15442.31 | 15139.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 14767.00 | 15307.25 | 15105.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 14:30:00 | 14737.00 | 15307.25 | 15105.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 14779.00 | 15201.60 | 15075.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 09:15:00 | 14968.00 | 15201.60 | 15075.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 11:15:00 | 14711.00 | 14991.95 | 15001.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 11:15:00 | 14711.00 | 14991.95 | 15001.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 11:15:00 | 14515.00 | 14686.71 | 14810.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 10:15:00 | 14390.00 | 14356.08 | 14478.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-07 11:00:00 | 14390.00 | 14356.08 | 14478.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 14796.00 | 14444.06 | 14507.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:00:00 | 14796.00 | 14444.06 | 14507.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 14859.00 | 14527.05 | 14539.36 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2025-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 13:15:00 | 14792.00 | 14580.04 | 14562.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 10:15:00 | 14902.00 | 14735.08 | 14650.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 16507.00 | 16544.91 | 16183.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 10:15:00 | 16392.00 | 16544.91 | 16183.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 16252.00 | 16428.24 | 16215.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 14:30:00 | 16398.00 | 16396.55 | 16236.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 14:15:00 | 16615.00 | 16804.74 | 16829.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 14:15:00 | 16615.00 | 16804.74 | 16829.32 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 12:15:00 | 17047.00 | 16864.96 | 16841.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 13:15:00 | 17090.00 | 16909.97 | 16864.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 11:15:00 | 17329.00 | 17398.95 | 17253.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-23 11:45:00 | 17327.00 | 17398.95 | 17253.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 17110.00 | 17341.16 | 17240.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 12:45:00 | 17133.00 | 17341.16 | 17240.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 17195.00 | 17311.93 | 17236.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:15:00 | 17103.00 | 17311.93 | 17236.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 17112.00 | 17271.94 | 17225.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 19005.00 | 17259.55 | 17223.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 10:45:00 | 17380.00 | 18417.36 | 18170.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 17823.00 | 18034.83 | 18062.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 09:15:00 | 17823.00 | 18034.83 | 18062.89 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 10:15:00 | 18270.00 | 18081.86 | 18081.72 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 13:15:00 | 17951.00 | 18103.28 | 18118.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 14:15:00 | 17807.00 | 18044.03 | 18089.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 17021.00 | 16884.13 | 17143.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 17021.00 | 16884.13 | 17143.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 17021.00 | 16884.13 | 17143.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:30:00 | 17047.00 | 16884.13 | 17143.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 17045.00 | 16999.85 | 17087.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:30:00 | 17008.00 | 17032.28 | 17093.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 11:15:00 | 17422.00 | 17110.22 | 17123.78 | SL hit (close>static) qty=1.00 sl=17280.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 12:15:00 | 17463.00 | 17180.78 | 17154.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 13:15:00 | 17642.00 | 17273.02 | 17198.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 09:15:00 | 17804.00 | 17957.66 | 17716.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-07 10:00:00 | 17804.00 | 17957.66 | 17716.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 17810.00 | 17985.50 | 17906.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 17810.00 | 17985.50 | 17906.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 17750.00 | 17938.40 | 17892.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 09:15:00 | 18091.00 | 17938.40 | 17892.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-13 13:15:00 | 19900.10 | 19379.17 | 18929.13 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 13:15:00 | 20443.00 | 20829.81 | 20844.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 20188.00 | 20701.45 | 20784.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 09:15:00 | 19803.00 | 19666.14 | 19892.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 09:15:00 | 19803.00 | 19666.14 | 19892.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 19803.00 | 19666.14 | 19892.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:45:00 | 19948.00 | 19666.14 | 19892.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 20176.00 | 19768.11 | 19918.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 20176.00 | 19768.11 | 19918.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 19961.00 | 19806.69 | 19922.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:45:00 | 19829.00 | 19813.75 | 19914.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 11:15:00 | 18837.55 | 19414.05 | 19651.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 19285.00 | 19253.24 | 19467.98 | SL hit (close>ema200) qty=0.50 sl=19253.24 alert=retest2 |

### Cycle 97 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 19930.00 | 19491.25 | 19490.82 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 10:15:00 | 19042.00 | 19480.59 | 19532.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 18100.00 | 18910.51 | 19221.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 17616.00 | 17428.24 | 17891.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 10:15:00 | 17668.00 | 17428.24 | 17891.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 17908.00 | 17618.43 | 17870.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:30:00 | 17807.00 | 17618.43 | 17870.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 17868.00 | 17668.34 | 17870.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:30:00 | 17900.00 | 17668.34 | 17870.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 18150.00 | 17764.67 | 17896.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:00:00 | 18150.00 | 17764.67 | 17896.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 18325.00 | 17876.74 | 17935.00 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 18510.00 | 18052.55 | 18007.85 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 15:15:00 | 18000.00 | 18104.18 | 18113.21 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 10:15:00 | 18347.00 | 18150.80 | 18132.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 11:15:00 | 18417.00 | 18204.04 | 18158.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 09:15:00 | 19168.00 | 19609.23 | 19388.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 19168.00 | 19609.23 | 19388.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 19168.00 | 19609.23 | 19388.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:00:00 | 19168.00 | 19609.23 | 19388.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 19453.00 | 19577.99 | 19394.06 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2025-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 14:15:00 | 19000.00 | 19310.37 | 19314.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 15:15:00 | 18930.00 | 19234.29 | 19279.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 09:15:00 | 19236.00 | 19234.64 | 19275.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 19236.00 | 19234.64 | 19275.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 19236.00 | 19234.64 | 19275.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:45:00 | 19260.00 | 19234.64 | 19275.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 19143.00 | 19216.31 | 19263.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 11:45:00 | 19036.00 | 19154.05 | 19230.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 15:15:00 | 18084.20 | 18938.23 | 19090.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-23 13:15:00 | 18205.00 | 18198.27 | 18475.01 | SL hit (close>ema200) qty=0.50 sl=18198.27 alert=retest2 |

### Cycle 103 — BUY (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 12:15:00 | 15892.00 | 15708.72 | 15686.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 13:15:00 | 16060.00 | 15778.98 | 15720.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 13:15:00 | 16590.00 | 16602.35 | 16369.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-15 13:30:00 | 16582.00 | 16602.35 | 16369.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 16530.00 | 16636.65 | 16532.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 09:15:00 | 17162.00 | 16636.65 | 16532.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 14:15:00 | 16850.00 | 17180.64 | 17224.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 16850.00 | 17180.64 | 17224.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 15:15:00 | 16400.00 | 17024.51 | 17149.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 16379.00 | 16374.67 | 16567.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 10:00:00 | 16379.00 | 16374.67 | 16567.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 16380.00 | 16370.46 | 16479.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 16658.00 | 16370.46 | 16479.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 16750.00 | 16446.37 | 16503.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 16750.00 | 16446.37 | 16503.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 16566.00 | 16470.29 | 16509.58 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 16650.00 | 16534.03 | 16533.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 09:15:00 | 16757.00 | 16624.54 | 16580.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 11:15:00 | 18130.00 | 18235.77 | 17893.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 12:00:00 | 18130.00 | 18235.77 | 17893.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 18157.00 | 18304.78 | 18138.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:45:00 | 18122.00 | 18304.78 | 18138.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 17879.00 | 18219.63 | 18115.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:00:00 | 17879.00 | 18219.63 | 18115.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 17890.00 | 18153.70 | 18094.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:15:00 | 17725.00 | 18153.70 | 18094.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 10:15:00 | 17930.00 | 18046.61 | 18052.40 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 11:15:00 | 18265.00 | 18090.29 | 18071.73 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 17610.00 | 18053.57 | 18093.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 10:15:00 | 17510.00 | 17944.86 | 18040.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 17794.00 | 17702.17 | 17850.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 09:15:00 | 17794.00 | 17702.17 | 17850.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 17794.00 | 17702.17 | 17850.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:30:00 | 17864.00 | 17702.17 | 17850.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 17838.00 | 17729.33 | 17849.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:45:00 | 17905.00 | 17729.33 | 17849.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 17701.00 | 17723.67 | 17835.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 12:30:00 | 17664.00 | 17689.93 | 17810.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 17725.00 | 17553.44 | 17533.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 17725.00 | 17553.44 | 17533.65 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 17300.00 | 17521.24 | 17533.13 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 13:15:00 | 17621.00 | 17522.49 | 17513.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 14:15:00 | 17654.00 | 17548.79 | 17525.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 09:15:00 | 17480.00 | 17536.83 | 17524.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 17480.00 | 17536.83 | 17524.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 17480.00 | 17536.83 | 17524.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:45:00 | 17461.00 | 17536.83 | 17524.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 17509.00 | 17531.26 | 17523.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 11:45:00 | 17585.00 | 17552.01 | 17533.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 14:30:00 | 17612.00 | 17548.72 | 17537.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 15:15:00 | 17464.00 | 17531.78 | 17530.45 | SL hit (close<static) qty=1.00 sl=17470.00 alert=retest2 |

### Cycle 112 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 17289.00 | 17483.22 | 17508.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 17196.00 | 17425.78 | 17480.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 13:15:00 | 16808.00 | 16764.53 | 16920.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 14:00:00 | 16808.00 | 16764.53 | 16920.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 17006.00 | 16821.53 | 16908.10 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 17056.00 | 16946.73 | 16941.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 10:15:00 | 17182.00 | 16993.78 | 16963.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 11:15:00 | 18035.00 | 18121.95 | 17894.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 12:00:00 | 18035.00 | 18121.95 | 17894.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 17959.00 | 18089.36 | 17900.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:30:00 | 17923.00 | 18089.36 | 17900.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 17948.00 | 18061.09 | 17904.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:00:00 | 17948.00 | 18061.09 | 17904.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 17906.00 | 18030.07 | 17904.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:15:00 | 17850.00 | 18030.07 | 17904.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 17850.00 | 17994.05 | 17899.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 17602.00 | 17994.05 | 17899.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 17570.00 | 17909.24 | 17869.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 17570.00 | 17909.24 | 17869.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 17535.00 | 17834.39 | 17839.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 09:15:00 | 17168.00 | 17568.34 | 17664.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 09:15:00 | 17328.00 | 17286.49 | 17438.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-08 10:00:00 | 17328.00 | 17286.49 | 17438.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 17190.00 | 17190.42 | 17295.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:30:00 | 17266.00 | 17190.42 | 17295.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 17419.00 | 17236.13 | 17307.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 17419.00 | 17236.13 | 17307.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 17229.00 | 17234.71 | 17300.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:30:00 | 17134.00 | 17220.41 | 17282.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 10:15:00 | 17352.00 | 17055.27 | 17030.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 17352.00 | 17055.27 | 17030.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 11:15:00 | 17515.00 | 17147.22 | 17074.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 10:15:00 | 17300.00 | 17321.33 | 17215.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 11:00:00 | 17300.00 | 17321.33 | 17215.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 17200.00 | 17297.06 | 17214.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:45:00 | 17200.00 | 17297.06 | 17214.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 17199.00 | 17277.45 | 17213.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:30:00 | 17190.00 | 17277.45 | 17213.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 17342.00 | 17283.89 | 17226.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 17393.00 | 17287.31 | 17233.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 11:45:00 | 17406.00 | 17319.96 | 17262.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:30:00 | 17348.00 | 17315.22 | 17284.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 10:15:00 | 17181.00 | 17288.38 | 17275.24 | SL hit (close<static) qty=1.00 sl=17219.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 17205.00 | 17259.00 | 17263.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 17100.00 | 17227.20 | 17248.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 17293.00 | 17125.58 | 17178.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 17293.00 | 17125.58 | 17178.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 17293.00 | 17125.58 | 17178.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:45:00 | 17220.00 | 17125.58 | 17178.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 17309.00 | 17162.26 | 17189.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:00:00 | 17309.00 | 17162.26 | 17189.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 15:15:00 | 17302.00 | 17219.90 | 17211.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 09:15:00 | 18126.00 | 17401.12 | 17295.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 13:15:00 | 18376.00 | 18393.70 | 18256.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 14:00:00 | 18376.00 | 18393.70 | 18256.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 18886.00 | 18551.16 | 18426.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 10:45:00 | 19189.00 | 18690.73 | 18501.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 13:45:00 | 19038.00 | 18850.02 | 18629.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 19081.00 | 18877.73 | 18680.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:00:00 | 19010.00 | 18904.19 | 18710.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-12-31 09:15:00 | 20941.80 | 19568.47 | 19158.52 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 10:15:00 | 20770.00 | 20817.87 | 20823.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 15:15:00 | 20580.00 | 20735.33 | 20778.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 19820.00 | 19263.25 | 19621.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 12:15:00 | 19820.00 | 19263.25 | 19621.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 19820.00 | 19263.25 | 19621.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:45:00 | 19750.00 | 19263.25 | 19621.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 19790.00 | 19368.60 | 19636.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 13:30:00 | 19815.00 | 19368.60 | 19636.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 11:15:00 | 19975.00 | 19787.33 | 19769.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 09:15:00 | 20370.00 | 20008.81 | 19891.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 20325.00 | 20327.32 | 20145.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 10:00:00 | 20325.00 | 20327.32 | 20145.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 20415.00 | 20324.49 | 20174.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 15:00:00 | 20560.00 | 20377.18 | 20236.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-21 10:15:00 | 19980.00 | 20624.99 | 20685.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 19980.00 | 20624.99 | 20685.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 13:15:00 | 19590.00 | 20190.79 | 20452.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 20480.00 | 20148.89 | 20358.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 20480.00 | 20148.89 | 20358.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 20480.00 | 20148.89 | 20358.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 20475.00 | 20148.89 | 20358.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 20465.00 | 20212.11 | 20368.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:30:00 | 20550.00 | 20212.11 | 20368.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 20455.00 | 20260.69 | 20376.09 | EMA400 retest candle locked (from downside) |

### Cycle 121 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 20655.00 | 20450.87 | 20444.25 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 19930.00 | 20351.70 | 20404.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 14:15:00 | 19395.00 | 20014.27 | 20224.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 09:15:00 | 19100.00 | 18650.14 | 18863.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 19100.00 | 18650.14 | 18863.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 19100.00 | 18650.14 | 18863.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:00:00 | 19100.00 | 18650.14 | 18863.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 19290.00 | 18778.11 | 18902.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:00:00 | 19290.00 | 18778.11 | 18902.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 19350.00 | 19035.43 | 19002.04 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 18686.00 | 19017.44 | 19037.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 18601.00 | 18929.76 | 18993.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 18897.00 | 18809.52 | 18912.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 13:00:00 | 18897.00 | 18809.52 | 18912.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 19221.00 | 18882.93 | 18927.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 19221.00 | 18882.93 | 18927.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 19300.00 | 18966.34 | 18961.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 19754.00 | 19123.88 | 19033.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 15:15:00 | 21197.00 | 21260.99 | 20898.81 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 09:15:00 | 21881.00 | 21260.99 | 20898.81 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:15:00 | 22975.05 | 22323.81 | 21716.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-02-11 09:15:00 | 24069.10 | 23478.82 | 22689.07 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 126 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 24308.00 | 24993.18 | 25003.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 24168.00 | 24648.19 | 24828.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 24335.00 | 24265.30 | 24557.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 11:30:00 | 24275.00 | 24265.30 | 24557.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 24475.00 | 24290.67 | 24493.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:45:00 | 24482.00 | 24290.67 | 24493.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 24385.00 | 24309.54 | 24483.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:15:00 | 24500.00 | 24309.54 | 24483.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 24467.00 | 24341.03 | 24482.07 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 09:15:00 | 25029.00 | 24520.56 | 24500.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 15:15:00 | 25145.00 | 24863.49 | 24705.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 15:15:00 | 25418.00 | 25425.45 | 25229.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 25200.00 | 25380.36 | 25227.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 25200.00 | 25380.36 | 25227.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:30:00 | 25320.00 | 25380.36 | 25227.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 24782.00 | 25260.69 | 25186.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:00:00 | 24782.00 | 25260.69 | 25186.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 24554.00 | 25119.35 | 25129.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 12:15:00 | 24440.00 | 24983.48 | 25066.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 22415.00 | 22090.65 | 22902.88 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 14:30:00 | 21570.00 | 22173.64 | 22655.62 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 09:15:00 | 21305.00 | 22101.91 | 22579.20 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 20491.50 | 21181.12 | 21754.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 20239.75 | 21181.12 | 21754.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 21300.00 | 20879.76 | 21262.01 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 21300.00 | 20879.76 | 21262.01 | SL hit (close>ema200) qty=0.50 sl=20879.76 alert=retest1 |

### Cycle 129 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 22320.00 | 21500.29 | 21460.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 22685.00 | 21945.15 | 21691.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 22000.00 | 22060.39 | 21819.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 12:30:00 | 22135.00 | 22060.39 | 21819.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 21650.00 | 21948.65 | 21808.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 21650.00 | 21948.65 | 21808.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 21640.00 | 21886.92 | 21793.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 21050.00 | 21886.92 | 21793.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — SELL (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 10:15:00 | 21380.00 | 21667.63 | 21702.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 20880.00 | 21499.47 | 21615.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 20640.00 | 20479.55 | 20802.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 20640.00 | 20479.55 | 20802.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 20640.00 | 20479.55 | 20802.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 20790.00 | 20479.55 | 20802.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 20750.00 | 20533.64 | 20797.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 09:15:00 | 20430.00 | 20533.64 | 20797.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 10:15:00 | 21085.00 | 20645.73 | 20803.14 | SL hit (close>static) qty=1.00 sl=20800.00 alert=retest2 |

### Cycle 131 — BUY (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 15:15:00 | 21040.00 | 20890.63 | 20880.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 21610.00 | 21034.50 | 20946.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 21230.00 | 21432.10 | 21254.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 21230.00 | 21432.10 | 21254.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 21230.00 | 21432.10 | 21254.31 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 20885.00 | 21152.07 | 21162.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 20605.00 | 21042.66 | 21112.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 21310.00 | 21062.10 | 21106.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 21310.00 | 21062.10 | 21106.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 21310.00 | 21062.10 | 21106.90 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 21565.00 | 21162.68 | 21148.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 15:15:00 | 21750.00 | 21395.71 | 21276.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 20625.00 | 21241.57 | 21216.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 09:15:00 | 20625.00 | 21241.57 | 21216.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 20625.00 | 21241.57 | 21216.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 20625.00 | 21241.57 | 21216.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 20525.00 | 21098.25 | 21153.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 20415.00 | 20857.88 | 21028.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 21020.00 | 20716.03 | 20890.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 21020.00 | 20716.03 | 20890.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 21020.00 | 20716.03 | 20890.71 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 21460.00 | 21000.69 | 20987.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 21960.00 | 21378.01 | 21186.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 21510.00 | 21649.01 | 21421.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 15:00:00 | 21510.00 | 21649.01 | 21421.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 21350.00 | 21589.21 | 21415.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 20680.00 | 21589.21 | 21415.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 20725.00 | 21416.37 | 21352.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:30:00 | 20600.00 | 21416.37 | 21352.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 20620.00 | 21257.09 | 21285.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 20545.00 | 21114.67 | 21218.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 20338.00 | 19880.66 | 20288.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 20338.00 | 19880.66 | 20288.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 20338.00 | 19880.66 | 20288.00 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 15:15:00 | 20850.00 | 20435.90 | 20426.82 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 20305.00 | 20409.72 | 20415.74 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 20758.00 | 20469.71 | 20438.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 21085.00 | 20592.77 | 20497.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 10:15:00 | 20580.00 | 20675.34 | 20568.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 10:15:00 | 20580.00 | 20675.34 | 20568.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 20580.00 | 20675.34 | 20568.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 12:30:00 | 20838.00 | 20713.18 | 20603.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 10:15:00 | 20315.00 | 20755.04 | 20694.15 | SL hit (close<static) qty=1.00 sl=20459.00 alert=retest2 |

### Cycle 140 — SELL (started 2026-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 12:15:00 | 20301.00 | 20596.87 | 20628.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 13:15:00 | 20093.00 | 20496.09 | 20579.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 20950.00 | 20421.92 | 20509.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 20950.00 | 20421.92 | 20509.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 20950.00 | 20421.92 | 20509.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:00:00 | 20950.00 | 20421.92 | 20509.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 21440.00 | 20625.54 | 20594.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 11:15:00 | 21701.00 | 20840.63 | 20694.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 10:15:00 | 21945.00 | 22061.97 | 21708.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 11:00:00 | 21945.00 | 22061.97 | 21708.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 22150.00 | 22306.78 | 22007.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 22280.00 | 22306.78 | 22007.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 22983.00 | 22270.47 | 22130.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 13:15:00 | 22230.00 | 22357.67 | 22324.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 09:15:00 | 22041.00 | 22342.43 | 22365.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2026-04-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 09:15:00 | 22041.00 | 22342.43 | 22365.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 14:15:00 | 21862.00 | 21985.76 | 22103.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 11:15:00 | 21914.00 | 21910.79 | 22024.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-22 11:30:00 | 21939.00 | 21910.79 | 22024.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 12:15:00 | 21944.00 | 21917.43 | 22017.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 14:00:00 | 21792.00 | 21892.35 | 21997.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 14:15:00 | 20702.40 | 21087.03 | 21470.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 20800.00 | 20529.79 | 20865.35 | SL hit (close>ema200) qty=0.50 sl=20529.79 alert=retest2 |

### Cycle 143 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 21194.00 | 20614.34 | 20585.20 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 19626.00 | 20540.81 | 20630.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 10:15:00 | 19561.00 | 19963.41 | 20243.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 20342.00 | 19398.22 | 19567.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 20342.00 | 19398.22 | 19567.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 20342.00 | 19398.22 | 19567.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:00:00 | 20342.00 | 19398.22 | 19567.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 20018.00 | 19522.17 | 19608.13 | EMA400 retest candle locked (from downside) |

### Cycle 145 — BUY (started 2026-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 12:15:00 | 20057.00 | 19723.19 | 19690.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 20151.00 | 19808.75 | 19732.28 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-21 13:30:00 | 8625.70 | 2024-05-28 11:15:00 | 8236.12 | PARTIAL | 0.50 | 4.52% |
| SELL | retest2 | 2024-05-21 15:15:00 | 8669.60 | 2024-05-28 12:15:00 | 8194.42 | PARTIAL | 0.50 | 5.48% |
| SELL | retest2 | 2024-05-22 09:30:00 | 8518.05 | 2024-05-28 12:15:00 | 8192.51 | PARTIAL | 0.50 | 3.82% |
| SELL | retest2 | 2024-05-23 12:30:00 | 8623.70 | 2024-05-28 15:15:00 | 8092.15 | PARTIAL | 0.50 | 6.16% |
| SELL | retest2 | 2024-05-21 13:30:00 | 8625.70 | 2024-05-29 09:15:00 | 8269.95 | STOP_HIT | 0.50 | 4.12% |
| SELL | retest2 | 2024-05-21 15:15:00 | 8669.60 | 2024-05-29 09:15:00 | 8269.95 | STOP_HIT | 0.50 | 4.61% |
| SELL | retest2 | 2024-05-22 09:30:00 | 8518.05 | 2024-05-29 09:15:00 | 8269.95 | STOP_HIT | 0.50 | 2.91% |
| SELL | retest2 | 2024-05-23 12:30:00 | 8623.70 | 2024-05-29 09:15:00 | 8269.95 | STOP_HIT | 0.50 | 4.10% |
| BUY | retest2 | 2024-05-31 12:45:00 | 8388.40 | 2024-06-04 11:15:00 | 7892.95 | STOP_HIT | 1.00 | -5.91% |
| BUY | retest2 | 2024-05-31 15:00:00 | 8789.30 | 2024-06-04 11:15:00 | 7892.95 | STOP_HIT | 1.00 | -10.20% |
| BUY | retest2 | 2024-06-11 11:15:00 | 8533.30 | 2024-06-19 13:15:00 | 8968.50 | STOP_HIT | 1.00 | 5.10% |
| SELL | retest2 | 2024-06-21 13:00:00 | 8926.70 | 2024-06-24 12:15:00 | 9078.85 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2024-06-24 09:30:00 | 8969.90 | 2024-06-24 12:15:00 | 9078.85 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-06-26 09:15:00 | 9215.85 | 2024-06-26 13:15:00 | 9050.00 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2024-06-26 11:15:00 | 9223.30 | 2024-06-26 13:15:00 | 9050.00 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-07-09 15:15:00 | 8411.00 | 2024-07-11 15:15:00 | 8531.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-07-10 09:30:00 | 8477.40 | 2024-07-11 15:15:00 | 8531.00 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-07-19 15:15:00 | 8192.00 | 2024-07-23 14:15:00 | 8301.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-07-23 09:15:00 | 8173.80 | 2024-07-23 14:15:00 | 8301.00 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2024-07-23 12:15:00 | 8075.65 | 2024-07-23 14:15:00 | 8301.00 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2024-08-02 11:30:00 | 9097.90 | 2024-08-02 15:15:00 | 9030.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-08-02 12:45:00 | 9048.10 | 2024-08-02 15:15:00 | 9030.00 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2024-08-02 13:30:00 | 9115.80 | 2024-08-02 15:15:00 | 9030.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-08-12 09:15:00 | 8368.20 | 2024-08-19 11:15:00 | 8299.40 | STOP_HIT | 1.00 | 0.82% |
| SELL | retest2 | 2024-08-12 09:45:00 | 8320.00 | 2024-08-19 11:15:00 | 8299.40 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2024-08-12 11:15:00 | 8370.00 | 2024-08-19 11:15:00 | 8299.40 | STOP_HIT | 1.00 | 0.84% |
| SELL | retest2 | 2024-08-12 13:00:00 | 8362.90 | 2024-08-19 11:15:00 | 8299.40 | STOP_HIT | 1.00 | 0.76% |
| SELL | retest2 | 2024-08-13 14:30:00 | 8252.15 | 2024-08-19 11:15:00 | 8299.40 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2024-08-16 10:45:00 | 8265.15 | 2024-08-19 11:15:00 | 8299.40 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2024-08-16 11:30:00 | 8283.35 | 2024-08-19 11:15:00 | 8299.40 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2024-08-16 14:30:00 | 8273.95 | 2024-08-19 11:15:00 | 8299.40 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2024-08-20 09:15:00 | 8517.95 | 2024-08-26 15:15:00 | 8572.00 | STOP_HIT | 1.00 | 0.63% |
| SELL | retest2 | 2024-09-13 15:15:00 | 7189.00 | 2024-09-19 09:15:00 | 6829.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-13 15:15:00 | 7189.00 | 2024-09-20 09:15:00 | 6863.90 | STOP_HIT | 0.50 | 4.52% |
| SELL | retest2 | 2024-10-17 10:00:00 | 6958.00 | 2024-10-22 10:15:00 | 6610.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 10:00:00 | 6958.00 | 2024-10-23 13:15:00 | 6262.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-18 10:15:00 | 6715.00 | 2024-11-25 09:15:00 | 6947.80 | STOP_HIT | 1.00 | -3.47% |
| SELL | retest2 | 2024-11-18 12:45:00 | 6731.80 | 2024-11-25 09:15:00 | 6947.80 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2024-11-18 14:00:00 | 6724.20 | 2024-11-25 09:15:00 | 6947.80 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2024-11-19 14:15:00 | 6732.30 | 2024-11-25 09:15:00 | 6947.80 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2024-11-22 13:30:00 | 6676.20 | 2024-11-25 09:15:00 | 6947.80 | STOP_HIT | 1.00 | -4.07% |
| BUY | retest1 | 2024-12-05 09:15:00 | 7040.00 | 2024-12-05 10:15:00 | 6900.00 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-01-09 10:45:00 | 6737.40 | 2025-01-13 11:15:00 | 6400.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 10:45:00 | 6737.40 | 2025-01-14 09:15:00 | 6418.50 | STOP_HIT | 0.50 | 4.73% |
| SELL | retest2 | 2025-01-29 13:00:00 | 6384.35 | 2025-01-30 09:15:00 | 6470.35 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-02-01 10:30:00 | 6327.95 | 2025-02-01 14:15:00 | 6490.30 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-02-01 11:45:00 | 6307.85 | 2025-02-01 14:15:00 | 6490.30 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2025-02-01 12:15:00 | 6325.00 | 2025-02-01 14:15:00 | 6490.30 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2025-02-01 13:00:00 | 6317.65 | 2025-02-01 14:15:00 | 6490.30 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-03-17 11:00:00 | 7179.15 | 2025-03-18 11:15:00 | 7324.40 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-03-18 09:45:00 | 7131.55 | 2025-03-18 11:15:00 | 7324.40 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2025-03-28 09:15:00 | 9195.05 | 2025-04-04 09:15:00 | 8855.55 | STOP_HIT | 1.00 | -3.69% |
| BUY | retest2 | 2025-04-01 12:15:00 | 8883.00 | 2025-04-04 09:15:00 | 8855.55 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-04-01 15:00:00 | 8885.30 | 2025-04-04 11:15:00 | 8862.90 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-04-02 10:30:00 | 9035.00 | 2025-04-04 11:15:00 | 8862.90 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-04-03 14:00:00 | 9116.80 | 2025-04-04 11:15:00 | 8862.90 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2025-04-03 14:45:00 | 9135.30 | 2025-04-04 11:15:00 | 8862.90 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2025-04-23 09:15:00 | 9214.00 | 2025-04-23 09:15:00 | 8972.50 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-04-23 15:15:00 | 9179.00 | 2025-04-25 09:15:00 | 8962.50 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-04-24 09:30:00 | 9249.50 | 2025-04-25 09:15:00 | 8962.50 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest2 | 2025-05-07 11:30:00 | 10190.00 | 2025-05-08 14:15:00 | 9747.00 | STOP_HIT | 1.00 | -4.35% |
| BUY | retest2 | 2025-05-07 13:15:00 | 10128.00 | 2025-05-08 14:15:00 | 9747.00 | STOP_HIT | 1.00 | -3.76% |
| BUY | retest2 | 2025-05-15 09:15:00 | 10549.00 | 2025-05-19 13:15:00 | 10605.50 | STOP_HIT | 1.00 | 0.54% |
| BUY | retest1 | 2025-06-05 09:15:00 | 12498.00 | 2025-06-06 09:15:00 | 12397.00 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-06-10 11:30:00 | 12328.00 | 2025-06-12 11:15:00 | 12687.00 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2025-06-12 11:00:00 | 12328.00 | 2025-06-12 11:15:00 | 12687.00 | STOP_HIT | 1.00 | -2.91% |
| BUY | retest2 | 2025-06-18 11:30:00 | 14180.00 | 2025-06-19 15:15:00 | 13659.00 | STOP_HIT | 1.00 | -3.67% |
| BUY | retest2 | 2025-07-02 09:15:00 | 14968.00 | 2025-07-02 11:15:00 | 14711.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-07-11 14:30:00 | 16398.00 | 2025-07-18 14:15:00 | 16615.00 | STOP_HIT | 1.00 | 1.32% |
| BUY | retest2 | 2025-07-24 09:15:00 | 19005.00 | 2025-07-28 09:15:00 | 17823.00 | STOP_HIT | 1.00 | -6.22% |
| BUY | retest2 | 2025-07-25 10:45:00 | 17380.00 | 2025-07-28 09:15:00 | 17823.00 | STOP_HIT | 1.00 | 2.55% |
| SELL | retest2 | 2025-08-05 10:30:00 | 17008.00 | 2025-08-05 11:15:00 | 17422.00 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2025-08-11 09:15:00 | 18091.00 | 2025-08-13 13:15:00 | 19900.10 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-29 12:45:00 | 19829.00 | 2025-09-01 11:15:00 | 18837.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-29 12:45:00 | 19829.00 | 2025-09-02 09:15:00 | 19285.00 | STOP_HIT | 0.50 | 2.74% |
| SELL | retest2 | 2025-09-19 11:45:00 | 19036.00 | 2025-09-19 15:15:00 | 18084.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 11:45:00 | 19036.00 | 2025-09-23 13:15:00 | 18205.00 | STOP_HIT | 0.50 | 4.37% |
| BUY | retest2 | 2025-10-17 09:15:00 | 17162.00 | 2025-10-23 14:15:00 | 16850.00 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-11-12 12:30:00 | 17664.00 | 2025-11-17 09:15:00 | 17725.00 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2025-11-20 11:45:00 | 17585.00 | 2025-11-20 15:15:00 | 17464.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-11-20 14:30:00 | 17612.00 | 2025-11-20 15:15:00 | 17464.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-12-09 14:30:00 | 17134.00 | 2025-12-12 10:15:00 | 17352.00 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-12-16 09:15:00 | 17393.00 | 2025-12-17 10:15:00 | 17181.00 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-12-16 11:45:00 | 17406.00 | 2025-12-17 10:15:00 | 17181.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-12-17 09:30:00 | 17348.00 | 2025-12-17 10:15:00 | 17181.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-12-29 10:45:00 | 19189.00 | 2025-12-31 09:15:00 | 20941.80 | TARGET_HIT | 1.00 | 9.13% |
| BUY | retest2 | 2025-12-29 13:45:00 | 19038.00 | 2025-12-31 09:15:00 | 20911.00 | TARGET_HIT | 1.00 | 9.84% |
| BUY | retest2 | 2025-12-30 09:15:00 | 19081.00 | 2026-01-02 09:15:00 | 21107.90 | TARGET_HIT | 1.00 | 10.62% |
| BUY | retest2 | 2025-12-30 10:00:00 | 19010.00 | 2026-01-02 09:15:00 | 20989.10 | TARGET_HIT | 1.00 | 10.41% |
| BUY | retest2 | 2026-01-07 09:45:00 | 21005.00 | 2026-01-07 10:15:00 | 20770.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2026-01-16 15:00:00 | 20560.00 | 2026-01-21 10:15:00 | 19980.00 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest1 | 2026-02-09 09:15:00 | 21881.00 | 2026-02-10 09:15:00 | 22975.05 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-02-09 09:15:00 | 21881.00 | 2026-02-11 09:15:00 | 24069.10 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-13 12:15:00 | 23868.00 | 2026-02-18 09:15:00 | 26254.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-13 12:45:00 | 23741.00 | 2026-02-18 09:15:00 | 26115.10 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2026-03-05 14:30:00 | 21570.00 | 2026-03-09 09:15:00 | 20491.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-03-06 09:15:00 | 21305.00 | 2026-03-09 09:15:00 | 20239.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-03-05 14:30:00 | 21570.00 | 2026-03-10 09:15:00 | 21300.00 | STOP_HIT | 0.50 | 1.25% |
| SELL | retest1 | 2026-03-06 09:15:00 | 21305.00 | 2026-03-10 09:15:00 | 21300.00 | STOP_HIT | 0.50 | 0.02% |
| SELL | retest2 | 2026-03-17 09:15:00 | 20430.00 | 2026-03-17 10:15:00 | 21085.00 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2026-04-06 12:30:00 | 20838.00 | 2026-04-07 10:15:00 | 20315.00 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2026-04-13 10:15:00 | 22280.00 | 2026-04-20 09:15:00 | 22041.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2026-04-15 09:15:00 | 22983.00 | 2026-04-20 09:15:00 | 22041.00 | STOP_HIT | 1.00 | -4.10% |
| BUY | retest2 | 2026-04-16 13:15:00 | 22230.00 | 2026-04-20 09:15:00 | 22041.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-04-22 14:00:00 | 21792.00 | 2026-04-23 14:15:00 | 20702.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-22 14:00:00 | 21792.00 | 2026-04-27 09:15:00 | 20800.00 | STOP_HIT | 0.50 | 4.55% |
