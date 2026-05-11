# Bajaj Auto Ltd. (BAJAJ-AUTO)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 10696.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 81 |
| ALERT1 | 57 |
| ALERT2 | 56 |
| ALERT2_SKIP | 34 |
| ALERT3 | 153 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 86 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 86 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 87 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 62
- **Target hits / Stop hits / Partials:** 0 / 86 / 1
- **Avg / median % per leg:** -0.21% / -0.74%
- **Sum % (uncompounded):** -18.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 41 | 11 | 26.8% | 0 | 41 | 0 | -0.48% | -19.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 41 | 11 | 26.8% | 0 | 41 | 0 | -0.48% | -19.8% |
| SELL (all) | 46 | 14 | 30.4% | 0 | 45 | 1 | 0.03% | 1.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 46 | 14 | 30.4% | 0 | 45 | 1 | 0.03% | 1.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 87 | 25 | 28.7% | 0 | 86 | 1 | -0.21% | -18.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 7926.00 | 7815.83 | 7804.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 8000.00 | 7852.66 | 7822.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 13:15:00 | 8068.00 | 8072.39 | 8011.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 14:00:00 | 8068.00 | 8072.39 | 8011.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 8606.50 | 8725.33 | 8607.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 8606.50 | 8725.33 | 8607.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 8557.00 | 8691.66 | 8602.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 8557.00 | 8691.66 | 8602.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 8600.00 | 8673.33 | 8602.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:15:00 | 8573.00 | 8673.33 | 8602.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 8735.00 | 8685.66 | 8614.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 10:15:00 | 8750.00 | 8685.66 | 8614.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 10:15:00 | 8531.00 | 8628.96 | 8622.45 | SL hit (close<static) qty=1.00 sl=8535.50 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 13:15:00 | 8819.00 | 8858.95 | 8864.31 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 15:15:00 | 8899.00 | 8870.57 | 8868.88 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 8662.00 | 8828.86 | 8850.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 8608.00 | 8701.96 | 8771.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 10:15:00 | 8535.00 | 8529.76 | 8606.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-03 10:30:00 | 8539.00 | 8529.76 | 8606.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 8586.00 | 8553.03 | 8599.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 14:45:00 | 8565.00 | 8554.92 | 8595.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 09:15:00 | 8616.00 | 8569.39 | 8595.56 | SL hit (close>static) qty=1.00 sl=8600.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 10:15:00 | 8634.00 | 8580.76 | 8580.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 09:15:00 | 8743.00 | 8651.98 | 8631.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 8696.00 | 8708.88 | 8678.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 8696.00 | 8708.88 | 8678.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 8696.00 | 8708.88 | 8678.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:00:00 | 8696.00 | 8708.88 | 8678.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 8684.50 | 8704.01 | 8678.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:45:00 | 8675.00 | 8704.01 | 8678.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 8664.50 | 8696.11 | 8677.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 8664.50 | 8696.11 | 8677.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 8675.00 | 8691.88 | 8677.23 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 8561.50 | 8649.03 | 8659.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 15:15:00 | 8550.00 | 8629.22 | 8649.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 8500.00 | 8492.21 | 8547.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 10:15:00 | 8514.00 | 8492.21 | 8547.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 8566.00 | 8506.97 | 8549.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 8566.00 | 8506.97 | 8549.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 8554.50 | 8516.47 | 8549.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 15:00:00 | 8530.00 | 8534.96 | 8551.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 8496.50 | 8535.57 | 8550.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 14:15:00 | 8527.50 | 8513.33 | 8530.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 09:15:00 | 8635.00 | 8534.05 | 8534.89 | SL hit (close>static) qty=1.00 sl=8595.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 10:15:00 | 8556.50 | 8538.54 | 8536.86 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 12:15:00 | 8521.00 | 8533.50 | 8534.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 15:15:00 | 8447.50 | 8513.28 | 8525.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 10:15:00 | 8542.50 | 8519.08 | 8525.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 10:15:00 | 8542.50 | 8519.08 | 8525.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 8542.50 | 8519.08 | 8525.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:00:00 | 8542.50 | 8519.08 | 8525.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 8542.00 | 8523.66 | 8527.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:45:00 | 8517.00 | 8523.66 | 8527.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 8500.00 | 8518.93 | 8524.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 13:15:00 | 8494.00 | 8518.93 | 8524.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 15:15:00 | 8493.00 | 8512.62 | 8520.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 12:15:00 | 8397.50 | 8383.83 | 8383.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 12:15:00 | 8397.50 | 8383.83 | 8383.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 09:15:00 | 8428.00 | 8395.00 | 8388.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 11:15:00 | 8388.50 | 8395.86 | 8390.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 11:15:00 | 8388.50 | 8395.86 | 8390.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 8388.50 | 8395.86 | 8390.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:00:00 | 8388.50 | 8395.86 | 8390.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 8370.50 | 8390.79 | 8388.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:45:00 | 8361.50 | 8390.79 | 8388.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 8411.50 | 8394.93 | 8390.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 14:15:00 | 8422.50 | 8394.93 | 8390.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 14:15:00 | 8380.00 | 8419.81 | 8422.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-06-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 14:15:00 | 8380.00 | 8419.81 | 8422.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 10:15:00 | 8338.50 | 8390.76 | 8407.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 12:15:00 | 8404.50 | 8392.10 | 8405.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 12:15:00 | 8404.50 | 8392.10 | 8405.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 8404.50 | 8392.10 | 8405.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:00:00 | 8404.50 | 8392.10 | 8405.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 8389.50 | 8391.58 | 8403.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 09:45:00 | 8370.00 | 8387.78 | 8399.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 10:30:00 | 8362.00 | 8380.83 | 8394.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 09:15:00 | 8428.00 | 8375.68 | 8383.85 | SL hit (close>static) qty=1.00 sl=8409.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 11:15:00 | 8406.50 | 8388.94 | 8388.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 12:15:00 | 8437.00 | 8398.55 | 8393.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 14:15:00 | 8385.50 | 8399.61 | 8394.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 14:15:00 | 8385.50 | 8399.61 | 8394.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 8385.50 | 8399.61 | 8394.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:45:00 | 8385.00 | 8399.61 | 8394.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 8392.50 | 8398.19 | 8394.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:15:00 | 8393.00 | 8398.19 | 8394.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 8414.00 | 8412.38 | 8403.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:00:00 | 8414.00 | 8412.38 | 8403.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 8401.00 | 8410.11 | 8403.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 14:00:00 | 8401.00 | 8410.11 | 8403.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 8434.00 | 8414.89 | 8406.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 09:30:00 | 8441.50 | 8421.57 | 8410.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 8367.50 | 8433.75 | 8426.35 | SL hit (close<static) qty=1.00 sl=8400.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 10:15:00 | 8309.50 | 8408.90 | 8415.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 15:15:00 | 8271.50 | 8315.32 | 8342.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 8135.00 | 8103.64 | 8155.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 8135.00 | 8103.64 | 8155.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 8135.00 | 8103.64 | 8155.37 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 13:15:00 | 8247.00 | 8189.43 | 8184.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 8315.00 | 8214.55 | 8196.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 8210.00 | 8229.71 | 8207.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 8210.00 | 8229.71 | 8207.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 8210.00 | 8229.71 | 8207.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:45:00 | 8209.50 | 8229.71 | 8207.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 8266.50 | 8237.07 | 8212.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 11:30:00 | 8270.00 | 8244.85 | 8218.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 14:15:00 | 8301.00 | 8345.10 | 8349.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 14:15:00 | 8301.00 | 8345.10 | 8349.47 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 10:15:00 | 8385.00 | 8351.32 | 8350.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 11:15:00 | 8395.50 | 8360.16 | 8354.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 09:15:00 | 8376.00 | 8379.76 | 8368.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 8376.00 | 8379.76 | 8368.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 8376.00 | 8379.76 | 8368.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:30:00 | 8382.50 | 8379.76 | 8368.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 8355.00 | 8374.80 | 8366.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:45:00 | 8355.50 | 8374.80 | 8366.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 8360.00 | 8371.84 | 8366.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:00:00 | 8360.00 | 8371.84 | 8366.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 8351.50 | 8367.77 | 8365.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:00:00 | 8351.50 | 8367.77 | 8365.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 13:15:00 | 8315.00 | 8357.22 | 8360.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 14:15:00 | 8285.50 | 8342.88 | 8353.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 11:15:00 | 8104.50 | 8101.65 | 8145.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 11:30:00 | 8089.00 | 8101.65 | 8145.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 8123.50 | 8113.93 | 8140.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 8123.50 | 8113.93 | 8140.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 8127.00 | 8116.54 | 8139.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 09:15:00 | 8106.50 | 8116.54 | 8139.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 09:45:00 | 8107.00 | 8116.23 | 8137.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 10:00:00 | 8115.50 | 8057.04 | 8059.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 10:15:00 | 8152.00 | 8076.03 | 8067.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 10:15:00 | 8152.00 | 8076.03 | 8067.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 12:15:00 | 8179.00 | 8108.46 | 8084.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 13:15:00 | 8186.00 | 8212.05 | 8179.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 13:15:00 | 8186.00 | 8212.05 | 8179.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 8186.00 | 8212.05 | 8179.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:00:00 | 8186.00 | 8212.05 | 8179.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 8182.00 | 8206.04 | 8179.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:30:00 | 8176.50 | 8206.04 | 8179.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 8175.00 | 8199.83 | 8179.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 8133.00 | 8199.83 | 8179.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 8126.50 | 8185.16 | 8174.71 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 8045.00 | 8157.13 | 8162.92 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 8221.00 | 8166.68 | 8164.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 15:15:00 | 8245.00 | 8182.35 | 8171.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 13:15:00 | 8203.50 | 8210.34 | 8192.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 13:15:00 | 8203.50 | 8210.34 | 8192.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 13:15:00 | 8203.50 | 8210.34 | 8192.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 14:00:00 | 8203.50 | 8210.34 | 8192.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 8225.50 | 8213.37 | 8195.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:15:00 | 8218.00 | 8213.37 | 8195.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 8218.00 | 8214.30 | 8197.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:30:00 | 8200.50 | 8207.14 | 8195.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 8146.00 | 8194.91 | 8190.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:00:00 | 8146.00 | 8194.91 | 8190.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 8169.00 | 8189.73 | 8188.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:45:00 | 8159.00 | 8189.73 | 8188.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 8252.50 | 8203.93 | 8195.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 14:30:00 | 8258.50 | 8217.94 | 8202.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 8282.00 | 8221.35 | 8205.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 12:30:00 | 8258.50 | 8246.67 | 8224.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 13:45:00 | 8263.00 | 8244.83 | 8225.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 8196.00 | 8235.07 | 8223.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 15:00:00 | 8196.00 | 8235.07 | 8223.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 8186.00 | 8225.25 | 8219.78 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-12 15:15:00 | 8186.00 | 8225.25 | 8219.78 | SL hit (close<static) qty=1.00 sl=8194.50 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 13:15:00 | 8229.00 | 8233.13 | 8233.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 15:15:00 | 8210.50 | 8225.30 | 8229.39 | Break + close below crossover candle low |

### Cycle 21 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 8573.00 | 8294.84 | 8260.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 8783.00 | 8551.73 | 8426.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 09:15:00 | 8794.00 | 8806.33 | 8712.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 09:30:00 | 8745.00 | 8806.33 | 8712.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 8715.50 | 8783.47 | 8718.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 8715.50 | 8783.47 | 8718.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 8649.50 | 8756.68 | 8712.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:00:00 | 8649.50 | 8756.68 | 8712.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 8718.50 | 8749.04 | 8712.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:30:00 | 8639.00 | 8749.04 | 8712.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 8679.00 | 8735.03 | 8709.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 8679.00 | 8735.03 | 8709.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 8690.00 | 8726.03 | 8707.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 8651.00 | 8726.03 | 8707.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 8607.00 | 8680.30 | 8688.79 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 12:15:00 | 8735.00 | 8693.24 | 8690.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 14:15:00 | 8750.00 | 8711.91 | 8700.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 14:15:00 | 8668.50 | 8740.42 | 8727.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 14:15:00 | 8668.50 | 8740.42 | 8727.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 8668.50 | 8740.42 | 8727.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 15:00:00 | 8668.50 | 8740.42 | 8727.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 8676.00 | 8727.53 | 8722.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 8691.50 | 8727.53 | 8722.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 13:15:00 | 8701.00 | 8717.38 | 8719.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 13:15:00 | 8701.00 | 8717.38 | 8719.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 8625.50 | 8690.99 | 8706.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 8732.50 | 8673.45 | 8686.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 8732.50 | 8673.45 | 8686.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 8732.50 | 8673.45 | 8686.62 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 8847.00 | 8708.16 | 8701.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 8913.50 | 8749.23 | 8720.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 14:15:00 | 9056.50 | 9115.84 | 9062.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 14:15:00 | 9056.50 | 9115.84 | 9062.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 9056.50 | 9115.84 | 9062.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 9056.50 | 9115.84 | 9062.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 9095.00 | 9111.67 | 9065.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:15:00 | 9113.50 | 9111.67 | 9065.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 10:15:00 | 9126.50 | 9107.24 | 9067.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:15:00 | 9132.50 | 9090.53 | 9075.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 15:15:00 | 9232.50 | 9278.42 | 9283.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 15:15:00 | 9232.50 | 9278.42 | 9283.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 09:15:00 | 9171.50 | 9257.04 | 9272.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 14:15:00 | 9029.00 | 9017.28 | 9071.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 15:00:00 | 9029.00 | 9017.28 | 9071.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 9066.50 | 9029.24 | 9067.60 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 9111.00 | 9077.32 | 9074.53 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 10:15:00 | 9067.00 | 9075.55 | 9075.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 12:15:00 | 9033.50 | 9065.93 | 9071.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 14:15:00 | 9075.50 | 9066.18 | 9070.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 14:15:00 | 9075.50 | 9066.18 | 9070.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 9075.50 | 9066.18 | 9070.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 15:00:00 | 9075.50 | 9066.18 | 9070.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 9080.00 | 9068.94 | 9071.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 9004.00 | 9068.94 | 9071.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 9109.00 | 9026.99 | 9038.99 | SL hit (close>static) qty=1.00 sl=9080.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 10:15:00 | 9130.00 | 9047.59 | 9047.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 11:15:00 | 9175.50 | 9073.18 | 9058.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 9063.00 | 9093.01 | 9073.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 14:15:00 | 9063.00 | 9093.01 | 9073.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 9063.00 | 9093.01 | 9073.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:45:00 | 9074.00 | 9093.01 | 9073.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 9033.00 | 9081.01 | 9069.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 09:15:00 | 9143.00 | 9081.01 | 9069.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 10:00:00 | 9103.00 | 9085.41 | 9072.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 11:15:00 | 9072.00 | 9081.33 | 9072.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 13:15:00 | 8998.50 | 9055.96 | 9062.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 13:15:00 | 8998.50 | 9055.96 | 9062.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 14:15:00 | 8992.50 | 9043.27 | 9056.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 13:15:00 | 8867.00 | 8860.12 | 8913.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 13:45:00 | 8865.50 | 8860.12 | 8913.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 8835.50 | 8855.20 | 8906.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 15:15:00 | 8827.50 | 8855.20 | 8906.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 10:00:00 | 8815.00 | 8842.73 | 8891.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 13:00:00 | 8815.00 | 8838.30 | 8877.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:30:00 | 8806.00 | 8710.47 | 8742.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 8684.00 | 8705.18 | 8737.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 11:45:00 | 8664.50 | 8698.14 | 8731.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 12:30:00 | 8644.00 | 8687.21 | 8723.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 13:30:00 | 8638.00 | 8625.41 | 8660.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 09:15:00 | 8664.00 | 8644.80 | 8663.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 8722.00 | 8668.43 | 8671.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:00:00 | 8722.00 | 8668.43 | 8671.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-06 11:15:00 | 8700.00 | 8674.75 | 8674.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 11:15:00 | 8700.00 | 8674.75 | 8674.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 12:15:00 | 8727.00 | 8685.20 | 8679.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 8804.00 | 8838.17 | 8786.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 10:00:00 | 8804.00 | 8838.17 | 8786.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 8816.50 | 8833.84 | 8789.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:30:00 | 8823.00 | 8833.84 | 8789.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 8797.50 | 8825.73 | 8799.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 8797.50 | 8825.73 | 8799.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 8769.00 | 8814.39 | 8796.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 8778.00 | 8814.39 | 8796.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 8723.50 | 8796.21 | 8790.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:00:00 | 8723.50 | 8796.21 | 8790.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 8782.00 | 8793.37 | 8789.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 11:45:00 | 8795.00 | 8793.49 | 8789.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 14:30:00 | 8792.00 | 8796.40 | 8792.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 9120.50 | 9125.25 | 9125.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 10:15:00 | 9120.50 | 9125.25 | 9125.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 13:15:00 | 9078.00 | 9105.94 | 9115.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 13:15:00 | 9073.50 | 9067.62 | 9085.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-24 14:00:00 | 9073.50 | 9067.62 | 9085.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 9062.00 | 9066.50 | 9083.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 12:00:00 | 9040.00 | 9068.25 | 9078.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:45:00 | 9039.00 | 9065.88 | 9074.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 9003.50 | 9064.11 | 9072.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 13:00:00 | 9035.50 | 9055.70 | 9065.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 9000.50 | 9035.68 | 9052.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 13:15:00 | 8957.50 | 9017.39 | 9038.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 12:45:00 | 8893.00 | 8966.22 | 8997.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 15:15:00 | 8770.00 | 8750.02 | 8747.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 15:15:00 | 8770.00 | 8750.02 | 8747.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 11:15:00 | 8866.00 | 8779.61 | 8762.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 13:15:00 | 8815.50 | 8857.49 | 8828.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 13:15:00 | 8815.50 | 8857.49 | 8828.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 8815.50 | 8857.49 | 8828.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:00:00 | 8815.50 | 8857.49 | 8828.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 8870.50 | 8860.09 | 8832.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 8876.50 | 8857.07 | 8833.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:45:00 | 8886.50 | 8863.26 | 8838.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 10:45:00 | 8882.00 | 8868.01 | 8843.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 11:30:00 | 8873.50 | 8867.71 | 8845.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 8861.50 | 8864.94 | 8850.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 8804.00 | 8864.94 | 8850.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 8815.00 | 8854.95 | 8847.62 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-14 10:15:00 | 8802.00 | 8844.36 | 8843.48 | SL hit (close<static) qty=1.00 sl=8814.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 8810.50 | 8837.59 | 8840.48 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 8980.00 | 8864.81 | 8851.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 11:15:00 | 9003.00 | 8911.52 | 8875.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 14:15:00 | 8917.00 | 8958.45 | 8932.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 14:15:00 | 8917.00 | 8958.45 | 8932.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 8917.00 | 8958.45 | 8932.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:00:00 | 8917.00 | 8958.45 | 8932.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 8915.00 | 8949.76 | 8930.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:15:00 | 8957.50 | 8949.76 | 8930.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 8920.00 | 8942.73 | 8930.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:00:00 | 8920.00 | 8942.73 | 8930.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 8926.00 | 8939.38 | 8930.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:15:00 | 8926.50 | 8939.38 | 8930.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 8929.50 | 8937.40 | 8930.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:15:00 | 8891.00 | 8937.40 | 8930.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 8900.00 | 8929.92 | 8927.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:45:00 | 8876.50 | 8929.92 | 8927.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2025-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 14:15:00 | 8884.00 | 8920.74 | 8923.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 15:15:00 | 8868.50 | 8910.29 | 8918.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 8918.00 | 8911.83 | 8918.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 8918.00 | 8911.83 | 8918.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 8918.00 | 8911.83 | 8918.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:30:00 | 8916.00 | 8911.83 | 8918.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 8997.00 | 8928.87 | 8925.55 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 8900.50 | 8932.49 | 8934.36 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 10:15:00 | 9006.00 | 8942.86 | 8935.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 09:15:00 | 9052.00 | 9001.66 | 8972.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 14:15:00 | 9045.00 | 9057.45 | 9015.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-25 15:00:00 | 9045.00 | 9057.45 | 9015.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 9050.00 | 9124.57 | 9091.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:00:00 | 9050.00 | 9124.57 | 9091.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 9022.00 | 9104.06 | 9084.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:00:00 | 9022.00 | 9104.06 | 9084.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 15:15:00 | 9040.00 | 9067.26 | 9070.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 8955.50 | 9044.91 | 9060.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 12:15:00 | 9034.00 | 9024.08 | 9044.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-28 13:00:00 | 9034.00 | 9024.08 | 9044.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 9042.00 | 9027.66 | 9044.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:15:00 | 9060.00 | 9027.66 | 9044.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 9073.50 | 9036.83 | 9047.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 15:00:00 | 9073.50 | 9036.83 | 9047.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 9080.00 | 9045.47 | 9050.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:15:00 | 9089.50 | 9045.47 | 9050.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 9162.50 | 9068.87 | 9060.44 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 9023.00 | 9065.78 | 9068.49 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 14:15:00 | 9088.50 | 9068.09 | 9067.64 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 9014.00 | 9057.90 | 9063.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 12:15:00 | 8980.00 | 9029.52 | 9047.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 9059.00 | 9022.25 | 9036.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 9059.00 | 9022.25 | 9036.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 9059.00 | 9022.25 | 9036.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:45:00 | 9070.00 | 9022.25 | 9036.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 9035.00 | 9024.80 | 9036.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 11:30:00 | 9017.00 | 9027.84 | 9036.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 14:15:00 | 9082.50 | 9049.25 | 9045.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 14:15:00 | 9082.50 | 9049.25 | 9045.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 10:15:00 | 9097.00 | 9069.67 | 9056.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 9016.00 | 9078.66 | 9069.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 9016.00 | 9078.66 | 9069.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 9016.00 | 9078.66 | 9069.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 9016.00 | 9078.66 | 9069.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 9041.50 | 9071.23 | 9067.17 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 9004.00 | 9057.79 | 9061.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 8980.00 | 9038.58 | 9051.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 13:15:00 | 8968.50 | 8962.16 | 8986.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-10 14:00:00 | 8968.50 | 8962.16 | 8986.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 8993.00 | 8968.33 | 8987.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 15:00:00 | 8993.00 | 8968.33 | 8987.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 8980.00 | 8970.67 | 8986.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 8988.50 | 8970.67 | 8986.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 9031.00 | 8982.73 | 8990.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:00:00 | 9031.00 | 8982.73 | 8990.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 9038.50 | 8993.89 | 8995.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:00:00 | 9038.50 | 8993.89 | 8995.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 9080.00 | 9011.11 | 9002.90 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 8925.00 | 8997.16 | 9006.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 10:15:00 | 8900.00 | 8977.73 | 8997.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 11:15:00 | 8958.50 | 8948.80 | 8967.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 11:15:00 | 8958.50 | 8948.80 | 8967.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 8958.50 | 8948.80 | 8967.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:45:00 | 8960.50 | 8948.80 | 8967.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 8974.50 | 8953.94 | 8968.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 8974.50 | 8953.94 | 8968.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 8990.00 | 8961.15 | 8970.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:45:00 | 8990.50 | 8961.15 | 8970.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 9014.50 | 8971.82 | 8974.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 15:00:00 | 9014.50 | 8971.82 | 8974.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 8989.00 | 8975.26 | 8975.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 09:30:00 | 8961.50 | 8970.61 | 8973.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 12:15:00 | 8963.50 | 8900.26 | 8898.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 12:15:00 | 8963.50 | 8900.26 | 8898.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 8999.00 | 8932.13 | 8913.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 12:15:00 | 9094.00 | 9110.80 | 9056.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 13:00:00 | 9094.00 | 9110.80 | 9056.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 9110.50 | 9106.11 | 9071.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 11:15:00 | 9156.00 | 9110.39 | 9076.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:45:00 | 9150.00 | 9147.68 | 9114.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 14:15:00 | 9065.00 | 9112.52 | 9108.77 | SL hit (close<static) qty=1.00 sl=9070.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 15:15:00 | 9069.00 | 9103.82 | 9105.15 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 09:15:00 | 9166.50 | 9104.79 | 9102.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 10:15:00 | 9207.00 | 9125.23 | 9112.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 10:15:00 | 9421.00 | 9448.15 | 9372.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 10:45:00 | 9416.50 | 9448.15 | 9372.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 9484.50 | 9536.46 | 9482.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 9484.50 | 9536.46 | 9482.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 9499.00 | 9521.45 | 9484.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 9650.00 | 9521.45 | 9484.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 13:15:00 | 9563.00 | 9684.36 | 9700.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 9563.00 | 9684.36 | 9700.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 14:15:00 | 9555.00 | 9658.49 | 9686.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 9558.50 | 9509.00 | 9546.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 14:15:00 | 9558.50 | 9509.00 | 9546.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 9558.50 | 9509.00 | 9546.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 15:00:00 | 9558.50 | 9509.00 | 9546.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 9550.00 | 9517.20 | 9546.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 9603.00 | 9517.20 | 9546.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 9541.00 | 9521.96 | 9545.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:30:00 | 9566.00 | 9521.96 | 9545.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 9570.50 | 9531.67 | 9548.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 9570.50 | 9531.67 | 9548.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 9539.00 | 9533.14 | 9547.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:30:00 | 9561.50 | 9533.14 | 9547.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 9549.50 | 9536.41 | 9547.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:45:00 | 9517.00 | 9535.03 | 9545.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:15:00 | 9518.50 | 9535.03 | 9545.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 14:15:00 | 9588.50 | 9545.72 | 9549.79 | SL hit (close>static) qty=1.00 sl=9562.50 alert=retest2 |

### Cycle 53 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 9559.50 | 9552.36 | 9552.35 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 10:15:00 | 9471.00 | 9536.09 | 9544.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 13:15:00 | 9403.50 | 9493.64 | 9522.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 9291.00 | 9211.69 | 9268.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 9291.00 | 9211.69 | 9268.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 9291.00 | 9211.69 | 9268.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 9304.50 | 9211.69 | 9268.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 9212.50 | 9211.85 | 9263.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:45:00 | 9188.00 | 9212.88 | 9259.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 9368.00 | 9262.99 | 9272.31 | SL hit (close>static) qty=1.00 sl=9297.50 alert=retest2 |

### Cycle 55 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 9387.00 | 9287.80 | 9282.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 9411.50 | 9312.54 | 9294.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 15:15:00 | 9390.00 | 9392.34 | 9352.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-27 09:15:00 | 9398.50 | 9392.34 | 9352.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 9400.00 | 9393.87 | 9356.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 15:00:00 | 9503.00 | 9424.11 | 9386.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 9270.50 | 9405.53 | 9385.17 | SL hit (close<static) qty=1.00 sl=9330.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 9280.50 | 9461.21 | 9475.97 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 9654.50 | 9494.35 | 9479.35 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 11:15:00 | 9517.50 | 9587.97 | 9595.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 12:15:00 | 9494.00 | 9569.18 | 9586.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 9592.50 | 9552.64 | 9570.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 9592.50 | 9552.64 | 9570.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 9592.50 | 9552.64 | 9570.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 9592.50 | 9552.64 | 9570.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 9574.50 | 9557.01 | 9570.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 11:45:00 | 9572.00 | 9557.81 | 9570.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 9710.00 | 9597.10 | 9584.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 09:15:00 | 9710.00 | 9597.10 | 9584.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 10:15:00 | 9761.00 | 9629.88 | 9600.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 13:15:00 | 9836.00 | 9866.06 | 9809.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-12 14:00:00 | 9836.00 | 9866.06 | 9809.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 9773.00 | 9838.47 | 9809.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 9765.00 | 9838.47 | 9809.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 9790.50 | 9828.88 | 9808.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 11:45:00 | 9854.00 | 9832.30 | 9811.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 13:30:00 | 9808.50 | 9823.01 | 9810.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 15:15:00 | 9748.00 | 9798.41 | 9801.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 15:15:00 | 9748.00 | 9798.41 | 9801.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 10:15:00 | 9675.00 | 9762.86 | 9783.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 9769.00 | 9725.84 | 9750.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 9769.00 | 9725.84 | 9750.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 9769.00 | 9725.84 | 9750.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 9769.00 | 9725.84 | 9750.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 9753.00 | 9731.27 | 9750.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:15:00 | 9765.00 | 9731.27 | 9750.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 9825.00 | 9750.02 | 9757.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:00:00 | 9825.00 | 9750.02 | 9757.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 9830.00 | 9766.02 | 9764.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 9889.50 | 9812.36 | 9788.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 10:15:00 | 9906.00 | 9914.08 | 9865.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 10:30:00 | 9920.50 | 9914.08 | 9865.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 9852.50 | 9901.77 | 9863.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 9852.50 | 9901.77 | 9863.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 9837.00 | 9888.81 | 9861.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:30:00 | 9813.50 | 9888.81 | 9861.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 9726.50 | 9845.74 | 9846.03 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 9871.50 | 9821.37 | 9818.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 14:15:00 | 9905.00 | 9858.72 | 9839.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 09:15:00 | 9785.50 | 9848.20 | 9837.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 09:15:00 | 9785.50 | 9848.20 | 9837.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 9785.50 | 9848.20 | 9837.92 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 9780.00 | 9824.53 | 9828.32 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 9921.00 | 9844.03 | 9835.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 10:15:00 | 10121.50 | 9899.53 | 9861.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 10033.50 | 10073.35 | 10020.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 09:45:00 | 10038.00 | 10073.35 | 10020.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 9985.50 | 10055.78 | 10017.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:45:00 | 9967.50 | 10055.78 | 10017.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 9994.50 | 10043.53 | 10015.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 13:00:00 | 10042.00 | 10043.22 | 10017.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 14:00:00 | 10035.50 | 10041.68 | 10019.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 14:15:00 | 9973.00 | 10027.94 | 10015.30 | SL hit (close<static) qty=1.00 sl=9976.50 alert=retest2 |

### Cycle 66 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 9770.00 | 9964.76 | 9988.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 9672.50 | 9840.31 | 9919.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 9704.00 | 9676.03 | 9751.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 9704.00 | 9676.03 | 9751.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 9704.00 | 9676.03 | 9751.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 11:45:00 | 9648.50 | 9676.28 | 9738.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 9784.00 | 9707.18 | 9738.26 | SL hit (close>static) qty=1.00 sl=9773.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 9849.00 | 9766.80 | 9759.78 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 9482.00 | 9736.92 | 9753.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 10:15:00 | 9388.50 | 9667.24 | 9720.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 10:15:00 | 9577.00 | 9491.73 | 9579.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-10 11:00:00 | 9577.00 | 9491.73 | 9579.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 9505.00 | 9494.38 | 9572.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 12:15:00 | 9499.50 | 9494.38 | 9572.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-10 14:15:00 | 9602.50 | 9531.70 | 9571.67 | SL hit (close>static) qty=1.00 sl=9599.00 alert=retest2 |

### Cycle 69 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 9147.50 | 9084.73 | 9079.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 9212.50 | 9117.02 | 9096.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 9079.50 | 9185.46 | 9153.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 9079.50 | 9185.46 | 9153.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 9079.50 | 9185.46 | 9153.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 9059.00 | 9185.46 | 9153.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 9001.00 | 9112.61 | 9125.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 8883.50 | 9066.79 | 9103.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 9006.50 | 9000.23 | 9058.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 9006.50 | 9000.23 | 9058.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 9006.50 | 9000.23 | 9058.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 8910.00 | 9041.62 | 9057.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:00:00 | 8870.00 | 8854.20 | 8928.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 14:30:00 | 8919.00 | 8897.45 | 8922.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 9081.50 | 8934.99 | 8935.57 | SL hit (close>static) qty=1.00 sl=9074.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 9138.50 | 8975.69 | 8954.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 9139.50 | 9008.45 | 8970.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 8894.50 | 9013.29 | 8992.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 8894.50 | 9013.29 | 8992.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 8894.50 | 9013.29 | 8992.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:45:00 | 8900.50 | 9013.29 | 8992.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 8865.00 | 8983.63 | 8981.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 8854.50 | 8983.63 | 8981.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 8877.00 | 8962.30 | 8971.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 8837.50 | 8916.00 | 8944.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 9008.50 | 8865.93 | 8894.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 9008.50 | 8865.93 | 8894.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 9008.50 | 8865.93 | 8894.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 9046.50 | 8865.93 | 8894.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 8972.50 | 8912.03 | 8911.03 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 15:15:00 | 8893.00 | 8911.61 | 8911.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 8698.00 | 8868.89 | 8892.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 8815.00 | 8784.10 | 8836.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 14:00:00 | 8815.00 | 8784.10 | 8836.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 8769.50 | 8781.18 | 8830.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 14:30:00 | 8778.00 | 8781.18 | 8830.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 8954.50 | 8809.25 | 8834.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:00:00 | 8954.50 | 8809.25 | 8834.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 8867.00 | 8820.80 | 8837.31 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 8942.00 | 8856.83 | 8851.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 12:15:00 | 8975.00 | 8928.53 | 8897.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 10:15:00 | 9720.00 | 9727.82 | 9578.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-13 11:00:00 | 9720.00 | 9727.82 | 9578.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 9752.00 | 9819.70 | 9776.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 13:00:00 | 9752.00 | 9819.70 | 9776.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 9776.50 | 9811.06 | 9776.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 14:30:00 | 9820.00 | 9813.95 | 9780.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 15:15:00 | 9750.00 | 9775.74 | 9775.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2026-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 15:15:00 | 9750.00 | 9775.74 | 9775.95 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2026-04-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 10:15:00 | 9799.00 | 9777.39 | 9776.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 12:15:00 | 9833.00 | 9790.93 | 9782.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 15:15:00 | 9740.00 | 9792.48 | 9786.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 15:15:00 | 9740.00 | 9792.48 | 9786.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 9740.00 | 9792.48 | 9786.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 10:00:00 | 9858.50 | 9805.68 | 9793.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 10:45:00 | 9824.50 | 9806.55 | 9794.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 09:15:00 | 9752.50 | 9790.04 | 9791.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 9752.50 | 9790.04 | 9791.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 12:15:00 | 9699.50 | 9761.22 | 9777.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 09:15:00 | 9634.00 | 9585.78 | 9642.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 09:15:00 | 9634.00 | 9585.78 | 9642.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 9634.00 | 9585.78 | 9642.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 10:00:00 | 9634.00 | 9585.78 | 9642.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 9592.50 | 9587.12 | 9637.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 10:30:00 | 9617.50 | 9587.12 | 9637.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 9618.00 | 9593.30 | 9635.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:30:00 | 9638.50 | 9593.30 | 9635.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 9558.00 | 9586.24 | 9628.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 13:30:00 | 9539.50 | 9577.79 | 9621.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 9668.00 | 9595.86 | 9618.51 | SL hit (close>static) qty=1.00 sl=9637.50 alert=retest2 |

### Cycle 79 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 9704.00 | 9635.74 | 9632.50 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 11:15:00 | 9543.50 | 9628.62 | 9634.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 13:15:00 | 9536.00 | 9596.95 | 9617.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 9644.50 | 9574.54 | 9599.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 09:15:00 | 9644.50 | 9574.54 | 9599.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 9644.50 | 9574.54 | 9599.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:45:00 | 9674.00 | 9574.54 | 9599.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 9626.00 | 9584.83 | 9602.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:15:00 | 9596.00 | 9596.31 | 9604.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 10:15:00 | 9678.00 | 9560.53 | 9580.24 | SL hit (close>static) qty=1.00 sl=9653.00 alert=retest2 |

### Cycle 81 — BUY (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 11:15:00 | 9755.00 | 9599.42 | 9596.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 12:15:00 | 9835.00 | 9646.54 | 9617.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 10:15:00 | 10067.00 | 10087.60 | 9962.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 11:00:00 | 10067.00 | 10087.60 | 9962.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-21 10:15:00 | 8750.00 | 2025-05-22 10:15:00 | 8531.00 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2025-05-22 15:15:00 | 8744.00 | 2025-05-29 13:15:00 | 8819.00 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest2 | 2025-05-23 14:15:00 | 8764.00 | 2025-05-29 13:15:00 | 8819.00 | STOP_HIT | 1.00 | 0.63% |
| BUY | retest2 | 2025-05-23 15:00:00 | 8744.00 | 2025-05-29 13:15:00 | 8819.00 | STOP_HIT | 1.00 | 0.86% |
| SELL | retest2 | 2025-06-03 14:45:00 | 8565.00 | 2025-06-04 09:15:00 | 8616.00 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-06-04 11:45:00 | 8561.50 | 2025-06-06 10:15:00 | 8634.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-06-04 14:45:00 | 8565.00 | 2025-06-06 10:15:00 | 8634.00 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-06-05 10:30:00 | 8564.50 | 2025-06-06 10:15:00 | 8634.00 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-06-05 14:00:00 | 8549.00 | 2025-06-06 10:15:00 | 8634.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-06-06 09:30:00 | 8552.00 | 2025-06-06 10:15:00 | 8634.00 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-06-16 15:00:00 | 8530.00 | 2025-06-18 09:15:00 | 8635.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-06-17 09:15:00 | 8496.50 | 2025-06-18 09:15:00 | 8635.00 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-06-17 14:15:00 | 8527.50 | 2025-06-18 09:15:00 | 8635.00 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-06-19 13:15:00 | 8494.00 | 2025-06-25 12:15:00 | 8397.50 | STOP_HIT | 1.00 | 1.14% |
| SELL | retest2 | 2025-06-19 15:15:00 | 8493.00 | 2025-06-25 12:15:00 | 8397.50 | STOP_HIT | 1.00 | 1.12% |
| BUY | retest2 | 2025-06-26 14:15:00 | 8422.50 | 2025-06-30 14:15:00 | 8380.00 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-07-02 09:45:00 | 8370.00 | 2025-07-03 09:15:00 | 8428.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-07-02 10:30:00 | 8362.00 | 2025-07-03 09:15:00 | 8428.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-07-07 09:30:00 | 8441.50 | 2025-07-08 09:15:00 | 8367.50 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-07-16 11:30:00 | 8270.00 | 2025-07-22 14:15:00 | 8301.00 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2025-07-30 09:15:00 | 8106.50 | 2025-08-04 10:15:00 | 8152.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-07-30 09:45:00 | 8107.00 | 2025-08-04 10:15:00 | 8152.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-08-04 10:00:00 | 8115.50 | 2025-08-04 10:15:00 | 8152.00 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-08-11 14:30:00 | 8258.50 | 2025-08-12 15:15:00 | 8186.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-08-12 09:15:00 | 8282.00 | 2025-08-12 15:15:00 | 8186.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-08-12 12:30:00 | 8258.50 | 2025-08-12 15:15:00 | 8186.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-08-12 13:45:00 | 8263.00 | 2025-08-12 15:15:00 | 8186.00 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-08-13 09:15:00 | 8217.00 | 2025-08-14 13:15:00 | 8229.00 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2025-08-13 09:45:00 | 8260.00 | 2025-08-14 13:15:00 | 8229.00 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-08-28 09:15:00 | 8691.50 | 2025-08-28 13:15:00 | 8701.00 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2025-09-05 09:15:00 | 9113.50 | 2025-09-10 15:15:00 | 9232.50 | STOP_HIT | 1.00 | 1.31% |
| BUY | retest2 | 2025-09-05 10:15:00 | 9126.50 | 2025-09-10 15:15:00 | 9232.50 | STOP_HIT | 1.00 | 1.16% |
| BUY | retest2 | 2025-09-08 09:15:00 | 9132.50 | 2025-09-10 15:15:00 | 9232.50 | STOP_HIT | 1.00 | 1.09% |
| SELL | retest2 | 2025-09-19 09:15:00 | 9004.00 | 2025-09-22 09:15:00 | 9109.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-09-23 09:15:00 | 9143.00 | 2025-09-23 13:15:00 | 8998.50 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-09-23 10:00:00 | 9103.00 | 2025-09-23 13:15:00 | 8998.50 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-09-23 11:15:00 | 9072.00 | 2025-09-23 13:15:00 | 8998.50 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-09-25 15:15:00 | 8827.50 | 2025-10-06 11:15:00 | 8700.00 | STOP_HIT | 1.00 | 1.44% |
| SELL | retest2 | 2025-09-26 10:00:00 | 8815.00 | 2025-10-06 11:15:00 | 8700.00 | STOP_HIT | 1.00 | 1.30% |
| SELL | retest2 | 2025-09-26 13:00:00 | 8815.00 | 2025-10-06 11:15:00 | 8700.00 | STOP_HIT | 1.00 | 1.30% |
| SELL | retest2 | 2025-10-01 09:30:00 | 8806.00 | 2025-10-06 11:15:00 | 8700.00 | STOP_HIT | 1.00 | 1.20% |
| SELL | retest2 | 2025-10-01 11:45:00 | 8664.50 | 2025-10-06 11:15:00 | 8700.00 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-10-01 12:30:00 | 8644.00 | 2025-10-06 11:15:00 | 8700.00 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-10-03 13:30:00 | 8638.00 | 2025-10-06 11:15:00 | 8700.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-10-06 09:15:00 | 8664.00 | 2025-10-06 11:15:00 | 8700.00 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-10-09 11:45:00 | 8795.00 | 2025-10-23 10:15:00 | 9120.50 | STOP_HIT | 1.00 | 3.70% |
| BUY | retest2 | 2025-10-09 14:30:00 | 8792.00 | 2025-10-23 10:15:00 | 9120.50 | STOP_HIT | 1.00 | 3.74% |
| SELL | retest2 | 2025-10-28 12:00:00 | 9040.00 | 2025-11-10 15:15:00 | 8770.00 | STOP_HIT | 1.00 | 2.99% |
| SELL | retest2 | 2025-10-28 14:45:00 | 9039.00 | 2025-11-10 15:15:00 | 8770.00 | STOP_HIT | 1.00 | 2.98% |
| SELL | retest2 | 2025-10-29 09:15:00 | 9003.50 | 2025-11-10 15:15:00 | 8770.00 | STOP_HIT | 1.00 | 2.59% |
| SELL | retest2 | 2025-10-29 13:00:00 | 9035.50 | 2025-11-10 15:15:00 | 8770.00 | STOP_HIT | 1.00 | 2.94% |
| SELL | retest2 | 2025-10-30 13:15:00 | 8957.50 | 2025-11-10 15:15:00 | 8770.00 | STOP_HIT | 1.00 | 2.09% |
| SELL | retest2 | 2025-10-31 12:45:00 | 8893.00 | 2025-11-10 15:15:00 | 8770.00 | STOP_HIT | 1.00 | 1.38% |
| BUY | retest2 | 2025-11-13 09:15:00 | 8876.50 | 2025-11-14 10:15:00 | 8802.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-11-13 09:45:00 | 8886.50 | 2025-11-14 10:15:00 | 8802.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-11-13 10:45:00 | 8882.00 | 2025-11-14 10:15:00 | 8802.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-11-13 11:30:00 | 8873.50 | 2025-11-14 10:15:00 | 8802.00 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-12-04 11:30:00 | 9017.00 | 2025-12-04 14:15:00 | 9082.50 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-12-17 09:30:00 | 8961.50 | 2025-12-19 12:15:00 | 8963.50 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2025-12-24 11:15:00 | 9156.00 | 2025-12-26 14:15:00 | 9065.00 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-12-26 09:45:00 | 9150.00 | 2025-12-26 14:15:00 | 9065.00 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2026-01-06 09:15:00 | 9650.00 | 2026-01-09 13:15:00 | 9563.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2026-01-14 13:45:00 | 9517.00 | 2026-01-14 14:15:00 | 9588.50 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2026-01-14 14:15:00 | 9518.50 | 2026-01-14 14:15:00 | 9588.50 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-01-22 11:45:00 | 9188.00 | 2026-01-22 14:15:00 | 9368.00 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2026-01-27 15:00:00 | 9503.00 | 2026-01-28 09:15:00 | 9270.50 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2026-01-29 09:15:00 | 9513.50 | 2026-02-02 09:15:00 | 9280.50 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2026-01-29 15:00:00 | 9513.00 | 2026-02-02 09:15:00 | 9280.50 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2026-01-30 11:45:00 | 9488.00 | 2026-02-02 09:15:00 | 9280.50 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2026-02-09 11:45:00 | 9572.00 | 2026-02-10 09:15:00 | 9710.00 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2026-02-13 11:45:00 | 9854.00 | 2026-02-13 15:15:00 | 9748.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2026-02-13 13:30:00 | 9808.50 | 2026-02-13 15:15:00 | 9748.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2026-02-27 13:00:00 | 10042.00 | 2026-02-27 14:15:00 | 9973.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2026-02-27 14:00:00 | 10035.50 | 2026-02-27 14:15:00 | 9973.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2026-02-27 14:30:00 | 10035.50 | 2026-02-27 15:15:00 | 9955.50 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-03-05 11:45:00 | 9648.50 | 2026-03-05 14:15:00 | 9784.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-03-10 12:15:00 | 9499.50 | 2026-03-10 14:15:00 | 9602.50 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2026-03-11 10:00:00 | 9498.00 | 2026-03-13 09:15:00 | 9023.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 10:00:00 | 9498.00 | 2026-03-16 11:15:00 | 8925.00 | STOP_HIT | 0.50 | 6.03% |
| SELL | retest2 | 2026-03-23 09:15:00 | 8910.00 | 2026-03-25 09:15:00 | 9081.50 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2026-03-24 10:00:00 | 8870.00 | 2026-03-25 09:15:00 | 9081.50 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2026-03-24 14:30:00 | 8919.00 | 2026-03-25 09:15:00 | 9081.50 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2026-04-16 14:30:00 | 9820.00 | 2026-04-17 15:15:00 | 9750.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2026-04-21 10:00:00 | 9858.50 | 2026-04-22 09:15:00 | 9752.50 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2026-04-21 10:45:00 | 9824.50 | 2026-04-22 09:15:00 | 9752.50 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2026-04-24 13:30:00 | 9539.50 | 2026-04-27 09:15:00 | 9668.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2026-04-29 13:15:00 | 9596.00 | 2026-04-30 10:15:00 | 9678.00 | STOP_HIT | 1.00 | -0.85% |
