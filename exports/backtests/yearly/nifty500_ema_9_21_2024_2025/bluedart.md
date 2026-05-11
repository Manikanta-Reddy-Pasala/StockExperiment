# Blue Dart Express Ltd. (BLUEDART)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 5695.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 149 |
| ALERT1 | 98 |
| ALERT2 | 99 |
| ALERT2_SKIP | 55 |
| ALERT3 | 281 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 126 |
| PARTIAL | 6 |
| TARGET_HIT | 1 |
| STOP_HIT | 128 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 135 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 37 / 98
- **Target hits / Stop hits / Partials:** 1 / 128 / 6
- **Avg / median % per leg:** -0.07% / -0.82%
- **Sum % (uncompounded):** -9.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 63 | 18 | 28.6% | 1 | 62 | 0 | 0.20% | 12.6% |
| BUY @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 2 | 0 | -0.07% | -0.1% |
| BUY @ 3rd Alert (retest2) | 61 | 17 | 27.9% | 1 | 60 | 0 | 0.21% | 12.8% |
| SELL (all) | 72 | 19 | 26.4% | 0 | 66 | 6 | -0.30% | -21.6% |
| SELL @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 3 | 1 | 1.68% | 6.7% |
| SELL @ 3rd Alert (retest2) | 68 | 17 | 25.0% | 0 | 63 | 5 | -0.42% | -28.4% |
| retest1 (combined) | 6 | 3 | 50.0% | 0 | 5 | 1 | 1.10% | 6.6% |
| retest2 (combined) | 129 | 34 | 26.4% | 1 | 123 | 5 | -0.12% | -15.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 12:15:00 | 7090.00 | 7039.10 | 7036.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 10:15:00 | 7164.35 | 7133.30 | 7103.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-18 11:15:00 | 7180.60 | 7204.48 | 7178.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-18 11:15:00 | 7180.60 | 7204.48 | 7178.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 11:15:00 | 7180.60 | 7204.48 | 7178.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 11:30:00 | 7170.05 | 7204.48 | 7178.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 12:15:00 | 7158.00 | 7195.18 | 7176.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:15:00 | 7142.85 | 7195.18 | 7176.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 7049.70 | 7166.08 | 7164.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:00:00 | 7049.70 | 7166.08 | 7164.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 7219.00 | 7176.67 | 7169.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 11:45:00 | 7224.60 | 7188.26 | 7175.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 10:45:00 | 7230.85 | 7234.53 | 7210.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 11:00:00 | 7225.55 | 7219.70 | 7215.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 11:30:00 | 7239.90 | 7223.59 | 7217.48 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 7221.25 | 7236.58 | 7227.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 10:45:00 | 7257.95 | 7240.28 | 7230.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 09:15:00 | 7182.75 | 7279.47 | 7282.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 09:15:00 | 7182.75 | 7279.47 | 7282.06 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 10:15:00 | 7348.85 | 7278.63 | 7271.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 14:15:00 | 7394.00 | 7330.56 | 7300.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 15:15:00 | 7195.10 | 7303.47 | 7290.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 15:15:00 | 7195.10 | 7303.47 | 7290.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 15:15:00 | 7195.10 | 7303.47 | 7290.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 11:30:00 | 7421.70 | 7340.88 | 7317.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-31 15:15:00 | 7190.00 | 7303.40 | 7309.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 15:15:00 | 7190.00 | 7303.40 | 7309.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 10:15:00 | 7150.00 | 7216.87 | 7252.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 11:15:00 | 7049.15 | 7027.77 | 7108.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 12:00:00 | 7049.15 | 7027.77 | 7108.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 7122.85 | 7046.79 | 7109.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 12:45:00 | 7137.20 | 7046.79 | 7109.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 7123.15 | 7062.06 | 7110.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:30:00 | 7160.10 | 7062.06 | 7110.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 7100.00 | 7069.65 | 7109.80 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 11:15:00 | 7191.45 | 7133.22 | 7129.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 12:15:00 | 7267.55 | 7160.08 | 7142.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 13:15:00 | 7772.45 | 7800.66 | 7712.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 14:00:00 | 7772.45 | 7800.66 | 7712.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 14:15:00 | 7932.95 | 8008.39 | 7933.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 15:00:00 | 7932.95 | 8008.39 | 7933.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 15:15:00 | 7944.65 | 7995.64 | 7934.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 10:00:00 | 7913.80 | 7979.27 | 7932.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 7944.50 | 7972.32 | 7933.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 11:30:00 | 7982.55 | 7980.93 | 7940.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 09:15:00 | 7871.65 | 7969.63 | 7952.51 | SL hit (close<static) qty=1.00 sl=7913.80 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 11:15:00 | 7887.00 | 7933.80 | 7937.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 12:15:00 | 7801.00 | 7883.29 | 7907.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 10:15:00 | 7833.65 | 7810.18 | 7854.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 10:15:00 | 7833.65 | 7810.18 | 7854.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 7833.65 | 7810.18 | 7854.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 11:00:00 | 7833.65 | 7810.18 | 7854.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 11:15:00 | 7898.00 | 7827.74 | 7858.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 12:00:00 | 7898.00 | 7827.74 | 7858.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 12:15:00 | 7839.15 | 7830.02 | 7856.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 12:45:00 | 7924.05 | 7830.02 | 7856.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 7826.85 | 7829.39 | 7854.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 13:30:00 | 7810.10 | 7829.39 | 7854.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 7720.25 | 7807.56 | 7841.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 15:15:00 | 7715.00 | 7755.23 | 7793.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 10:45:00 | 7701.05 | 7741.56 | 7776.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 11:15:00 | 7705.05 | 7741.56 | 7776.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 09:45:00 | 7676.20 | 7695.50 | 7734.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 7746.00 | 7705.68 | 7732.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:45:00 | 7735.50 | 7705.68 | 7732.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 7761.20 | 7716.79 | 7735.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 13:00:00 | 7761.20 | 7716.79 | 7735.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 13:15:00 | 7787.05 | 7730.84 | 7739.91 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-06-26 14:15:00 | 7818.10 | 7748.29 | 7747.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 14:15:00 | 7818.10 | 7748.29 | 7747.02 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 15:15:00 | 7720.00 | 7742.63 | 7744.56 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 09:15:00 | 7890.10 | 7772.13 | 7757.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 09:15:00 | 7973.15 | 7844.98 | 7803.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 09:15:00 | 8181.00 | 8186.85 | 8111.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-03 09:30:00 | 8179.45 | 8186.85 | 8111.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 12:15:00 | 8107.00 | 8159.40 | 8117.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 13:00:00 | 8107.00 | 8159.40 | 8117.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 13:15:00 | 8145.75 | 8156.67 | 8119.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 15:00:00 | 8164.95 | 8158.32 | 8123.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-04 10:15:00 | 8082.35 | 8137.99 | 8122.86 | SL hit (close<static) qty=1.00 sl=8100.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 13:15:00 | 8430.05 | 8604.67 | 8618.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 8226.00 | 8480.87 | 8554.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-19 15:15:00 | 8300.00 | 8290.32 | 8406.68 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-22 09:15:00 | 8093.65 | 8290.32 | 8406.68 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 7688.97 | 7827.46 | 8021.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-07-23 14:15:00 | 7839.95 | 7829.12 | 7988.33 | SL hit (close>ema200) qty=0.50 sl=7829.12 alert=retest1 |

### Cycle 11 — BUY (started 2024-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 09:15:00 | 7955.25 | 7847.81 | 7846.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 10:15:00 | 8231.35 | 7924.51 | 7881.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 09:15:00 | 8138.00 | 8174.17 | 8102.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 12:15:00 | 8122.20 | 8155.22 | 8110.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 12:15:00 | 8122.20 | 8155.22 | 8110.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 12:30:00 | 8108.50 | 8155.22 | 8110.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 14:15:00 | 8125.95 | 8144.55 | 8113.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 14:30:00 | 8115.00 | 8144.55 | 8113.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 15:15:00 | 8110.00 | 8137.64 | 8113.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 09:15:00 | 7906.10 | 8137.64 | 8113.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 8047.90 | 8119.69 | 8107.19 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 7931.00 | 8081.95 | 8091.17 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 10:15:00 | 8187.25 | 8067.14 | 8066.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 10:15:00 | 8238.25 | 8143.75 | 8110.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-07 13:15:00 | 8131.55 | 8154.59 | 8125.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 13:15:00 | 8131.55 | 8154.59 | 8125.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 8131.55 | 8154.59 | 8125.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:00:00 | 8131.55 | 8154.59 | 8125.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 8125.00 | 8148.67 | 8125.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:45:00 | 8125.00 | 8148.67 | 8125.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 8135.00 | 8145.94 | 8126.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 09:30:00 | 8142.95 | 8145.17 | 8127.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 10:15:00 | 8150.05 | 8146.15 | 8129.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-08 12:00:00 | 8212.70 | 8159.46 | 8137.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 12:15:00 | 8122.65 | 8152.10 | 8135.93 | SL hit (close<static) qty=1.00 sl=8125.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 14:15:00 | 8025.65 | 8121.82 | 8124.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 14:15:00 | 7923.85 | 8034.76 | 8073.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-12 10:15:00 | 8143.10 | 8046.05 | 8068.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 10:15:00 | 8143.10 | 8046.05 | 8068.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 8143.10 | 8046.05 | 8068.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 11:00:00 | 8143.10 | 8046.05 | 8068.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 11:15:00 | 8095.00 | 8055.84 | 8070.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 12:45:00 | 8090.05 | 8064.36 | 8073.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 13:15:00 | 8092.25 | 8064.36 | 8073.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 14:15:00 | 8200.00 | 8100.62 | 8088.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 14:15:00 | 8200.00 | 8100.62 | 8088.87 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 12:15:00 | 8030.15 | 8082.80 | 8086.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 13:15:00 | 8009.80 | 8068.20 | 8079.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 8122.05 | 7939.43 | 7976.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 8122.05 | 7939.43 | 7976.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 8122.05 | 7939.43 | 7976.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:00:00 | 8122.05 | 7939.43 | 7976.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 8018.20 | 7955.18 | 7980.12 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 13:15:00 | 8040.20 | 8000.75 | 7997.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 15:15:00 | 8065.00 | 8021.52 | 8007.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 09:15:00 | 7983.20 | 8013.85 | 8005.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-19 09:15:00 | 7983.20 | 8013.85 | 8005.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 7983.20 | 8013.85 | 8005.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 10:00:00 | 7983.20 | 8013.85 | 8005.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 10:15:00 | 7969.15 | 8004.91 | 8002.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 10:30:00 | 7955.60 | 8004.91 | 8002.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 11:15:00 | 7950.00 | 7993.93 | 7997.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-19 12:15:00 | 7940.60 | 7983.26 | 7992.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 10:15:00 | 8010.05 | 7966.44 | 7977.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 10:15:00 | 8010.05 | 7966.44 | 7977.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 8010.05 | 7966.44 | 7977.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 10:45:00 | 8014.45 | 7966.44 | 7977.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 7993.95 | 7971.94 | 7978.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 12:15:00 | 8004.35 | 7971.94 | 7978.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 12:15:00 | 8000.00 | 7977.55 | 7980.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 12:45:00 | 7995.95 | 7977.55 | 7980.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 7955.70 | 7968.29 | 7975.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 14:45:00 | 7958.80 | 7968.29 | 7975.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 15:15:00 | 7990.00 | 7972.63 | 7976.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:15:00 | 7990.00 | 7972.63 | 7976.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 7983.00 | 7974.71 | 7977.46 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2024-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 13:15:00 | 8000.60 | 7979.18 | 7978.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 15:15:00 | 8050.00 | 8000.25 | 7988.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 10:15:00 | 8101.10 | 8112.63 | 8071.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 11:00:00 | 8101.10 | 8112.63 | 8071.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 8083.00 | 8105.49 | 8078.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 13:45:00 | 8090.55 | 8105.49 | 8078.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 8094.05 | 8103.20 | 8079.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 15:00:00 | 8094.05 | 8103.20 | 8079.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 8105.00 | 8103.56 | 8081.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 8077.45 | 8103.56 | 8081.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 8040.75 | 8091.00 | 8078.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 10:00:00 | 8040.75 | 8091.00 | 8078.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 8034.15 | 8079.63 | 8074.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:15:00 | 8027.85 | 8079.63 | 8074.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 8134.40 | 8091.65 | 8080.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 09:15:00 | 8312.30 | 8098.76 | 8087.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 12:00:00 | 8180.35 | 8222.47 | 8208.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 14:15:00 | 8126.15 | 8194.40 | 8198.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 14:15:00 | 8126.15 | 8194.40 | 8198.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 09:15:00 | 8102.25 | 8164.87 | 8183.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-02 11:15:00 | 8117.60 | 8068.21 | 8108.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 11:15:00 | 8117.60 | 8068.21 | 8108.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 8117.60 | 8068.21 | 8108.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 11:45:00 | 8232.20 | 8068.21 | 8108.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 8115.25 | 8077.62 | 8108.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:45:00 | 8106.50 | 8077.62 | 8108.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 13:15:00 | 8089.75 | 8080.05 | 8107.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 14:30:00 | 8063.60 | 8075.90 | 8102.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 15:00:00 | 8059.30 | 8075.90 | 8102.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 09:15:00 | 8213.75 | 8100.12 | 8108.95 | SL hit (close>static) qty=1.00 sl=8123.45 alert=retest2 |

### Cycle 21 — BUY (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 10:15:00 | 8283.95 | 8136.89 | 8124.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 15:15:00 | 8352.00 | 8254.61 | 8194.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 13:15:00 | 8314.45 | 8320.72 | 8255.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-04 14:00:00 | 8314.45 | 8320.72 | 8255.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 14:15:00 | 8200.20 | 8296.62 | 8250.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 15:00:00 | 8200.20 | 8296.62 | 8250.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 15:15:00 | 8250.00 | 8287.29 | 8250.56 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2024-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 12:15:00 | 8186.55 | 8232.87 | 8233.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 13:15:00 | 8100.85 | 8206.46 | 8221.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-06 10:15:00 | 8159.95 | 8153.40 | 8186.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-06 10:30:00 | 8153.70 | 8153.40 | 8186.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 12:15:00 | 8161.30 | 8156.39 | 8182.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 12:30:00 | 8217.70 | 8156.39 | 8182.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 8184.95 | 8164.07 | 8181.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 14:45:00 | 8188.65 | 8164.07 | 8181.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 15:15:00 | 8160.00 | 8163.25 | 8179.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:15:00 | 8101.35 | 8163.25 | 8179.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 8136.80 | 8157.96 | 8175.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 14:30:00 | 8060.00 | 8133.83 | 8155.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 12:15:00 | 8176.05 | 8157.72 | 8156.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2024-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 12:15:00 | 8176.05 | 8157.72 | 8156.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 09:15:00 | 8209.25 | 8172.16 | 8164.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 14:15:00 | 8211.70 | 8215.59 | 8190.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 14:15:00 | 8211.70 | 8215.59 | 8190.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 8211.70 | 8215.59 | 8190.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 14:45:00 | 8220.30 | 8215.59 | 8190.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 15:15:00 | 8181.35 | 8208.74 | 8189.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 09:15:00 | 8233.90 | 8208.74 | 8189.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 12:15:00 | 8160.05 | 8180.58 | 8181.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-09-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 12:15:00 | 8160.05 | 8180.58 | 8181.14 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-09-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 13:15:00 | 8218.00 | 8188.06 | 8184.49 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 11:15:00 | 8145.90 | 8178.00 | 8180.80 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 12:15:00 | 8204.95 | 8183.39 | 8182.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 13:15:00 | 8224.20 | 8191.55 | 8186.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 09:15:00 | 8865.80 | 8904.49 | 8642.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-18 10:00:00 | 8865.80 | 8904.49 | 8642.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 8585.00 | 8771.09 | 8671.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 15:00:00 | 8585.00 | 8771.09 | 8671.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 15:15:00 | 8481.00 | 8713.08 | 8654.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:30:00 | 8453.45 | 8636.86 | 8624.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 10:15:00 | 8223.05 | 8554.10 | 8588.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 09:15:00 | 8090.00 | 8206.83 | 8253.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 12:15:00 | 8110.05 | 8106.12 | 8157.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 13:15:00 | 8128.20 | 8110.53 | 8154.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 13:15:00 | 8128.20 | 8110.53 | 8154.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 13:30:00 | 8146.35 | 8110.53 | 8154.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 8183.80 | 8125.19 | 8157.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 14:30:00 | 8166.10 | 8125.19 | 8157.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 8214.95 | 8143.14 | 8162.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 09:30:00 | 8150.05 | 8144.13 | 8161.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 12:45:00 | 8158.40 | 8157.86 | 8164.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 10:15:00 | 8154.05 | 8141.50 | 8152.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 11:15:00 | 8223.10 | 8163.82 | 8161.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 11:15:00 | 8223.10 | 8163.82 | 8161.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 13:15:00 | 8224.60 | 8183.77 | 8171.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 15:15:00 | 8171.05 | 8183.82 | 8173.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 15:15:00 | 8171.05 | 8183.82 | 8173.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 8171.05 | 8183.82 | 8173.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 09:15:00 | 8533.95 | 8183.82 | 8173.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 14:15:00 | 8300.05 | 8475.00 | 8475.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-10-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 14:15:00 | 8300.05 | 8475.00 | 8475.94 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-07 14:15:00 | 8648.95 | 8494.88 | 8475.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 10:15:00 | 8710.45 | 8572.37 | 8518.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-08 14:15:00 | 8592.00 | 8620.96 | 8563.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-08 15:00:00 | 8592.00 | 8620.96 | 8563.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 8554.50 | 8607.67 | 8562.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-09 09:15:00 | 8729.50 | 8607.67 | 8562.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 09:15:00 | 8699.95 | 8656.89 | 8619.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-10 12:15:00 | 8530.00 | 8601.85 | 8602.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 12:15:00 | 8530.00 | 8601.85 | 8602.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 13:15:00 | 8488.00 | 8579.08 | 8592.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 09:15:00 | 8509.45 | 8487.54 | 8523.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 09:15:00 | 8509.45 | 8487.54 | 8523.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 8509.45 | 8487.54 | 8523.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:30:00 | 8519.30 | 8487.54 | 8523.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 8529.70 | 8495.97 | 8523.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:30:00 | 8527.45 | 8495.97 | 8523.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 11:15:00 | 8588.85 | 8514.55 | 8529.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 11:45:00 | 8632.00 | 8514.55 | 8529.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 12:15:00 | 8549.30 | 8521.50 | 8531.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 12:30:00 | 8581.90 | 8521.50 | 8531.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 14:15:00 | 8500.05 | 8515.21 | 8526.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 15:00:00 | 8500.05 | 8515.21 | 8526.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 15:15:00 | 8502.00 | 8512.57 | 8524.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 09:15:00 | 8522.55 | 8512.57 | 8524.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 8508.45 | 8511.74 | 8523.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:15:00 | 8552.75 | 8511.74 | 8523.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 8570.40 | 8523.47 | 8527.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:45:00 | 8569.30 | 8523.47 | 8527.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 8550.00 | 8528.78 | 8529.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 12:30:00 | 8512.45 | 8525.94 | 8528.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 09:45:00 | 8509.00 | 8516.65 | 8522.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 11:15:00 | 8564.85 | 8531.11 | 8528.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 11:15:00 | 8564.85 | 8531.11 | 8528.45 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 15:15:00 | 8493.75 | 8522.58 | 8526.05 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-17 10:15:00 | 8590.00 | 8539.99 | 8533.61 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 14:15:00 | 8346.20 | 8497.30 | 8515.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 15:15:00 | 8310.20 | 8459.88 | 8497.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 13:15:00 | 8407.90 | 8385.65 | 8438.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 14:00:00 | 8407.90 | 8385.65 | 8438.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 8425.20 | 8393.56 | 8437.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 15:00:00 | 8425.20 | 8393.56 | 8437.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 8390.05 | 8392.86 | 8433.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:15:00 | 8403.60 | 8392.86 | 8433.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 8332.45 | 8380.78 | 8424.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 15:15:00 | 8216.00 | 8329.75 | 8379.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-24 12:15:00 | 7805.20 | 7923.46 | 8025.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-25 14:15:00 | 7737.95 | 7719.49 | 7835.58 | SL hit (close>ema200) qty=0.50 sl=7719.49 alert=retest2 |

### Cycle 37 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 7889.90 | 7748.34 | 7736.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 09:15:00 | 7902.40 | 7821.29 | 7784.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 7979.75 | 8025.06 | 7934.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 7979.75 | 8025.06 | 7934.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 7979.75 | 8025.06 | 7934.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 09:45:00 | 7920.15 | 8025.06 | 7934.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 13:15:00 | 7966.85 | 8007.65 | 7953.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 13:45:00 | 7967.70 | 8007.65 | 7953.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 15:15:00 | 7960.10 | 7996.89 | 7958.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:15:00 | 7944.25 | 7996.89 | 7958.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 7919.85 | 7981.48 | 7954.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:45:00 | 7937.00 | 7981.48 | 7954.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 7891.30 | 7963.44 | 7948.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 10:45:00 | 7879.10 | 7963.44 | 7948.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2024-11-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 12:15:00 | 7866.00 | 7930.22 | 7935.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 13:15:00 | 7849.20 | 7914.02 | 7927.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 14:15:00 | 7924.40 | 7916.09 | 7927.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 14:15:00 | 7924.40 | 7916.09 | 7927.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 7924.40 | 7916.09 | 7927.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 15:00:00 | 7924.40 | 7916.09 | 7927.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 7906.40 | 7914.62 | 7924.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:30:00 | 7951.85 | 7914.62 | 7924.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 7825.95 | 7896.89 | 7915.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:30:00 | 7895.90 | 7896.89 | 7915.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 12:15:00 | 7919.45 | 7892.38 | 7909.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 13:00:00 | 7919.45 | 7892.38 | 7909.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 13:15:00 | 7935.00 | 7900.91 | 7912.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 13:30:00 | 7938.85 | 7900.91 | 7912.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 15:15:00 | 7920.20 | 7908.45 | 7913.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 09:15:00 | 7981.00 | 7908.45 | 7913.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 09:15:00 | 7975.00 | 7921.76 | 7919.41 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 7876.15 | 7924.27 | 7927.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 7842.90 | 7907.99 | 7919.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 7881.30 | 7878.48 | 7896.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 10:15:00 | 7881.30 | 7878.48 | 7896.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 7881.30 | 7878.48 | 7896.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:15:00 | 7913.05 | 7878.48 | 7896.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 7938.60 | 7890.50 | 7900.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:30:00 | 7962.55 | 7890.50 | 7900.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 7946.85 | 7901.77 | 7904.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 12:45:00 | 7940.45 | 7901.77 | 7904.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2024-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 13:15:00 | 7947.60 | 7910.94 | 7908.37 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 09:15:00 | 7760.00 | 7884.66 | 7897.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 12:15:00 | 7735.85 | 7814.36 | 7858.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 11:15:00 | 7624.95 | 7515.34 | 7602.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 11:15:00 | 7624.95 | 7515.34 | 7602.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 11:15:00 | 7624.95 | 7515.34 | 7602.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 12:00:00 | 7624.95 | 7515.34 | 7602.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 7639.50 | 7540.18 | 7606.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:00:00 | 7639.50 | 7540.18 | 7606.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 7646.80 | 7561.50 | 7609.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:45:00 | 7630.00 | 7561.50 | 7609.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 15:15:00 | 7649.00 | 7594.30 | 7617.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:15:00 | 7400.00 | 7594.30 | 7617.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 13:00:00 | 7500.00 | 7421.73 | 7477.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-21 09:15:00 | 7125.00 | 7355.97 | 7430.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-21 12:15:00 | 7334.50 | 7313.12 | 7388.58 | SL hit (close>ema200) qty=0.50 sl=7313.12 alert=retest2 |

### Cycle 43 — BUY (started 2024-11-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 15:15:00 | 7490.00 | 7390.25 | 7382.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 7513.10 | 7414.82 | 7394.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 14:15:00 | 7491.40 | 7523.42 | 7486.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 14:15:00 | 7491.40 | 7523.42 | 7486.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 14:15:00 | 7491.40 | 7523.42 | 7486.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 15:00:00 | 7491.40 | 7523.42 | 7486.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 7486.00 | 7515.94 | 7486.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 09:30:00 | 7528.20 | 7521.29 | 7491.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 11:30:00 | 7533.00 | 7507.18 | 7490.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-27 13:15:00 | 7450.00 | 7494.59 | 7487.40 | SL hit (close<static) qty=1.00 sl=7460.15 alert=retest2 |

### Cycle 44 — SELL (started 2024-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 15:15:00 | 7455.00 | 7481.12 | 7482.18 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 09:15:00 | 7499.90 | 7484.88 | 7483.79 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 11:15:00 | 7478.50 | 7482.53 | 7482.85 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 14:15:00 | 7498.20 | 7484.14 | 7483.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 15:15:00 | 7510.00 | 7489.31 | 7485.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 14:15:00 | 7513.95 | 7522.17 | 7506.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 14:15:00 | 7513.95 | 7522.17 | 7506.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 7513.95 | 7522.17 | 7506.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 15:00:00 | 7513.95 | 7522.17 | 7506.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 7490.00 | 7515.73 | 7504.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:15:00 | 7512.95 | 7515.73 | 7504.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 7518.75 | 7516.34 | 7505.98 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2024-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 10:15:00 | 7425.00 | 7498.07 | 7498.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 13:15:00 | 7411.90 | 7459.41 | 7478.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 14:15:00 | 7473.95 | 7462.32 | 7478.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 14:15:00 | 7473.95 | 7462.32 | 7478.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 7473.95 | 7462.32 | 7478.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 15:00:00 | 7473.95 | 7462.32 | 7478.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 15:15:00 | 7555.00 | 7480.85 | 7485.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 14:45:00 | 7419.05 | 7456.87 | 7469.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 15:15:00 | 7418.05 | 7456.87 | 7469.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-04 11:15:00 | 7503.25 | 7478.13 | 7475.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2024-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 11:15:00 | 7503.25 | 7478.13 | 7475.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 14:15:00 | 7527.95 | 7491.48 | 7482.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 11:15:00 | 7508.00 | 7516.70 | 7499.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 11:15:00 | 7508.00 | 7516.70 | 7499.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 7508.00 | 7516.70 | 7499.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:45:00 | 7493.05 | 7516.70 | 7499.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 12:15:00 | 7482.40 | 7509.84 | 7498.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 13:00:00 | 7482.40 | 7509.84 | 7498.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 13:15:00 | 7474.95 | 7502.86 | 7496.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 13:45:00 | 7469.90 | 7502.86 | 7496.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 7488.45 | 7499.98 | 7495.36 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2024-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 15:15:00 | 7460.20 | 7492.02 | 7492.16 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 09:15:00 | 7511.80 | 7495.98 | 7493.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 11:15:00 | 7552.85 | 7513.11 | 7502.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-10 09:15:00 | 7691.00 | 7717.18 | 7648.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-10 09:30:00 | 7681.05 | 7717.18 | 7648.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 14:15:00 | 7669.00 | 7722.47 | 7679.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 15:00:00 | 7669.00 | 7722.47 | 7679.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 7689.90 | 7715.96 | 7680.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 09:15:00 | 7770.00 | 7715.96 | 7680.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 11:00:00 | 7716.15 | 7760.00 | 7745.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 11:15:00 | 7732.50 | 7766.95 | 7768.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 11:15:00 | 7732.50 | 7766.95 | 7768.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 13:15:00 | 7714.00 | 7754.13 | 7762.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 11:15:00 | 7720.75 | 7709.10 | 7734.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 11:15:00 | 7720.75 | 7709.10 | 7734.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 11:15:00 | 7720.75 | 7709.10 | 7734.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 12:00:00 | 7720.75 | 7709.10 | 7734.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 7683.80 | 7198.19 | 7219.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 10:00:00 | 7683.80 | 7198.19 | 7219.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2024-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 10:15:00 | 7500.35 | 7258.62 | 7245.45 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2024-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 10:15:00 | 7106.75 | 7279.81 | 7281.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-27 12:15:00 | 7087.00 | 7213.96 | 7249.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 13:15:00 | 6692.20 | 6636.76 | 6802.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-31 14:00:00 | 6692.20 | 6636.76 | 6802.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 6996.95 | 6708.80 | 6820.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 15:00:00 | 6996.95 | 6708.80 | 6820.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 15:15:00 | 6917.85 | 6750.61 | 6829.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 09:30:00 | 6915.10 | 6785.68 | 6838.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 11:15:00 | 6912.05 | 6813.13 | 6845.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 13:15:00 | 6912.55 | 6871.28 | 6867.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 13:15:00 | 6912.55 | 6871.28 | 6867.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 15:15:00 | 7018.00 | 6908.66 | 6885.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 14:15:00 | 6904.85 | 6924.01 | 6905.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 14:15:00 | 6904.85 | 6924.01 | 6905.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 14:15:00 | 6904.85 | 6924.01 | 6905.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 15:00:00 | 6904.85 | 6924.01 | 6905.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 15:15:00 | 6887.45 | 6916.70 | 6904.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 09:15:00 | 6944.90 | 6916.70 | 6904.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-03 12:15:00 | 6840.10 | 6896.98 | 6898.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-01-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 12:15:00 | 6840.10 | 6896.98 | 6898.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 15:15:00 | 6819.15 | 6863.73 | 6881.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 15:15:00 | 6250.00 | 6233.59 | 6312.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 09:15:00 | 6235.00 | 6233.59 | 6312.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 6246.55 | 6236.18 | 6306.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 13:30:00 | 6181.35 | 6224.36 | 6278.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 10:15:00 | 6336.30 | 6242.02 | 6267.34 | SL hit (close>static) qty=1.00 sl=6320.05 alert=retest2 |

### Cycle 57 — BUY (started 2025-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 12:15:00 | 6374.40 | 6284.70 | 6283.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 13:15:00 | 6400.80 | 6307.92 | 6294.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 15:15:00 | 6430.00 | 6434.67 | 6388.64 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 09:15:00 | 6464.05 | 6434.67 | 6388.64 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 14:15:00 | 6635.00 | 6438.68 | 6408.43 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 6591.00 | 6540.18 | 6491.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 6516.75 | 6540.18 | 6491.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 6523.60 | 6536.87 | 6494.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 10:15:00 | 6605.50 | 6536.87 | 6494.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 09:30:00 | 6572.35 | 6597.32 | 6555.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 10:15:00 | 6544.05 | 6586.67 | 6554.74 | SL hit (close<ema400) qty=1.00 sl=6554.74 alert=retest1 |

### Cycle 58 — SELL (started 2025-01-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 11:15:00 | 6502.00 | 6543.66 | 6547.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 6290.75 | 6465.61 | 6506.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 14:15:00 | 6536.45 | 6434.41 | 6468.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-27 14:15:00 | 6536.45 | 6434.41 | 6468.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 14:15:00 | 6536.45 | 6434.41 | 6468.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 15:00:00 | 6536.45 | 6434.41 | 6468.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 15:15:00 | 6400.10 | 6427.55 | 6462.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 09:15:00 | 6392.10 | 6427.55 | 6462.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 15:00:00 | 6351.00 | 6408.95 | 6437.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 15:15:00 | 6396.00 | 6415.11 | 6424.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 10:45:00 | 6395.45 | 6420.08 | 6425.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 11:15:00 | 6411.60 | 6418.39 | 6423.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 13:15:00 | 6383.05 | 6421.11 | 6424.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 14:15:00 | 6498.80 | 6433.29 | 6429.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 14:15:00 | 6498.80 | 6433.29 | 6429.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 6536.40 | 6452.11 | 6438.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 6551.50 | 6558.01 | 6514.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 6551.50 | 6558.01 | 6514.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 6551.50 | 6558.01 | 6514.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 6550.00 | 6558.01 | 6514.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 6509.65 | 6548.33 | 6514.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 6509.65 | 6548.33 | 6514.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 6506.10 | 6539.89 | 6513.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 15:00:00 | 6573.05 | 6546.52 | 6518.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 11:15:00 | 6466.30 | 6527.84 | 6519.42 | SL hit (close<static) qty=1.00 sl=6490.40 alert=retest2 |

### Cycle 60 — SELL (started 2025-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 12:15:00 | 6450.70 | 6512.41 | 6513.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 09:15:00 | 6421.65 | 6473.72 | 6492.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 10:15:00 | 6459.75 | 6426.68 | 6452.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 10:15:00 | 6459.75 | 6426.68 | 6452.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 6459.75 | 6426.68 | 6452.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:45:00 | 6462.10 | 6426.68 | 6452.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 11:15:00 | 6482.55 | 6437.85 | 6454.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 11:30:00 | 6486.25 | 6437.85 | 6454.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 09:15:00 | 6509.00 | 6471.02 | 6466.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 11:15:00 | 6581.45 | 6501.90 | 6481.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 13:15:00 | 6498.70 | 6504.14 | 6486.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 13:15:00 | 6498.70 | 6504.14 | 6486.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 6498.70 | 6504.14 | 6486.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:45:00 | 6500.00 | 6504.14 | 6486.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 6558.15 | 6514.94 | 6493.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:30:00 | 6501.80 | 6514.94 | 6493.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 6507.00 | 6513.35 | 6494.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:15:00 | 6526.40 | 6513.35 | 6494.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 6529.00 | 6516.48 | 6497.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:30:00 | 6562.05 | 6531.19 | 6505.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-11 09:15:00 | 6368.75 | 6586.47 | 6591.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 09:15:00 | 6368.75 | 6586.47 | 6591.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 10:15:00 | 6278.15 | 6524.80 | 6562.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 13:15:00 | 5922.80 | 5909.11 | 6004.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 13:45:00 | 5910.70 | 5909.11 | 6004.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 5964.85 | 5920.26 | 6000.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 5964.85 | 5920.26 | 6000.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 5995.00 | 5935.21 | 6000.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 09:15:00 | 5936.20 | 5935.21 | 6000.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 09:15:00 | 6033.05 | 5967.35 | 5978.72 | SL hit (close>static) qty=1.00 sl=6020.00 alert=retest2 |

### Cycle 63 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 6053.35 | 5998.98 | 5992.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 12:15:00 | 6110.30 | 6021.25 | 6002.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 09:15:00 | 6248.05 | 6313.32 | 6245.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 09:15:00 | 6248.05 | 6313.32 | 6245.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 6248.05 | 6313.32 | 6245.72 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2025-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 13:15:00 | 6202.00 | 6248.09 | 6249.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 14:15:00 | 6155.20 | 6229.51 | 6241.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 14:15:00 | 5973.55 | 5951.55 | 6009.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 15:00:00 | 5973.55 | 5951.55 | 6009.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 5998.00 | 5963.79 | 6004.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:45:00 | 6012.40 | 5963.79 | 6004.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 5917.85 | 5954.60 | 5997.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 11:15:00 | 5912.20 | 5954.60 | 5997.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 10:00:00 | 5907.60 | 5910.38 | 5951.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 14:15:00 | 6010.00 | 5968.19 | 5967.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 14:15:00 | 6010.00 | 5968.19 | 5967.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 10:15:00 | 6023.10 | 6001.64 | 5990.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 14:15:00 | 5970.05 | 5997.99 | 5992.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 14:15:00 | 5970.05 | 5997.99 | 5992.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 5970.05 | 5997.99 | 5992.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 15:00:00 | 5970.05 | 5997.99 | 5992.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 5959.95 | 5990.38 | 5989.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 5857.15 | 5990.38 | 5989.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2025-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 09:15:00 | 5896.30 | 5971.56 | 5980.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 13:15:00 | 5822.35 | 5913.50 | 5948.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 15:15:00 | 5677.00 | 5669.14 | 5734.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-13 09:15:00 | 5635.80 | 5669.14 | 5734.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 13:15:00 | 5705.70 | 5686.31 | 5719.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 14:00:00 | 5705.70 | 5686.31 | 5719.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 5700.25 | 5689.09 | 5717.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 14:30:00 | 5671.35 | 5689.09 | 5717.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 5741.70 | 5702.16 | 5718.69 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2025-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 15:15:00 | 5748.00 | 5731.13 | 5728.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 5756.95 | 5735.39 | 5732.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 10:15:00 | 5892.55 | 5898.89 | 5874.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 11:00:00 | 5892.55 | 5898.89 | 5874.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 6208.75 | 6315.53 | 6247.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:30:00 | 6183.30 | 6315.53 | 6247.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 6223.40 | 6297.10 | 6245.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:30:00 | 6204.20 | 6297.10 | 6245.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 6228.60 | 6283.40 | 6243.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:30:00 | 6225.70 | 6283.40 | 6243.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 6233.15 | 6273.35 | 6242.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 09:30:00 | 6270.20 | 6244.77 | 6236.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 10:15:00 | 6212.00 | 6238.21 | 6234.04 | SL hit (close<static) qty=1.00 sl=6213.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 12:15:00 | 6161.00 | 6219.38 | 6225.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 13:15:00 | 6089.45 | 6193.39 | 6213.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 09:15:00 | 6221.50 | 6182.94 | 6202.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 6221.50 | 6182.94 | 6202.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 6221.50 | 6182.94 | 6202.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:45:00 | 6220.05 | 6182.94 | 6202.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 6186.80 | 6183.72 | 6200.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:45:00 | 6210.50 | 6183.72 | 6200.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 11:15:00 | 6238.40 | 6194.65 | 6204.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:45:00 | 6221.20 | 6194.65 | 6204.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 12:15:00 | 6215.30 | 6198.78 | 6205.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 13:15:00 | 6210.00 | 6198.78 | 6205.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 14:15:00 | 6189.70 | 6201.69 | 6206.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 14:45:00 | 6206.70 | 6202.81 | 6206.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 09:15:00 | 6181.75 | 6204.65 | 6206.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 6193.00 | 6202.32 | 6205.45 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-02 10:15:00 | 6255.00 | 6212.85 | 6209.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 10:15:00 | 6255.00 | 6212.85 | 6209.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 12:15:00 | 6268.75 | 6230.45 | 6218.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 14:15:00 | 6204.35 | 6230.43 | 6221.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 14:15:00 | 6204.35 | 6230.43 | 6221.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 6204.35 | 6230.43 | 6221.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 6204.35 | 6230.43 | 6221.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 6241.00 | 6232.54 | 6222.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:15:00 | 6206.25 | 6232.54 | 6222.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 6166.95 | 6219.42 | 6217.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 10:00:00 | 6166.95 | 6219.42 | 6217.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2025-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 10:15:00 | 6192.75 | 6214.09 | 6215.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 09:15:00 | 6051.50 | 6160.37 | 6187.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 5947.50 | 5875.98 | 5971.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-07 15:00:00 | 5947.50 | 5875.98 | 5971.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 5904.00 | 5881.59 | 5965.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:15:00 | 5980.10 | 5881.59 | 5965.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 5934.35 | 5892.14 | 5962.60 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2025-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 14:15:00 | 6071.65 | 6002.18 | 5994.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 6193.85 | 6085.62 | 6047.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 12:15:00 | 6370.00 | 6383.82 | 6339.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 12:45:00 | 6385.00 | 6383.82 | 6339.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 13:15:00 | 6363.00 | 6379.66 | 6341.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 14:00:00 | 6363.00 | 6379.66 | 6341.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 6370.00 | 6374.58 | 6345.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:15:00 | 6405.00 | 6374.58 | 6345.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 10:00:00 | 6392.00 | 6522.60 | 6518.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 10:15:00 | 6347.50 | 6487.58 | 6503.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 6347.50 | 6487.58 | 6503.02 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 15:15:00 | 6499.50 | 6463.69 | 6463.11 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 09:15:00 | 6431.50 | 6457.25 | 6460.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 11:15:00 | 6406.00 | 6444.56 | 6453.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 13:15:00 | 6440.00 | 6423.72 | 6441.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 13:15:00 | 6440.00 | 6423.72 | 6441.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 13:15:00 | 6440.00 | 6423.72 | 6441.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 13:45:00 | 6448.00 | 6423.72 | 6441.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 6401.50 | 6419.28 | 6437.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 09:15:00 | 6252.50 | 6415.62 | 6434.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 13:15:00 | 6456.00 | 6327.80 | 6318.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2025-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 13:15:00 | 6456.00 | 6327.80 | 6318.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 09:15:00 | 6543.50 | 6418.02 | 6365.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 6513.50 | 6556.14 | 6480.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 12:15:00 | 6494.50 | 6533.37 | 6487.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 6494.50 | 6533.37 | 6487.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 13:00:00 | 6494.50 | 6533.37 | 6487.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 6492.00 | 6525.10 | 6488.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 13:30:00 | 6486.50 | 6525.10 | 6488.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 6646.50 | 6549.38 | 6502.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 15:15:00 | 6700.50 | 6549.38 | 6502.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 12:00:00 | 6694.50 | 6684.97 | 6637.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 15:15:00 | 6731.50 | 6673.05 | 6643.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 11:15:00 | 6904.50 | 6938.65 | 6940.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 6904.50 | 6938.65 | 6940.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 13:15:00 | 6870.00 | 6919.45 | 6931.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 09:15:00 | 7020.50 | 6923.15 | 6928.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 7020.50 | 6923.15 | 6928.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 7020.50 | 6923.15 | 6928.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:00:00 | 7020.50 | 6923.15 | 6928.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 10:15:00 | 7000.00 | 6938.52 | 6934.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 7129.50 | 7025.46 | 6992.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 6868.50 | 7087.32 | 7053.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 6868.50 | 7087.32 | 7053.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 6868.50 | 7087.32 | 7053.90 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 11:15:00 | 6845.00 | 6999.04 | 7017.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 15:15:00 | 6810.00 | 6894.26 | 6955.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 14:15:00 | 6382.00 | 6368.75 | 6439.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 15:00:00 | 6382.00 | 6368.75 | 6439.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 6561.00 | 6412.12 | 6446.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:00:00 | 6561.00 | 6412.12 | 6446.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 6544.00 | 6438.50 | 6455.81 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 12:15:00 | 6523.50 | 6469.50 | 6467.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 14:15:00 | 6535.50 | 6487.58 | 6476.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 14:15:00 | 6500.00 | 6515.07 | 6506.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 14:15:00 | 6500.00 | 6515.07 | 6506.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 6500.00 | 6515.07 | 6506.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 15:00:00 | 6500.00 | 6515.07 | 6506.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 6497.50 | 6511.55 | 6505.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 09:15:00 | 6529.00 | 6511.55 | 6505.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 11:00:00 | 6527.50 | 6519.13 | 6510.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 13:45:00 | 6526.00 | 6548.05 | 6536.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 14:15:00 | 6529.50 | 6548.05 | 6536.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 6549.00 | 6548.24 | 6538.07 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 6477.00 | 6523.97 | 6528.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 6477.00 | 6523.97 | 6528.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 6474.00 | 6513.98 | 6523.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 12:15:00 | 6514.00 | 6513.98 | 6522.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 12:15:00 | 6514.00 | 6513.98 | 6522.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 6514.00 | 6513.98 | 6522.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:45:00 | 6533.00 | 6513.98 | 6522.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 6383.00 | 6487.78 | 6510.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 09:15:00 | 6325.50 | 6455.62 | 6490.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 11:15:00 | 6378.00 | 6430.52 | 6472.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 10:15:00 | 6312.00 | 6226.11 | 6215.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 6312.00 | 6226.11 | 6215.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 11:15:00 | 6396.00 | 6260.09 | 6231.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 6258.50 | 6281.02 | 6250.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 14:15:00 | 6258.50 | 6281.02 | 6250.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 6258.50 | 6281.02 | 6250.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 6258.50 | 6281.02 | 6250.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 6240.50 | 6272.92 | 6249.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 6339.00 | 6272.92 | 6249.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-02 09:15:00 | 6972.90 | 6772.43 | 6602.96 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 13:15:00 | 6713.00 | 6770.57 | 6774.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 15:15:00 | 6675.00 | 6741.45 | 6759.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 13:15:00 | 6704.50 | 6703.84 | 6730.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-07 14:00:00 | 6704.50 | 6703.84 | 6730.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 6664.00 | 6687.08 | 6715.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 10:45:00 | 6646.00 | 6677.57 | 6708.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 6729.00 | 6652.48 | 6658.06 | SL hit (close>static) qty=1.00 sl=6719.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 10:15:00 | 6744.00 | 6670.79 | 6665.87 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 12:15:00 | 6670.00 | 6678.08 | 6678.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 14:15:00 | 6630.00 | 6668.29 | 6673.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 6680.50 | 6663.57 | 6670.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 6680.50 | 6663.57 | 6670.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 6680.50 | 6663.57 | 6670.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:00:00 | 6680.50 | 6663.57 | 6670.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 6640.50 | 6658.95 | 6667.43 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 6816.00 | 6694.28 | 6680.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 6901.50 | 6751.04 | 6709.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 10:15:00 | 6849.50 | 6861.62 | 6805.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 11:00:00 | 6849.50 | 6861.62 | 6805.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 6902.00 | 6891.98 | 6848.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 13:45:00 | 6920.00 | 6900.79 | 6866.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 14:15:00 | 6919.00 | 6900.79 | 6866.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 6836.00 | 6864.34 | 6865.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 6836.00 | 6864.34 | 6865.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 13:15:00 | 6754.00 | 6814.68 | 6838.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 10:15:00 | 6825.50 | 6807.24 | 6826.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 11:00:00 | 6825.50 | 6807.24 | 6826.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 6768.00 | 6799.39 | 6821.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:30:00 | 6838.50 | 6799.39 | 6821.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 6834.00 | 6782.66 | 6801.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 6834.00 | 6782.66 | 6801.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 6847.00 | 6795.53 | 6805.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:30:00 | 6846.00 | 6795.53 | 6805.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 6799.00 | 6802.01 | 6807.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:45:00 | 6795.00 | 6802.01 | 6807.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2025-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 14:15:00 | 6861.50 | 6813.91 | 6812.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 10:15:00 | 6898.00 | 6843.04 | 6826.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 6791.00 | 6856.95 | 6844.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 09:15:00 | 6791.00 | 6856.95 | 6844.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 6791.00 | 6856.95 | 6844.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:15:00 | 6755.00 | 6856.95 | 6844.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 6770.00 | 6839.56 | 6837.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:30:00 | 6761.00 | 6839.56 | 6837.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 6768.00 | 6825.25 | 6831.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 15:15:00 | 6725.00 | 6784.42 | 6808.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 09:15:00 | 5805.00 | 5803.52 | 5864.61 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 10:15:00 | 5765.00 | 5786.23 | 5825.16 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 10:45:00 | 5763.00 | 5779.39 | 5818.51 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 5804.50 | 5772.59 | 5801.50 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 5804.50 | 5772.59 | 5801.50 | SL hit (close>ema400) qty=1.00 sl=5801.50 alert=retest1 |

### Cycle 89 — BUY (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 09:15:00 | 5828.50 | 5811.68 | 5810.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 14:15:00 | 5877.00 | 5835.94 | 5823.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 10:15:00 | 5899.00 | 5912.77 | 5882.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 10:45:00 | 5898.00 | 5912.77 | 5882.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 5894.50 | 5902.46 | 5884.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 10:45:00 | 5915.00 | 5892.06 | 5884.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 11:15:00 | 5870.00 | 5887.65 | 5883.14 | SL hit (close<static) qty=1.00 sl=5882.50 alert=retest2 |

### Cycle 90 — SELL (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 13:15:00 | 5855.50 | 5877.92 | 5879.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 15:15:00 | 5850.00 | 5870.67 | 5875.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 5889.50 | 5874.43 | 5876.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 5889.50 | 5874.43 | 5876.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 5889.50 | 5874.43 | 5876.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:45:00 | 5889.00 | 5874.43 | 5876.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 5878.00 | 5875.15 | 5877.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 13:00:00 | 5842.00 | 5866.73 | 5872.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 13:30:00 | 5831.00 | 5859.99 | 5869.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 14:30:00 | 5841.00 | 5854.39 | 5865.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 11:15:00 | 5893.50 | 5862.50 | 5865.46 | SL hit (close>static) qty=1.00 sl=5893.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 12:15:00 | 5914.00 | 5872.80 | 5869.87 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 12:15:00 | 5847.50 | 5868.70 | 5869.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 14:15:00 | 5840.00 | 5859.49 | 5865.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 5854.50 | 5854.33 | 5861.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-21 11:15:00 | 5868.50 | 5854.33 | 5861.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 5873.00 | 5858.07 | 5862.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 15:15:00 | 5842.00 | 5858.56 | 5861.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 5810.00 | 5825.55 | 5830.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 5549.90 | 5615.75 | 5686.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 11:15:00 | 5519.50 | 5587.68 | 5660.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 13:15:00 | 5628.50 | 5592.94 | 5650.31 | SL hit (close>ema200) qty=0.50 sl=5592.94 alert=retest2 |

### Cycle 93 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 5802.50 | 5684.03 | 5669.41 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 5672.50 | 5726.83 | 5733.56 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 12:15:00 | 5724.50 | 5720.40 | 5720.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 09:15:00 | 5735.50 | 5725.29 | 5722.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 14:15:00 | 5732.00 | 5749.09 | 5741.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 14:15:00 | 5732.00 | 5749.09 | 5741.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 5732.00 | 5749.09 | 5741.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:45:00 | 5735.00 | 5749.09 | 5741.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 5730.00 | 5745.27 | 5740.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 5775.00 | 5745.27 | 5740.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 5761.00 | 5766.85 | 5758.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 11:15:00 | 5710.50 | 5756.18 | 5755.78 | SL hit (close<static) qty=1.00 sl=5729.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 12:15:00 | 5728.50 | 5750.65 | 5753.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 14:15:00 | 5698.50 | 5716.91 | 5730.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 5757.00 | 5722.06 | 5730.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 5757.00 | 5722.06 | 5730.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 5757.00 | 5722.06 | 5730.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:30:00 | 5750.00 | 5722.06 | 5730.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 5746.50 | 5726.95 | 5731.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:30:00 | 5753.00 | 5726.95 | 5731.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 5735.00 | 5730.89 | 5732.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:45:00 | 5724.00 | 5730.11 | 5732.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 12:15:00 | 5737.00 | 5733.72 | 5733.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 12:15:00 | 5737.00 | 5733.72 | 5733.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 14:15:00 | 5749.50 | 5737.08 | 5735.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 09:15:00 | 5817.00 | 5824.68 | 5793.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-19 09:45:00 | 5804.00 | 5824.68 | 5793.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 5778.50 | 5815.45 | 5792.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:45:00 | 5780.00 | 5815.45 | 5792.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 5781.00 | 5808.56 | 5791.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:00:00 | 5781.00 | 5808.56 | 5791.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 5756.00 | 5793.88 | 5787.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:00:00 | 5756.00 | 5793.88 | 5787.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 5761.00 | 5787.30 | 5784.84 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2025-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 15:15:00 | 5753.00 | 5780.44 | 5781.95 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 5822.00 | 5788.75 | 5785.59 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 5767.00 | 5786.38 | 5787.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 13:15:00 | 5727.00 | 5766.08 | 5776.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 5790.50 | 5767.42 | 5774.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 5790.50 | 5767.42 | 5774.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 5790.50 | 5767.42 | 5774.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 5790.50 | 5767.42 | 5774.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 5778.50 | 5769.64 | 5774.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 11:15:00 | 5752.50 | 5769.64 | 5774.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 12:00:00 | 5770.00 | 5769.71 | 5774.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 13:45:00 | 5770.00 | 5771.09 | 5774.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:30:00 | 5772.50 | 5772.07 | 5774.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 5770.50 | 5771.76 | 5774.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 09:15:00 | 5769.00 | 5771.76 | 5774.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 12:15:00 | 5800.00 | 5767.65 | 5770.42 | SL hit (close>static) qty=1.00 sl=5790.50 alert=retest2 |

### Cycle 101 — BUY (started 2025-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 13:15:00 | 5800.00 | 5774.12 | 5773.11 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 10:15:00 | 5742.00 | 5768.05 | 5771.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 5722.50 | 5758.94 | 5766.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 5723.50 | 5722.83 | 5743.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 5723.50 | 5722.83 | 5743.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 5723.50 | 5722.83 | 5743.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:45:00 | 5731.50 | 5722.83 | 5743.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 5714.50 | 5721.16 | 5740.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 5690.50 | 5714.53 | 5735.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 13:30:00 | 5651.50 | 5679.34 | 5716.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 5952.50 | 5708.82 | 5717.94 | SL hit (close>static) qty=1.00 sl=5750.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 10:15:00 | 5914.00 | 5749.85 | 5735.76 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 10:15:00 | 5699.50 | 5742.22 | 5746.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-03 10:15:00 | 5646.50 | 5696.73 | 5718.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 14:15:00 | 5677.50 | 5674.29 | 5699.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-03 15:00:00 | 5677.50 | 5674.29 | 5699.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 5692.00 | 5677.83 | 5698.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 09:15:00 | 5624.50 | 5677.83 | 5698.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 11:00:00 | 5640.50 | 5663.03 | 5687.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 09:15:00 | 5634.00 | 5656.35 | 5672.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 15:15:00 | 5554.50 | 5531.18 | 5530.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 15:15:00 | 5554.50 | 5531.18 | 5530.78 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 5472.50 | 5519.44 | 5525.48 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 5540.00 | 5511.54 | 5508.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 14:15:00 | 5564.50 | 5530.19 | 5518.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 5522.00 | 5530.92 | 5521.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 5522.00 | 5530.92 | 5521.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 5522.00 | 5530.92 | 5521.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:00:00 | 5522.00 | 5530.92 | 5521.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 5568.00 | 5538.34 | 5525.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:30:00 | 5527.00 | 5538.34 | 5525.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 5536.00 | 5553.23 | 5542.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:00:00 | 5536.00 | 5553.23 | 5542.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 5538.50 | 5550.28 | 5541.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:45:00 | 5539.00 | 5550.28 | 5541.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 5548.00 | 5549.83 | 5542.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:00:00 | 5548.00 | 5549.83 | 5542.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 5531.50 | 5546.16 | 5541.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:30:00 | 5529.00 | 5546.16 | 5541.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 5546.00 | 5546.13 | 5541.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 5579.00 | 5552.76 | 5545.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 14:30:00 | 5565.50 | 5577.38 | 5570.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 5577.00 | 5571.91 | 5568.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 5534.00 | 5564.33 | 5565.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 09:15:00 | 5534.00 | 5564.33 | 5565.60 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 09:15:00 | 5572.00 | 5563.93 | 5563.58 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 10:15:00 | 5555.00 | 5562.14 | 5562.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 12:15:00 | 5532.50 | 5553.63 | 5558.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 15:15:00 | 5575.00 | 5553.66 | 5557.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 15:15:00 | 5575.00 | 5553.66 | 5557.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 5575.00 | 5553.66 | 5557.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 6007.50 | 5553.66 | 5557.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 6136.00 | 5670.13 | 5609.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 10:15:00 | 6319.00 | 5799.90 | 5674.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 15:15:00 | 6597.00 | 6620.97 | 6341.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 09:15:00 | 6698.50 | 6620.97 | 6341.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 6290.00 | 6524.71 | 6443.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 6290.00 | 6524.71 | 6443.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 6359.00 | 6491.57 | 6436.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 13:00:00 | 6375.00 | 6447.93 | 6424.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 6420.00 | 6416.99 | 6415.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 6391.50 | 6411.90 | 6412.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 09:15:00 | 6391.50 | 6411.90 | 6412.99 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 10:15:00 | 6435.50 | 6416.62 | 6415.03 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 6393.00 | 6411.89 | 6413.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 12:15:00 | 6374.50 | 6404.41 | 6409.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 10:15:00 | 6403.50 | 6378.03 | 6392.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 10:15:00 | 6403.50 | 6378.03 | 6392.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 6403.50 | 6378.03 | 6392.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:00:00 | 6403.50 | 6378.03 | 6392.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 6415.50 | 6385.53 | 6394.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:00:00 | 6415.50 | 6385.53 | 6394.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 6378.00 | 6384.02 | 6393.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 14:15:00 | 6361.00 | 6384.82 | 6392.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 15:15:00 | 6042.95 | 6134.19 | 6183.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 13:15:00 | 6141.50 | 6121.40 | 6156.46 | SL hit (close>ema200) qty=0.50 sl=6121.40 alert=retest2 |

### Cycle 115 — BUY (started 2025-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 15:15:00 | 5435.00 | 5410.21 | 5409.52 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 5399.00 | 5407.33 | 5408.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 5396.00 | 5404.13 | 5406.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 15:15:00 | 5296.50 | 5294.46 | 5325.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 09:15:00 | 5330.50 | 5294.46 | 5325.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 5357.00 | 5306.97 | 5328.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:00:00 | 5357.00 | 5306.97 | 5328.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 5360.00 | 5317.58 | 5331.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:45:00 | 5364.00 | 5317.58 | 5331.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 5374.00 | 5342.13 | 5340.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 5394.00 | 5352.51 | 5345.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 12:15:00 | 5457.50 | 5460.49 | 5436.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 12:45:00 | 5457.50 | 5460.49 | 5436.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 5459.00 | 5460.73 | 5442.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 5482.00 | 5460.73 | 5442.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 12:45:00 | 5485.50 | 5468.61 | 5452.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 11:15:00 | 5426.00 | 5446.07 | 5447.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 5426.00 | 5446.07 | 5447.58 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 15:15:00 | 5470.00 | 5448.76 | 5447.58 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 09:15:00 | 5421.50 | 5443.30 | 5445.21 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 10:15:00 | 5499.00 | 5454.44 | 5450.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 12:15:00 | 5516.50 | 5473.34 | 5459.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 09:15:00 | 5513.00 | 5514.21 | 5486.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:45:00 | 5507.00 | 5514.21 | 5486.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 5509.50 | 5519.77 | 5500.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:30:00 | 5506.00 | 5519.77 | 5500.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 5591.00 | 5658.55 | 5609.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:30:00 | 5595.50 | 5658.55 | 5609.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 5592.00 | 5645.24 | 5607.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:45:00 | 5585.50 | 5645.24 | 5607.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 5601.00 | 5636.39 | 5607.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:30:00 | 5592.50 | 5636.39 | 5607.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 5593.00 | 5627.72 | 5605.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:00:00 | 5593.00 | 5627.72 | 5605.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 5600.50 | 5622.27 | 5605.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 15:00:00 | 5611.00 | 5620.02 | 5605.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 5554.00 | 5602.01 | 5599.80 | SL hit (close<static) qty=1.00 sl=5585.00 alert=retest2 |

### Cycle 122 — SELL (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 10:15:00 | 5556.00 | 5592.81 | 5595.82 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 12:15:00 | 5659.00 | 5597.52 | 5596.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 14:15:00 | 5670.50 | 5621.23 | 5608.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 5596.00 | 5625.35 | 5613.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 5596.00 | 5625.35 | 5613.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 5596.00 | 5625.35 | 5613.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:15:00 | 5588.00 | 5625.35 | 5613.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 5575.50 | 5615.38 | 5609.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:00:00 | 5575.50 | 5615.38 | 5609.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 12:15:00 | 5565.00 | 5598.84 | 5602.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 14:15:00 | 5543.00 | 5580.66 | 5593.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 5372.00 | 5368.16 | 5397.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:45:00 | 5372.50 | 5368.16 | 5397.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 5360.00 | 5366.53 | 5394.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 5400.00 | 5366.53 | 5394.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 5436.50 | 5380.52 | 5398.10 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 5441.50 | 5409.94 | 5406.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 11:15:00 | 5442.00 | 5416.35 | 5409.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 11:15:00 | 5410.00 | 5425.27 | 5419.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 11:15:00 | 5410.00 | 5425.27 | 5419.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 5410.00 | 5425.27 | 5419.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:45:00 | 5414.50 | 5425.27 | 5419.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 5398.00 | 5419.81 | 5417.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:00:00 | 5398.00 | 5419.81 | 5417.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2026-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 13:15:00 | 5393.00 | 5414.45 | 5415.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 5381.00 | 5405.26 | 5410.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 5310.50 | 5293.26 | 5320.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 5310.50 | 5293.26 | 5320.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 5310.50 | 5293.26 | 5320.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 5334.00 | 5293.26 | 5320.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 5314.00 | 5297.41 | 5319.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:30:00 | 5296.00 | 5316.60 | 5321.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:30:00 | 5304.00 | 5313.28 | 5319.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 14:15:00 | 5339.50 | 5311.32 | 5310.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2026-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 14:15:00 | 5339.50 | 5311.32 | 5310.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 15:15:00 | 5565.00 | 5362.05 | 5333.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 5440.00 | 5461.89 | 5412.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 10:00:00 | 5440.00 | 5461.89 | 5412.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 5413.50 | 5452.21 | 5412.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:30:00 | 5410.50 | 5452.21 | 5412.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 5425.50 | 5446.87 | 5413.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 09:30:00 | 5451.00 | 5432.12 | 5416.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:00:00 | 5455.00 | 5432.12 | 5416.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 10:00:00 | 5469.50 | 5480.56 | 5455.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:30:00 | 5453.00 | 5507.01 | 5485.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 5480.00 | 5501.61 | 5484.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 5480.00 | 5501.61 | 5484.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 5485.00 | 5498.29 | 5484.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 13:30:00 | 5499.00 | 5494.83 | 5485.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 14:30:00 | 5496.50 | 5495.56 | 5486.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 5569.50 | 5494.75 | 5487.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 5735.00 | 5811.76 | 5818.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 5735.00 | 5811.76 | 5818.63 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2026-02-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 15:15:00 | 5846.00 | 5801.25 | 5800.22 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 12:15:00 | 5784.50 | 5797.98 | 5799.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-17 15:15:00 | 5760.00 | 5785.78 | 5793.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 11:15:00 | 5772.00 | 5771.33 | 5783.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 12:00:00 | 5772.00 | 5771.33 | 5783.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 12:15:00 | 5773.50 | 5771.76 | 5782.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 13:00:00 | 5773.50 | 5771.76 | 5782.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 5798.50 | 5777.11 | 5784.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:00:00 | 5798.50 | 5777.11 | 5784.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 5825.00 | 5786.69 | 5787.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 5825.00 | 5786.69 | 5787.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 15:15:00 | 5825.00 | 5794.35 | 5791.28 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 5760.00 | 5787.06 | 5788.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 12:15:00 | 5737.50 | 5777.15 | 5784.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 15:15:00 | 5570.00 | 5569.03 | 5608.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 09:15:00 | 5545.00 | 5569.03 | 5608.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 5607.00 | 5576.62 | 5608.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:00:00 | 5607.00 | 5576.62 | 5608.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 5617.00 | 5584.70 | 5609.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:30:00 | 5620.50 | 5584.70 | 5609.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 5617.00 | 5591.16 | 5609.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 11:45:00 | 5615.50 | 5591.16 | 5609.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 5578.50 | 5588.63 | 5606.94 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 5639.50 | 5615.23 | 5613.89 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 12:15:00 | 5602.00 | 5612.15 | 5612.69 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-02-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 15:15:00 | 5630.50 | 5615.81 | 5614.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 09:15:00 | 5642.00 | 5621.05 | 5616.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 11:15:00 | 5611.00 | 5619.91 | 5617.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 11:15:00 | 5611.00 | 5619.91 | 5617.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 5611.00 | 5619.91 | 5617.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:30:00 | 5620.50 | 5619.91 | 5617.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 5620.50 | 5620.03 | 5617.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:30:00 | 5622.00 | 5620.03 | 5617.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 5637.00 | 5623.42 | 5619.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:45:00 | 5632.50 | 5623.42 | 5619.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 5581.00 | 5630.66 | 5624.97 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 5533.50 | 5611.22 | 5616.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 5512.00 | 5576.22 | 5598.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 10:15:00 | 5462.00 | 5456.59 | 5499.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 11:00:00 | 5462.00 | 5456.59 | 5499.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 5348.50 | 5333.87 | 5362.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 12:30:00 | 5365.00 | 5333.87 | 5362.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 13:15:00 | 5374.50 | 5342.00 | 5363.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:00:00 | 5374.50 | 5342.00 | 5363.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 14:15:00 | 5363.50 | 5346.30 | 5363.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 15:15:00 | 5374.50 | 5346.30 | 5363.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 5374.50 | 5351.94 | 5364.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 5394.50 | 5351.94 | 5364.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 5365.50 | 5354.65 | 5364.61 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 14:15:00 | 5406.00 | 5369.68 | 5367.44 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 13:15:00 | 5350.00 | 5369.36 | 5369.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 14:15:00 | 5301.00 | 5355.69 | 5363.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 5127.50 | 5126.41 | 5199.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 15:00:00 | 5127.50 | 5126.41 | 5199.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 5181.50 | 5137.36 | 5191.81 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 5248.50 | 5199.77 | 5194.39 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 5111.50 | 5184.22 | 5191.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 5100.00 | 5154.30 | 5175.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 10:15:00 | 5126.00 | 5114.49 | 5143.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 10:45:00 | 5116.00 | 5114.49 | 5143.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 5120.00 | 5115.59 | 5141.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:30:00 | 5131.50 | 5115.59 | 5141.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 4905.00 | 4959.86 | 5025.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 4863.00 | 4940.79 | 5010.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 12:15:00 | 5077.00 | 5014.82 | 5011.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 5077.00 | 5014.82 | 5011.26 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 4913.50 | 4995.42 | 5006.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 12:15:00 | 4879.50 | 4956.65 | 4985.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 4941.90 | 4804.76 | 4855.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 4941.90 | 4804.76 | 4855.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 4941.90 | 4804.76 | 4855.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 4919.90 | 4804.76 | 4855.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 4962.00 | 4836.21 | 4865.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:00:00 | 4962.00 | 4836.21 | 4865.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 4932.90 | 4881.92 | 4881.58 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 4825.00 | 4879.16 | 4881.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 10:15:00 | 4773.50 | 4858.03 | 4871.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 4848.20 | 4837.56 | 4856.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 14:00:00 | 4848.20 | 4837.56 | 4856.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 4909.90 | 4852.03 | 4861.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 15:00:00 | 4909.90 | 4852.03 | 4861.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 4894.00 | 4860.42 | 4864.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:15:00 | 4875.30 | 4860.42 | 4864.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 14:15:00 | 4913.90 | 4841.88 | 4850.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 15:00:00 | 4913.90 | 4841.88 | 4850.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — BUY (started 2026-04-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 15:15:00 | 4937.50 | 4861.00 | 4858.03 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 10:15:00 | 4801.00 | 4846.44 | 4851.77 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2026-04-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 14:15:00 | 4928.70 | 4861.66 | 4856.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 5062.50 | 4912.61 | 4881.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 5009.00 | 5030.15 | 4983.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:45:00 | 5010.00 | 5030.15 | 4983.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 5058.00 | 5035.72 | 4990.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:45:00 | 5044.10 | 5035.72 | 4990.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 5137.60 | 5129.04 | 5073.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 5168.80 | 5129.04 | 5073.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:30:00 | 5145.00 | 5114.58 | 5092.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 12:15:00 | 5346.00 | 5420.16 | 5426.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 12:15:00 | 5346.00 | 5420.16 | 5426.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 5340.80 | 5394.66 | 5413.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 14:15:00 | 5352.90 | 5342.49 | 5372.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 14:45:00 | 5350.00 | 5342.49 | 5372.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 5351.00 | 5344.19 | 5370.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 5400.20 | 5344.19 | 5370.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 5443.80 | 5364.11 | 5377.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 5477.50 | 5364.11 | 5377.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 5442.40 | 5379.77 | 5383.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 5460.30 | 5379.77 | 5383.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 5422.20 | 5388.26 | 5386.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 5469.90 | 5418.85 | 5403.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 5418.60 | 5430.89 | 5415.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 13:15:00 | 5418.60 | 5430.89 | 5415.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 5418.60 | 5430.89 | 5415.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:45:00 | 5407.20 | 5430.89 | 5415.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 5405.50 | 5425.81 | 5414.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:15:00 | 5401.00 | 5425.81 | 5414.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 5401.00 | 5420.85 | 5413.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 5433.80 | 5420.85 | 5413.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 10:00:00 | 5415.00 | 5419.68 | 5413.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-21 11:45:00 | 7224.60 | 2024-05-28 09:15:00 | 7182.75 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-05-22 10:45:00 | 7230.85 | 2024-05-28 09:15:00 | 7182.75 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-05-23 11:00:00 | 7225.55 | 2024-05-28 09:15:00 | 7182.75 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2024-05-23 11:30:00 | 7239.90 | 2024-05-28 09:15:00 | 7182.75 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-05-24 10:45:00 | 7257.95 | 2024-05-28 09:15:00 | 7182.75 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-05-31 11:30:00 | 7421.70 | 2024-05-31 15:15:00 | 7190.00 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2024-06-18 11:30:00 | 7982.55 | 2024-06-19 09:15:00 | 7871.65 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-06-24 15:15:00 | 7715.00 | 2024-06-26 14:15:00 | 7818.10 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-06-25 10:45:00 | 7701.05 | 2024-06-26 14:15:00 | 7818.10 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-06-25 11:15:00 | 7705.05 | 2024-06-26 14:15:00 | 7818.10 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-06-26 09:45:00 | 7676.20 | 2024-06-26 14:15:00 | 7818.10 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2024-07-03 15:00:00 | 8164.95 | 2024-07-04 10:15:00 | 8082.35 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-07-04 11:45:00 | 8208.50 | 2024-07-18 13:15:00 | 8430.05 | STOP_HIT | 1.00 | 2.70% |
| BUY | retest2 | 2024-07-04 14:30:00 | 8193.45 | 2024-07-18 13:15:00 | 8430.05 | STOP_HIT | 1.00 | 2.89% |
| SELL | retest1 | 2024-07-22 09:15:00 | 8093.65 | 2024-07-23 12:15:00 | 7688.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-07-22 09:15:00 | 8093.65 | 2024-07-23 14:15:00 | 7839.95 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2024-07-25 09:15:00 | 7820.00 | 2024-07-31 09:15:00 | 7955.25 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2024-07-25 11:30:00 | 7854.45 | 2024-07-31 09:15:00 | 7955.25 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-07-25 13:30:00 | 7850.00 | 2024-07-31 09:15:00 | 7955.25 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-07-25 14:15:00 | 7809.95 | 2024-07-31 09:15:00 | 7955.25 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2024-07-29 13:30:00 | 7826.60 | 2024-07-31 09:15:00 | 7955.25 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2024-08-08 12:00:00 | 8212.70 | 2024-08-08 12:15:00 | 8122.65 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-08-12 12:45:00 | 8090.05 | 2024-08-12 14:15:00 | 8200.00 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-08-12 13:15:00 | 8092.25 | 2024-08-12 14:15:00 | 8200.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-08-27 09:15:00 | 8312.30 | 2024-08-29 14:15:00 | 8126.15 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2024-08-29 12:00:00 | 8180.35 | 2024-08-29 14:15:00 | 8126.15 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-09-02 14:30:00 | 8063.60 | 2024-09-03 09:15:00 | 8213.75 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2024-09-02 15:00:00 | 8059.30 | 2024-09-03 09:15:00 | 8213.75 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-09-09 14:30:00 | 8060.00 | 2024-09-11 12:15:00 | 8176.05 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-09-13 09:15:00 | 8233.90 | 2024-09-13 12:15:00 | 8160.05 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-09-27 09:30:00 | 8150.05 | 2024-09-30 11:15:00 | 8223.10 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-09-27 12:45:00 | 8158.40 | 2024-09-30 11:15:00 | 8223.10 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-09-30 10:15:00 | 8154.05 | 2024-09-30 11:15:00 | 8223.10 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-10-01 09:15:00 | 8533.95 | 2024-10-04 14:15:00 | 8300.05 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2024-10-09 09:15:00 | 8729.50 | 2024-10-10 12:15:00 | 8530.00 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2024-10-10 09:15:00 | 8699.95 | 2024-10-10 12:15:00 | 8530.00 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2024-10-15 12:30:00 | 8512.45 | 2024-10-16 11:15:00 | 8564.85 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-10-16 09:45:00 | 8509.00 | 2024-10-16 11:15:00 | 8564.85 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-10-21 15:15:00 | 8216.00 | 2024-10-24 12:15:00 | 7805.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 15:15:00 | 8216.00 | 2024-10-25 14:15:00 | 7737.95 | STOP_HIT | 0.50 | 5.82% |
| SELL | retest2 | 2024-11-18 09:15:00 | 7400.00 | 2024-11-21 09:15:00 | 7125.00 | PARTIAL | 0.50 | 3.72% |
| SELL | retest2 | 2024-11-18 09:15:00 | 7400.00 | 2024-11-21 12:15:00 | 7334.50 | STOP_HIT | 0.50 | 0.89% |
| SELL | retest2 | 2024-11-19 13:00:00 | 7500.00 | 2024-11-22 15:15:00 | 7490.00 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2024-11-27 09:30:00 | 7528.20 | 2024-11-27 13:15:00 | 7450.00 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-11-27 11:30:00 | 7533.00 | 2024-11-27 13:15:00 | 7450.00 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-12-03 14:45:00 | 7419.05 | 2024-12-04 11:15:00 | 7503.25 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-12-03 15:15:00 | 7418.05 | 2024-12-04 11:15:00 | 7503.25 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-12-11 09:15:00 | 7770.00 | 2024-12-17 11:15:00 | 7732.50 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2024-12-13 11:00:00 | 7716.15 | 2024-12-17 11:15:00 | 7732.50 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2025-01-01 09:30:00 | 6915.10 | 2025-01-01 13:15:00 | 6912.55 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-01-01 11:15:00 | 6912.05 | 2025-01-01 13:15:00 | 6912.55 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2025-01-03 09:15:00 | 6944.90 | 2025-01-03 12:15:00 | 6840.10 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-01-15 13:30:00 | 6181.35 | 2025-01-16 10:15:00 | 6336.30 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest1 | 2025-01-20 09:15:00 | 6464.05 | 2025-01-23 10:15:00 | 6544.05 | STOP_HIT | 1.00 | 1.24% |
| BUY | retest1 | 2025-01-20 14:15:00 | 6635.00 | 2025-01-23 10:15:00 | 6544.05 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-01-22 10:15:00 | 6605.50 | 2025-01-24 11:15:00 | 6502.00 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-01-23 09:30:00 | 6572.35 | 2025-01-24 11:15:00 | 6502.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-01-23 11:15:00 | 6575.90 | 2025-01-24 11:15:00 | 6502.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-01-24 09:30:00 | 6588.00 | 2025-01-24 11:15:00 | 6502.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-01-28 09:15:00 | 6392.10 | 2025-01-30 14:15:00 | 6498.80 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-01-28 15:00:00 | 6351.00 | 2025-01-30 14:15:00 | 6498.80 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-01-29 15:15:00 | 6396.00 | 2025-01-30 14:15:00 | 6498.80 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-01-30 10:45:00 | 6395.45 | 2025-01-30 14:15:00 | 6498.80 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-01-30 13:15:00 | 6383.05 | 2025-01-30 14:15:00 | 6498.80 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-02-01 15:00:00 | 6573.05 | 2025-02-03 11:15:00 | 6466.30 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-02-07 10:30:00 | 6562.05 | 2025-02-11 09:15:00 | 6368.75 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2025-02-18 09:15:00 | 5936.20 | 2025-02-19 09:15:00 | 6033.05 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-03-04 11:15:00 | 5912.20 | 2025-03-05 14:15:00 | 6010.00 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-03-05 10:00:00 | 5907.60 | 2025-03-05 14:15:00 | 6010.00 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-03-28 09:30:00 | 6270.20 | 2025-03-28 10:15:00 | 6212.00 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-04-01 13:15:00 | 6210.00 | 2025-04-02 10:15:00 | 6255.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-04-01 14:15:00 | 6189.70 | 2025-04-02 10:15:00 | 6255.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-04-01 14:45:00 | 6206.70 | 2025-04-02 10:15:00 | 6255.00 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-04-02 09:15:00 | 6181.75 | 2025-04-02 10:15:00 | 6255.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-04-21 09:15:00 | 6405.00 | 2025-04-25 10:15:00 | 6347.50 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-04-25 10:00:00 | 6392.00 | 2025-04-25 10:15:00 | 6347.50 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-04-30 09:15:00 | 6252.50 | 2025-05-05 13:15:00 | 6456.00 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2025-05-07 15:15:00 | 6700.50 | 2025-05-21 11:15:00 | 6904.50 | STOP_HIT | 1.00 | 3.04% |
| BUY | retest2 | 2025-05-09 12:00:00 | 6694.50 | 2025-05-21 11:15:00 | 6904.50 | STOP_HIT | 1.00 | 3.14% |
| BUY | retest2 | 2025-05-09 15:15:00 | 6731.50 | 2025-05-21 11:15:00 | 6904.50 | STOP_HIT | 1.00 | 2.57% |
| BUY | retest2 | 2025-06-10 09:15:00 | 6529.00 | 2025-06-12 10:15:00 | 6477.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-06-10 11:00:00 | 6527.50 | 2025-06-12 10:15:00 | 6477.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-06-11 13:45:00 | 6526.00 | 2025-06-12 10:15:00 | 6477.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-06-11 14:15:00 | 6529.50 | 2025-06-12 10:15:00 | 6477.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-06-13 09:15:00 | 6325.50 | 2025-06-24 10:15:00 | 6312.00 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2025-06-13 11:15:00 | 6378.00 | 2025-06-24 10:15:00 | 6312.00 | STOP_HIT | 1.00 | 1.03% |
| BUY | retest2 | 2025-06-25 09:15:00 | 6339.00 | 2025-07-02 09:15:00 | 6972.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-08 10:45:00 | 6646.00 | 2025-07-10 09:15:00 | 6729.00 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-07-17 13:45:00 | 6920.00 | 2025-07-21 09:15:00 | 6836.00 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-07-17 14:15:00 | 6919.00 | 2025-07-21 09:15:00 | 6836.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest1 | 2025-08-07 10:15:00 | 5765.00 | 2025-08-07 14:15:00 | 5804.50 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest1 | 2025-08-07 10:45:00 | 5763.00 | 2025-08-07 14:15:00 | 5804.50 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-08-14 10:45:00 | 5915.00 | 2025-08-14 11:15:00 | 5870.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-08-18 13:00:00 | 5842.00 | 2025-08-19 11:15:00 | 5893.50 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-08-18 13:30:00 | 5831.00 | 2025-08-19 11:15:00 | 5893.50 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-08-18 14:30:00 | 5841.00 | 2025-08-19 11:15:00 | 5893.50 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-08-21 15:15:00 | 5842.00 | 2025-08-29 09:15:00 | 5549.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 09:15:00 | 5810.00 | 2025-08-29 11:15:00 | 5519.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 15:15:00 | 5842.00 | 2025-08-29 13:15:00 | 5628.50 | STOP_HIT | 0.50 | 3.65% |
| SELL | retest2 | 2025-08-26 09:15:00 | 5810.00 | 2025-08-29 13:15:00 | 5628.50 | STOP_HIT | 0.50 | 3.12% |
| BUY | retest2 | 2025-09-11 09:15:00 | 5775.00 | 2025-09-12 11:15:00 | 5710.50 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-09-12 09:15:00 | 5761.00 | 2025-09-12 11:15:00 | 5710.50 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-09-16 13:45:00 | 5724.00 | 2025-09-17 12:15:00 | 5737.00 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-09-24 11:15:00 | 5752.50 | 2025-09-25 12:15:00 | 5800.00 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-09-24 12:00:00 | 5770.00 | 2025-09-25 12:15:00 | 5800.00 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-09-24 13:45:00 | 5770.00 | 2025-09-25 12:15:00 | 5800.00 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-09-24 14:30:00 | 5772.50 | 2025-09-25 12:15:00 | 5800.00 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-09-25 09:15:00 | 5769.00 | 2025-09-25 12:15:00 | 5800.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-09-29 11:30:00 | 5690.50 | 2025-09-30 09:15:00 | 5952.50 | STOP_HIT | 1.00 | -4.60% |
| SELL | retest2 | 2025-09-29 13:30:00 | 5651.50 | 2025-09-30 09:15:00 | 5952.50 | STOP_HIT | 1.00 | -5.33% |
| SELL | retest2 | 2025-10-06 09:15:00 | 5624.50 | 2025-10-13 15:15:00 | 5554.50 | STOP_HIT | 1.00 | 1.24% |
| SELL | retest2 | 2025-10-06 11:00:00 | 5640.50 | 2025-10-13 15:15:00 | 5554.50 | STOP_HIT | 1.00 | 1.52% |
| SELL | retest2 | 2025-10-07 09:15:00 | 5634.00 | 2025-10-13 15:15:00 | 5554.50 | STOP_HIT | 1.00 | 1.41% |
| BUY | retest2 | 2025-10-21 13:45:00 | 5579.00 | 2025-10-27 09:15:00 | 5534.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-10-24 14:30:00 | 5565.50 | 2025-10-27 09:15:00 | 5534.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-10-27 09:15:00 | 5577.00 | 2025-10-27 09:15:00 | 5534.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-11-03 13:00:00 | 6375.00 | 2025-11-04 09:15:00 | 6391.50 | STOP_HIT | 1.00 | 0.26% |
| BUY | retest2 | 2025-11-04 09:15:00 | 6420.00 | 2025-11-04 09:15:00 | 6391.50 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-11-06 14:15:00 | 6361.00 | 2025-11-11 15:15:00 | 6042.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 14:15:00 | 6361.00 | 2025-11-12 13:15:00 | 6141.50 | STOP_HIT | 0.50 | 3.45% |
| BUY | retest2 | 2025-12-26 09:15:00 | 5482.00 | 2025-12-29 11:15:00 | 5426.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-12-26 12:45:00 | 5485.50 | 2025-12-29 11:15:00 | 5426.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2026-01-02 15:00:00 | 5611.00 | 2026-01-05 09:15:00 | 5554.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2026-01-23 09:30:00 | 5296.00 | 2026-01-27 14:15:00 | 5339.50 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-01-23 10:30:00 | 5304.00 | 2026-01-27 14:15:00 | 5339.50 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2026-01-30 09:30:00 | 5451.00 | 2026-02-13 09:15:00 | 5735.00 | STOP_HIT | 1.00 | 5.21% |
| BUY | retest2 | 2026-01-30 10:00:00 | 5455.00 | 2026-02-13 09:15:00 | 5735.00 | STOP_HIT | 1.00 | 5.13% |
| BUY | retest2 | 2026-02-01 10:00:00 | 5469.50 | 2026-02-13 09:15:00 | 5735.00 | STOP_HIT | 1.00 | 4.85% |
| BUY | retest2 | 2026-02-02 09:30:00 | 5453.00 | 2026-02-13 09:15:00 | 5735.00 | STOP_HIT | 1.00 | 5.17% |
| BUY | retest2 | 2026-02-02 13:30:00 | 5499.00 | 2026-02-13 09:15:00 | 5735.00 | STOP_HIT | 1.00 | 4.29% |
| BUY | retest2 | 2026-02-02 14:30:00 | 5496.50 | 2026-02-13 09:15:00 | 5735.00 | STOP_HIT | 1.00 | 4.34% |
| BUY | retest2 | 2026-02-03 09:15:00 | 5569.50 | 2026-02-13 09:15:00 | 5735.00 | STOP_HIT | 1.00 | 2.97% |
| SELL | retest2 | 2026-03-24 10:30:00 | 4863.00 | 2026-03-25 12:15:00 | 5077.00 | STOP_HIT | 1.00 | -4.40% |
| BUY | retest2 | 2026-04-13 10:15:00 | 5168.80 | 2026-04-23 12:15:00 | 5346.00 | STOP_HIT | 1.00 | 3.43% |
| BUY | retest2 | 2026-04-15 09:30:00 | 5145.00 | 2026-04-23 12:15:00 | 5346.00 | STOP_HIT | 1.00 | 3.91% |
