# Polycab India Ltd. (POLYCAB.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 8088.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 13 |
| ALERT2 | 12 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / EMA400 exits:** 3 / 3
- **Total realized P&L (per unit):** 431.21
- **Avg P&L per closed trade:** 71.87

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 11:15:00 | 6538.05 | 6616.39 | 6616.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 13:15:00 | 6470.25 | 6614.13 | 6615.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 11:15:00 | 6635.80 | 6609.33 | 6612.96 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 11:15:00 | 6649.35 | 6616.40 | 6616.39 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 14:15:00 | 6578.85 | 6616.23 | 6616.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 09:15:00 | 6559.00 | 6615.30 | 6615.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 10:15:00 | 6622.45 | 6590.89 | 6603.15 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 13:15:00 | 6781.60 | 6614.08 | 6613.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 10:15:00 | 6818.95 | 6620.96 | 6617.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 09:15:00 | 6620.00 | 6693.39 | 6661.46 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-11 09:15:00 | 6791.50 | 6681.24 | 6658.89 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-09-18 11:15:00 | 6675.00 | 6707.07 | 6677.19 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-11-01 17:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-01 17:15:00 | 6524.95 | 6802.89 | 6804.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 6419.15 | 6796.09 | 6800.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 6824.10 | 6760.07 | 6781.66 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 15:15:00 | 6698.90 | 6770.72 | 6785.34 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 6766.45 | 6769.57 | 6784.54 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-11 13:15:00 | 6719.75 | 6768.92 | 6784.07 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-11-25 09:15:00 | 6733.00 | 6653.82 | 6715.21 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 09:15:00 | 7360.10 | 6767.55 | 6765.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 14:15:00 | 7409.05 | 6834.24 | 6800.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 14:15:00 | 7156.05 | 7189.49 | 7028.69 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-30 09:15:00 | 7263.95 | 7173.74 | 7042.61 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-06 10:15:00 | 7045.90 | 7204.78 | 7081.19 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-01-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 12:15:00 | 6529.30 | 6991.74 | 6993.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-15 13:15:00 | 6481.10 | 6986.66 | 6990.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 10:15:00 | 5379.85 | 5353.17 | 5765.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-20 09:15:00 | 4954.90 | 5352.00 | 5752.36 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-22 09:15:00 | 5529.00 | 5207.14 | 5456.01 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 13:15:00 | 5946.00 | 5571.86 | 5571.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 14:15:00 | 5979.50 | 5575.91 | 5573.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 09:15:00 | 5891.00 | 5953.68 | 5839.54 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-23 09:15:00 | 6139.50 | 5946.86 | 5843.64 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 7496.00 | 7607.56 | 7460.38 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-24 11:15:00 | 7430.50 | 7604.53 | 7460.32 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 7305.50 | 7393.33 | 7393.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 7270.00 | 7392.10 | 7392.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 13:15:00 | 7384.50 | 7349.88 | 7370.56 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 09:15:00 | 7653.00 | 7390.11 | 7389.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 7805.00 | 7459.35 | 7427.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 09:15:00 | 7536.50 | 7571.57 | 7498.47 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2026-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 12:15:00 | 7038.50 | 7442.15 | 7443.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-22 14:15:00 | 6991.50 | 7433.49 | 7439.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 7437.50 | 7224.68 | 7319.05 | EMA200 retest candle locked |

### Cycle 12 — BUY (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 14:15:00 | 7812.50 | 7390.98 | 7390.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 10:15:00 | 7827.50 | 7472.70 | 7434.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-10 11:15:00 | 7959.50 | 7964.14 | 7741.89 | EMA200 retest candle locked |

### Cycle 13 — SELL (started 2026-03-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 15:15:00 | 6800.00 | 7592.16 | 7592.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 6769.50 | 7416.95 | 7497.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 7540.00 | 7345.80 | 7451.73 | EMA200 retest candle locked |

### Cycle 14 — BUY (started 2026-04-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 11:15:00 | 8253.00 | 7531.77 | 7529.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 15:15:00 | 8271.00 | 7733.81 | 7642.27 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-09-11 09:15:00 | 6791.50 | 2024-09-18 11:15:00 | 6675.00 | EXIT_EMA400 | -116.50 |
| SELL | 2024-11-11 13:15:00 | 6719.75 | 2024-11-12 14:15:00 | 6526.80 | TARGET | 192.95 |
| SELL | 2024-11-08 15:15:00 | 6698.90 | 2024-11-13 09:15:00 | 6439.58 | TARGET | 259.32 |
| BUY | 2024-12-30 09:15:00 | 7263.95 | 2025-01-06 10:15:00 | 7045.90 | EXIT_EMA400 | -218.05 |
| SELL | 2025-03-20 09:15:00 | 4954.90 | 2025-04-22 09:15:00 | 5529.00 | EXIT_EMA400 | -574.10 |
| BUY | 2025-06-23 09:15:00 | 6139.50 | 2025-07-18 09:15:00 | 7027.09 | TARGET | 887.59 |
