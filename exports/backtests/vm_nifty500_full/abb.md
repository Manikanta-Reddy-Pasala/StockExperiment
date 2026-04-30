# ABB India Ltd. (ABB.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 7230.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 9 |
| ENTRY1 | 7 |
| ENTRY2 | 5 |
| EXIT | 6 |

## P&L

- **Trades closed:** 12
- **Trades open at end:** 0
- **Winners / losers:** 3 / 9
- **Target hits / EMA400 exits:** 3 / 9
- **Total realized P&L (per unit):** 2082.08
- **Avg P&L per closed trade:** 173.51

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-27 15:15:00 | 4233.35 | 4328.40 | 4328.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 09:15:00 | 4227.55 | 4327.39 | 4328.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-13 10:15:00 | 4228.60 | 4228.53 | 4268.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-10-18 11:15:00 | 4139.25 | 4234.96 | 4267.95 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 09:15:00 | 4185.00 | 4131.84 | 4193.42 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2023-11-06 11:15:00 | 4156.75 | 4132.58 | 4193.18 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2023-11-06 14:15:00 | 4205.95 | 4133.97 | 4192.98 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-11-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 14:15:00 | 4424.85 | 4226.88 | 4226.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 15:15:00 | 4450.00 | 4229.10 | 4227.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-02 10:15:00 | 4609.95 | 4616.29 | 4491.02 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-02 12:15:00 | 4667.60 | 4616.93 | 4492.59 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 09:15:00 | 4727.15 | 4741.75 | 4620.90 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-01-24 10:15:00 | 4798.00 | 4742.31 | 4621.79 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 11:15:00 | 4645.05 | 4741.11 | 4637.38 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-01-31 13:15:00 | 4672.90 | 4739.63 | 4637.67 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 09:15:00 | 4658.35 | 4737.39 | 4638.06 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-02-01 10:15:00 | 4620.70 | 4736.23 | 4637.97 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 09:15:00 | 7482.00 | 7812.55 | 7813.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 11:15:00 | 7391.85 | 7763.10 | 7783.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 15:15:00 | 7757.00 | 7744.22 | 7772.54 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 09:15:00 | 8070.00 | 7798.68 | 7798.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 09:15:00 | 8218.80 | 7856.26 | 7828.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 10:15:00 | 7850.00 | 7902.05 | 7855.76 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-08 10:15:00 | 7989.50 | 7896.31 | 7854.42 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 8089.95 | 8230.38 | 8066.34 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-23 10:15:00 | 8062.05 | 8228.70 | 8066.32 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 7289.60 | 7940.49 | 7941.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 09:15:00 | 7006.35 | 7902.48 | 7922.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 10:15:00 | 7347.70 | 7325.69 | 7563.31 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-20 12:15:00 | 7105.10 | 7519.72 | 7573.70 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-17 10:15:00 | 5617.00 | 5319.03 | 5574.88 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 5997.50 | 5621.80 | 5621.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 12:15:00 | 6047.00 | 5702.93 | 5663.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 5885.00 | 5922.37 | 5817.98 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 10:15:00 | 5962.50 | 5921.33 | 5820.02 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 5897.50 | 5956.65 | 5866.08 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-02 14:15:00 | 5907.50 | 5956.16 | 5866.29 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 5875.00 | 5951.89 | 5867.21 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-03 15:15:00 | 5866.50 | 5951.04 | 5867.21 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 10:15:00 | 5691.50 | 5820.86 | 5821.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 09:15:00 | 5663.50 | 5813.58 | 5817.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 10:15:00 | 5233.30 | 5204.64 | 5363.89 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-26 09:15:00 | 5148.70 | 5269.12 | 5352.27 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 5316.50 | 5244.00 | 5322.51 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-07 14:15:00 | 5215.00 | 5244.49 | 5320.83 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 5269.50 | 5211.68 | 5270.33 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-29 11:15:00 | 5294.00 | 5212.50 | 5270.45 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 12:15:00 | 5778.00 | 5143.68 | 5141.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-05 11:15:00 | 5811.50 | 5179.14 | 5159.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 12:15:00 | 5997.00 | 6014.84 | 5767.05 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-24 09:15:00 | 6114.50 | 6016.82 | 5772.95 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-10-18 11:15:00 | 4139.25 | 2023-11-06 14:15:00 | 4205.95 | EXIT_EMA400 | -66.70 |
| SELL | 2023-11-06 11:15:00 | 4156.75 | 2023-11-06 14:15:00 | 4205.95 | EXIT_EMA400 | -49.20 |
| BUY | 2024-01-02 12:15:00 | 4667.60 | 2024-02-01 10:15:00 | 4620.70 | EXIT_EMA400 | -46.90 |
| BUY | 2024-01-24 10:15:00 | 4798.00 | 2024-02-01 10:15:00 | 4620.70 | EXIT_EMA400 | -177.30 |
| BUY | 2024-01-31 13:15:00 | 4672.90 | 2024-02-01 10:15:00 | 4620.70 | EXIT_EMA400 | -52.20 |
| BUY | 2024-10-08 10:15:00 | 7989.50 | 2024-10-09 10:15:00 | 8394.75 | TARGET | 405.25 |
| SELL | 2024-12-20 12:15:00 | 7105.10 | 2025-01-30 11:15:00 | 5699.31 | TARGET | 1405.79 |
| BUY | 2025-06-20 10:15:00 | 5962.50 | 2025-07-03 15:15:00 | 5866.50 | EXIT_EMA400 | -96.00 |
| BUY | 2025-07-02 14:15:00 | 5907.50 | 2025-07-03 15:15:00 | 5866.50 | EXIT_EMA400 | -41.00 |
| SELL | 2025-09-26 09:15:00 | 5148.70 | 2025-10-29 11:15:00 | 5294.00 | EXIT_EMA400 | -145.30 |
| SELL | 2025-10-07 14:15:00 | 5215.00 | 2025-10-29 11:15:00 | 5294.00 | EXIT_EMA400 | -79.00 |
| BUY | 2026-03-24 09:15:00 | 6114.50 | 2026-04-20 14:15:00 | 7139.14 | TARGET | 1024.64 |
