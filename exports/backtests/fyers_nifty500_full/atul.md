# Atul Ltd. (ATUL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 6825.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 1 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** -817.61
- **Avg P&L per closed trade:** -163.52

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 14:15:00 | 7238.10 | 7610.40 | 7611.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 7065.00 | 7446.40 | 7506.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 11:15:00 | 5777.50 | 5733.25 | 6104.47 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-19 11:15:00 | 5649.00 | 5730.14 | 6077.82 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-28 14:15:00 | 6051.00 | 5762.81 | 6015.85 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 13:15:00 | 6985.50 | 6010.17 | 6009.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 14:15:00 | 7027.50 | 6020.30 | 6014.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 12:15:00 | 6998.00 | 7005.31 | 6742.06 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-18 14:15:00 | 7055.00 | 7005.46 | 6744.75 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-18 13:15:00 | 6986.50 | 7335.93 | 7098.83 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 13:15:00 | 6673.00 | 6948.83 | 6949.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 6571.00 | 6939.72 | 6944.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 10:15:00 | 6480.00 | 6478.32 | 6618.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-15 12:15:00 | 6420.00 | 6477.85 | 6617.30 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-11-18 09:15:00 | 6227.00 | 5931.17 | 6099.19 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 09:15:00 | 6165.00 | 6022.96 | 6022.34 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 5821.00 | 6021.90 | 6022.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 10:15:00 | 5755.50 | 6019.25 | 6020.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 11:15:00 | 5974.00 | 5952.79 | 5983.97 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-02 10:15:00 | 5886.00 | 5995.01 | 6002.77 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 5963.50 | 5992.10 | 6001.10 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-03 09:15:00 | 6403.50 | 5996.20 | 6003.10 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 15:15:00 | 6200.00 | 6010.32 | 6010.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 13:15:00 | 6310.00 | 6021.90 | 6015.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 6383.50 | 6405.66 | 6268.59 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-04 15:15:00 | 6700.00 | 6405.26 | 6272.41 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-09 09:15:00 | 6278.50 | 6412.44 | 6285.78 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-19 11:15:00 | 5649.00 | 2025-03-28 14:15:00 | 6051.00 | EXIT_EMA400 | -402.00 |
| BUY | 2025-06-18 14:15:00 | 7055.00 | 2025-07-18 13:15:00 | 6986.50 | EXIT_EMA400 | -68.50 |
| SELL | 2025-09-15 12:15:00 | 6420.00 | 2025-10-14 13:15:00 | 5828.11 | TARGET | 591.89 |
| SELL | 2026-02-02 10:15:00 | 5886.00 | 2026-02-03 09:15:00 | 6403.50 | EXIT_EMA400 | -517.50 |
| BUY | 2026-03-04 15:15:00 | 6700.00 | 2026-03-09 09:15:00 | 6278.50 | EXIT_EMA400 | -421.50 |
