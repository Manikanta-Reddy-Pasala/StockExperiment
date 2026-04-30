# Eicher Motors Ltd. (EICHERMOT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 7110.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 1
- **Winners / losers:** 5 / 1
- **Target hits / EMA400 exits:** 5 / 1
- **Total realized P&L (per unit):** 1077.81
- **Avg P&L per closed trade:** 179.64

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 12:15:00 | 4744.60 | 4828.25 | 4828.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 13:15:00 | 4717.60 | 4827.15 | 4828.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 10:15:00 | 4843.25 | 4791.37 | 4808.30 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-22 09:15:00 | 4770.70 | 4791.99 | 4808.11 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 4770.70 | 4791.99 | 4808.11 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-10-22 14:15:00 | 4752.95 | 4791.65 | 4807.54 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 4735.30 | 4755.98 | 4786.22 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-10-29 11:15:00 | 4823.00 | 4756.58 | 4786.22 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 5003.75 | 4808.13 | 4807.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 5014.20 | 4829.86 | 4818.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 10:15:00 | 4820.80 | 4851.30 | 4831.55 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-06 09:15:00 | 4864.50 | 4841.24 | 4829.69 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 4864.50 | 4841.24 | 4829.69 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-09 11:15:00 | 4830.15 | 4843.29 | 4831.26 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 10:15:00 | 4715.80 | 4823.12 | 4823.13 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 09:15:00 | 4897.40 | 4821.83 | 4821.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 10:15:00 | 4960.10 | 4823.21 | 4822.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-13 10:15:00 | 4943.40 | 4960.93 | 4901.02 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-14 10:15:00 | 5004.25 | 4960.70 | 4902.96 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-02-11 09:15:00 | 5055.50 | 5184.27 | 5069.76 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-03-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 13:15:00 | 4904.90 | 4993.89 | 4994.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-04 09:15:00 | 4847.00 | 4990.64 | 4992.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 11:15:00 | 4986.40 | 4980.63 | 4987.33 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-03-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-07 15:15:00 | 5130.00 | 4994.61 | 4993.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 5138.45 | 5007.79 | 5001.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 5104.50 | 5193.17 | 5114.82 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-08 09:15:00 | 5181.30 | 5186.24 | 5113.99 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 5181.30 | 5186.24 | 5113.99 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-04-08 11:15:00 | 5225.00 | 5186.60 | 5114.89 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-05-09 09:15:00 | 5298.00 | 5443.23 | 5320.65 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 11:15:00 | 6854.00 | 7352.51 | 7353.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 6745.00 | 7329.07 | 7342.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 11:15:00 | 7072.00 | 7049.48 | 7177.84 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-30 09:15:00 | 6982.50 | 7123.00 | 7172.48 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-22 09:15:00 | 4770.70 | 2024-10-23 14:15:00 | 4658.47 | TARGET | 112.23 |
| SELL | 2024-10-22 14:15:00 | 4752.95 | 2024-10-25 09:15:00 | 4589.17 | TARGET | 163.78 |
| BUY | 2024-12-06 09:15:00 | 4864.50 | 2024-12-09 11:15:00 | 4830.15 | EXIT_EMA400 | -34.35 |
| BUY | 2025-01-14 10:15:00 | 5004.25 | 2025-02-01 12:15:00 | 5308.13 | TARGET | 303.88 |
| BUY | 2025-04-08 09:15:00 | 5181.30 | 2025-04-11 09:15:00 | 5383.23 | TARGET | 201.93 |
| BUY | 2025-04-08 11:15:00 | 5225.00 | 2025-04-15 15:15:00 | 5555.34 | TARGET | 330.34 |
