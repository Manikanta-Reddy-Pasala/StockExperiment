# Nifty 500 — All Entries & Exits (Validation File)

_Source: Fyers production data, 720 days, 1H bars._
_Backtest run on container `trading_system_app` at 77.42.45.12._

## Files

| File | Rows | Description |
|------|-----:|-------------|
| `ENTRIES_EXITS_TRADES.csv` | **2,695** | One row per closed trade — entry/exit time, price, P&L, exit reason |
| `ENTRIES_EXITS_SIGNALS.csv` | **14,800** | Every state-machine event — CROSSOVER, ALERT1-3, ENTRY1-2, EXIT |
| `fyers_nifty500_full/<symbol>.md` | — | Per-stock report with cycle-level breakdown |

## TRADES CSV schema

```
symbol,trend,entry_date,entry_time,entry_price,exit_date,exit_time,exit_price,exit_reason,pnl
```

| Column | Notes |
|--------|-------|
| `symbol` | Yahoo format (e.g. `RELIANCE.NS`) — NSE symbol with `.NS` suffix |
| `trend` | `BUY` (long) or `SELL` (short) |
| `entry_date` | YYYY-MM-DD (IST) |
| `entry_time` | HH:MM:SS — start of the 1H candle whose close triggered entry |
| `entry_price` | 1H close at entry candle |
| `exit_date` / `exit_time` | When trade closed |
| `exit_price` | Target price (TARGET) OR 1H close past EMA 400 (EXIT_EMA400) |
| `exit_reason` | `TARGET` (always profitable) or `EXIT_EMA400` (mostly losers) |
| `pnl` | Per-unit P&L: `(exit - entry)` for BUY, `(entry - exit)` for SELL |

## SIGNALS CSV schema

```
symbol,cycle,trend,stage,date,time,price,ema_200,ema_400,note
```

| Column | Notes |
|--------|-------|
| `cycle` | 1-based cycle index per stock; new cycle on each CROSSOVER |
| `stage` | `CROSSOVER` / `ALERT1` / `ALERT2` / `ENTRY1` / `ALERT3` / `ENTRY2` / `EXIT` |
| `price` | 1H close at signal candle (entry signals = entry price) |
| `ema_200` / `ema_400` | EMA values at signal candle |
| `note` | Human-readable stage description |

## Entry/Exit rules to validate

| Rule | Code |
|------|------|
| **CROSSOVER (BUY)** | EMA 200 crosses **above** EMA 400 (prev_ema200 ≤ prev_ema400 AND ema200 > ema400) |
| **CROSSOVER (SELL)** | EMA 200 crosses **below** EMA 400 |
| **ALERT1 (BUY)** | 1H `close > crossover_candle.high` AND `high > crossover_candle.high` |
| **ALERT1 (SELL)** | 1H `close < crossover_candle.low` AND `low < crossover_candle.low` |
| **ALERT2 (BUY)** | 1H `close < EMA 200` AND `low < EMA 200` (price retests EMA 200 from above) |
| **ALERT2 (SELL)** | 1H `close > EMA 200` AND `high > EMA 200` |
| **ENTRY1 (BUY)** | 1H `close > retest1.high` (entry price = that 1H close) |
| **ENTRY1 (SELL)** | 1H `close < retest1.low` |
| **ALERT3 (BUY)** | 1H `low ≤ EMA 400` (price touches/crosses EMA 400) |
| **ALERT3 (SELL)** | 1H `high ≥ EMA 400` |
| **ENTRY2 (BUY)** | 1H `close > retest2.high` (pyramid) |
| **ENTRY2 (SELL)** | 1H `close < retest2.low` |
| **EXIT (BUY)** | 1H `close < EMA 400` (closes ALL open BUY entries) |
| **EXIT (SELL)** | 1H `close > EMA 400` |
| **TARGET (BUY)** | Bar `high ≥ entry_price + 3 × |entry_price − ema_400_at_entry|` |
| **TARGET (SELL)** | Bar `low ≤ entry_price − 3 × |entry_price − ema_400_at_entry|` |
| **TARGET (index)** | Replace `3 × |distance|` with `5000` absolute pts |

Code: `src/services/technical/ema_crossover_strategy.py`
P&L logic: `tools/backtests/run_ema_200_400_backtest.py::simulate_pnl`

## Headline numbers

| Metric | Value |
|--------|-------|
| Symbols processed | 498 / 504 |
| Closed trades | 2,695 |
| Wins | 852 (31.6%) |
| Losses | 1,843 (68.4%) |
| Reward : Risk | 2.77 : 1 |
| Net P&L per unit | **+9,429** |
| Target hits | 758 (always wins) |
| EMA-exit closes | 1,937 (1,843 losers + 94 winners) |

## Per-stock totals

| Status | Stocks |
|--------|-------:|
| Profitable | 247 |
| Losing | 236 |
| Flat / no signals | 15 |

Top winners: HONAUT.NS, ABBOTINDIA.NS, ABB.NS, SOLARINDS.NS, PERSISTENT.NS
Top losers: 3MINDIA.NS, PAGEIND.NS, SHREECEM.NS, DIXON.NS, FORCEMOT.NS

## Validation tips

1. Pick any symbol from `ENTRIES_EXITS_TRADES.csv`
2. Open `fyers_nifty500_full/<symbol_lower>.md` for the cycle context
3. Cross-check signal timestamps against your charting tool's 1H candles
4. NSE 1H candles align to: 09:15, 10:15, 11:15, 12:15, 13:15, 14:15, 15:15
5. Last "1H" bar may close at 15:30 (15-min residual after 15:15 candle)
6. Expected: every ENTRY in signals CSV should map to a row in trades CSV
   (1:1 if no opens at end of window; 1:0 for trades still open at last bar)

## Reproducing on production VM

```bash
ssh root@77.42.45.12

# Container has the harness + Fyers token
docker exec -w /app trading_system_app /usr/local/bin/python \
    /app/tools/backtests/run_ema_200_400_backtest.py \
    --days 720 --source fyers --user-id 1 \
    --universe nifty500 --out /app/exports/backtests_fyers/nifty500_full
```

Or via UI: `/backtest` page (any single symbol, custom window).
