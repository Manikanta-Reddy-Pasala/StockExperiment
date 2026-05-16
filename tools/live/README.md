# Live Execution Infrastructure

Generic broker + execution scripts. Used by Model 3 (momentum_n100_top5_max1)
for live Fyers order placement. **No paper trading.**

## Files

| File | Purpose |
|---|---|
| `fyers_executor.py` | Real Fyers order placement (gated by `LIVE_TRADING=true`) |
| `risk_manager.py` | Pre-trade risk check (capital lock, kill-switch) |
| `daily_summary.py` | Read live ledger from `/app/logs/momrot/ledger/`, emit NAV/P&L |
| `telegram_notify.py` | Optional Telegram notifications for signals/fills |
| `run_daily.sh` | Cron wrapper — see `--help` |

## Daily flow (cron)

```
09:00 IST  ./run_daily.sh prefetch        # cache OHLCV (N100)
09:30 IST  ./run_daily.sh signals         # emit signals (rebalance-gated)
09:35 IST  LIVE_TRADING=true ./run_daily.sh live    # place Fyers orders
15:35 IST  ./run_daily.sh summary         # post-close NAV + P&L
```

## Live trading safety

- `LIVE_TRADING=true` must be set explicitly. Default refuses.
- `USER_ID` env selects which Fyers session (broker_configurations row).
- Capital + kill-switch enforced in `risk_manager.py`.

## Bootstrap (one-time)

1. Build the N100 universe snapshot:
   ```bash
   python tools/models/momentum_n100_top5_max1/build_universe.py \
       --top 100 --out /app/logs/momrot/universes/n100_current.json
   ```
2. Ensure Fyers token is fresh (`tools/refresh_fyers_token.py`).
3. Test signals dry-run: `./run_daily.sh signals-force` then inspect JSON.
4. Enable live: `LIVE_TRADING=true ./run_daily.sh live`.

## Refresh universe

Run weekly (Sunday) or after universe drift:
```bash
python tools/models/momentum_n100_top5_max1/build_universe.py \
    --top 100 --out /app/logs/momrot/universes/n100_current.json
```
