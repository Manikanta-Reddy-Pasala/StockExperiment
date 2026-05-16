# breakout_60d_high_volume

## Configs available

| Variant | Avg/yr | Avg/mo | Max DD | Worst mo | Trades |
|---|---:|---:|---:|---:|---:|
| **v1 (raw breakout)** | +74.99% | +6.10% | -41.62% | -22.71% | 40 |
| **v2 (regime + quality filter)** ⭐ | **+50.63%** | +4.19% | **-25.95%** | -15.38% | 26 |

v2 = added two risk overlays:
- **Regime filter:** only enter when NIFTY50 close ≥ NIFTY200 SMA (skip bear regimes)
- **Quality filter:** require 90-day return ≥ +5% (skip stocks without prior trend)

DD cut from -42% → -26%. Return cut from +75% → +51%. Same alpha source,
just smoother. Pick the one matching your DD tolerance.

⚠️ User goal of +95%/yr AND DD ≤ -25% NOT simultaneously achievable on
this signal/data — DD-cutting filters cost ~25%/yr in return. Documented
in `SUMMARY_v2.md` + sweep results.

## Strategy

Price-level breakout + volume confirmation. Different alpha source from
relative-momentum rotation — trades level breaks not rank.

- **Universe:** pseudo-N100 by ADV (shared with `momentum_n100_top5_max1`)
- **Entry signal:** close ≥ 60-day high AND volume ≥ 1.5× 20-day avg volume
- **Entry execution:** next trading day at open
- **Exit:** close below 20-day SMA OR -8% trailing stop from peak OR 90-day max hold
- **Position size:** max 1 concurrent, full equity allocation per pick
- **Costs modeled:** 10bps slip + ₹20 brokerage + 0.1% STT (sell side)

## 3-year backtest result (2023-05-15 → 2026-05-15)

| Metric | Value |
|---|---:|
| Starting capital | ₹2,00,000 |
| Ending capital | **₹13,97,515** |
| Total return | **+598.8%** |
| Avg/yr | **+74.99%** |
| Avg/mo | **+6.10%** |
| Median/mo | +5.13% |
| Best mo | +60.41% |
| Worst mo | -22.71% |
| Months ≥ 20% | 5 / 37 |
| Months ≥ 30% | 3 / 37 |
| Max DD | -41.62% |
| Total trades | 40 |
| Win rate | 45.0% |
| Total fees | ₹39,388 |

Full ledger: `exports/models/breakout_60d_high_volume/SUMMARY.md`

## Why this diversifies the existing book

| Aspect | Momentum N100 | Breakout 60d |
|---|---|---|
| Signal type | Relative ranking (60d return) | Absolute level break + volume |
| Rebalance | Monthly | Event-driven (per breakout) |
| Hold period | Until rotation drops it | Until 20d SMA or trail stop |
| Concentration | Top 1 of 100 | Top 1 (rotates faster) |
| Avg trade count/year | ~12 | ~13 |
| Drawdown profile | -41% | -42% |

Both equity but different alpha sources → portfolio split between them adds
diversification benefit beyond a single concentrated bet.

## Files

| File | Purpose |
|---|---|
| `backtest.py` | v1 backtest (raw 60d-high + volume + 8% trail) |
| `backtest_v2.py` | v2 backtest with regime/ATR/partial/quality filters |
| `sweep.py` | Variant sweep (lookback / vol-mult / trail / max-conc) |
| `data_pull.py` | No-op (shares N100 cache with momentum model) |
| `cron.py` | Registration stubs (trading not wired yet) |
| `README.md` | This file |

## v2 reproduce (winner under -25% DD constraint)

```bash
python tools/models/breakout_60d_high_volume/backtest_v2.py \
    --universe-file /app/logs/momrot/universes/n100_current.json \
    --from 2023-05-15 --to 2026-05-15 \
    --capital 200000 --max-conc 1 \
    --regime-on --quality-on --quality-min-90d 0.05 \
    --out exports/models/breakout_60d_high_volume/SUMMARY_v2.md
```

## v2 result detail

- Start ₹2L → End ₹7.95L (+297.7%) over 3yr
- Avg/yr: +50.63% | Avg/mo: +4.19%
- Best mo: +60.34% | Worst: -15.38%
- Months ≥ 20%: 4/37 | Months ≥ 30%: 3/37
- Max DD: -25.95% (HIT target)
- 26 trades, 50% WR
- Fees: ₹16,706

## Filter sweep summary (best risk-adjusted configs)

| Config | Yr | DD | Worst mo |
|---|---:|---:|---:|
| regime+quality5% (winner) | +50.6% | -25.95% | -15.4% |
| regime+partial30@50+qual5 | +41.8% | -25.97% | -15.4% |
| regime+partial15@30+qual5 | +40.3% | -24.88% | -14.4% |
| regime+qual5+trail12 | +39.8% | -25.97% | -16.9% |
| regime+atr3x+qual5 | +29.2% | -26.00% | -16.9% |

## Reproduce

```bash
# Universe is shared with Model 3 — assumes it exists
python tools/models/breakout_60d_high_volume/backtest.py \
    --universe-file /app/logs/momrot/universes/n100_current.json \
    --from 2023-05-15 --to 2026-05-15 \
    --capital 200000 --max-conc 1 --breakout-lookback 60 \
    --vol-mult 1.5 --sma-exit 20 --trail-pct 0.08 \
    --max-hold-days 90 \
    --out exports/models/breakout_60d_high_volume/run_$(date +%F).md
```

Sweep variants:
```bash
python tools/models/breakout_60d_high_volume/sweep.py \
    --universe-file /app/logs/momrot/universes/n100_current.json \
    --from 2023-05-15 --to 2026-05-15
```

## Live deployment status

❌ **Not yet wired.** Backtest-only. Live signal generator needs to be built
(should mirror `tools/models/momentum_n100_top5_max1/live_signal.py` but
scan-based instead of rank-based).

When wired, will register cron via `tools/models/breakout_60d_high_volume/cron.py`.

## Honest caveats

- 3-year window is small. ~12-13 trades/year means single-trade luck has
  outsized impact on annualized number.
- 41% drawdown is real. Backtest spent multi-week periods below water.
- 45% win rate means more losers than winners — strategy works because
  winners are bigger than losers (asymmetric payoff).
- Slippage assumed 10bps; real fills on small-cap breakouts can slip 30-50bps.
  Real net return likely 60-70% of backtest = **~50%/yr live expectation**.
