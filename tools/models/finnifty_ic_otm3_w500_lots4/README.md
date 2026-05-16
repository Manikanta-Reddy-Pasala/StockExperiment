# finnifty_ic_otm3_w500_lots4

## Goal smashed: +193%/yr at -9.7% max DD

3-yr backtest delivers **+193.14%/yr at -9.70% max DD** — both
inside user goal (+100-200%/yr at sub-25% DD).

After fixing the seen_exp bug (missed-month recovery), this model is
the **risk-adjusted star** of the portfolio.

## Strategy

- Underlying: FINNIFTY (Nifty Financial Services Index)
- **SELL** OTM 3% Call + OTM 3% Put (body)
- **BUY** wings ±500 points further out (caps risk)
- 4 lots per cycle
- Stop loss: combined pair value ≥ 3× entry credit
- Otherwise hold to monthly expiry (last Thursday)
- Slippage: 1% per leg

## Capital + margin

- Capital: ₹2,00,000
- Lot size: 65 (post Sep 2024) / 40 (pre)
- Max loss per trade (defined by wings): (500 − net_credit) × 65 × 4 ≈ ₹1,25,000
  = ~62% of ₹2L capital per trade
- Backtest never approached max loss (worst trade ≈ -₹98k = 49% of cap)

## 3-year backtest (2023-05-15 → 2026-05-15)

| Metric | Value |
|---|---:|
| Start | ₹2,00,000 |
| End | **₹17,45,118** |
| Total return | **+772.56%** |
| **Avg/yr** | **+193.14%** ✅ (target 100-200%) |
| **Max DD** | **-9.70%** ✅ (target ≤ -25%) |
| Calmar | **19.9** |
| Avg/mo | +22.07% |
| Best mo | +100.31% |
| Worst mo | -49.09% |
| Win rate | 83.3% |
| Trades | 36 |

### Yearly ROI

| Year | Trades | WR | P&L | ROI |
|---|---:|---:|---:|---:|
| 2023 (May-Dec) | 8 | 87.5% | ₹5,65,338 | **+282.67%** |
| 2024 | 12 | 91.7% | ₹4,85,809 | **+242.90%** |
| 2025 | 12 | 83.3% | ₹5,76,360 | **+288.18%** |
| 2026 (Jan-May) | 4 | 50.0% | ₹-82,389 | -41.19% |

2026 partial-year drag from single tail trade. 3 of 4 years strongly positive.

## Entry/exit logic

```
Each Monday d:
  exp = nearest monthly expiry > d
  if exp already used  → SKIP
  spot = FINNIFTY close on d
  CE_strike  = round(spot × 1.03, step=50)
  PE_strike  = round(spot × 0.97, step=50)
  wing_CE    = CE_strike + 500
  wing_PE    = PE_strike − 500

  validate 4 strikes + bars exist on entry day
  if wing's first bar > d → RETRY next Monday  (recovered ~30% of months)

  ENTER:
    SELL CE × 4 lots × 65, SELL PE × 4 lots × 65
    BUY wing_CE × 4 lots × 65, BUY wing_PE × 4 lots × 65
    net_credit = (CE_px + PE_px) − (wCE_px + wPE_px)

  EXIT (whichever fires first):
    pair_value ≥ 3 × net_credit → STOP (buy-back losers)
    else hold to expiry Thursday → settle intrinsic
```

## Forward applicability

✅ FinNifty MONTHLY options still trade post-SEBI weekly cut (Nov 2024).
Strategy is **forward-deployable** without modification. Live signal
emitter wired in `live_signal.py`, daily cron at 09:25 + 14:30 IST.

## Files

| File | Purpose |
|---|---|
| `run_winner.py` | Run config + emit per-trade ledger |
| `live_signal.py` | Monday entry scan + daily stop monitor + expiry settle |
| `data_pull.py` | No-op (shares bhav cache with finnifty_ic_otm4_w300) |
| `cron.py` | Signal + execute job registrations (LIVE_TRADING_OPTIONS gated) |
| `README.md` | This file |

`exports/models/finnifty_ic_otm3_w500_lots4/`:
| `SUMMARY.md` | Full report with every trade + monthly equity curve |
| `trades.csv` | Per-trade ledger |
| `monthly.csv` | Monthly stats |

## How to reproduce

```bash
# Ensure FinNifty bhavcopy + spot data ingested (shared infra)
docker exec trading_system_app python tools/shared/fetch_index_spot.py \
    --symbol NSE:FINNIFTY-INDEX --from 2023-01-01 --to 2026-05-15
docker exec trading_system_app python tools/shared/prefetch_bhav.py \
    --from 2023-05-15 --to 2026-05-15 \
    --underlying FINNIFTY --instrument OPTIDX

# Run the winner — produces exports/models/.../SUMMARY.md + trades.csv
docker exec trading_system_app python \
    tools/models/finnifty_ic_otm3_w500_lots4/run_winner.py \
    --from 2023-05-15 --to 2026-05-15 --capital 200000 --lots 4
```

## Honest caveats

- 36 trades over 3 yrs ≈ 1/month. Strategy hits every monthly cycle now.
- Worst single trade -₹98k (49% of capital). Live could exceed if
  execution slips on illiquid wings during fast moves.
- 2026-05 tail trade -41% — strategy still has positive expectancy
  across full sample (Calmar 19.9 over 3 years).
- Live realistic estimate: 70-80% of backtest = **+135-155%/yr live,
  -10-15% live DD**.
- Margin requirement = wing_width × lot × lots ≈ ₹1.30L. Capital +
  buffer must support this. Won't enter if available cash < ₹1.5L.
- 4 lots is leveraged. If broker margin per IC unit larger, scale to
  lots=3 ≈ +145%/yr at -7.4% DD (still meets DD target).

## Comparison vs sibling model

| Model | Avg/yr | Max DD | Calmar | Trades | WR |
|---|---:|---:|---:|---:|---:|
| **otm3_w500_lots4** (this) | **+193%** | **-9.7%** | **19.9** | 36 | 83% |
| otm4_w300_lots5 | +337% avg | -42.8% | 7.9 | 35 | 77% |

otm4 has higher raw return but ~5× the DD. **This model is the
risk-adjusted sweet spot** for the portfolio.

## Position in portfolio (4 models)

- **Equity**: `momentum_n100_top5_max1` (monthly N100, wired) + `midcap_narrow_60d_breakout` (60d swing, unwired)
- **Options**: `finnifty_ic_otm4_w300_lots5` (aggressive) + `finnifty_ic_otm3_w500_lots4` (this — balanced)
