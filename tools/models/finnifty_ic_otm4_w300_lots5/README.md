# finnifty_ic_otm4_w300_lots5

## Strategy

FINNIFTY (Nifty Financial Services Index) monthly Iron Condor.

- **SELL** OTM 4% Call (body) + OTM 4% Put (body)
- **BUY** Call wing +300 points further + Put wing −300 points further (caps risk)
- **5 lots** per cycle
- **Stop loss:** exit if combined pair value ≥ 3× entry credit
- Otherwise hold to monthly expiry (last Thursday)
- **Defined risk:** max loss bounded by wing width → margin ≈ ₹85k-100k for 5 lots

## Backtest result (2023-05-15 → 2026-05-15, 3 years)

| Metric | Value |
|---|---:|
| Start capital | ₹2,00,000 |
| Final NAV | ₹22,25,673 |
| Total return | **+1012.84%** |
| Avg/yr | **+337.6%** ✅ |
| **Max DD (portfolio)** | **-13.88%** ✅ (target ≤ 25%) |
| Calmar | **24.3** ⭐ |
| Win rate | 77.1% (27W / 8L) |
| Trades | 35 over 33 months |

| Year | Trades | WR | P&L | ROI on ₹2L |
|---|---:|---:|---:|---:|
| 2023 (May-Dec) | 8 | 62.5% | ₹3,43,816 | **+171.91%** |
| 2024 | 12 | 91.7% | ₹4,34,490 | **+217.25%** |
| 2025 | 12 | 75.0% | ₹6,49,567 | **+324.78%** |
| 2026 (Jan-May) | 3 | 66.7% | ₹5,97,799 | **+298.90%** |
| **3-yr total** | **35** | **77.1%** | **₹20,25,673** | **+1012.84%** |

- Final NAV: ₹22,25,673 from ₹2L (10.1× growth)
- Avg/mo: **+30.69%** | Best mo: +316.3% | Worst mo: -42.8%
- **Max portfolio DD: -13.88%** (peak-to-trough equity curve)
- Months ≥20%: 12/33 | Months ≥30%: 10/33
- Max single-trade loss: ₹96,325 (48.2% of capital, defined-risk by wings)
- Calmar (Avg-yr/MaxDD): **~24** ⭐

Full trade ledger + monthly equity curve: `exports/models/finnifty_ic_otm4_w300_lots5/SUMMARY.md`
Per-month invested capital + credit received: `exports/models/finnifty_ic_otm4_w300_lots5/MONTHLY_INVESTED.md`

## Capital invested per cycle

| Period | Lot size | Margin/cycle (5 lots) | Net credit typical | Defined max loss |
|---|---:|---:|---:|---:|
| Pre Sep 2024 | 40 | wing_width × 40 × 5 = ₹60,000 | ~₹15-200k | ₹40-60k |
| Post Sep 2024 | 65 | wing_width × 65 × 5 = ₹97,500 | ~₹25-280k | ₹65-95k |

Each IC cycle deploys the full margin amount as defined-risk capital. Multiple cycles overlap only when wings drift across expiries — single-trade max loss is bounded by `wing_width × lot × lots − net_credit`.

## Entry/exit logic per cycle

```
Each Monday d:
  exp = nearest monthly expiry > d
  if exp already used → SKIP
  spot = FINNIFTY close on d
  CE_strike  = round(spot × 1.04, step=50)
  PE_strike  = round(spot × 0.96, step=50)
  wing_CE    = CE_strike + 300
  wing_PE    = PE_strike − 300

  validate 4 strikes exist in option_universe + have daily bars on d
    (if wing strike's first bar > d, RETRY next Monday — recovers
     ~30% of months that earlier versions missed)

  enter on next valid day:
    SELL CE × 5 lots, SELL PE × 5 lots
    BUY wing_CE × 5 lots, BUY wing_PE × 5 lots
    net_credit = (CE_px + PE_px) − (wCE_px + wPE_px)

  hold until:
    pair_value ≥ 3 × net_credit → STOP exit (buy-back losers)
    else → hold to expiry Thursday → settle intrinsic
```

## Forward applicability

✅ FinNifty monthly options still trade through 2030+. (Weekly killed by SEBI Nov 2024.)

## Files

| File | Purpose |
|---|---|
| `schema.sql` | `historical_options` + `option_universe` DDL |
| `sweep.py` | Iron Condor variant sweep (multiple OTM/width/lots combos) |
| `run_winner.py` | Run winning config + emit per-trade ledger CSV/MD |
| `live_signal.py` | Monday entry scan + daily stop monitor + expiry settle (DB ledger) |
| `data_pull.py` | Daily index spots + monthly option bhavcopy ingest |
| `cron.py` | Schedule registrations (data + trading) |

## How to reproduce

```bash
# 1. Create tables (one-time)
docker exec -i trading_system_db psql -U trader -d trading_system \
    < tools/models/finnifty_ic_otm4_w300_lots5/schema.sql

# 2. Fetch FINNIFTY spot history
python tools/shared/fetch_index_spot.py \
    --symbol NSE:FINNIFTY-INDEX --from 2023-01-01 --to 2026-05-15

# 3. Ingest FINNIFTY option chain bhavcopy
python tools/shared/prefetch_bhav.py \
    --from 2023-05-15 --to 2026-05-15 \
    --underlying FINNIFTY --instrument OPTIDX

# 4. Run the winning config and produce trade ledger
python tools/models/finnifty_ic_otm4_w300_lots5/run_winner.py \
    --from 2023-05-15 --to 2026-05-15 --capital 200000

# 5. (optional) sweep all variants
python tools/models/finnifty_ic_otm4_w300_lots5/sweep.py \
    --from 2023-05-15 --to 2026-05-15 --capital 200000
```

## Live signal output

```bash
# Monday entry scan + daily stop monitor (called by tech_scheduler cron)
python tools/models/finnifty_ic_otm4_w300_lots5/live_signal.py \
    --signals-out /app/logs/finnifty_ic_otm4_w300_lots5/signals/$(date +%F).json
```

Emits 4 ENTRY signals (sell CE+PE, buy wings) on first Monday of new
monthly cycle. On subsequent days, checks if combined pair value crossed
stop or if expiry reached → emits 4 EXIT signals.

## Honest caveats

1. **33 of 36 months** had IC entry — 3 months still missing because OTM 4% + 300pt wing strikes never traded for those expiries.
2. **3 down months** in 33: 2023-06 (-18%), 2023-10 (-12%), 2023-12 (-43%). Tail risk real.
3. **Single-trade max loss = 48.2% of capital** (defined by wing width, can't go higher).
4. **Live realistic ≈ 70% of backtest** = ~+20-22%/mo, +180-220%/yr post-slippage.
5. **High win-rate trap**: 77% WR but ONE losing month can erase 3+ winners. Position sizing is binary (5 lots all in / out).
6. **Last 2 years FINNIFTY weekly killed**; monthly cycle untouched.
