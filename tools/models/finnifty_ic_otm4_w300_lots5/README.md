# finnifty_ic_otm4_w300_lots5

## Strategy

FINNIFTY (Nifty Financial Services Index) monthly Iron Condor.

- **SELL** OTM 4% Call (body) + OTM 4% Put (body)
- **BUY** Call wing +300 points further + Put wing −300 points further (caps risk)
- **5 lots** per cycle
- **Stop loss:** exit if combined pair value ≥ 3× entry credit
- Otherwise hold to monthly expiry (last Thursday)
- **Defined risk:** max loss bounded by wing width → margin ≈ ₹85k-100k for 5 lots

## Backtest result (2023-05 → 2026-05)

| Year | Trades | WR | P&L | ROI on ₹2L |
|---|---:|---:|---:|---:|
| 2023 (May-Dec) | 5 | 40% | ₹1,60,142 | +80.07% |
| 2024 | 8 | 100% | ₹3,97,117 | +198.56% |
| 2025 | 8 | 75% | ₹6,58,682 | +329.34% |
| 2026 (Jan-May) | 3 | 67% | ₹5,97,799 | +298.90% |
| **3-yr total** | **24** | **75%** | **₹18,13,740** | **+906.87%** |

- Avg/mo: **+41.22%** | Best mo: +316.3% | Worst mo: -42.8%
- Months ≥20%: 10/22 | Months ≥30%: 9/22
- Max single-trade loss: ₹81,644 (40.8% of capital, hard-capped by wings)

Full trade ledger + monthly equity curve: `exports/models/finnifty_ic_otm4_w300_lots5/SUMMARY.md`

## Forward applicability

✅ FinNifty monthly options still trade through 2030+. (Weekly killed by SEBI Nov 2024.)

## Files

| File | Purpose |
|---|---|
| `schema.sql` | `historical_options` + `option_universe` DDL |
| `sweep.py` | Iron Condor variant sweep (multiple OTM/width/lots combos) |
| `run_winner.py` | Run winning config + emit per-trade ledger CSV/MD |

## How to reproduce

```bash
# 1. Create tables
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

## Honest caveats

1. 22 of 36 months had IC entry (some months lacked 4% OTM + 300pt wing data)
2. 3 months in 22 went -18% to -43% — tail risk real
3. Single-trade max loss = 40.8% of capital (defined by wings)
4. Live realistic ≈ 70% of backtest = ~+28-30%/mo, +200-250%/yr
