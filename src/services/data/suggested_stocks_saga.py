"""
Suggested Stocks Saga (EMA 200/400 1H crossover)

Thin orchestrator that delegates to ``EMACrossoverRunner``. Kept under the
existing module path so callers (web routes, schedulers, providers) do not
need to be rewired.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy import text

try:
    from ...models.database import get_database_manager
    from ..technical.ema_crossover_runner import get_ema_crossover_runner
except ImportError:
    from src.models.database import get_database_manager
    from src.services.technical.ema_crossover_runner import get_ema_crossover_runner

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Lightweight saga primitives (kept for backwards compatibility)
# ----------------------------------------------------------------------
class SagaStepStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class SagaStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SagaStep:
    step_id: str
    name: str
    description: str = ""
    status: SagaStepStatus = SagaStepStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    input_count: int = 0
    output_count: int = 0
    filtered_count: int = 0
    rejected_count: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    additional_info: Dict[str, Any] = field(default_factory=dict)
    results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SuggestedStocksSaga:
    saga_id: str
    user_id: int
    strategies: List[str] = field(default_factory=lambda: ["ema_200_400"])
    limit: int = 50
    search_query: Optional[str] = None
    sort_by: Optional[str] = None
    sort_order: str = "desc"
    sector: Optional[str] = None
    status: SagaStatus = SagaStatus.PENDING
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_duration_seconds: Optional[float] = None
    steps: List[SagaStep] = field(default_factory=list)
    final_results: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    model_type: str = "crossover"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "saga_id": self.saga_id,
            "user_id": self.user_id,
            "strategies": self.strategies,
            "limit": self.limit,
            "search_query": self.search_query,
            "sort_by": self.sort_by,
            "sort_order": self.sort_order,
            "sector": self.sector,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_seconds": self.total_duration_seconds,
            "steps": [
                {
                    "step_id": s.step_id,
                    "name": s.name,
                    "description": s.description,
                    "status": s.status.value,
                    "input_count": s.input_count,
                    "output_count": s.output_count,
                    "metadata": s.metadata,
                }
                for s in self.steps
            ],
            "final_results": self.final_results,
            "summary": self.summary,
            "errors": self.errors,
        }


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------
class SuggestedStocksSagaOrchestrator:
    """Drives the EMA 200/400 strategy and returns today's picks."""

    def __init__(self):
        self.runner = get_ema_crossover_runner()
        self.db = get_database_manager()

    def execute_suggested_stocks_saga(
        self,
        user_id: int,
        strategies: Optional[List[str]] = None,
        limit: int = 50,
        search: Optional[str] = None,
        sort_by: Optional[str] = None,
        sort_order: str = "desc",
        sector: Optional[str] = None,
        model_type: str = "crossover",
    ) -> Dict[str, Any]:
        saga_id = f"ema200x400_{user_id}_{int(datetime.now().timestamp())}"
        saga = SuggestedStocksSaga(
            saga_id=saga_id,
            user_id=user_id,
            strategies=strategies or ["ema_200_400"],
            limit=limit,
            search_query=search,
            sort_by=sort_by,
            sort_order=sort_order,
            sector=sector,
            model_type=model_type,
        )
        saga.status = SagaStatus.RUNNING

        try:
            # Step 1: refresh + run strategy
            run_step = SagaStep(
                step_id="step1_run_strategy",
                name="Run EMA 200/400 strategy",
                status=SagaStepStatus.IN_PROGRESS,
                start_time=datetime.now(),
            )
            saga.steps.append(run_step)

            run_result = self.runner.run_for_user(user_id)
            run_step.status = SagaStepStatus.COMPLETED
            run_step.end_time = datetime.now()
            run_step.metadata = {
                "symbols_processed": run_result.get("symbols_processed", 0),
                "signals_emitted": run_result.get("signals_emitted", 0),
            }

            # Step 2: load today's picks
            load_step = SagaStep(
                step_id="step2_load_picks",
                name="Load picks from daily_suggested_stocks",
                status=SagaStepStatus.IN_PROGRESS,
                start_time=datetime.now(),
            )
            saga.steps.append(load_step)

            picks = self._load_picks(limit=limit, search=search, sector=sector)
            saga.final_results = picks
            load_step.status = SagaStepStatus.COMPLETED
            load_step.end_time = datetime.now()
            load_step.output_count = len(picks)

            saga.status = SagaStatus.COMPLETED
            saga.end_time = datetime.now()
            saga.total_duration_seconds = (
                saga.end_time - saga.start_time
            ).total_seconds()
            saga.summary = {
                "total_picks": len(picks),
                "strategy": "ema_200_400",
                "timeframe": "1H",
            }
            return saga.to_dict()
        except Exception as e:
            logger.error(f"EMA crossover saga failed: {e}", exc_info=True)
            saga.status = SagaStatus.FAILED
            saga.errors.append(str(e))
            saga.end_time = datetime.now()
            return saga.to_dict()

    def _load_picks(
        self,
        limit: int = 50,
        search: Optional[str] = None,
        sector: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        with self.db.get_session() as session:
            base_sql = """
                SELECT date, symbol, stock_name, current_price, target_price,
                       stop_loss, recommendation, selection_score,
                       reason, sector
                FROM daily_suggested_stocks
                WHERE strategy = 'ema_200_400'
                  AND date = (SELECT MAX(date) FROM daily_suggested_stocks
                              WHERE strategy = 'ema_200_400')
            """
            params: Dict[str, Any] = {}
            if search:
                base_sql += " AND (LOWER(symbol) LIKE :q OR LOWER(stock_name) LIKE :q)"
                params["q"] = f"%{search.lower()}%"
            if sector:
                base_sql += " AND LOWER(sector) = :sector"
                params["sector"] = sector.lower()
            base_sql += " ORDER BY selection_score DESC NULLS LAST, created_at DESC"
            base_sql += " LIMIT :limit"
            params["limit"] = limit
            rows = session.execute(text(base_sql), params).fetchall()
            return [dict(r._mapping) for r in rows]


_saga_orchestrator: Optional[SuggestedStocksSagaOrchestrator] = None


def get_suggested_stocks_saga_orchestrator() -> SuggestedStocksSagaOrchestrator:
    global _saga_orchestrator
    if _saga_orchestrator is None:
        _saga_orchestrator = SuggestedStocksSagaOrchestrator()
    return _saga_orchestrator
