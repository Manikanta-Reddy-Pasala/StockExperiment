"""
Admin Routes for Manual Triggers
Provides UI controls to manually trigger automated processes.
"""

from flask import Blueprint, render_template, jsonify, request
from datetime import datetime
import subprocess
import threading
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

# Track running tasks
running_tasks = {}


def run_command_async(task_id, command, description):
    """Run a command asynchronously and track its status."""
    try:
        running_tasks[task_id] = {
            'status': 'running',
            'description': description,
            'start_time': datetime.now().isoformat(),
            'output': '',
            'error': ''
        }

        # Get the project root directory (where run_pipeline.py lives)
        project_root = '/app' if os.path.exists('/app/run_pipeline.py') else os.getcwd()

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            cwd=project_root
        )
        
        running_tasks[task_id]['status'] = 'completed' if result.returncode == 0 else 'failed'
        running_tasks[task_id]['output'] = result.stdout
        running_tasks[task_id]['error'] = result.stderr
        running_tasks[task_id]['return_code'] = result.returncode
        running_tasks[task_id]['end_time'] = datetime.now().isoformat()
        
    except subprocess.TimeoutExpired:
        running_tasks[task_id]['status'] = 'timeout'
        running_tasks[task_id]['error'] = 'Task timeout after 1 hour'
        running_tasks[task_id]['end_time'] = datetime.now().isoformat()
    except Exception as e:
        running_tasks[task_id]['status'] = 'error'
        running_tasks[task_id]['error'] = str(e)
        running_tasks[task_id]['end_time'] = datetime.now().isoformat()


@admin_bp.route('/')
def admin_dashboard():
    """Admin dashboard with manual trigger controls."""
    return render_template('admin/dashboard.html')


def run_function_async(task_id, func, description, *args, **kwargs):
    """Run a Python function asynchronously and track its status."""
    try:
        running_tasks[task_id] = {
            'status': 'running',
            'description': description,
            'start_time': datetime.now().isoformat(),
            'output': '',
            'error': ''
        }

        # Run the function
        result = func(*args, **kwargs)

        running_tasks[task_id]['status'] = 'completed'
        running_tasks[task_id]['output'] = str(result) if result else 'Completed successfully'
        running_tasks[task_id]['end_time'] = datetime.now().isoformat()

    except Exception as e:
        running_tasks[task_id]['status'] = 'failed'
        running_tasks[task_id]['error'] = str(e)
        running_tasks[task_id]['end_time'] = datetime.now().isoformat()
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)


@admin_bp.route('/trigger/pipeline', methods=['POST'])
def trigger_pipeline():
    """Trigger complete data pipeline."""
    task_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def run_pipeline():
        from src.services.data.pipeline_saga import PipelineSaga

        saga = PipelineSaga()
        saga.execute()

    thread = threading.Thread(
        target=run_function_async,
        args=(task_id, run_pipeline, 'Data Pipeline (6-step saga)')
    )
    thread.start()

    return jsonify({
        'success': True,
        'task_id': task_id,
        'message': 'Data pipeline started'
    })


@admin_bp.route('/trigger/ml-training', methods=['POST'])
def trigger_ml_training():
    """Trigger ML model training."""
    task_id = f"ml_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def run_ml_training():
        from src.services.ml.stock_predictor import StockMLPredictor
        from src.models.database import get_database_manager

        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            predictor = StockMLPredictor(session)
            predictor.train(lookback_days=365)
            return "ML models trained successfully"

    thread = threading.Thread(
        target=run_function_async,
        args=(task_id, run_ml_training, 'ML Model Training')
    )
    thread.start()

    return jsonify({
        'success': True,
        'task_id': task_id,
        'message': 'ML training started'
    })


@admin_bp.route('/trigger/all', methods=['POST'])
def trigger_all():
    """Trigger all processes in sequence."""
    base_task_id = f"all_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def run_all_tasks():
        """Run all tasks sequentially."""
        from src.services.data.pipeline_saga import PipelineSaga
        from src.services.ml.stock_predictor import StockMLPredictor
        from src.models.database import get_database_manager

        overall_task_id = f"{base_task_id}_all"
        running_tasks[overall_task_id] = {
            'status': 'running',
            'description': 'Complete Data + ML Pipeline',
            'start_time': datetime.now().isoformat(),
            'steps': [],
            'output': '',
            'error': ''
        }

        db_manager = get_database_manager()

        # Step 1: Data Pipeline
        try:
            running_tasks[overall_task_id]['steps'].append({
                'name': 'pipeline',
                'description': 'Data Pipeline',
                'status': 'running',
                'start_time': datetime.now().isoformat()
            })

            saga = PipelineSaga()
            saga.execute()

            running_tasks[overall_task_id]['steps'][-1]['status'] = 'completed'
            running_tasks[overall_task_id]['steps'][-1]['end_time'] = datetime.now().isoformat()

        except Exception as e:
            running_tasks[overall_task_id]['steps'][-1]['status'] = 'failed'
            running_tasks[overall_task_id]['steps'][-1]['error'] = str(e)
            running_tasks[overall_task_id]['steps'][-1]['end_time'] = datetime.now().isoformat()
            running_tasks[overall_task_id]['error'] += f"\nPipeline failed: {str(e)}"
            logger.error(f"Pipeline failed: {e}", exc_info=True)

        # Step 2: ML Training
        try:
            running_tasks[overall_task_id]['steps'].append({
                'name': 'ml_training',
                'description': 'ML Training',
                'status': 'running',
                'start_time': datetime.now().isoformat()
            })

            with db_manager.get_session() as session:
                predictor = StockMLPredictor(session)
                predictor.train(lookback_days=365)

            running_tasks[overall_task_id]['steps'][-1]['status'] = 'completed'
            running_tasks[overall_task_id]['steps'][-1]['end_time'] = datetime.now().isoformat()

        except Exception as e:
            running_tasks[overall_task_id]['steps'][-1]['status'] = 'failed'
            running_tasks[overall_task_id]['steps'][-1]['error'] = str(e)
            running_tasks[overall_task_id]['steps'][-1]['end_time'] = datetime.now().isoformat()
            running_tasks[overall_task_id]['error'] += f"\nML Training failed: {str(e)}"
            logger.error(f"ML Training failed: {e}", exc_info=True)

        # Mark overall task as completed
        failed_count = len([s for s in running_tasks[overall_task_id]['steps'] if s['status'] == 'failed'])
        running_tasks[overall_task_id]['status'] = 'completed' if failed_count == 0 else 'failed'
        running_tasks[overall_task_id]['end_time'] = datetime.now().isoformat()

    thread = threading.Thread(target=run_all_tasks)
    thread.start()

    return jsonify({
        'success': True,
        'task_id': f"{base_task_id}_all",
        'message': 'All tasks started sequentially'
    })


@admin_bp.route('/task/<task_id>/status', methods=['GET'])
def get_task_status(task_id):
    """Get status of a running task."""
    if task_id not in running_tasks:
        return jsonify({
            'success': False,
            'error': 'Task not found'
        }), 404
    
    return jsonify({
        'success': True,
        'task': running_tasks[task_id]
    })


@admin_bp.route('/tasks', methods=['GET'])
def list_tasks():
    """List all tasks."""
    return jsonify({
        'success': True,
        'tasks': running_tasks
    })


@admin_bp.route('/task/<task_id>/retry-failed', methods=['POST'])
def retry_failed_steps(task_id):
    """Retry only the failed steps of a task."""
    if task_id not in running_tasks:
        return jsonify({
            'success': False,
            'error': 'Task not found'
        }), 404

    task = running_tasks[task_id]

    # Get failed steps
    failed_steps = [step for step in task.get('steps', []) if step.get('status') == 'failed']

    if not failed_steps:
        return jsonify({
            'success': False,
            'error': 'No failed steps to retry'
        }), 400

    # Create a new task for retrying failed steps
    new_task_id = f"retry_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def retry_steps():
        """Retry failed steps only."""
        running_tasks[new_task_id] = {
            'status': 'running',
            'description': 'Retry Failed Steps',
            'start_time': datetime.now().isoformat(),
            'steps': [],
            'output': '',
            'error': ''
        }

        for step in failed_steps:
            step_name = step['name']
            command = None
            description = step['description']

            # Map step name to command
            if step_name == 'pipeline':
                command = ['python3', 'run_pipeline.py']
            elif step_name == 'fill_data':
                command = ['python3', 'fill_data_sql.py']
            elif step_name == 'business_logic':
                command = ['python3', 'fix_business_logic.py']
            elif step_name == 'ml_training':
                command = ['python3', 'train_ml_model.py']
            elif step_name == 'csv_export':
                command = ['python3', '-c', 'from data_scheduler import export_daily_csv; export_daily_csv()']

            if not command:
                continue

            try:
                running_tasks[new_task_id]['steps'].append({
                    'name': step_name,
                    'description': description,
                    'status': 'running',
                    'start_time': datetime.now().isoformat()
                })

                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=3600,
                    cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                )

                step_status = 'completed' if result.returncode == 0 else 'failed'
                running_tasks[new_task_id]['steps'][-1]['status'] = step_status
                running_tasks[new_task_id]['steps'][-1]['end_time'] = datetime.now().isoformat()
                running_tasks[new_task_id]['steps'][-1]['return_code'] = result.returncode

                if result.returncode != 0:
                    running_tasks[new_task_id]['error'] += f"\n{step_name} failed: {result.stderr}"

            except Exception as e:
                running_tasks[new_task_id]['steps'][-1]['status'] = 'error'
                running_tasks[new_task_id]['steps'][-1]['error'] = str(e)
                running_tasks[new_task_id]['error'] += f"\n{step_name} error: {str(e)}"

        running_tasks[new_task_id]['status'] = 'completed'
        running_tasks[new_task_id]['end_time'] = datetime.now().isoformat()

    thread = threading.Thread(target=retry_steps)
    thread.start()

    return jsonify({
        'success': True,
        'task_id': new_task_id,
        'message': f'Retrying {len(failed_steps)} failed steps',
        'failed_steps': [s['name'] for s in failed_steps]
    })


@admin_bp.route('/system/status', methods=['GET'])
def system_status():
    """Get system status."""
    try:
        from src.models.database import get_database_manager
        from sqlalchemy import text

        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            # Get database stats
            stocks_stats = session.execute(text("""
                SELECT
                    COUNT(*) as total_stocks,
                    COUNT(current_price) as with_price,
                    COUNT(market_cap) as with_market_cap,
                    MAX(last_updated) as last_updated
                FROM stocks
            """)).fetchone()

            history_stats = session.execute(text("""
                SELECT
                    COUNT(DISTINCT symbol) as symbols,
                    COUNT(*) as records,
                    MAX(date) as latest_date
                FROM historical_data
            """)).fetchone()

            tech_stats = session.execute(text("""
                SELECT
                    COUNT(DISTINCT symbol) as symbols,
                    MAX(date) as latest_date
                FROM technical_indicators
            """)).fetchone()

            snapshot_stats = session.execute(text("""
                SELECT
                    COUNT(*) as total_snapshots,
                    COUNT(DISTINCT date) as unique_dates,
                    MAX(date) as latest_date
                FROM daily_suggested_stocks
            """)).fetchone()

            return jsonify({
                'success': True,
                'status': {
                    'stocks': {
                        'total': stocks_stats.total_stocks,
                        'with_price': stocks_stats.with_price,
                        'with_market_cap': stocks_stats.with_market_cap,
                        'last_updated': stocks_stats.last_updated.isoformat() if stocks_stats.last_updated else None
                    },
                    'historical_data': {
                        'symbols': history_stats.symbols,
                        'records': history_stats.records,
                        'latest_date': history_stats.latest_date.isoformat() if history_stats.latest_date else None
                    },
                    'technical_indicators': {
                        'symbols': tech_stats.symbols,
                        'latest_date': tech_stats.latest_date.isoformat() if tech_stats.latest_date else None
                    },
                    'daily_snapshots': {
                        'total': snapshot_stats.total_snapshots,
                        'unique_dates': snapshot_stats.unique_dates,
                        'latest_date': snapshot_stats.latest_date.isoformat() if snapshot_stats.latest_date else None
                    }
                }
            })

    except Exception as e:
        logger.error(f"System status error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
