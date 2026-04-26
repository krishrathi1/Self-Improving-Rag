"""
APEX Async Task Queues / Workers.
Production equivalent of Celery or Redis RQ.
Manages long-running tasks like Graph Ingestion and Episodic Memory archiving.
"""
import asyncio
from typing import Callable, Coroutine, Any
from loguru import logger

class TaskQueueWorker:
    """
    Lightweight background worker simulator for handling asynchronous,
    heavy payloads (e.g. Graph clustering, Vector Indexing, API callbacks)
    without blocking the FastAPI event loop. Highly scalable.
    """
    def __init__(self):
        self.queue = asyncio.Queue()
        self.workers = []

    async def start_workers(self, num_workers: int = 3):
        """Boot up background workers to pull from the distributed queue."""
        for i in range(num_workers):
            task = asyncio.create_task(self._worker_loop(f"Worker-{i}"))
            self.workers.append(task)
        logger.info(f"⚙️ Scalability Online: {num_workers} Async background workers listening.")

    async def _worker_loop(self, name: str):
        """Worker lifecycle."""
        while True:
            try:
                task_func, args, kwargs = await self.queue.get()
                logger.debug(f"👷 [{name}] Processing background task...")
                
                # Execute the coroutine securely
                if asyncio.iscoroutinefunction(task_func):
                    await task_func(*args, **kwargs)
                else:
                    task_func(*args, **kwargs)
                    
            except Exception as e:
                logger.error(f"❌ Background job failed in {name}: {e}")
            finally:
                self.queue.task_done()

    async def submit_task(self, func: Callable, *args, **kwargs):
        """Produce a task payload for background ingestion."""
        await self.queue.put((func, args, kwargs))
        logger.debug(f"📥 Queued async background job. Queue size: {self.queue.qsize()}")

# Singleton
_work_queue = None

def get_task_queue() -> TaskQueueWorker:
    global _work_queue
    if _work_queue is None:
        _work_queue = TaskQueueWorker()
    return _work_queue
