from celery import Celery
from config import settings
from kombu import Queue

celery_app = Celery(
    "ingestion",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["ingestion.tasks"]
)

# Define your named queues
celery_app.conf.task_queues = [
    Queue('ingestion', routing_key='ingestion'),
]

celery_app.conf.update(
    task_track_started=True,
    result_expires=3600
)