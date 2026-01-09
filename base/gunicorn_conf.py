import os
import multiprocessing

bind = f"0.0.0.0:{os.environ.get('PORT', '8080')}"

worker_class = "gthread"

default_workers = (multiprocessing.cpu_count() * 2) + 1
default_threads = 25

workers = int(os.environ.get("GUNICORN_WORKERS", str(default_workers)))
threads = int(os.environ.get("GUNICORN_THREADS", str(default_threads)))

timeout = int(os.environ.get("GUNICORN_TIMEOUT", "120"))

accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info").lower()

print("--- Gunicorn Config (Synchronous Multi-Threaded) ---")
print(f"  Binding to: {bind}")
print(f"  Worker Class: {worker_class}")
print(f"  Workers: {workers}")
print(f"  Threads per worker: {threads}")
print(f"  Timeout: {timeout}s")
print(f"  Log Level: {loglevel}")
print("----------------------------------------------------")
