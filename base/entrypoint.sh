#!/bin/bash

set -e

cleanup() {
    echo "--- [ENTRYPOINT] Caught Signal! Shutting down Gunicorn and Subscriber... ---"
    kill -s SIGTERM -$(jobs -p)
    wait
    echo "--- [ENTRYPOINT] All processes stopped. ---"
}

trap cleanup SIGINT SIGTERM

# Note the change to 'app.main:app' to correctly locate the Flask app object
echo "--- [ENTRYPOINT] Starting Gunicorn server in background... ---"
# gunicorn --bind 0.0.0.0:8080 --workers 2 --threads 8 app.main:app &
gunicorn -c gunicorn_conf.py app.main:app &

# Note the change to 'python -m app.subscriber' to correctly run the subscriber module
echo "--- [ENTRYPOINT] Starting Pub/Sub subscriber in background... ---"
python -m app.subscriber &

echo "--- [ENTRYPOINT] Services started. Waiting for signals... ---"
wait