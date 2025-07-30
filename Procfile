gunicorn app:app \
  -k gevent \        # use gevent workers for non-blocking I/O
  -w 8 \             # 8 vCPUs → run 8 workers
  --timeout 120 \    # restart a worker if a request runs >120 s
  --log-level info \ # keep Gunicorn logs at INFO level
  --bind 0.0.0.0:$PORT   # Railway injects $PORT at runtime
