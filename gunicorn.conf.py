# gunicorn.conf.py
bind = '0.0.0.0:8000'  # Listen on all interfaces on port 8000
workers = 4  # Number of worker processes to handle requests
timeout = 120  # Timeout for worker processes in seconds
