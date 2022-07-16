# Logs_to_user

This programme is for grouping system log messages by user session.
Usage: `go run logs_to_user.go /path/to/in_dir > logs_with_user.log`.
The directors `/path/to/in_dir` must contain files messages-\*.gz (which are logs from /var/log/messages) and files sacct-\*.gz (which are Slurm logs).