#!/usr/bin/env bash
#
# Test hypothesis: changing innodb_log_file_size from 5GB -> 48MB causes slow MySQL restart
#
set -euo pipefail

MYSQL_BASE="/work/dpl-sfc/users/tatsu/tmp/DBTune/mysql_build"
MYSQLD="$MYSQL_BASE/bin/mysqld"
MYSQLADMIN="$MYSQL_BASE/bin/mysqladmin"
MYSQL_CLI="$MYSQL_BASE/bin/mysql"
CNF="$MYSQL_BASE/cnf/my.cnf"
CNF_CLEAN="$MYSQL_BASE/cnf/my.cnf.clean"
CLIENT_CNF="$MYSQL_BASE/cnf/mysql_client.cnf"
SOCK="$MYSQL_BASE/mysql.sock"
PIDFILE="$MYSQL_BASE/mysql.pid"
TIMEOUT=300

LOG_5GB=5368709120
LOG_48MB=50331648

wait_for_mysql() {
    local elapsed=0
    while ! "$MYSQLADMIN" --socket="$SOCK" -u root ping &>/dev/null; do
        sleep 1
        elapsed=$((elapsed + 1))
        if [ "$elapsed" -ge "$TIMEOUT" ]; then
            echo "ERROR: MySQL did not start within ${TIMEOUT}s"
            exit 1
        fi
    done
}

shutdown_mysql() {
    if "$MYSQLADMIN" --socket="$SOCK" -u root ping &>/dev/null; then
        "$MYSQLADMIN" --socket="$SOCK" -u root shutdown
        # Wait for process to exit
        while [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; do
            sleep 0.5
        done
    fi
}

write_cnf() {
    local log_file_size=$1
    cp "$CNF_CLEAN" "$CNF"
    # Append the log file size setting before the [client] section or at end of [mysqld]
    sed -i "/^\[mysqld\]/a innodb_log_file_size\t\t= $log_file_size" "$CNF"
}

start_mysql() {
    "$MYSQLD" --defaults-file="$CNF" &
    wait_for_mysql
}

echo "============================================"
echo " innodb_log_file_size restart slowdown test"
echo "============================================"
echo ""

# --- Step 1: Start MySQL with 5GB log file size ---
echo "Step 1: Starting MySQL with innodb_log_file_size = 5GB..."
shutdown_mysql
write_cnf $LOG_5GB
start_mysql

actual=$("$MYSQL_CLI" --defaults-file="$CLIENT_CNF" -u root -N -e "SELECT @@innodb_log_file_size")
echo "  Verified innodb_log_file_size = $actual"
if [ "$actual" != "$LOG_5GB" ]; then
    echo "  WARNING: expected $LOG_5GB, got $actual"
fi
echo ""

# --- Step 2: Run sysbench RW ---
echo "Step 2: Running sysbench oltp_read_write (5s warmup + 15s run)..."
sysbench oltp_read_write \
    --mysql-host=localhost \
    --mysql-socket="$SOCK" \
    --mysql-user=root \
    --mysql-password="" \
    --mysql-db=sbtest \
    --db-driver=mysql \
    --mysql-storage-engine=innodb \
    --range-size=100 \
    --events=0 \
    --rand-type=uniform \
    --tables=150 \
    --table-size=800000 \
    --db-ps-mode=disable \
    --warmup-time=5 \
    --threads=32 \
    --time=15 \
    run 2>&1 | tail -5
echo ""

# --- Step 3: Change to 48MB and measure restart ---
echo "Step 3: Changing innodb_log_file_size to 48MB and restarting..."
write_cnf $LOG_48MB

T_start=$(date +%s.%N)

echo "  Shutting down MySQL..."
"$MYSQLADMIN" --socket="$SOCK" -u root shutdown
while [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; do
    sleep 0.5
done

T_after_shutdown=$(date +%s.%N)

echo "  Starting MySQL with 48MB log file size..."
"$MYSQLD" --defaults-file="$CNF" &
wait_for_mysql

T_after_start=$(date +%s.%N)

actual=$("$MYSQL_CLI" --defaults-file="$CLIENT_CNF" -u root -N -e "SELECT @@innodb_log_file_size")
echo "  Verified innodb_log_file_size = $actual"
echo ""

# --- Results ---
echo "============================================"
echo " Results (5GB -> 48MB restart)"
echo "============================================"
python3 -c "
s, m, e = $T_start, $T_after_shutdown, $T_after_start
print(f'Shutdown time: {m-s:6.2f} seconds')
print(f'Startup time:  {e-m:6.2f} seconds')
print(f'Total time:    {e-s:6.2f} seconds')
"
echo "============================================"

# --- Step 4: Restore clean config ---
cp "$CNF_CLEAN" "$CNF"
echo ""
echo "Restored my.cnf to clean config."
echo "MySQL is still running with 48MB log file size."
