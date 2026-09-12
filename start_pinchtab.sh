#!/bin/bash
# start_pinchtab.sh -- start pinchtab for LAN access from the Reachy robot
#
# Run this on the pinchtab PC (not the robot). Binds to 0.0.0.0 so the
# robot can reach it over the LAN, and starts a headed Chrome instance.
#
# The token is hardcoded below - change it if you want a different one.
# The bind address is set once via config (persists across restarts).
#
# Usage:
#   ./start_pinchtab.sh
#   TOKEN=mys3cret ./start_pinchtab.sh        # override token
#   ALLOWED_DOMAINS="*" ./start_pinchtab.sh   # restrict domains (default: all)

set -euo pipefail

# Hardcoded token - change this to whatever you want.
# The robot's .env must have the same value in PINCHTAB_TOKEN.
TOKEN="${TOKEN:-reachy-mini-formfill-2026}"
DISPLAY="${DISPLAY:-:0}"
export DISPLAY

echo "=== configuring pinchtab for LAN access ==="
# Bind is config-only (no env var), so set it once. Idempotent.
pinchtab config set server.bind 0.0.0.0

# Allowed domains: default to all. Restrict with e.g. ALLOWED_DOMAINS="example.com,reachy-mini.local"
ALLOWED_DOMAINS="${ALLOWED_DOMAINS:-*}"
pinchtab config set security.allowedDomains "$ALLOWED_DOMAINS"

echo "  bind: 0.0.0.0"
echo "  token: $TOKEN"
echo "  display: $DISPLAY"
echo "  allowed_domains: $ALLOWED_DOMAINS"
echo ""

# Token via env var - overrides server.token at runtime, no config set needed.
export PINCHTAB_TOKEN="$TOKEN"

echo "=== starting pinchtab server ==="
# Lower rate limit since we're exposed on the LAN.
PINCHTAB_RATE_LIMIT_MAX=300 pinchtab server &
SERVER_PID=$!

# Server config spawns a headed default instance automatically; give it a moment.
sleep 3

echo ""
echo "=== checking pinchtab instances ==="
pinchtab instance list
echo ""

echo "=== pinchtab is running ==="
echo "  server PID: $SERVER_PID"
echo "  API: http://$(hostname -I 2>/dev/null | awk '{print $1}' || echo 'this-pc'):9867"
echo "  token: $TOKEN"
echo "  default instance mode: headed (set in pinchtab config via instanceDefaults.mode)"
echo ""
echo "  Put these in the robot's .env:"
echo "    PINCHTAB_URL=http://<this-pc-ip>:9867"
echo "    PINCHTAB_TOKEN=$TOKEN"
echo ""
echo "  Press Ctrl+C to stop."
wait $SERVER_PID
