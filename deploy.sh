#!/bin/bash
# deploy.sh -- scp changed files to the Reachy Mini and sync the venv
#
# Compares SHA256 hashes to skip unchanged files.
# Prompts for .env values if .env is missing or --env is passed.
#
# Usage:
#   ./deploy.sh
#   ./deploy.sh --env          # re-enter .env values
#   ROBOT=pollen@reachy-mini.local ./deploy.sh

set -euo pipefail

ROBOT="${ROBOT:-pollen@reachy-mini.local}"
ROBOT_DIR="${ROBOT_DIR:-reachy-control}"
FORCE_ENV=false

for arg in "$@"; do
	case $arg in
	--env) FORCE_ENV=true ;;
	esac
done

# Load SSH key into agent so we only enter the passphrase once.
if ! ssh-add -l 2>/dev/null | grep -q "id_ed25519"; then
	ssh-add ~/.ssh/id_ed25519 2>/dev/null || true
fi

# --- .env generation ---
generate_env() {
	echo "=== .env setup ==="
	echo "  Current values shown in brackets. Press Enter to keep, or type a new value."
	echo ""

	# Read current .env values into plain variables (bash 3 compatible).
	V_ELEVENLABS_API_KEY=""
	V_OPENROUTER_API_KEY=""
	V_EXA_API_KEY=""
	V_REACHY_HOST=""
	V_REACHY_PORT=""
	if [ -f .env ]; then
		while IFS='=' read -r key value; do
			[ -z "$key" ] && continue
			case "$key" in \#*) continue ;; esac
			case "$key" in
			ELEVENLABS_API_KEY) V_ELEVENLABS_API_KEY="$value" ;;
			OPENROUTER_API_KEY) V_OPENROUTER_API_KEY="$value" ;;
			EXA_API_KEY) V_EXA_API_KEY="$value" ;;
			REACHY_HOST) V_REACHY_HOST="$value" ;;
			REACHY_PORT) V_REACHY_PORT="$value" ;;
			esac
		done <.env
	fi

	read -p "  ELEVENLABS_API_KEY [${V_ELEVENLABS_API_KEY:-(required)}]: " INPUT
	ELEVENLABS_API_KEY="${INPUT:-$V_ELEVENLABS_API_KEY}"
	if [ -z "$ELEVENLABS_API_KEY" ]; then
		echo "  ERROR: ELEVENLABS_API_KEY is required"
		exit 1
	fi

	read -p "  OPENROUTER_API_KEY [${V_OPENROUTER_API_KEY:-(required)}]: " INPUT
	OPENROUTER_API_KEY="${INPUT:-$V_OPENROUTER_API_KEY}"
	if [ -z "$OPENROUTER_API_KEY" ]; then
		echo "  ERROR: OPENROUTER_API_KEY is required"
		exit 1
	fi

	read -p "  EXA_API_KEY [${V_EXA_API_KEY:-(optional)}]: " INPUT
	EXA_API_KEY="${INPUT:-$V_EXA_API_KEY}"

	read -p "  REACHY_HOST [${V_REACHY_HOST:-localhost}]: " INPUT
	REACHY_HOST="${INPUT:-${V_REACHY_HOST:-localhost}}"

	read -p "  REACHY_PORT [${V_REACHY_PORT:-8000}]: " INPUT
	REACHY_PORT="${INPUT:-${V_REACHY_PORT:-8000}}"

	echo ""
	echo "=== .env summary ==="
	echo "  ELEVENLABS_API_KEY = $ELEVENLABS_API_KEY"
	echo "  OPENROUTER_API_KEY = $OPENROUTER_API_KEY"
	echo "  EXA_API_KEY        = ${EXA_API_KEY:-(not set)}"
	echo "  REACHY_HOST        = $REACHY_HOST"
	echo "  REACHY_PORT        = $REACHY_PORT"
	echo ""

	# Heredoc must not be indented - EOF at column 0.
	cat >.env <<EOF
ELEVENLABS_API_KEY=$ELEVENLABS_API_KEY   # STT (Scribe realtime)
OPENROUTER_API_KEY=$OPENROUTER_API_KEY   # LLM + TTS
EXA_API_KEY=$EXA_API_KEY                 # web search context (optional)
REACHY_HOST=$REACHY_HOST
REACHY_PORT=$REACHY_PORT
EOF
	echo "  written to .env"
	echo ""
}

if [ "$FORCE_ENV" = true ] || [ ! -f .env ]; then
	generate_env
fi

# Check if .env has placeholder or missing required values.
NEEDS_ENV=false
if [ -f .env ]; then
	# Check for placeholder values that need real input.
	if grep -q "your-elevenlabs-api-key\|your-openrouter-api-key" .env 2>/dev/null; then
		NEEDS_ENV=true
	fi
	# Check for missing required vars.
	for var in ELEVENLABS_API_KEY OPENROUTER_API_KEY REACHY_HOST REACHY_PORT; do
		if ! grep -q "^${var}=" .env 2>/dev/null; then
			NEEDS_ENV=true
		fi
	done
fi

if [ "$NEEDS_ENV" = true ]; then
	echo "=== .env has missing or placeholder values ==="
	generate_env
fi

# Python source + config files to sync.
FILES="main.py reachy_audio.py motion.py stt.py llm.py tts.py context_tools.py pinchtab_tools.py agent_tools.py pyproject.toml .env"

echo "=== checking robot prerequisites ==="
# Ensure the daemon's Python 3.12 exists (uv-managed, can get deleted).
ssh "$ROBOT" "/venvs/mini_daemon/bin/python --version 2>/dev/null || /opt/uv/uv python install 3.12"
# Max out the speaker volume.
ssh "$ROBOT" "amixer -c 0 cset numid=5 60,60 >/dev/null 2>&1 || true"
echo "done"

echo ""
echo "=== checking which files changed ==="

# Single SSH call: get all remote hashes at once.
REMOTE_HASHES=$(ssh "$ROBOT" "mkdir -p $ROBOT_DIR && cd $ROBOT_DIR 2>/dev/null && for f in $FILES; do if [ -f \"\$f\" ]; then echo \$(sha256sum \"\$f\"); else echo \"MISSING \$f\"; fi; done" 2>/dev/null || echo "")

CHANGED=""
for f in $FILES; do
	if [ "$f" = ".env" ] && [ ! -f "$f" ]; then
		echo "  SKIP $f (not found locally)"
		continue
	fi
	if [ ! -f "$f" ]; then
		echo "  SKIP $f (not found locally)"
		continue
	fi
	local_hash=$(shasum -a 256 "$f" | awk '{print $1}')
	remote_line=$(echo "$REMOTE_HASHES" | grep " $f$" || true)
	if [ -z "$remote_line" ]; then
		if echo "$REMOTE_HASHES" | grep -q "MISSING $f"; then
			echo "  CHG  $f (not on robot)"
			CHANGED="$CHANGED $f"
		else
			echo "  CHG  $f (no remote info)"
			CHANGED="$CHANGED $f"
		fi
	else
		remote_hash=$(echo "$remote_line" | awk '{print $1}')
		if [ "$local_hash" = "$remote_hash" ]; then
			echo "  OK   $f (unchanged)"
		else
			echo "  CHG  $f"
			CHANGED="$CHANGED $f"
		fi
	fi
done

echo ""
CHANGED=$(echo $CHANGED | xargs)
if [ -z "$CHANGED" ]; then
	echo "=== nothing to copy -- all files match ==="
else
	# Stop the running agent so we can overwrite files.
	ssh "$ROBOT" "pkill -f 'python.*main.py' 2>/dev/null; sleep 1" || true
	echo "=== scp changed file(s) to $ROBOT ==="
	scp $CHANGED "$ROBOT:$ROBOT_DIR/"
	echo "done"
fi

echo ""
echo "=== syncing Python venv on robot ==="
ssh "$ROBOT" "cd $ROBOT_DIR && /opt/uv/uv sync"
echo "venv ready"

echo ""
echo "=== deploy complete ==="
echo ""
echo "To run on the robot:"
echo "  ssh $ROBOT"
echo "  cd $ROBOT_DIR && /opt/uv/uv run reachy-agent"
echo ""
echo "Or with CLI overrides:"
echo "  cd $ROBOT_DIR && /opt/uv/uv run reachy-agent --no-motors   # test without motors"
echo "  cd $ROBOT_DIR && /opt/uv/uv run reachy-agent --no-audio   # text-only test"
