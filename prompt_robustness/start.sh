#!/bin/zsh

# ─────────────────────────────────────────────
#  Prompt Robustness Evaluation Framework
#  Start Script
# ─────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🧪 Prompt Robustness Evaluation Platform"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check venv exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Virtual environment not found at: $VENV_PYTHON"
    echo "   Run: python3 -m venv venv && venv/bin/pip install -r requirements.txt"
    exit 1
fi

echo "✅ Using Python: $($VENV_PYTHON --version)"
echo ""

# Parse mode argument
MODE=${1:-benchmark}

case "$MODE" in
    benchmark)
        echo "🚀 Running Benchmark Pipeline..."
        echo ""
        shift
        "$VENV_PYTHON" "$SCRIPT_DIR/main.py" "$@"
        ;;
    dashboard)
        echo "📊 Launching Streamlit Dashboard..."
        echo ""
        "$SCRIPT_DIR/venv/bin/streamlit" run "$SCRIPT_DIR/dashboard/app.py"
        ;;
    fetch)
        echo "📥 Fetching Data..."
        echo ""
        "$VENV_PYTHON" "$SCRIPT_DIR/fetch_data.py"
        ;;
    *)
        echo "Usage: ./start.sh [mode] [options]"
        echo ""
        echo "Modes:"
        echo "  benchmark   Run the evaluation benchmark (default)"
        echo "  dashboard   Launch the Streamlit dashboard"
        echo "  fetch       Fetch/update dataset"
        echo ""
        echo "Benchmark options (passed through):"
        echo "  --models <model1> <model2>   Specify models to benchmark"
        echo "  --no-cache                   Disable caching"
        echo "  --no-parallel                Disable parallel processing"
        echo "  --dynamic-weighting          Enable dynamic score weighting"
        echo "  --no-advanced                Disable advanced metrics"
        echo "  --enable-bertscore           Enable BERTScore"
        echo "  --no-rouge                   Disable ROUGE computation"
        echo "  --no-correlation             Disable correlation analysis"
        echo ""
        echo "Examples:"
        echo "  ./start.sh"
        echo "  ./start.sh benchmark --no-cache --no-parallel"
        echo "  ./start.sh dashboard"
        echo "  ./start.sh fetch"
        exit 1
        ;;
esac
