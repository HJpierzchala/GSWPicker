#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Creates a Conda environment (Python 3.11), installs packages from
#   <script‑directory>/config/requirements.txt,
# and finally runs  make_tree.py  using that environment.
# -----------------------------------------------------------------------------

ENV_NAME="gsw_env"                               # change if you like

# ----- Paths -----------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="$SCRIPT_DIR/config/requirements.txt"
MAKE_TREE="$SCRIPT_DIR/make_tree.py"           # script to execute at the end

# Abort if requirements.txt not found
if [[ ! -f "$REQ_FILE" ]]; then
    echo "❌  File not found: $REQ_FILE"
    exit 1
fi

# -----------------------------------------------------------------------------
setup_env () {
    echo "Creating Conda environment \"$ENV_NAME\"…"
    conda create -n "$ENV_NAME" python=3.11 -y

    # Activate (works in Bash on macOS, Linux, Git‑Bash, WSL)
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"

    echo "Installing packages from $REQ_FILE…"
    pip install -r "$REQ_FILE"

    # ------------------------------------------------------
    # Run make_tree.py if it exists
    # ------------------------------------------------------
    if [[ -f "$MAKE_TREE" ]]; then
        echo "Running make_tree.py"
        python3.11 "$MAKE_TREE"
        # or simply:  python "$MAKE_TREE"   (same interpreter inside env)
    else
        echo "⚠️  make_tree.py not found at $MAKE_TREE – skipping."
    fi
}

# -----------------------------------------------------------------------------
# OS detection
# -----------------------------------------------------------------------------
case "$OSTYPE" in
  darwin*)                   echo "macOS detected."                        ; setup_env ;;
  msys*|cygwin*|win32*)      echo "Windows detected (Git‑Bash/Cygwin/WSL)." ; setup_env ;;
  linux*)                    echo "Linux detected."                        ; setup_env ;;
  *)                         echo "Unsupported OS: $OSTYPE"                ; exit 1 ;;
esac

echo "✅  Done! Environment \"$ENV_NAME\" is still active."
echo "   (Later: run  conda activate $ENV_NAME  to activate it again.)"
