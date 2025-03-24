SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MODULE_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"

SAVE_DIR="/home/wph52/earthnet2021/"
N_SAMPLES=10000

python "$MODULE_ROOT/causal_earth/data/download/download.py" \
  --save_dir "$SAVE_DIR" \
  --n_samples "$N_SAMPLES" \