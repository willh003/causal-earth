SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MODULE_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"

BASE_DIR="/home/wph52/earthnet2021/earthnet2021x/"
VAL_PERCENT=20

python "$MODULE_ROOT/causal_earth/data/create_val_split.py" \
  --base_dir "$BASE_DIR" \
  --val_percent "$VAL_PERCENT" \