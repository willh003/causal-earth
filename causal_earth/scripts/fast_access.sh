SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MODULE_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"

DATA_PATH="/home/wph52/greenearthnet/earthnet2021x/"
OUT_PATH="/home/wph52/greenearthnet/earthnet2021x/"
DATASET="train"

python "$SCRIPT_DIR/fast_access.py" \
  "$DATA_PATH" \
  "$OUT_PATH" \
  "$DATASET"