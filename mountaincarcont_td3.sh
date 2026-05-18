set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"
pip install gymnasium
pip install matplotlib
pip install tqdm
pip install pandas
python main.py --env MountainCarContinuous-v0 --alg TD3