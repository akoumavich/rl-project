set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"
pip install gymnasium
pip install matplotlib
pip install tqdm
pip install pandas
pip install scipy
pip install optuna
python main.py --env Pendulum-v1 --alg TD3