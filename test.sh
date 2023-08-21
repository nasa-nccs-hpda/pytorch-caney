set -xeuo pipefail

# Install deps in a virtual env.
readonly VENV_DIR=/tmp/pytorch-caney-env
# rm -rf "${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python --version

pip install --upgrade pip
pip install --upgrade pip setuptools wheel
pip install flake8 pytype pylint pylint-exit
pip install -r requirements/requirements-test.txt

flake8 `find pytorch_caney -name '*.py' | xargs` --count --show-source --statistics

# Run tests using unittest.
python -m unittest discover pytorch_caney/tests

set +u
deactivate
echo "All tests passed. Congrats!"