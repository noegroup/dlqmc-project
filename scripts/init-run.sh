echo "python = $(which python)"
[[ -d venv ]] || { python -m venv venv --system-site-packages && venv/bin/pip install deepqmc.tar.gz[train]; }
source venv/bin/activate
echo "python = $(which python)"
