DLQMC_VERSION = 0.1.0
KFAC_VERSION = 0.1.0
DEEPQMC_VERSION = 0.1.0

LOCAL_VENV = .venv

-include local.mk

PYTHON ?= python3

SHELL = /bin/bash  # need brace expansion

TODAY := $(shell date +"%Y-%W/%Y-%m-%d")
RSYNC_CMD = rsync \
			-cirl --relative --delete --rsync-path="cd $(REMOTE_PATH) && rsync" $(RSYNC_OPTS) \
			--exclude={venv/,OUTPUT,__pycache__/}

.PHONY: bundle

all:

go:
	ssh -t $(REMOTE) 'cd $(REMOTE_PATH) && exec $$SHELL'

update: bundle
	$(RSYNC_CMD) --ignore-missing-args Makefile bundle data/extern states notebooks/preamble*.py \
		$(UPDATE_EXTRA) $(REMOTE):./
	@ssh $(REMOTE) 'make -C $(REMOTE_PATH) deploy'

bundle:
	rm -rf bundle && mkdir bundle
	poetry export -f requirements.txt --without-hashes | grep -v deepqmc >bundle/requirements.txt
	cd extern/deepqmc && poetry build -f wheel
	cp extern/deepqmc/dist/deepqmc-$(DEEPQMC_VERSION)-py3-none-any.whl bundle/
	cd extern/torch-kfac && poetry build -f wheel
	cp extern/torch-kfac/dist/torch_kfac-$(KFAC_VERSION)-py3-none-any.whl bundle/
	poetry build -f wheel
	cp dist/dlqmc-$(DLQMC_VERSION)-py3-none-any.whl bundle/
	rsync -aR --exclude=__pycache__ extern/./deepqmc/tests bundle/tests/
	$(PYTHON) -m pytest bundle/tests

deploy:
	$(PYTHON) -m pip install -q -r bundle/requirements.txt
	$(PYTHON) -m pip install --force-reinstall --no-deps bundle/*.whl

local_venv:
	mkdir -p $(RUN)
	rsync -a bundle $(RUN)/
	$(PYTHON) -m venv $(RUN)/$(LOCAL_VENV) --system-site-packages
	$(MAKE) -C $(RUN) -f $(CURDIR)/Makefile deploy_local PYTHON=$(LOCAL_VENV)/bin/python

deploy_local:
	$(PYTHON) -m pip install --ignore-installed --no-deps bundle/*.whl

fetch:
	$(RSYNC_CMD) -K $(REMOTE):$(RUNS) ./

push:
	$(RSYNC_CMD) -K $(RUNS) $(REMOTE):./

today:
	mkdir -p runs/$(TODAY) runs/Current
	ln -fns $(TODAY) runs/Today
	ln -s ../$(TODAY) runs/Current/

