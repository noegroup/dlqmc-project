-include local.mk
PYTHON ?= python3

DLQMC_VERSION = 0.1.0
DEEPQMC_VERSION = 0.1.0

SHELL = /bin/bash  # need brace expansion

RSYNC_CMD = rsync \
			-cirl --delete --rsync-path="cd $(REMOTE_PATH) && rsync" $(RSYNC_OPTS) \
			--exclude={venv/,OUTPUT,__pycache__/}

.PHONY: bundle

all:

go:
	ssh -t $(REMOTE) 'cd $(REMOTE_PATH) && exec $$SHELL'

update: bundle
	$(RSYNC_CMD) Makefile bundle extern/deepqmc/tests $(REMOTE):./
	@ssh $(REMOTE) 'make -C $(REMOTE_PATH) deploy'

bundle:
	rm -rf bundle && mkdir bundle
	poetry export -f requirements.txt | grep -v deepqmc >bundle/requirements.txt
	cd extern/deepqmc && poetry build -f wheel
	cp extern/deepqmc/dist/deepqmc-$(DEEPQMC_VERSION)-py3-none-any.whl bundle/
	poetry build -f wheel
	cp dist/dlqmc-$(DLQMC_VERSION)-py3-none-any.whl bundle/

fetch:
	$(RSYNC_CMD) -K --relative $(REMOTE):$(RUNS) ./

push:
	$(RSYNC_CMD) -K --relative $(RUNS) $(REMOTE):./

deploy:
	$(PYTHON) -m pip install -q -r bundle/requirements.txt
	$(PYTHON) -m pip install --force-reinstall --no-deps bundle/*.whl
