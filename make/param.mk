run:
	deepqmc train .

prepare:
	rsync -a $(SRCDIR)/param.toml ./
	rsync -a --ignore-missing-args $(SRCDIR)/hooks.py ./
