run:
	python script.py

prepare:
	rsync -a $(SRCDIR)/script.py ./
	ln -s $(SRCDIR)/states ./
