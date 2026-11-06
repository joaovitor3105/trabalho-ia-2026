.PHONY: help install data eda part1 part2 test all clean

PYTHON ?= python3

help:
	@echo "Comandos disponíveis:"
	@echo "  make install  - Instala as dependências do projeto (requirements.txt)"
	@echo "  make part1    - Executa o pipeline de Aprendizado Supervisionado e Ensembles (Parte 1)"
	@echo "  make part2    - Executa os experimentos de Computação Natural: AG, BPSO e CLONALG (Parte 2)"
	@echo "  make test     - Executa os testes automatizados (avaliador, restrições e algoritmos)"
	@echo "  make all      - Executa testes, Parte 1 e Parte 2 completos"
	@echo "  make clean    - Remove arquivos temporários e caches Python"

install:
	$(PYTHON) -m pip install -r requirements.txt

part1:
	$(PYTHON) -m src.supervised.train_models

part2:
	$(PYTHON) -m src.optimization.experiment

test:
	$(PYTHON) -m pytest tests/ -v

all: test part1 part2

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
