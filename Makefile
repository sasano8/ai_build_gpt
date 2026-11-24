# 使い方: `make help`
PYTHON  ?= uv run python
PRESET  ?= small

.PHONY: help setup data test bigram train train-tiny train-full generate chat clean

help:
	@echo "make setup       uv sync で依存関係インストール(pyproject.toml)"
	@echo "make data        tiny shakespeare を data/input.txt に取得"
	@echo "make test        単体テスト"
	@echo "make bigram      Phase3: Bigramモデルを学習・生成"
	@echo "make train-tiny  GPTを数十秒で動作確認 (preset=tiny)"
	@echo "make train       GPTを学習・生成 (PRESET=small 既定。tiny/small/full)"
	@echo "make generate    out/gpt.pt を使って生成のみ"
	@echo "make chat        out/gpt.pt と対話(続きを生成)するCLI"

setup:
	uv sync

data: data/input.txt
data/input.txt:
	mkdir -p data
	curl -sSfL -o $@ https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

test:
	$(PYTHON) -m pytest -q tests

bigram: data
	$(PYTHON) -m gpt.train --model bigram --preset bigram

train-tiny: data
	$(PYTHON) -m gpt.train --model gpt --preset tiny

train: data
	$(PYTHON) -m gpt.train --model gpt --preset $(PRESET)

train-full:
	$(MAKE) train PRESET=full

generate:
	$(PYTHON) -m gpt.train --model gpt --preset $(PRESET) --generate-only

chat:
	$(PYTHON) -m gpt.chat --preset $(PRESET)

clean:
	rm -rf out __pycache__ gpt/__pycache__ .pytest_cache
