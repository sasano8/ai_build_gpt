# 一発実行:  make            (= make chat)
#   依存関係(DAG):
#     .venv/.synced ─┐
#     data/input.txt ┼─> out/gpt-$(PRESET).pt ─> chat / generate
#     gpt/*.py ──────┘
#   足りないものだけ自動で実行される(環境構築 → データ取得 → 学習 → 対話)。
#   2回目以降は、学習済みなら学習をスキップして即チャットが始まる。
#
# 例: make PRESET=tiny   数十秒で試す / make PRESET=full  GPU向けフル学習
PRESET ?= small
UV      = uv run
DATA    = data/input.txt
SYNCED  = .venv/.synced
CKPT    = out/gpt-$(PRESET).pt
BIGRAM  = out/bigram-bigram.pt
# 学習結果に影響するソース(変更されたら再学習。chat.py などは含めない)
TRAIN_SRC = gpt/config.py gpt/data.py gpt/model.py gpt/bigram.py gpt/train.py

.DEFAULT_GOAL := chat
.PHONY: help setup data test bigram train retrain generate chat estimate clean

help:
	@echo "make              全自動: セットアップ → データ取得 → 学習 → 対話 (= make chat)"
	@echo "make PRESET=tiny  プリセット指定 (tiny: 数十秒 / small: CPUで約2分 / full: GPU向け)"
	@echo "make train        学習のみ(学習済みならスキップ)"
	@echo "make retrain      学習済みでも強制的に再学習"
	@echo "make generate     学習済みモデルで文章を生成"
	@echo "make bigram       Bigramベースラインを学習・生成"
	@echo "make estimate    LLM学習の費用見積もり (PARAMS=7e9 [TOKENS=1e12])"
	@echo "make test         単体テスト"
	@echo "make clean        学習結果・キャッシュを削除"

# --- 部品(ファイルターゲット) ---------------------------------
$(SYNCED): pyproject.toml uv.lock
	uv sync
	@touch $@

$(DATA):
	mkdir -p data
	curl -sSfL -o $@ https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

$(CKPT): $(SYNCED) $(DATA) $(TRAIN_SRC)
	$(UV) python -m gpt.train --model gpt --preset $(PRESET)

$(BIGRAM): $(SYNCED) $(DATA) $(TRAIN_SRC)
	$(UV) python -m gpt.train --model bigram --preset bigram

# --- ユーザー向けターゲット ------------------------------------
setup: $(SYNCED)
data: $(DATA)
train: $(CKPT)
bigram: $(BIGRAM)

test: $(SYNCED)
	$(UV) python -m pytest -q tests

retrain:
	rm -f $(CKPT)
	$(MAKE) train

generate: $(CKPT)
	$(UV) python -m gpt.train --model gpt --preset $(PRESET) --generate-only

estimate:
	python3 scripts/estimate_cost.py --params $(PARAMS) $(if $(TOKENS),--tokens $(TOKENS))

chat: $(CKPT)
	$(UV) python -m gpt.chat --preset $(PRESET)

clean:
	rm -rf out .pytest_cache gpt/__pycache__ tests/__pycache__
