# Standalone host build of the text tests (no gufo build). Run `make -j3 test`.
CXX      ?= /usr/bin/c++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra
ICU_LIBS ?= -licuuc
S1       ?= /server/staging/ripple-gufo/s1-text
STAGE    ?= /server/staging/ripple-gufo
VOCAB    ?= $(STAGE)/fixtures/vocab-only.gguf
TOKGOLD  ?= $(STAGE)/oracle/tok-goldens.jsonl
CHATGOLD ?= $(STAGE)/oracle/chat-goldens.jsonl
PY       ?= $(STAGE)/venv/bin/python
HFPY     ?= $(S1)/hfvenv/bin/python
LLAMA    ?= /server/programming/llama.cpp-ripple
BUILD    ?= build
SRC       = ../../src/text
TOOLS     = ../../tools/text

all: $(BUILD)/test_tokenizer $(BUILD)/test_chat

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/test_tokenizer: test_tokenizer.cpp gguf_vocab.hpp $(SRC)/tokenizer.cpp $(SRC)/tokenizer.hpp $(SRC)/json_value.cpp $(SRC)/json_value.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) test_tokenizer.cpp $(SRC)/tokenizer.cpp $(SRC)/json_value.cpp $(ICU_LIBS) -o $@

$(BUILD)/test_chat: test_chat.cpp gguf_vocab.hpp $(SRC)/chat_template.cpp $(SRC)/chat_template.hpp $(SRC)/json_value.cpp $(SRC)/json_value.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) test_chat.cpp $(SRC)/chat_template.cpp $(SRC)/json_value.cpp -o $@

# ---- oracle data (seeded, regenerated into $(S1)) ----
$(S1)/prop.hex: $(TOOLS)/gen_tok_prop.py
	$(PY) $< $@ 8000
$(S1)/llama_tok_batch: $(TOOLS)/llama_tok_batch.cpp
	$(CXX) -O2 -std=c++17 -I$(LLAMA)/include -I$(LLAMA)/ggml/include $< -o $@ -L$(LLAMA)/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$(LLAMA)/build/bin
$(S1)/prop-llama.txt: $(S1)/llama_tok_batch $(S1)/prop.hex
	$< $(VOCAB) $(S1)/prop.hex $@
$(S1)/prop-hf.txt: $(S1)/prop.hex $(TOOLS)/hf_tok_batch.py
	$(HFPY) $(TOOLS)/hf_tok_batch.py $(STAGE)/ref/hf/tokenizer.json $< $@

tok-data: $(S1)/prop-llama.txt $(S1)/prop-hf.txt $(S1)/prop-hf-nospecial.txt

test-tok: $(BUILD)/test_tokenizer tok-data
	$(BUILD)/test_tokenizer goldens $(VOCAB) $(TOKGOLD)
	$(BUILD)/test_tokenizer prop $(VOCAB) $(S1)/prop.hex $(S1)/prop-llama.txt $(S1)/prop-hf.txt $(S1)/prop-hf-nospecial.txt
	$(BUILD)/test_tokenizer stream $(VOCAB) $(S1)/prop.hex
	$(BUILD)/test_tokenizer bench $(VOCAB) $(S1)/prop.hex

# ---- chat template ----
$(S1)/chat-prop.jsonl: $(TOOLS)/gen_chat_prop.py
	$(PY) $< $(STAGE)/ref/hf/chat_template.jinja $@ 1000

test-chat: $(BUILD)/test_chat $(S1)/chat-prop.jsonl
	$(BUILD)/test_chat cases $(CHATGOLD)
	$(BUILD)/test_chat cases $(S1)/chat-prop.jsonl
	$(BUILD)/test_chat template $(VOCAB) $(STAGE)/ref/hf/chat_template.jinja
	sha256sum $(STAGE)/ref/hf/chat_template.jinja

# ---- exhaustive code-point checks against Hugging Face (slow: ~1.5 min, writes ~1 GB under $(S1)) ----
$(S1)/cp8.hex: $(TOOLS)/gen_cp_probes.py
	$(PY) $< $@ 8
$(S1)/cp8-hfpieces.txt: $(S1)/cp8.hex $(TOOLS)/hf_pieces_batch.py
	$(HFPY) $(TOOLS)/hf_pieces_batch.py $(STAGE)/ref/hf/tokenizer.json $< $@
$(S1)/hf-nospecial.json: $(TOOLS)/make_hf_nospecial.py
	$(PY) $< $(STAGE)/ref/hf/tokenizer.json $@
$(S1)/prop-hf-nospecial.txt: $(S1)/prop.hex $(S1)/hf-nospecial.json
	$(HFPY) $(TOOLS)/hf_tok_batch.py $(S1)/hf-nospecial.json $< $@

test-pieces: $(BUILD)/test_tokenizer $(S1)/cp8-hfpieces.txt
	$(BUILD)/test_tokenizer pieces $(S1)/cp8.hex $(S1)/cp8-hfpieces.txt

negative: $(BUILD)/test_tokenizer $(BUILD)/test_chat tok-data $(S1)/chat-prop.jsonl $(S1)/cp8-hfpieces.txt
	CXX="$(CXX)" CXXFLAGS="$(CXXFLAGS)" ICU_LIBS="$(ICU_LIBS)" VOCAB="$(VOCAB)" TOKGOLD="$(TOKGOLD)" CHATGOLD="$(CHATGOLD)" S1="$(S1)" BUILD="$(BUILD)" ./negative.sh

test: test-tok test-pieces test-chat negative
	@echo ALL TESTS PASSED

clean:
	rm -rf $(BUILD)
.PHONY: all tok-data test-tok test-chat test-pieces negative test clean
