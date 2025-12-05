# 11697 HW6: RAG Pipeline (Clean, Runnable)

Minimal repo with required code/data/outputs. Prompts/plots/docs were removed.

## Layout
```
README.md
topic.txt
data/
  ├── question.tsv
  ├── answer.tsv
  ├── evidence.tsv
  ├── corpus/        # supporting docs (text)
  ├── raw_corpus/    # source PDFs/html/json (reference only)
  ├── db*/           # vector DBs (gitignored)
  └── documents.json
output/
  ├── evaluation/    # six required eval TSVs
  ├── prediction/    # six required prediction TSVs
  └── */ablation/    # ablations isolated
src/
  ├── rag_pipeline.py
  ├── evaluate.py
  ├── llm/llm.py
  ├── rag/...
  └── data/ (scraper utilities kept)
requirements.txt
setup.sh
```

## Setup
```bash
pip install -r requirements.txt
# Set API key for CMU Gateway/OpenAI if needed:
export CMU_GATEWAY_API_KEY=sk-...
```

## Build embeddings (example)
```bash
python3 src/rag_pipeline.py \
  --mode embed \
  --retriever dense \
  --embedder azure \
  --db_path data/db_azure_1000_100 \
  --data_path data/documents.json \
  --chunk_size 1000 --chunk_overlap 100 --topk 20
```

## Run inference
- Closed-book (None):
```bash
python3 src/rag_pipeline.py \
  --mode infer \
  --retriever none \
  --generator qwen \
  --embedder azure \
  --db_path data/db_azure_1000_100 \
  --questions data/question.tsv \
  --output output/prediction/None_qwen.tsv
```

- RAG (dense, azure embeddings, qwen):
```bash
python3 src/rag_pipeline.py \
  --mode infer \
  --retriever dense \
  --generator qwen \
  --embedder azure \
  --db_path data/db_azure_1000_100 \
  --questions data/question.tsv \
  --output output/prediction/azure_qwen.tsv
```

- RAG + refined summary (k=3):
```bash
python3 src/rag_pipeline.py \
  --mode infer \
  --retriever dense \
  --generator gpt-5-mini \
  --embedder azure \
  --db_path data/db_azure_1000_100 \
  --questions data/question.tsv \
  --output output/prediction/azure_gpt-5-mini_refine3.tsv \
  --refine-summary True --refine-k 3
```

## Evaluate predictions
```bash
python3 src/evaluate.py \
  --prediction output/prediction/azure_gpt-5-mini.tsv \
  --reference data/answer.tsv \
  --output output/evaluation/azure_gpt-5-mini.tsv
```

## Notes
- Required TSVs kept at top-level output; ablations isolated under `output/*/ablation/`.
- Vector DBs under `data/db*` are gitignored.
- Raw PDFs/HTML live in `data/raw_corpus/` (not needed to run).
- CLIs tested with `--help` (no API calls). Ensure API keys before inference/eval.

