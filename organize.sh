#!/bin/bash
set -e

echo "Creating new directory structure..."
mkdir -p 01_tokenizer/scripts 01_tokenizer/artifacts
mkdir -p 02_data_pipeline/scripts
mkdir -p 03_model
mkdir -p 04_training/checkpoints
mkdir -p data

echo "--- Stage 1: Tokenizer ---"
mv tokenizer/README.md 01_tokenizer/README.md
mv tokenizer/TOKENIZER_DEEPDIVE.md 01_tokenizer/TOKENIZER_DEEPDIVE.md
mv tokenizer/test_tokenizer.py 01_tokenizer/test_tokenizer.py

mv tokenizer/10k-training.py 01_tokenizer/scripts/train_shakespeare_10k.py
mv tokenizer/training_time.py 01_tokenizer/scripts/train_shakespeare_5k.py
mv tokenizer/compare.py 01_tokenizer/scripts/compare_vocabs.py
mv tokenizer/create_sampler.py 01_tokenizer/scripts/create_sampler.py
mv tokenizer/inspection_learning.py 01_tokenizer/scripts/inspect_sample.py

mv tokenizer/shakespeare_full_tok.merges.txt 01_tokenizer/artifacts/
mv tokenizer/shakespeare_full_tok.vocab.json 01_tokenizer/artifacts/
mv tokenizer/shakespeare_full_tok_10k.merges.txt 01_tokenizer/artifacts/
mv tokenizer/shakespeare_full_tok_10k.vocab.json 01_tokenizer/artifacts/
mv tokenizer/shakespeare_tok.merges.txt 01_tokenizer/artifacts/
mv tokenizer/shakespeare_tok.vocab.json 01_tokenizer/artifacts/

mv tokenizer/input.txt data/shakespeare_full.txt
mv tokenizer/input_sample.txt data/shakespeare_sample.txt

echo "--- Stage 2: Data pipeline ---"
mv data-pipeline/1mb-chunk.py 02_data_pipeline/scripts/sample_1mb.py
mv data-pipeline/new-train-tokenizer.py 02_data_pipeline/scripts/train_tinystories_tokenizer.py
mv data-pipeline/tinystories_tok_compact.py 02_data_pipeline/scripts/compact_tokenizer.py
mv data-pipeline/token_encode.py 02_data_pipeline/scripts/tokenize_corpus.py

mv data-pipeline/tinystories_tok_32k.merges.txt 01_tokenizer/artifacts/
mv data-pipeline/tinystories_tok_32k.vocab.json 01_tokenizer/artifacts/
mv data-pipeline/tinystories_tok_compact.merges.txt 01_tokenizer/artifacts/
mv data-pipeline/tinystories_tok_compact.vocab.json 01_tokenizer/artifacts/
mv data-pipeline/tinystories_tokenizer_sample.txt data/tinystories_sample.txt

echo "--- Stage 3: Model ---"
mv data-pipeline/rmsnorm.py src/cs336_workspace/rmsnorm.py
mv data-pipeline/rope.py src/cs336_workspace/rope.py
mv data-pipeline/attention.py src/cs336_workspace/attention.py
mv data-pipeline/swiglu.py src/cs336_workspace/swiglu.py
mv data-pipeline/block.py src/cs336_workspace/block.py
mv data-pipeline/model.py src/cs336_workspace/model.py
mv data-pipeline/batching.py src/cs336_workspace/batching.py

mv data-pipeline/test_rmsnorm.py 03_model/test_rmsnorm.py
mv data-pipeline/test_rope.py 03_model/test_rope.py
mv data-pipeline/test_attention.py 03_model/test_attention.py
mv data-pipeline/test_swiglu.py 03_model/test_swiglu.py
mv data-pipeline/test_block.py 03_model/test_block.py
mv data-pipeline/test_model.py 03_model/test_model.py

echo "--- Stage 4: Training ---"
mv data-pipeline/train.py 04_training/train.py
mv data-pipeline/checkpoint_best.pt 04_training/checkpoints/
mv data-pipeline/checkpoint_step1000.pt 04_training/checkpoints/
mv data-pipeline/checkpoint_step2000.pt 04_training/checkpoints/
mv data-pipeline/checkpoint_step3000.pt 04_training/checkpoints/
mv data-pipeline/checkpoint_step4000.pt 04_training/checkpoints/
mv data-pipeline/checkpoint_step5000.pt 04_training/checkpoints/
mv data-pipeline/checkpoint_step6000.pt 04_training/checkpoints/

echo "--- Large raw data -> data/ ---"
mv data-pipeline/TinyStories-train.txt data/TinyStories-train.txt
mv data-pipeline/train.bin data/train.bin
mv data-pipeline/val.bin data/val.bin

echo "--- Cleanup ---"
rm -f data-pipeline/README.md data-pipeline/TRANSFORMER_DEEPDIVE.md data-pipeline/data-pipeline.md
rmdir data-pipeline 2>/dev/null || echo "NOTE: data-pipeline/ not empty, check remaining files manually"
rmdir tokenizer 2>/dev/null || echo "NOTE: tokenizer/ not empty, check remaining files manually"

echo "Staging everything with git..."
git add -A

echo "Done. Run 'git status' to review before committing."