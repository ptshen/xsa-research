# Source attribution

`nanogpt_model.py` and `nanogpt_train.py` are unmodified reference files from Andrej Karpathy's nanoGPT commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`; see `nanogpt_LICENSE` (MIT). Our model implementation derives its baseline conventions from these files.

`paper.html`, `benchmark_block_results.png`, and `learning_rate_train_loss.svg` are cached from Shuangfei Zhai, *Exclusive Self Attention*, arXiv:2603.09078v1, https://arxiv.org/html/2603.09078v1, under CC BY 4.0 as stated on the paper. The `.svg.png` is a Quick Look rasterization for reading the figure labels. They are reference material, not experiment results.

`fineweb_metadata.json` is dataset repository metadata from https://huggingface.co/api/datasets/HuggingFaceFW/fineweb, retrieved 2026-11-23, used to resolve the immutable dataset revision. No training corpus is included.

`harness_model.py` is an unmodified API reference from EleutherAI/lm-evaluation-harness commit `ad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9`; see `harness_LICENSE` (MIT).
