# Evaluating with LMM-R1

We provide a fork of [LMM-R1](https://github.com/TideDra/lmm-r1) to evaluate large multimodal models on reasoning-focused benchmarks. LMM-R1 is a framework designed to enhance reasoning capabilities in smaller (3B) multimodal models through rule-based reinforcement learning. 

>This is a fork from [lmms_eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), where we implement vllm for lmm inference. 

### Evaluation Process

The evaluation of models trained with LMM-R1 involves two main steps:

1. **Deploy the evaluation model** for answer extraction:

```bash
bash miscs/lmmr1/deploy_eval_model.sh
```

2. **Run evaluation scripts** for different benchmark types:

For text-based reasoning benchmarks:
```bash
bash miscs/lmmr1/eval_text_benchmark.sh
```

For multimodal reasoning benchmarks:
```bash
bash miscs/lmmr1/eval_mm_benchmark.sh
```

### Configuration

Before running the evaluation scripts, you need to modify the model path in the evaluation scripts:

```bash
# In the script files, change:
export MODEL_CKPT="/path/to/your/model"  # Change to your model path
```

The evaluation framework will automatically process the benchmarks and generate performance metrics for your model's reasoning capabilities across various tasks.




