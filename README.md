# Multimodal Multi-Task ML (Speech + Text + Images)

This project trains one model to solve three tasks at once:
- Emotion detection
- Spoken command classification
- Product classification

Inputs per sample:
- Speech (`audio/*.wav`)
- Words (`text/*.txt`)
- Picture (`image/*`)

## Dataset Format
For each split (`train`, `val`, `test`) and each sample id (example: `sample_0001`):

- `audio/sample_0001.wav`
- `text/sample_0001.txt`
- `image/sample_0001.jpg` (or png/jpeg/bmp)
- `labels_emotion/sample_0001.txt`
- `labels_command/sample_0001.txt`
- `labels_product/sample_0001.txt`

Each label file contains one class index integer.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m ensurepip --upgrade
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Optional Local LLM Setup
```bash
python3 -m pip install -r requirements-llm.txt
```

## Generate Dummy Data
```bash
python3 -m src.multimodal_ml.make_dummy_data \
  --data_dir data/sample \
  --emotion_classes 5 \
  --command_classes 5 \
  --product_classes 5
```

## Train
```bash
python3 -m src.multimodal_ml.train \
  --data_dir data/sample \
  --emotion_classes 5 \
  --command_classes 5 \
  --product_classes 5 \
  --epochs 10
```

## Predict (One Sample)
```bash
python3 -m src.multimodal_ml.predict \
  --checkpoint checkpoints/best_model.pt \
  --audio_path data/sample/test/audio/test_0000.wav \
  --text_path data/sample/test/text/test_0000.txt \
  --image_path data/sample/test/image/test_0000.jpg \
  --creator_profile creator_profile.json
```

Output example:
- `emotion_pred=... emotion_confidence=...`
- `command_pred=... command_confidence=...`
- `product_pred=... product_confidence=...`

## Creator Profile Format
- File: `creator_profile.json`
- Used during prediction to personalize model output headers.

## Math + Physics + Calculus Assistant
Use these commands:

```bash
python3 -m src.multimodal_ml.ask --task derivative --expr "sin(x)*x**2" --var x
python3 -m src.multimodal_ml.ask --task integral --expr "x**2 + 3*x" --var x
python3 -m src.multimodal_ml.ask --task limit --expr "sin(x)/x" --var x --point 0
python3 -m src.multimodal_ml.ask --task solve --equation "x**2 - 5*x + 6 = 0" --var x
python3 -m src.multimodal_ml.ask --task physics_kinematics --u 0 --a 9.8 --t 3
python3 -m src.multimodal_ml.ask --task physics_force --mass 10 --acceleration 9.8
```

Local LLM Q/A (math/physics explanation):
```bash
python3 -m src.multimodal_ml.ask \
  --task qa \
  --question "Solve d/dx (x^3 + 2x) and explain." \
  --llm_model "Qwen/Qwen2.5-1.5B-Instruct" \
  --creator_profile creator_profile.json
```

## Single Chat Command (Auto Route)
```bash
python3 -m src.multimodal_ml.chat --prompt "differentiate sin(x)*x^2"
python3 -m src.multimodal_ml.chat --prompt "integral of x^2 + 3x"
python3 -m src.multimodal_ml.chat --prompt "limit sin(x)/x as x->0"
python3 -m src.multimodal_ml.chat --prompt "force with mass=10 acceleration=9.8"
python3 -m src.multimodal_ml.chat --prompt "kinematics u=0 a=9.8 t=3"
python3 -m src.multimodal_ml.chat --prompt "Explain Gauss law in simple words" --llm_model "Qwen/Qwen2.5-1.5B-Instruct"
```

## Build Your Own Personal LLM (LoRA Fine-Tuning)
1. Install LLM dependencies:
```bash
python3 -m pip install -r requirements-llm.txt
```

2. Prepare data files:
- `data/llm/train.jsonl`
- `data/llm/val.jsonl`

Format (one JSON per line):
```json
{"instruction":"Differentiate x^2.","response":"Derivative is 2x."}
```

3. Fine-tune:
```bash
python3 -m src.multimodal_ml.llm.finetune_lora \
  --base_model "Qwen/Qwen2.5-1.5B-Instruct" \
  --train_file data/llm/train.jsonl \
  --val_file data/llm/val.jsonl \
  --output_dir checkpoints/personal_llm_lora \
  --epochs 3
```

4. Run your personalized model:
```bash
python3 -m src.multimodal_ml.llm.run_personal_llm \
  --adapter_dir checkpoints/personal_llm_lora \
  --question "Solve d/dx (x^3 + 2x) with steps" \
  --creator_profile creator_profile.json
```
The assistant is prompted to return:
- `Reasoning: ...`
- `Final: ...`

Interactive mode:
```bash
python3 -m src.multimodal_ml.llm.run_personal_llm \
  --adapter_dir checkpoints/personal_llm_lora_v2 \
  --creator_profile creator_profile.json \
  --interactive
```

Any-topic mode is default (`--domain_mode general`), including coding/reasoning/general Q&A.
For strict science-only behavior, use:
```bash
python3 -m src.multimodal_ml.llm.run_personal_llm \
  --adapter_dir checkpoints/personal_llm_lora_v2 \
  --creator_profile creator_profile.json \
  --interactive \
  --domain_mode specialized
```

To show reasoning steps in output:
```bash
python3 -m src.multimodal_ml.llm.run_personal_llm \
  --adapter_dir checkpoints/personal_llm_lora_v2 \
  --creator_profile creator_profile.json \
  --interactive \
  --show_reasoning
```

5. Evaluate your personalized model:
```bash
python3 -m src.multimodal_ml.llm.evaluate_personal_llm \
  --adapter_dir checkpoints/personal_llm_lora_v2 \
  --test_file data/llm/val.jsonl \
  --max_samples 200 \
  --output_report checkpoints/personal_llm_eval.json
```

## Continue From Best Adapter (v2 + Corrections)
1. Export mistakes from best adapter:
```bash
python3 -m src.multimodal_ml.llm.evaluate_personal_llm \
  --adapter_dir checkpoints/personal_llm_lora_v2 \
  --test_file data/llm/val.jsonl \
  --max_samples 200 \
  --mistakes_jsonl data/llm/corrections_v2.jsonl \
  --output_report checkpoints/personal_llm_eval_v2.json
```

2. Build correction-focused train file:
```bash
python3 -m src.multimodal_ml.llm.prepare_correction_dataset \
  --corrections_file data/llm/corrections_v2.jsonl \
  --base_file data/llm/train.jsonl \
  --output_file data/llm/train_correction_v2.jsonl \
  --repeat_corrections 6 \
  --base_sample 600
```

3. Continue training from adapter v2:
```bash
python3 -m src.multimodal_ml.llm.finetune_lora \
  --base_model "Qwen/Qwen2.5-1.5B-Instruct" \
  --init_adapter_dir checkpoints/personal_llm_lora_v2 \
  --train_file data/llm/train_correction_v2.jsonl \
  --val_file data/llm/val.jsonl \
  --output_dir checkpoints/personal_llm_lora_v2_fix1 \
  --epochs 2 \
  --learning_rate 5e-5 \
  --warmup_ratio 0.08 \
  --early_stopping_patience 1
```

## Hard-Case Boost (for better exact accuracy)
Generate hard examples:
```bash
python3 -m src.multimodal_ml.llm.generate_hard_dataset \
  --train_file data/llm/train_hard.jsonl \
  --val_file data/llm/val_hard.jsonl \
  --train_size 1200 \
  --val_size 200
```

Merge + dedupe with existing data:
```bash
python3 -m src.multimodal_ml.llm.merge_jsonl \
  --inputs data/llm/train.jsonl data/llm/train_hard.jsonl \
  --output data/llm/train_v3.jsonl

python3 -m src.multimodal_ml.llm.merge_jsonl \
  --inputs data/llm/val.jsonl data/llm/val_hard.jsonl \
  --output data/llm/val_v3.jsonl
```

Retrain on merged set:
```bash
python3 -m src.multimodal_ml.llm.finetune_lora \
  --base_model "Qwen/Qwen2.5-1.5B-Instruct" \
  --train_file data/llm/train_v3.jsonl \
  --val_file data/llm/val_v3.jsonl \
  --output_dir checkpoints/personal_llm_lora_v3 \
  --epochs 4
```

Recommended stable retrain settings:
```bash
python3 -m src.multimodal_ml.llm.finetune_lora \
  --base_model "Qwen/Qwen2.5-1.5B-Instruct" \
  --train_file data/llm/train.jsonl \
  --val_file data/llm/val.jsonl \
  --output_dir checkpoints/personal_llm_lora_v5 \
  --epochs 2 \
  --learning_rate 8e-5 \
  --warmup_ratio 0.08 \
  --max_grad_norm 0.3 \
  --early_stopping_patience 1
```

Higher-capacity LoRA (more adapter power, more memory use):
```bash
python3 -m src.multimodal_ml.llm.finetune_lora \
  --base_model "Qwen/Qwen2.5-1.5B-Instruct" \
  --train_file data/llm/train.jsonl \
  --val_file data/llm/val.jsonl \
  --output_dir checkpoints/personal_llm_lora_highcap \
  --epochs 2 \
  --learning_rate 5e-5 \
  --lora_r 16 \
  --lora_alpha 32 \
  --target_modules "q_proj,k_proj,v_proj,o_proj"
```

## Add Physics + Rocket + Space Concepts
Generate concept Q&A data:
```bash
python3 -m src.multimodal_ml.llm.generate_concept_dataset \
  --train_file data/llm/train_concepts.jsonl \
  --val_file data/llm/val_concepts.jsonl \
  --train_size 1200 \
  --val_size 200
```

Merge with existing datasets:
```bash
python3 -m src.multimodal_ml.llm.merge_jsonl \
  --inputs data/llm/train.jsonl data/llm/train_concepts.jsonl \
  --output data/llm/train_with_concepts.jsonl

python3 -m src.multimodal_ml.llm.merge_jsonl \
  --inputs data/llm/val.jsonl data/llm/val_concepts.jsonl \
  --output data/llm/val_with_concepts.jsonl
```

Fine-tune from best adapter (recommended):
```bash
python3 -m src.multimodal_ml.llm.finetune_lora \
  --base_model "Qwen/Qwen2.5-1.5B-Instruct" \
  --init_adapter_dir checkpoints/personal_llm_lora_v2 \
  --train_file data/llm/train_with_concepts.jsonl \
  --val_file data/llm/val_with_concepts.jsonl \
  --output_dir checkpoints/personal_llm_lora_v2_concepts \
  --epochs 1 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.05 \
  --early_stopping_patience 1
```

## Add Business + Finance Concepts
Generate business/finance Q&A:
```bash
python3 -m src.multimodal_ml.llm.generate_business_finance_dataset \
  --train_file data/llm/train_business_finance.jsonl \
  --val_file data/llm/val_business_finance.jsonl \
  --train_size 1000 \
  --val_size 200
```

Merge with core datasets:
```bash
python3 -m src.multimodal_ml.llm.merge_jsonl \
  --inputs data/llm/train.jsonl data/llm/train_business_finance.jsonl \
  --output data/llm/train_with_bizfin.jsonl

python3 -m src.multimodal_ml.llm.merge_jsonl \
  --inputs data/llm/val.jsonl data/llm/val_business_finance.jsonl \
  --output data/llm/val_with_bizfin.jsonl
```

Fine-tune from best adapter for stronger general mode:
```bash
python3 -m src.multimodal_ml.llm.finetune_lora \
  --base_model "Qwen/Qwen2.5-1.5B-Instruct" \
  --init_adapter_dir checkpoints/personal_llm_lora_v2 \
  --train_file data/llm/train_with_bizfin.jsonl \
  --val_file data/llm/val_with_bizfin.jsonl \
  --output_dir checkpoints/personal_llm_lora_v2_bizfin \
  --epochs 1 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.05 \
  --early_stopping_patience 1
```

## Important Reality Check
- This code is a strong starter, not a full AGI system.
- 95%+ accuracy is possible only with large, clean, real data and model upgrades.
- "Own LLM" for math/physics usually requires major compute, data curation, and weeks to months of training.

## Next Upgrade Path
- Replace text encoder with multilingual transformer.
- Replace audio encoder with wav2vec2/Whisper encoder.
- Replace image encoder with CLIP/ViT.
- Add a separate math/physics LLM fine-tuning pipeline.
