# Stage 1 B2: RGB + Text

## Change map

Every B2-only implementation is marked with `Stage-1 B2` or `B2 change` in
comments/docstrings.

| File | B2 change |
|---|---|
| `basicsr/data/rgb_text_paired_image_dataset.py` | Strict LR/GT/TXT basename pairing and UTF-8 caption loading |
| `basicsr/archs/rgb_text_mambairv2_arch.py` | Frozen CLIP ViT-L/14 sentence encoder and shallow FiLM |
| `basicsr/models/rgb_text_mambairv2_model.py` | Text-aware train/test calls and one-time caption encoding for tiled inference |
| `options/train/mambairv2/train_S1_B2_RGBText_MambaIRv2_x4.yml` | Full 500k B2 training configuration |
| `options/test/mambairv2/test_S1_B2_RGBText_MambaIRv2_x4.yml` | Five-dataset B2 evaluation configuration |

No MambaIRv2 block, scan, reconstruction layer, loss, crop, augmentation,
optimizer, scheduler, batch size, or metric was changed.

## Why frozen CLIP ViT-L/14

- OpenAI CLIP is trained to align natural-language descriptions with visual
  concepts, matching the current content-caption modality.
- CLIP-SR demonstrates that CLIP features are applicable to text-guided SR, but
  its prompt predictor, iterative refinement, and richer fusion are intentionally
  excluded from this Stage-1 value test.
- SeeSR and Text-guided Explorable SR use large generative/diffusion priors.
  Those methods target perceptual or explorable generation and would confound
  the current PSNR-oriented controlled comparison.
- InstructIR encodes restoration instructions such as denoising/dehazing tasks;
  the current TXT files instead describe image content.

Selected implementation: load the existing local Hugging Face-format
`clip-vit-large-patch14`, freeze `CLIPTextModel`, take `pooler_output`, and use
one zero-initialized projection to produce global FiLM gamma/beta. The zero
initialization makes B2 exactly B0 at iteration zero.

Trainable parameter comparison:

```text
B0:                         23,050,713
B2 excluding frozen CLIP:   23,319,861
Added trainable FiLM:          269,148
```

The frozen CLIP weights are loaded from `clip_model_path` and deliberately
excluded from every BasicSR checkpoint. Strict loading still checks all
MambaIRv2 and FiLM keys. This avoids storing the same large frozen encoder at
every 5,000-iteration checkpoint, but means the local CLIP directory remains
mandatory for both training and testing.

Research references:

- OpenAI CLIP: https://github.com/openai/CLIP
- CLIP-SR: https://github.com/Bingwen-Hu/CLIP-SR
- SeeSR: https://github.com/cswry/SeeSR
- InstructIR: https://github.com/mv-lab/InstructIR
- Text-guided Explorable SR: https://github.com/KVGandikota/Text-guidedSR

## Required external files

The code does not download models during training. This directory must already
exist on the server:

```text
/home/BRAIN/xukai/files/clip-vit-large-patch14/
```

It must be a complete Hugging Face CLIP text-model directory containing at
least the model config, tokenizer files, and model weights, typically:

```text
config.json
tokenizer_config.json
vocab.json
merges.txt
special_tokens_map.json
pytorch_model.bin or model.safetensors
```

Every HR image also needs one non-empty English UTF-8 caption. Example:

```text
HR:   0001.png
LR:   0001x4.png
Text: 0001.txt
```

The supplied dataset paths are reused exactly from the earlier reference
configuration. If any caption is missing, empty, not UTF-8, or incorrectly
named, dataset construction/loading fails with the exact path.

Important limitation retained from the Stage-1 plan: one full-image caption is
used for every randomly cropped patch. This is recorded as a controlled B2
limitation rather than hidden behind a new local-caption generator.

## Commands

Full training:

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py \
  -opt options/train/mambairv2/train_S1_B2_RGBText_MambaIRv2_x4.yml \
  --launcher none
```

Resume:

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py \
  -opt options/train/mambairv2/train_S1_B2_RGBText_MambaIRv2_x4.yml \
  --launcher none \
  --auto_resume
```

Optional 25% screening run in a separate experiment directory:

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py \
  -opt options/train/mambairv2/train_S1_B2_RGBText_MambaIRv2_x4.yml \
  --launcher none \
  --force_yml \
    name=S1_B2_RGBText_MambaIRv2_x4_screen25 \
    train:total_iter=125000 \
    train:scheduler:milestones=[62500,100000,112500,118750]
```

Five-dataset test after the full checkpoint is available:

```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/test.py \
  -opt options/test/mambairv2/test_S1_B2_RGBText_MambaIRv2_x4.yml \
  --launcher none
```
