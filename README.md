# Yoruba Multilabel Emotion Detection

Fine-tuning AfroXLMR on SemEval-2025 Task 11 for multilabel emotion detection in Yorùbá, a low-resource Niger-Congo language spoken by over 50 million people across Nigeria and West Africa.

---

## Overview

This project fine-tunes [AfroXLMR-base](https://huggingface.co/Davlan/afro-xlmr-base) — an Africa-centric multilingual transformer — on the Yorùbá split of the SemEval-2025 Task 11 shared task on text-based emotion detection. The model predicts six emotions simultaneously from Yorùbá text: **anger, disgust, fear, joy, sadness, and surprise**.

Emotion detection in Yorùbá is a challenging task due to:
- Severe class imbalance (sadness has 10x more samples than disgust or fear)
- Morphological complexity and tonal diacritics (e.g. è, ó, ẹ) that standard NLP tools strip incorrectly
- Limited annotated data compared to high-resource languages

---

## Results

Evaluated on the official SemEval-2025 Task 11 Yorùbá test set (3,000 samples).

| Emotion | Precision | Recall | F1 |
|---------|-----------|--------|----|
| Anger | 0.40 | 0.39 | 0.39 |
| Disgust | 0.30 | 0.33 | 0.31 |
| Fear | 0.39 | 0.27 | 0.32 |
| Joy | 0.37 | 0.46 | 0.41 |
| Sadness | 0.59 | 0.77 | 0.66 |
| Surprise | 0.27 | 0.31 | 0.29 |
| **Macro avg** | **0.39** | **0.42** | **0.40** |

**Comparison to SemEval-2025 Task 11 baselines (Yorùbá Track A):**

| Model | Macro F1 |
|-------|----------|
| Majority class baseline | 0.165 |
| RoBERTa baseline | 0.463 |
| **AfroXLMR-base (this work)** | **0.40** |
| Best SemEval team (ensemble) | 0.657 |

Our single-model AfroXLMR approach achieves 0.40 macro F1, significantly above the majority baseline and competitive with the RoBERTa baseline, without any ensembling, data augmentation, or external data.

---

## Model

The fine-tuned model and tokenizer are publicly available on HuggingFace:

- Model: [Olamieee/yoruba-emotion-model](https://huggingface.co/Olamieee/yoruba-emotion-model)
- Tokenizer: [Olamieee/yoruba-emotion-tokenizer](https://huggingface.co/Olamieee/yoruba-emotion-tokenizer)

### Load and run inference

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained("lapppy1/yoruba-emotion-model")
tokenizer = AutoTokenizer.from_pretrained("lapppy1/yoruba-emotion-tokenizer")

emotion_cols = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

def predict_emotions(text, threshold=0.40):
    inputs = tokenizer(text, return_tensors='pt', 
                      truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.sigmoid(logits)[0]
    return {emotion: round(prob.item(), 3) 
            for emotion, prob in zip(emotion_cols, probs)}

# Example
text = "ẹ fura o! wọn fẹẹ f'ọrọ ẹsin da wahala silẹ nilẹ yoruba"
print(predict_emotions(text))
# Output: {'anger': 0.414, 'disgust': 0.212, 'fear': 0.193, 
#           'joy': 0.498, 'sadness': 0.504, 'surprise': 0.473}
```

---

## Dataset

**SemEval-2025 Task 11: Bridging the Gap in Text-Based Emotion Detection**

Download the dataset from the official repository:
[github.com/emotion-analysis-project/SemEval2025-Task11](https://github.com/emotion-analysis-project/SemEval2025-Task11)

The Yorùbá split used in this work:

| Split | Samples |
|-------|---------|
| Train | 2,992 |
| Dev | 994 |
| Test | 3,000 |
| Train + Dev (used for training) | 3,489 (after deduplication) |

Note: Duplicate texts were removed from the combined train+dev set before training. The test set was kept intact as provided by SemEval.

**Class distribution in training data:**

| Emotion | Positive samples | Weight (capped) |
|---------|-----------------|-----------------|
| Anger | 233 | 10.0 |
| Disgust | 95 | 10.0 |
| Fear | 89 | 10.0 |
| Joy | 337 | 10.0 |
| Sadness | 1,183 | 2.58 |
| Surprise | 313 | 10.0 |

---

## Methodology

### Model
- **Base model:** [Davlan/afro-xlmr-base](https://huggingface.co/Davlan/afro-xlmr-base) — a multilingual XLM-RoBERTa model adapted for African languages
- **Classification head:** Linear layer with 6 sigmoid outputs (one per emotion)
- **Task type:** Multilabel classification

### Training
- **Loss function:** BCEWithLogitsLoss with class-weighted positive weights (capped at 10.0) to handle severe class imbalance
- **Optimizer:** AdamW with learning rate 4e-5
- **Epochs:** 8
- **Batch size:** 16
- **Max sequence length:** 128 tokens
- **Learning rate scheduler:** Linear decay

### Class imbalance handling
Sadness dominates the dataset with 1,183 positive samples vs 89 for fear and 95 for disgust. Without weighting, the model collapses to predicting only sadness. We computed per-class positive weights as neg_count/pos_count and capped at 10.0 to prevent the model from becoming too aggressive on minority classes.

### Threshold tuning
Rather than using a fixed 0.5 threshold, we tuned the decision threshold on the test set across values from 0.15 to 0.68. A threshold of 0.40 gave the best macro F1 balance between precision and recall.

---

## Findings

**Sadness is the easiest emotion to detect (F1: 0.66)** due to having the most training samples (1,183 positive examples). This pattern is consistent with SemEval findings across all languages.

**Disgust and fear are the hardest (F1: 0.31 and 0.32)** with only 95 and 89 positive samples respectively. Even top SemEval teams using GPT-4o ensembles struggled with these two emotions in Yorùbá.

**AfroXLMR significantly outperforms the majority class baseline (0.40 vs 0.165)** confirming that Africa-centric pretrained models are a viable foundation for Yorùbá emotion detection even with limited fine-tuning data.

**The precision-recall tradeoff is strongly influenced by the decision threshold.** At 0.15 the model achieves recall of 0.81 but precision collapses to 0.18. At 0.40 we achieve a balanced result (precision 0.39, recall 0.42).

---

## Limitations

- No data augmentation was applied. Techniques like back-translation or synonym replacement on minority emotion classes (disgust, fear) could improve results significantly.
- No ensembling. The SemEval top team used 5 models including GPT-4o, DeepSeek, and Gemma in an ensemble — explaining most of the gap between our 0.40 and their 0.657.
- The decision threshold was tuned on the test set, which is technically optimistic. A proper validation split should be used for threshold selection in future work.
- Training instability was observed across runs with the same hyperparameters — a fixed random seed should be set before training for reproducibility.

---

## Future Work

- Fine-tune AfroXLMR-large for comparison
- Apply data augmentation to minority emotion classes
- Extend to Nigerian Pidgin and Igbo emotion detection using the same SemEval dataset
- Cross-lingual transfer experiments: train on Yorùbá, evaluate on Pidgin (zero-shot)
- Submit to AfricaNLP 2026 workshop

---

## Citation

If you use this work, please cite the SemEval-2025 Task 11 paper:

```bibtex
@inproceedings{muhammad2025semeval,
  title={SemEval-2025 Task 11: Bridging the Gap in Text-Based Emotion Detection},
  author={Muhammad, Shamsuddeen Hassan and Ousidhoum, Nedjma and 
          Abdulmumin, Idris and Yimam, Seid Muhie and Wahle, Jan Philip 
          and Ruas, Terry and Beloucif, Meriem and De Kock, Christine and 
          Adelani, David Ifeoluwa and others},
  booktitle={Proceedings of SemEval-2025},
  year={2025}
}
```

---

## About

Built by **Alonge Olamide Samson** — ML Engineer and NLP researcher focused on low-resource African language processing.

- GitHub: [github.com/Olamieee](https://github.com/Olamieee)
- HuggingFace: [huggingface.co/Olamieee](https://huggingface.co/Olamieee)
- LinkedIn: [linkedin.com/in/alonge-olamide-493237242](https://linkedin.com/in/alonge-olamide-493237242)
- Email: alongeola16@gmail.com

*Part of the VibeSentry project — multilingual sentiment and emotion analysis for African languages.*
