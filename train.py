"""
train.py — Fine-tune DistilBERT for 6-class emotion classification.

Dataset : dair-ai/emotion   (Twitter messages -> sadness/joy/love/anger/fear/surprise)
Model   : distilbert-base-uncased

Usage:
    python train.py                                  # full run, default settings
    python train.py --epochs 5 --batch_size 32        # tune hyperparameters
    python train.py --sample_size 500                 # quick smoke test (~1 min)
    python train.py --push_to_hub --hub_model_id you/emotion-classifier
        (requires `huggingface-cli login` first)

Outputs (written to the project root):
    ./emotion-classifier-model/   -> saved model + tokenizer, loaded by app.py
    ./confusion_matrix.png        -> confusion matrix on the held-out test set
    ./eval_results.json           -> final test-set metrics
"""
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe backend, no display needed
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)

MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "dair-ai/emotion"
LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
NUM_LABELS = len(LABEL_NAMES)
MAX_LENGTH = 64  # generous for tweet-length text
OUTPUT_MODEL_DIR = "./emotion-classifier-model"


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune DistilBERT on the emotion dataset")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_size", type=int, default=None,
                    help="If set, subsample this many TRAIN examples for a fast smoke test.")
    p.add_argument("--push_to_hub", action="store_true",
                    help="Push the final model to the HF Hub (requires prior `huggingface-cli login`).")
    p.add_argument("--hub_model_id", type=str, default=None,
                    help="e.g. your-username/emotion-classifier (required if --push_to_hub is set).")
    return p.parse_args()


def main():
    args = parse_args()
    if args.push_to_hub and not args.hub_model_id:
        raise ValueError("--push_to_hub requires --hub_model_id, e.g. your-username/emotion-classifier")

    set_seed(args.seed)

    # ---- 1. Load data ----
    print(f"Loading dataset: {DATASET_NAME}")
    raw_datasets = load_dataset(DATASET_NAME)
    # Native splits are already train / validation / test.
    train_ds, val_ds, test_ds = raw_datasets["train"], raw_datasets["validation"], raw_datasets["test"]

    if args.sample_size:
        train_ds = train_ds.shuffle(seed=args.seed).select(range(min(args.sample_size, len(train_ds))))
        val_ds = val_ds.shuffle(seed=args.seed).select(range(min(max(args.sample_size // 5, 20), len(val_ds))))
        print(f"[smoke test] using {len(train_ds)} train / {len(val_ds)} val examples")

    print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # ---- 2. Tokenize ----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ---- 3. Model ----
    id2label = {i: name for i, name in enumerate(LABEL_NAMES)}
    label2id = {name: i for i, name in enumerate(LABEL_NAMES)}
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, id2label=id2label, label2id=label2id
    )

    # ---- 4. Metrics ----
    def compute_metrics(eval_pred):
        logits, refs = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(refs, preds),
            "f1_macro": f1_score(refs, preds, average="macro"),
        }

    # ---- 5. Train ----
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=0.1,  # NOTE: kept as warmup_ratio (not warmup_steps=0.1) for
        # compatibility with transformers 4.46-4.x; warmup_steps only accepts a
        # float ratio starting in v5. Harmless deprecation warning on v5+, still correct.
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    print("\n=== Training ===")
    trainer.train()

    # ---- 6. Final evaluation on the held-out TEST set ----
    print("\n=== Evaluating on test set ===")
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    print(test_metrics)

    preds_output = trainer.predict(test_ds)
    y_pred = np.argmax(preds_output.predictions, axis=-1)
    y_true = preds_output.label_ids

    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=3)
    print("\n" + report)

    # ---- 7. Confusion matrix ----
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_LABELS)))
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix — DistilBERT Emotion Classifier (test set)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved confusion_matrix.png")

    # ---- 8. Save results + model ----
    with open("eval_results.json", "w") as f:
        json.dump({**test_metrics, "classification_report": report}, f, indent=2)
    print("Saved eval_results.json")

    trainer.save_model(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
    print(f"Saved model + tokenizer to {OUTPUT_MODEL_DIR}")

    if args.push_to_hub:
        print(f"\nPushing to Hub: {args.hub_model_id}")
        trainer.push_to_hub()
        print("Done. Update MODEL_SOURCE in app.py to point at this Hub repo.")


if __name__ == "__main__":
    main()
