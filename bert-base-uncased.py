import drive
import tensorflow as tf
from transformers import BertTokenizer, TFBertForTokenClassification
from datasets import load_dataset
from transformers import TrainingArguments, Trainer

dataset = load_dataset("conll2003")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(dataset["train"].features["ner_tags"].feature.names))
def tokenize_and_align_labels(examples):
  tokenized_inputs = tokenizer(examples["tokens"], truncation=True, padding='max_length', max_length=128, is_split_into_words=True, return_special_tokens_mask=True)

    # Initialize label_ids with -100 (ignored label)
  label_ids = [-100] * len(tokenized_inputs['input_ids'])

  for i, label in enumerate(examples["ner_tags"]):
    label_idx = 1  # Skip [CLS] token
    for word_idx, word in enumerate(examples["tokens"][i]):
      token_idx = i * 128 + word_idx + 1
      if token_idx >= len(tokenized_inputs['input_ids']):
        break  # Exit loop if token index is out of range
      if tokenized_inputs['special_tokens_mask'][i][word_idx + 1] == 0:  # Ignore special tokens
        if label_idx < len(label) and word == tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'][i][word_idx + 1]):
          label_ids[token_idx] = label[label_idx]
          label_idx += 1

  tokenized_inputs["labels"] = label_ids
  return tokenized_inputs
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
train_dataset = tokenized_datasets["train"].to_tf_dataset(columns=["input_ids", "attention_mask", "labels"], batch_size=8)
val_dataset = tokenized_datasets["validation"].to_tf_dataset(columns=["input_ids", "attention_mask", "labels"], batch_size=8)
# from google.colab import drive
drive.mount('/content/drive')
training_args = TrainingArguments(
  output_dir="/content/drive/My Drive/123",
  num_train_epochs=3,
  per_device_train_batch_size=8,
  save_steps=500,
  save_total_limit=2,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    framework="tf"
)
trainer.train()