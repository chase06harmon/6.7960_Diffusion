from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # Or any model name you want
tokenizer.save_pretrained('tokenizers/bert-base-uncased')  # Save it locally