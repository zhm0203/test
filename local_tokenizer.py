from transformers import BertTokenizerFast

# 下载 BERT tokenizer 并保存到本地
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
tokenizer.save_pretrained('./tokenizer')