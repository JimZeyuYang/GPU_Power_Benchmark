#!/usr/bin/env python3

from transformers import BertTokenizer, BertModel
import torch
import time
from faker import Faker


import warnings

def main():
    REPEAT = 20

    warnings.filterwarnings('ignore')

    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your setup.")
        exit()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load pre-trained model tokenizer, to convert our text into tokens that correspond to BERT's vocabulary
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Batch of texts
    fake = Faker()
    texts = ["[CLS] " + fake.text(max_nb_chars=200) + " [SEP]" for _ in range(64)]

    # Tokenize the texts, map the token strings to their vocabulary indeces, add [CLS] and [SEP] tokens, and create attention masks
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
                            text,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 512,          # Pad & truncate all sentences.
                            truncation = True,
                            padding = 'max_length',
                            return_attention_mask = True,   # Construct attention masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors
    input_ids = torch.cat(input_ids, dim=0).to(device)
    attention_masks = torch.cat(attention_masks, dim=0).to(device)

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased')

    # Put the model in "evaluation" mode, meaning feed-forward operation
    model.eval()

    # Move the model to the GPU
    model = model.to(device)

    # warm up
    with torch.no_grad():  model(input_ids, attention_mask=attention_masks)
    torch.cuda.synchronize()

    start_ts = int(time.time() * 1_000_000)

    with torch.no_grad():
        for i in range(REPEAT):
            model(input_ids, attention_mask=attention_masks)
    torch.cuda.synchronize()

    end_ts = int(time.time() * 1_000_000)

    # Store timestamps in a file
    with open("timestamps.csv", "w") as f:
        f.write("timestamp\n")
        f.write(str(start_ts) + "\n")
        f.write(str(end_ts) + "\n")

    # print("Time spent per batch: {:.3f} ms".format((end_ts - start_ts) / 1000 / REPEAT))
    # print("Total runtime: {:.3f} ms".format((end_ts - start_ts) / 1000))


if __name__ == '__main__':
    main()