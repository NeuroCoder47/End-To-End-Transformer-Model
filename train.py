import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, random_split
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from dataset import BilingualDataset 
from model import *
from model import *
from config import get_config
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def greedy_decode(model, source, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) >= max_len:
            break
        output = model(source, decoder_input)  # Shape: [1, seq_len, vocab_size]
        prob = output[:, -1]  # Shape: [1, vocab_size]
        # Get the token with highest probability
        _, next_word = torch.max(prob, dim=1)  # Shape: [1]
        # Add the predicted token to decoder input
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
        # Stop if we predict [EOS] token
        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)  # Remove batch dimension



def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, writer, num_examples=2):
    model.eval()
    count = 0
    source_texts = []
    expected =[]
    predicted = []

    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count +=1
            encoder_input = batch['encoder_input'].to(device)
            assert encoder_input.size(0)==1 , "Batch size must be 1 for validation"
            model_out = greedy_decode(model, encoder_input, tokenizer_src, tokenizer_tgt, max_len, device)
            source_text = batch[ 'src_text'][0]
            target_txt = batch[ 'tgt_text'][0]
            model_out_text =  tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_txt}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

    

def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds,lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        ## tokenizer: create empty tokenizer object with WordLevel model (Tokenizer container, WordLevel model = whole-word tokenization), set unknown token (unk_token='[UNK]'); Purpose: initialize tokenizer shell before adding config
        tokenizer = Tokenizer(WordLevel(unk_token = '[UNK]'))
        ## tokenizer.pre_tokenizer: set text splitting strategy to Whitespace (splits on spaces/tabs/newlines); Purpose: define how raw text becomes word pieces before ID assignment
        ## Simplified: tells tokenizer "split sentences by spaces"; e.g., "Hello world" → ["Hello", "world"]
        tokenizer.pre_tokenizer= Whitespace()
        ## trainer: create training configuration with special tokens list ([UNK], [PAD], [SOS], [EOS]) and minimum word frequency (min_frequency=2 means ignore words appearing less than twice); Purpose: set vocabulary building rules
        ## Simplified: rules for training—always include 4 special tokens, only keep words that appear at least 2 times in dataset
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]","[EOS]"], min_frequency = 2)
        ## tokenizer.train_from_iterator: train tokenizer by reading all sentences (get_all_sentences iterator), applying trainer rules; Purpose: scan dataset, count words, build vocabulary with IDs
        ## Simplified: read all sentences, count word frequencies, keep words appearing ≥2 times + special tokens, assign each a unique ID (0, 1, 2, ...) to create vocabulary
        tokenizer.train_from_iterator(get_all_sentences(ds,lang), trainer=trainer)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('opus_books',f'{config["lang_src"]}-{config["lang_tgt"]}',split='train')
    tokenizer_src = get_or_build_tokenizer(config, ds_raw,config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw,config["lang_tgt"])
    
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size =len(ds_raw) - train_ds_size
    train_ds_raw , val_ds_raw = random_split(ds_raw,[train_ds_size,val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    train_dataloader = DataLoader(train_ds, batch_size = config['batch_size'],shuffle= True)
    val_dataloader = DataLoader(val_ds, batch_size = 1,shuffle= True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
    
def get_model(config, vocab_src_len, vocab_tgt_len):
    model = Transformer(vocab_src_len,vocab_tgt_len, d_model = config['d_model'], num_head= config['heads'], num_layers= config['layers'], d_ff= config['d_ff'], max_seq_length=config['seq_len'], dropout=config['dropout'])
    return model
def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataloader, val_dataloader, tokenizer_src , tokenizer_tgt = get_ds(config)
    model = get_model(config , tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(),lr = config['lr'], eps = 1e-9)
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    best_loss = float('inf')  # ADD THIS HERE

    for epoch in range ( config['num_epoch']):
        step = 0
        model.train()
        batch_iterator = tqdm(train_dataloader, desc = f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
## BEFORE PROJECTION (decoder_output):
## Shape: [batch, seq_len, d_model] e.g., [8, 20, 512]
## Rows (seq_len dimension): represent positions in the sequence (position 0, position 1, ..., position 19)
## Columns (d_model=512): abstract learned features encoding grammar, semantics, context, and patterns - NOT words or interpretable concepts
## Each position's 512 numbers are the model's "internal thoughts" - contextual representation containing:
##   - Information about the token at that position
##   - Context from all previous positions (via self-attention with causal mask)
##   - Relevant source sentence information (via cross-attention to encoder output)
## Example: position 3's vector [2.3, -1.5, 0.8, ..., 0.2] might encode "expects plural noun, animal context, subject position"
## These 512 features are compact and efficient for attention layers to work with during processing

## PROJECTION TRANSFORMATION (model.project):
## Linear layer: nn.Linear(d_model=512, vocab_size=5000)
## Operation: matrix multiplication (decoder_output @ weights) + bias
## Purpose: translate abstract 512 features → concrete 5000 vocabulary word scores
## Acts like a translator from "model's internal language" to "actual vocabulary words"

## AFTER PROJECTION (proj_output):
## Shape: [batch, seq_len, vocab_size] e.g., [8, 20, 5000]
## Rows (seq_len): still represent positions (position 0, 1, 2, ...) - SAME as before
## Columns (vocab_size=5000): NOW represent actual vocabulary words - COMPLETELY DIFFERENT meaning than before
##   - Column 0 = score for vocabulary word 0 (e.g., "the")
##   - Column 1 = score for vocabulary word 1 (e.g., "apple")
##   - Column 4999 = score for vocabulary word 4999 (e.g., "zebra")
## Each number is a logit (score) indicating confidence that specific word should appear at that position
## Higher score = model predicts that word is more likely to be the next token
## Example: position 3 row [0.2, 4.5, 1.1, ...] means word 1 (score 4.5) is top prediction, word 2 (1.1) second choice
## These scores are compared to labels during loss calculation to train the model
## Note: scores are logits (not probabilities yet); log_softmax applied later to get log probabilities

## Simplified: decoder gives abstract features [batch, positions, 512 internal patterns] → 
##            projection translates to [batch, positions, 5000 word scores] for each vocabulary word
            proj_output = model(encoder_input, decoder_input)
            # it contains the correct data it the oorignal data
            label = batch['label'].to(device)
## BEFORE RESHAPING:
## proj_output shape: [batch, seq_len, vocab_size] e.g., [8, 20, 5000]
##   - Each position (row) has 5000 scores (one per vocabulary word)
##   - Example: position 5 might have scores [0.2, 4.5, 1.1, 0.3, ...] predicting word 1 is most likely
## label shape: [batch, seq_len] e.g., [8, 20]
##   - Contains correct token IDs for each position
##   - Example: [45, 123, 67, 89, ...] means position 0 should be token 45, position 1 should be token 123, etc.

## RESHAPING WITH .view():
## proj_output.view(-1, tokenizer_tgt.get_vocab_size()):
##   - Flattens batch and seq_len dimensions together: [8, 20, 5000] → [8×20, 5000] = [160, 5000]
##   - '-1' means "calculate this dimension automatically" (PyTorch computes 8×20=160)
##   - 'tokenizer_tgt.get_vocab_size()' explicitly sets second dimension to vocab_size (5000)
##   - Result: 160 rows (one per position across all batches), 5000 columns (vocab scores)
##   - Each row is now treated as an independent prediction for one position
## label.view(-1):
##   - Flattens to 1D: [8, 20] → [160]
##   - Creates one long list of correct token IDs: [tok0_batch0_pos0, tok1_batch0_pos1, ..., tok19_batch0, tok0_batch1_pos0, ...]
##   - Each element is the correct answer for one position

## WHY RESHAPE:
## CrossEntropyLoss expects specific input shapes:
##   - Predictions: [N, C] where N=number of samples, C=number of classes
##   - Targets: [N] where N=number of correct class indices
## By reshaping, we treat each sequence position as an independent classification problem
## This allows comparing model predictions to correct answers position-by-position across all batches simultaneously

## LOSS CALCULATION (loss_fn = CrossEntropyLoss):
## Operation: compares predicted scores to correct token IDs for every position
## For each position (row):
##   1. Takes the 5000 predicted scores
##   2. Looks at the correct token ID from label
##   3. Calculates how confident the model was in predicting the correct token
##   4. Penalizes wrong predictions (high loss if predicted wrong token, low loss if predicted correct token)
## Example for one position:
##   - Predicted scores: [0.2, 4.5, 1.1, 0.3, ...]
##   - Correct token ID: 1 (should be word 1)
##   - Model gave word 1 score 4.5 (highest) → loss is low (good prediction!)
##   - If correct was token 3 (score 0.3), loss would be high (bad prediction)
## Final loss: average of all 160 position losses → single scalar number

## OUTPUT (loss):
## Shape: scalar (single number) e.g., 2.456
## Meaning: how wrong the model's predictions are on average across all positions
## Lower loss = better predictions (model correctly predicted most tokens)
## Higher loss = worse predictions (model got many tokens wrong)
## This loss is used for backpropagation (.backward()) to update model weights and improve predictions

## Simplified: flatten predictions [160, 5000] and labels [160], compare them position-by-position using CrossEntropyLoss,
##            get single loss number measuring "how wrong were all the predictions"; lower is better
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            step += 1 
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if loss.item() < best_loss:
            model_filename = Path(r"C:\Users\Ashmit Gupta\Desktop\Coding\Pytorch\weights\translation.pth")
            best_loss = loss.item()
            print(f"\nNew lowest loss: {best_loss:.4f} - Model saved!")
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
            }, model_filename)
            
        if epoch % 1 ==0:
            run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg),writer)
        
if __name__ == '__main__':
    print("MAIN BLOCK RUNNING") 
    config = get_config()
    train_model(config)

        

        

    
