from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
import os

# Custom DNA vocabulary including positional tokens
DNA_VOCAB = {
    "<pad>": 0,
    "<unk>": 1,
    "<task>": 2,
    "A": 3, "T": 4, "C": 5, "G": 6,
    "-": 7, ",": 8, "0": 9, "1": 10, "2": 11, "3": 12,
    "4": 13, "5": 14, "6": 15, "7": 16, "8": 17, "9": 18
}

tokenizer_core = Tokenizer(WordLevel(vocab=DNA_VOCAB, unk_token="<unk>"))
output_dir = "models"
tokenizer_core.save(os.path.join(output_dir,"dna_tokenizer.json"))