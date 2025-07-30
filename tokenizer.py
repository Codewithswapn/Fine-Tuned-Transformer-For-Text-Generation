# ----------------------------------------
# PART-1 : Word level Tokenization
# ----------------------------------------

import re
import json

def build_vocab(text, vocab_size=10000):

    text = text.lower()  # make all the text form given dataset in lowercase 

    # ------------------------------------------------------------------------------------------------------------------------------
    # words = re.findall(r'\w+', text)  
    # extract words from the given text or dataset also ignore the symbol and punction from text
    # because they are not important for building vocabulary as we have limit of 10k words.
    # Example - text = "My name is Swapnil Salunkhe and I'm enrolled in ChatGPT course. Thank You!"
    #           words = re.findall(r'\w+', text)
    #           print(words)
    # Output - (venv) D:\Water\ChatGPT\group-9\Transformer_Pretraining>python c.py
    #          ['My', 'name', 'is', 'Swapnil', 'Salunkhe', 'and', 'I', 'm', 'enrolled', 'in', 'ChatGPT', 'course', 'Thank', 'You']
    # -----------------------------------------------------------------------------------------------------------------------------

    words = re.findall(r'\w+', text)  

    # Here counting the most frequent and rare words occure in the given text
    word_counts = {}
    for word in words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1

    # Sort words by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    # Start vocab with special tokens 
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    # Adding top words to vocabulary with special tokens (10000-2= 9998 - frequent words appear in the text added to vocab)
    for i, (word, _) in enumerate(sorted_words[:vocab_size - 2]):
        vocab[word] = i + 2  # start indexing from 2

    return vocab

# Converting words to token IDs
def tokenize(text, vocab):

    text = text.lower()
    words = re.findall(r'\w+', text)
    token_ids = []
    for word in words:
        if word in vocab:
            token_ids.append(vocab[word])
        else:
            token_ids.append(vocab["<UNK>"])
    return token_ids

# Converting token IDs back to words
def detokenize(token_ids, vocab):
  
    id_to_word = {}
    for word in vocab:
        token_id = vocab[word]
        id_to_word[token_id] = word

    # Convert each ID to a word
    words = []
    for tid in token_ids:
        if tid in id_to_word:
            words.append(id_to_word[tid])
        else:
            words.append("<UNK>")

    return words

# Save vocabulary to a file - So we can use it for text generation 
def save_vocab(vocab, path='vocab.json'):
    with open(path, 'w') as f:
        json.dump(vocab, f)

# Load vocabulary 
def load_vocab(path='vocab.json'):
    with open(path, 'r') as f:
        return json.load(f)



# ----------------------------------------
# PART-2 : Character level Tokenization
# ----------------------------------------

# import json

# def build_vocab(text, vocab_size=10000):

#     text = text.lower()  # make all the text from given dataset in lowercase

#     # ------------------------------------------------------------------------------------------------------------------------------
#     # chars = list(text)  
#     # extract characters from the given text including spaces and symbols
#     # because in character-level tokenization each character is treated as a token
#     # Example - text = "Hello!"
#     #           chars = list(text)
#     #           print(chars)
#     # Output - ['h', 'e', 'l', 'l', 'o', '!']
#     # ------------------------------------------------------------------------------------------------------------------------------

#     chars = list(text)

#     # Here counting the most frequent and rare characters in the given text
#     char_counts = {}
#     for ch in chars:
#         if ch in char_counts:
#             char_counts[ch] += 1
#         else:
#             char_counts[ch] = 1

#     # Sort characters by frequency
#     sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)

#     # Start vocab with special tokens 
#     vocab = {
#         "<PAD>": 0,
#         "<UNK>": 1
#     }

#     # Adding top characters to vocabulary with special tokens (10000-2= 9998 - most frequent characters in the text added to vocab)
#     for i, (ch, _) in enumerate(sorted_chars[:vocab_size - 2]):
#         vocab[ch] = i + 2  # start indexing from 2

#     return vocab

# # Converting characters to token IDs
# def tokenize(text, vocab):

#     text = text.lower()
#     chars = list(text)  # convert text to list of characters
#     token_ids = []
#     for ch in chars:
#         if ch in vocab:
#             token_ids.append(vocab[ch])
#         else:
#             token_ids.append(vocab["<UNK>"])
#     return token_ids

# # Converting token IDs back to characters
# def detokenize(token_ids, vocab):
  
#     id_to_char = {}
#     for ch in vocab:
#         token_id = vocab[ch]
#         id_to_char[token_id] = ch

#     # Convert each ID to a character
#     chars = []
#     for tid in token_ids:
#         if tid in id_to_char:
#             chars.append(id_to_char[tid])
#         else:
#             chars.append("<UNK>")

#     return "".join(chars)

# # Save vocabulary to a file - So we can use it for text generation 
# def save_vocab(vocab, path='vocab_Charlevel.json'):
#     with open(path, 'w', encoding='utf-8') as f:
#         json.dump(vocab, f, ensure_ascii=False, indent=2)

# # Load vocabulary 
# def load_vocab(path='vocab_Charlevel.json'):
#     with open(path, 'r', encoding='utf-8') as f:
#         return json.load(f)
