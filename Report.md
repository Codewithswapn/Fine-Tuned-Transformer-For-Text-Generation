## ----------------------

## 1. Dataset Selection

## ----------------------

We chose the **Tiny Shakespeare** dataset because it contains a rich vocabulary of **12,632 unique tokens** with a total of **204,089 tokens**. So try to experiment with it and check how model generate text. 

```
(venv) D:\Water\ChatGPT\group-9\Transformer_Pretraining>python data_loader.py
Total tokens: 204089  
Total unique tokens: 12632
```

---

## ---------------------------

## 2. Tokenizer Selection

## ---------------------------

For the Tiny Shakespeare dataset, we experimented with both tokenization methods:

1. **Character-level**
2. **Word-level**

### Character-Level Tokenization

The issue with character-level tokenization is that it cannot satisfy the required model parameter range of **5–10 million**, even when using the largest architecture paramerts mentioned in project requirement. This is because the vocabulary only contains **37 characters**, including special tokens like `<PAD>` and `<UNK>`.

![Alt text](<./Results/TotalModelParam-With CharLevelTokenizationUsed.png>)

### Transformer Model Parameters Calculation (Using Character-Level Vocab)

---

#### Embedding Parameters:

* **Token Embedding** = vocab_size × embedding_dim = 37 × 256 = **9,472**
* **Positional Embedding** = sequence_length × embedding_dim = 256 × 256 = **65,536**

**Total Embedding Parameters:**
`9472 + 65536 = 74,752`

---

#### Single Transformer Block Parameters:

* **Normalization Layers (3 layers)** = 2 × (2 × embedding_dim) = **1,024**
* **Multi-head Attention** = 4 × (embedding_dim × embedding_dim) = 4 × (256 × 256) = **262,144**
* **Feedforward Network (2 layers):**

  * Layer 1: embedding_dim*(4*embedding_dim)+(4*embedding_dim)- 256 × (4 × 256) + (4 × 256) = **263,168**
  * Layer 2: (4*embedding_dim)*embedding_dim+embedding_dim- (4 × 256) × 256 + 256 = **262,400**

**Total per Transformer Block:**
`1024 + 262144 + 263168 + 262400 = 788,736`

With **3 Transformer blocks**:
`3 × 788,736 = 2,366,208`

---

#### Final Normalization and Output Layer:

* **Final Normalization** = 2 × embedding_dim = **512**
* **Output Layer** = embedding_dim × vocab_size + vocab_size = 256 × 37 + 37 = **9,509**

**Total Model Parameters:**
`2,366,208 + 9472 + 65536 + 512 + 9509 = 2,451,237`
**≈ 2.45M Parameters**

---

### Word-Level Tokenization

Using word-level tokenization helps us stay within the required parameter limits. The Tiny Shakespeare dataset contains **12,632 unique tokens**, so using the **10,000 most frequent tokens** for the vocabulary is feasible and effective.

![alt text](<Training.png>)
---

## --------------------------------------

## 3. Model Design and Attention Analysis

## --------------------------------------

For this part, we first studied the complete Transformer architecture using some excellent YouTube videos and Medium blogs. Since we are working on **text generation**, we only need the **decoder block** of the Transformer.

Initially, we designed and trained the model but forgot to return attention values for each layer, which are required for attention visualization. We then modified the model structure and retrained it to include the attention outputs.

### References for Model Design and Attention Visualization:

1. https://www.youtube.com/watch?v=bCz4OMemCcA
2. https://www.youtube.com/watch?v=ISNdQcPhsts
3. https://medium.com/@sjasmeet135/transforming-text-generation-the-power-of-transformers-in-llms-703b236fa03b
4. https://medium.com/@aadit.kshirsagar/building-a-text-generation-transformer-from-scratch-a-deep-dive-3dcde380013b
