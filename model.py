import torch
import torch.nn as nn

# Single transformer block that includes self-attention with a mask to block future tokens.
class SimpleTransformerBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, dropout_rate):
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_size)

        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True )

        self.norm2 = nn.LayerNorm(embedding_size)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.GELU(),
            nn.Linear(4 * embedding_size, embedding_size),
            nn.Dropout(dropout_rate) )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs, attention_mask):

        norm_inputs = self.norm1(inputs)
        attention_output, attn_weights = self.self_attention(
            norm_inputs, norm_inputs, norm_inputs,
            attn_mask=attention_mask,
            average_attn_weights=False )

        inputs = inputs + self.dropout(attention_output)
        norm_inputs = self.norm2(inputs)
        ff_output = self.feed_forward(norm_inputs)
        inputs = inputs + ff_output

        return inputs, attn_weights

# Decoder-only Transformer model
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_layers, num_heads, max_sequence_length, dropout_rate):
        super().__init__()

        # Embedding for tokens
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)

        # Positional embedding
        self.position_embedding = nn.Embedding(max_sequence_length, embedding_size)

        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            SimpleTransformerBlock(embedding_size, num_heads, dropout_rate)
            for _ in range(num_layers)])

        self.final_norm = nn.LayerNorm(embedding_size)

        # Output layer to predict the next token
        self.output_layer = nn.Linear(embedding_size, vocab_size)

        self.max_sequence_length = max_sequence_length

    def forward(self, input_tokens, return_attentions=False):

        batch_size, seq_length = input_tokens.shape
        token_embed = self.token_embedding(input_tokens)

        # Create position indices and get their embeddings
        positions = torch.arange(seq_length, device=input_tokens.device)
        position_embed = self.position_embedding(positions)

        # Add token and position embeddings
        x = token_embed + position_embed.unsqueeze(0)

        # Masking to prevent looking at future tokens
        mask = torch.triu(torch.ones(seq_length, seq_length).bool(), diagonal=1)
        mask = mask.to(input_tokens.device)
        
        all_attentions = []

        # Pass through each transformer block
        for block in self.transformer_blocks:
            x, attn = block(x, mask)
            all_attentions.append(attn)
        
        # Normalize and predict next tokens and return attentions 
        x = self.final_norm(x)

        if return_attentions:
            return self.output_layer(x), all_attentions
        return self.output_layer(x)