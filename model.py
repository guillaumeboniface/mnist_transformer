import torch
from torch import nn

class ImageEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=14, embedding_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Linear(patch_size ** 2, embedding_dim)

    def forward(self, x):
        x = x.view(x.shape[0], self.num_patches, -1)
        x = self.projection(x)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=10):
        super().__init__()
        self.embedding = nn.Embedding(max_len, embedding_dim)

    def forward(self, x):
        positions = torch.arange(0, x.shape[1], device=x.device)
        return self.embedding(positions) + x
    
class AttentionHead(nn.Module):
    def __init__(self, embedding_dim, head_dim, mask=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_dim = head_dim
        self.query = nn.Linear(embedding_dim, head_dim)
        self.key = nn.Linear(embedding_dim, head_dim)
        self.value = nn.Linear(embedding_dim, head_dim)
        self.mask = mask
        
    def forward(self, x, y):
        query = self.query(y) # [batch_size, num_patches, head_dim]
        key = self.key(x)
        value = self.value(x)
        
        attention_weights = torch.bmm(query, key.transpose(1, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if self.mask:
            attention_weights = torch.tril(attention_weights)
        attention_weights = torch.softmax(attention_weights, dim=-1)
        weighted_value = torch.bmm(attention_weights, value)

        return weighted_value # [batch_size, num_patches, head_dim]

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, mask=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.heads = nn.ModuleList([AttentionHead(embedding_dim, embedding_dim // num_heads, mask) for _ in range(num_heads)])

    def forward(self, x, y):
        return torch.cat([head(x, y) for head in self.heads], dim=-1)

class SelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.multi_head_attention = MultiHeadAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )

    def forward(self, x):
        x_attn = self.multi_head_attention(x, x)
        x = self.norm1(x + x_attn)
        x_ff = self.feed_forward(x)
        x = self.norm2(x + x_ff)
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.layers = nn.ModuleList([SelfAttentionBlock(embedding_dim, num_heads) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.masked_multi_head_attention = MultiHeadAttention(embedding_dim, num_heads, mask=True)
        self.multi_head_attention = MultiHeadAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )

    def forward(self, d_input, e_output):
        d_masked = self.masked_multi_head_attention(d_input, d_input)
        print(d_masked.shape)
        print(d_input.shape)
        d_input = self.norm1(d_input + d_masked)
        d_attn = self.multi_head_attention(e_output, d_input)
        d_input = self.norm2(d_input + d_attn)
        d_ff = self.feed_forward(d_input)
        d_input = self.norm3(d_input + d_ff)
        return d_input
    
class TransformerDecoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.layers = nn.ModuleList([CrossAttentionBlock(embedding_dim, num_heads) for _ in range(num_layers)])

    def forward(self, d_input, e_output):
        for layer in self.layers:
            d_input = layer(d_input, e_output)
        return d_input
    
class TransformerClassifier(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, num_classes):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.image_embedding = ImageEmbedding()
        self.positional_encoding = PositionalEncoding(embedding_dim)
        self.transformer_encoder = TransformerEncoder(embedding_dim, num_heads, num_layers)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.image_embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    dummy_image = torch.randn(3, 28, 28) # 3 images of 28x28
    image_embedding = ImageEmbedding()
    assert image_embedding(dummy_image).shape == torch.Size([3, 4, 64])

    positional_encoding = PositionalEncoding(64)
    assert positional_encoding(image_embedding(dummy_image)).shape == torch.Size([3, 4, 64])

    head = AttentionHead(64, 16)
    assert head(image_embedding(dummy_image), image_embedding(dummy_image)).shape == torch.Size([3, 4, 16])

    multi_head = MultiHeadAttention(64, 4)
    assert multi_head(image_embedding(dummy_image), image_embedding(dummy_image)).shape == torch.Size([3, 4, 64])

    self_attention_block = SelfAttentionBlock(64, 4)
    assert self_attention_block(image_embedding(dummy_image)).shape == torch.Size([3, 4, 64])

    transformer_encoder = TransformerEncoder(64, 4, 3)
    assert transformer_encoder(image_embedding(dummy_image)).shape == torch.Size([3, 4, 64])

    classifier = TransformerClassifier(64, 4, 3, 10)
    assert classifier(dummy_image).shape == torch.Size([3, 10])

    dummy_decoder_input = torch.randn(3, 6, 16) # batch size 3, 6 max sentence length, vocab size 16
    cross_attention_block = CrossAttentionBlock(16, 4)
    assert cross_attention_block(dummy_decoder_input, transformer_encoder(image_embedding(dummy_image))).shape == torch.Size([3, 6, 16])

    transformer_decoder = TransformerDecoder(16, 4, 3)
    assert transformer_decoder(dummy_decoder_input, transformer_encoder(image_embedding(dummy_image))).shape == torch.Size([3, 6, 16])


        
        
        