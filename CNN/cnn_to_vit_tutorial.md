# Building PyTorch Architectures: CNNs to Vision Transformers

This walkthrough breaks down how to construct machine learning architectures from scratch in PyTorch. Understanding these structures will help you smoothly transition into coding Vision Transformers (ViTs) and multimodal models (Vision + Text).

## 1. The Building Blocks of PyTorch Models

In PyTorch, all models inherit from `torch.nn.Module`. To define an architecture, you implement two required components:
1. **`__init__`**: This is where you declare all the layers containing learnable weights (e.g., convolutions, linear/dense layers, attention mechanisms).
2. **`forward`**: This function defines the "forward pass" — the exact path your data volume takes as it flows through the layers defined in `__init__`.

You can copy and run the following code directly in your `Testnb.ipynb` notebook to see it in action.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # 1st Convolutional Block
        # in_channels: 3 for RGB images
        # out_channels: 16 filters (features)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # MaxPool halves the height and width of the image
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2nd Convolutional Block
        # Notice that in_channels here matches out_channels from the previous block
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # 3rd Convolutional Block
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Fully Connected (Linear) Layers
        # Let's trace the spatial size for a 224x224 input image:
        # conv1 + pool -> 112x112
        # conv2 + pool -> 56x56
        # conv3 + pool -> 28x28
        # Final feature map size: 64 channels * 28 height * 28 width = 50176
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Pass data through: Conv -> ReLU Activation -> MaxPool
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the 3D volume into a 1D vector for the linear layers
        # The -1 tells PyTorch to dynamically fill in the batch size
        x = x.view(-1, 64 * 28 * 28) 
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Let's test the architecture!
model = SimpleCNN(num_classes=2)
dummy_image = torch.randn(1, 3, 224, 224) # Format: [Batch Size, Channels, Height, Width]
logits = model(dummy_image)

print(f"Output shape: {logits.shape}") # Expected: [1, 2]
```

> [!TIP]
> **Tracking Dimensionality**: The most common error in architecture coding is mismatched dimensions between the final convolutional layer's output and the first linear layer's input. 

---

## 2. Transitioning from CNNs to Vision Transformers (ViT)

CNNs excel at finding local patterns (edges, textures) thanks to their sliding kernel. A **Vision Transformer** takes a fundamentally different approach, heavily influenced by NLP text transformers: it treats an image as a sequence of input tokens.

In text, you tokenize an entire sentence into words. In a ViT, you divide the image into uniform **"patches"**.

### Patch Embeddings: The Bridge

The very first step of a ViT is transforming a standard `[B, C, H, W]` image into a sequence of flat patches `[B, N, D]` (where `N` is the number of patches, and `D` is the embedding dimension).

We can actually use what we just learned about `nn.Conv2d` to do this efficiently! A convolution with a kernel size AND a stride equal to the patch size will slice the image into non-overlapping patches and project them into an embedding vector simultaneously.

```python
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        
        # A Convolution layer where the stride equals the kernel size
        # This prevents the overlapping "sliding window" effect of standard CNNs
        # and instead extracts exact, non-overlapping PxP patches!
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x):
        # x.shape: [Batch, 3, 224, 224]
        x = self.proj(x)
        # x.shape: [Batch, embed_dim, 14, 14] 
        # (Since 224/16 = 14, we get a 14x14 grid of patches)
        
        # Transformers expect a sequence, not a spatial grid.
        # We flatten the spatial dimensions (14x14 = 196 tokens)
        x = x.flatten(2) # Flatten dims 2 and 3 into one -> [Batch, embed_dim, 196]
        
        # Finally, swap the dimensions to match standard Transformer inputs: 
        # [Batch Sequence Length, Embedding Dim]
        x = x.transpose(1, 2) # -> [Batch, 196, 768]
        
        return x

vit_stem = PatchEmbedding(patch_size=16, embed_dim=768)
dummy_image = torch.randn(1, 3, 224, 224)
tokens = vit_stem(dummy_image)

print(f"Vision Token Sequence Shape: {tokens.shape}") 
# Look familiar? It's the same dimension structure as embedded text tokens!
```

> [!NOTE]
> By flattening the image into sequence `[1, 196, 768]`, the image tokens look exactly like mathematical text embeddings. This makes it much easier down the road to combine your `vision_tokens` array with a `text_tokens` array (either by concatenating them or using Cross-Attention) since they share the same dimensional structures.
