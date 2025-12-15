```mermaid
graph TD
    A[Input Patches (144xQxQx1)] --> B{TimeDistributed}
    subgraph CNN Base (Shared)
        B --> C[Patch Input (41x41x1)]
        C --> Conv1_32x3x3[Conv2D (32 filters)]
        Conv1_32x3x3 --> Pool1_2x2[MaxPooling2D ((2, 2) pool)]
        Pool1_2x2 --> Conv2_64x3x3[Conv2D (64 filters)]
        Conv2_64x3x3 --> Pool2_2x2[MaxPooling2D ((2, 2) pool)]
        Pool2_2x2 --> Flatten[Flatten (6400)]
        Flatten --> PatchFeatures[Dense (64 features)]
    end
    B --> Concat[Flatten (Concatenate 144x64)]
    Concat --> DNN_Dense1_128[Dense (128 units)]
    DNN_Dense1_128 --> DNN_Dense2_64[Dense (64 units)]
    DNN_Dense2_64 --> Output[Output (Softmax, 3 bins)]
```