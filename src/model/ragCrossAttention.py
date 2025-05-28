import os
import torch
import torch.nn.functional as F

class RagCrossAttention(torch.nn.Module):
    """
    A class to perform cross-attention on embeddings using a multi-head attention mechanism.
    This class is designed to augment embeddings by applying a cross-attention mechanism.
    """

    def __init__(
        self,
        patchSize : int = 16,
        numHeads : int = 8,
        innerDim : int = 256,
        pretrainedModel : str = "",
        loadPretrainedModel : bool = False,
    ):
        """
        Initialize the EmbeddingAugmentation class.

        :param patchSize: Size of the patches to be used in the augmentation process.
        """
        super().__init__()
        self.__device : str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__patchSize : int = patchSize

        self.__dropout : torch.nn.Dropout = torch.nn.Dropout(p=0.25)

        self.__linealInput = torch.nn.Linear(patchSize, innerDim).to(self.__device)

        self.__crossAttentionModule : torch.nn.MultiheadAttention = torch.nn.MultiheadAttention(
            embed_dim=innerDim,
            num_heads=numHeads,
            batch_first=True
        ).to(self.__device)

        self.__linealOutput = torch.nn.Linear(innerDim, patchSize).to(self.__device)

        if loadPretrainedModel:
            if os.path.exists(pretrainedModel):
                self.load_state_dict(
                    torch.load(pretrainedModel, map_location=self.__device, weights_only=False),
                )

    def __concat(
        self,
        x : torch.Tensor,
        y : torch.Tensor,
    ) -> torch.Tensor:
        """
        Concatenate two tensors along the last dimension.
        This method is used to concatenate the augmented tensor with the original tensor.
        :param x: First tensor to be concatenated.
        :param y: Second tensor to be concatenated.
        :return: Concatenated tensor.
        """
        return torch.cat((x, y), dim=-1)

    def __patching(
        self,
        x : torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply patching to the input tensor.
        This method reshapes the input tensor into patches of a specified size.
        :param x: Input tensor to be patched.
        :return: Patches of the input tensor.
        """
        seqLength = x.shape[2]
        remainder : int  = (seqLength - self.__patchSize) % self.__patchSize
        padLen : int = self.__patchSize - remainder if remainder != 0 else 0
        x = F.pad(x, (0, padLen), value=0)
        return x.unfold(dimension=2, size=self.__patchSize, step=self.__patchSize)

    def __joinPatches(
        self,
        x : torch.Tensor,
    ) -> torch.Tensor:
        """
        Join patches back to the original tensor shape.
        This method reshapes the patches back to the original tensor shape.
        :param x: Patches to be joined.
        :return: Joined tensor.
        """
        return x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

    def __normalization(
        self,
        x : torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize the input tensor.
        This method applies layer normalization to the input tensor.
        :param x: Input tensor to be normalized.
        :return: Normalized tensor and its mean and std.
        """
        mean : torch.Tensor = x.mean(dim=(1,2,3), keepdim=True)
        std : torch.Tensor = x.std(dim=(1,2,3), keepdim=True) + 1e-6
        x = (x - mean) / std
        return x, mean, std

    def __denormalization(
        self,
        x : torch.Tensor,
        mean : torch.Tensor,
        std : torch.Tensor,
    ) -> torch.Tensor:
        """
        Denormalize the input tensor.
        This method applies inverse normalization to the input tensor.
        :param x: Input tensor to be denormalized.
        :param mean: Mean used for normalization.
        :param std: Standard deviation used for normalization.
        :return: Denormalized tensor.
        """
        return (x * std) + mean
    
    def __positionalEncoding(
        self,
        x : torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply positional encoding to the input tensor.
        This method adds positional encoding to the input tensor.
        :param x: Input tensor to be augmented with positional encoding.
        :return: Tensor with positional encoding applied.
        """
        dModel : int = self.__patchSize
        nPositions : int = x.shape[2]

        pos : torch.Tensor = torch.arange(
            nPositions,
            dtype=torch.float32,
            device=self.__device,
        ).unsqueeze(1)

        i : torch.Tensor = torch.arange(
            dModel,
            dtype=torch.float32,
            device=self.__device,
        ).unsqueeze(0)

        angleRates : torch.Tensor = 1 / torch.pow(10000, (2 * (i // 2)) / dModel)

        # Apply the formula
        angleRads = pos * angleRates

        # Apply sin to even indices in the array; cos to odd indices
        posEncoding : torch.Tensor = torch.zeros_like(angleRads)
        posEncoding[:, 0::2] = torch.sin(angleRads[:, 0::2])  # even dimensions
        posEncoding[:, 1::2] = torch.cos(angleRads[:, 1::2])  # odd dimensions
        return x + posEncoding.unsqueeze(0).unsqueeze(0)

    def __crossAttention(
        self,
        x : torch.Tensor,
        context : torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply cross-attention to the input tensor.
        This method applies cross-attention to the input tensor using the context tensor.
        :param x: Input tensor to be augmented with cross-attention.
        :param context: Context tensor to be used for cross-attention.
        :return: Tensor after applying cross-attention.
        """
        batchSize = x.shape[0]
        dModel = x.shape[-1]
        query = x.view(batchSize, -1, dModel)
        queryProj : torch.Tensor = self.__linealInput(query)
        queryProj = self.__dropout(queryProj)

        keyValue = context.view(batchSize, -1, dModel)
        keyValueProj : torch.Tensor = self.__linealInput(keyValue)
        keyValueProj = self.__dropout(keyValueProj)

        x = self.__crossAttentionModule(queryProj, keyValueProj, keyValueProj)
        x = self.__dropout(x[0])

        x = self.__linealOutput(x.unsqueeze(1))
        return x

    def gaussian_kernel_1d(self, kernel_size=5, sigma=1.0, device="cpu"):
        """Create a 1D Gaussian kernel."""
        half_size = kernel_size // 2
        x = torch.arange(-half_size, half_size + 1, device=device).float()
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        return kernel.view(1, 1, -1)  # shape: [1, 1, K]

    def iterative_gaussian_smooth_to_match(self, a, b, tol=1e-4, max_steps=20, kernel_size=7, sigma=1.5):
        """
        Smooth `a` using Gaussian kernel until it matches curvature of `b`.
        """
        a = a.clone()
        device = a.device

        original_mean = a.mean(dim=1, keepdim=True)
        kernel = self.gaussian_kernel_1d(kernel_size, sigma, device=device)

        for _ in range(max_steps):
            d2a = a[:, 2:] - 2 * a[:, 1:-1] + a[:, :-2]
            d2b = b[:, 2:] - 2 * b[:, 1:-1] + b[:, :-2]
            smoothness_a = d2a.abs().mean(dim=1)
            smoothness_b = d2b.abs().mean(dim=1)

            if torch.all(smoothness_a <= smoothness_b + tol):
                break

            # Apply 1D Gaussian smoothing
            a_unsq = a.unsqueeze(1)  # [B, 1, L]
            a_smooth = F.conv1d(a_unsq, kernel, padding=kernel_size//2, groups=1).squeeze(1)

            # Keep mean
            a = a_smooth - a_smooth.mean(dim=1, keepdim=True) + original_mean

        return a

    def forward(
        self, xInput : torch.Tensor,
        context : torch.Tensor,
        scores : torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the embedding augmentation.
        This method applies cross-attention to the input embeddings and returns the augmented embeddings.
        :param x: Input embeddings to be augmented. [batches, seqLength]
        :param context: Context embeddings to be used for augmentation. [batches, k, seqLength]
        :param scores: similarity scores to be used for augmentation. [batches, k]
        :return: Augmented embeddings after applying cross-attention.
        """
        xInput = xInput.to(self.__device)
        context = context.to(self.__device)
        scores = scores.to(self.__device)
        scores = scores.unsqueeze(-1)

        x : torch.Tensor = xInput.unsqueeze(1)

        x = self.__patching(x)
        x, xMean, xStd = self.__normalization(x)
        x = self.__positionalEncoding(x)

        context = self.__patching(context * scores)
        context, _, _ = self.__normalization(context)
        context = self.__positionalEncoding(context)
        x = self.__crossAttention(x, context)

        x = self.__denormalization(x, xMean, xStd)

        x = self.__joinPatches(x)

        xAugmented = x.squeeze(1)

        #x = self.iterative_gaussian_smooth_to_match(x, xInput)

        xAugmented : torch.Tensor = self.__concat(xAugmented, xInput)

        return xAugmented

    def inference(
        self, xInput : torch.Tensor,
        context : torch.Tensor,
        scores : torch.Tensor
    ) -> torch.Tensor:
        """
        Inference pass for the embedding augmentation.
        This method applies cross-attention to the input embeddings and returns the augmented embeddings.
        :param x: Input embeddings to be augmented. [batches, seqLength]
        :param context: Context embeddings to be used for augmentation. [batches, k, seqLength]
        :param scores: similarity scores to be used for augmentation. [batches, k, 1]
        :return: Augmented embeddings after applying cross-attention.
        """
        output : torch.Tensor = None
        self.eval()
        with torch.no_grad():
            output = self.forward(xInput, context, scores)

        return output

if __name__ == "__main__":
    a = RagCrossAttention()
    x = torch.randn(5, 31)
    context = torch.randn(5, 3, 33)
    scores = torch.randn(5, 3, 1)
    y = a(x, context, scores)