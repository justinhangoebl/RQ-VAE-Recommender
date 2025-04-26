import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sklearn.cluster import KMeans

from schemas.quantization import QuantizeOutput

class Quantization(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        codebook_size: int,
        commitment_weight: float = 0.25,
        do_kmeans_init: bool = True,
        sim_vq: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = latent_dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.do_kmeans_init = do_kmeans_init
        self.kmeans_initted = False

        self.embedding = nn.Embedding(codebook_size, latent_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim, bias=False) if sim_vq else nn.Identity(),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)

    @torch.no_grad
    def _kmeans_init(self, x: Tensor):
        x_np = x.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.codebook_size, n_init="auto", max_iter=50)
        kmeans.fit(x_np)
        centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, device=x.device)
        self.embedding.weight.data.copy_(centroids)
        self.kmeans_initted = True

    def get_item_embeddings(self, item_ids) -> Tensor:
        return self.out_proj(self.embedding(item_ids))
    
    def get_codebook(self) -> Tensor:
        return self.out_proj(self.embedding.weight)

    def forward(self, x: Tensor) -> QuantizeOutput:
        if self.do_kmeans_init and not self.kmeans_initted:
            self._kmeans_init(x)

        codebook = self.get_codebook()

        # Compute L2 distances
        x_flat = x.view(-1, self.embed_dim)
        dists = torch.cdist(x_flat, codebook, p=2)
        ids = dists.argmin(dim=1)

        quantized = codebook[ids]
        quantized = x + (quantized - x).detach()  # STE

        # Compute commitment loss
        loss = F.mse_loss(x, quantized.detach()) + self.commitment_weight * F.mse_loss(quantized, x.detach())

        return QuantizeOutput(
            embeddings=quantized.view_as(x),
            ids=ids.view(x.shape[:-1]),
            loss=loss,
        )