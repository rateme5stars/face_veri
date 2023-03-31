import torch


class ArcFace(torch.nn.Module):

    def __init__(
        self,
        backbone: torch.nn.Module,
        n_identities: int,
        n_embeddings: int = 512,
        n_subclasses: int = 1,
        margin=0.5,
        scale_factor=64,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self._w = torch.nn.init.xavier_uniform_(
            torch.zeros(n_embeddings, n_identities, n_subclasses)
        )
        self._w_param = torch.nn.Parameter(self._w)
        self._margin = margin
        self._scale_factor = scale_factor
        self._softmax = torch.nn.Softmax(dim=0)

    def forward(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
    ) -> torch.Tensor:
        embedding_batch = self.backbone(input_batch)
        self._w = torch.nn.functional.normalize(
            self._w, dim=0
        ).to(self._w_param.device)
        self._w_param = torch.nn.Parameter(self._w)
        cosine_similarities_per_sub_classes = cosine_similarity(
            embedding_batch, self._w_param
        )
        cosine_similarities = cosine_similarities_per_sub_classes.max(
            dim=2
        ).values
        theta = cosine_similarities.acos()
        margins = target_batch * self._margin
        return self._softmax(
            (theta + margins).cos() * self._scale_factor
        )


def cosine_similarity(
    embedding_batch: torch.Tensor, w_param: torch.Tensor
) -> torch.Tensor:
    embedding_batch = embedding_batch[:, :, None, None]
    return (embedding_batch * w_param).sum(dim=1).clamp(-1, 1)
