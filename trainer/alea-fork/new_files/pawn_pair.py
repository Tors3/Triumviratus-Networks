import torch
from torch import nn

from .input_feature import InputFeature


class PawnPair(InputFeature):
    """Pawn-pair inputs (rubicon-alea-v2 graft).

    Unordered pairs of pawns on the same or adjacent files. 96 pawn slots
    (48 own + 48 enemy; ranks 2-7 oriented per perspective, file-mirrored with
    the king like HalfKA/threats) -> index hi*(hi-1)/2 + lo = 4560 features.
    Spec identical to Stormphrax/Viridithas/Pawnocchio. Captures phalanx/
    chains/doubled/isolated pawns as learned patterns - geometry the threat
    features cannot see. Extraction lives in the C++ data loader ("PawnPair").
    """

    HASH = 0x50414952  # "PAIR" — must match the engine-side hash when porting
    FEATURE_NAME = "PawnPair"
    INPUT_FEATURE_NAME = "PawnPair"
    MAX_ACTIVE_FEATURES = 120  # C(16,2): all pawns inside one file band

    NUM_INPUTS = 4560
    NUM_REAL_FEATURES = 4560
    EXPORT_WEIGHT_DTYPE = torch.int8

    def __init__(self, num_outputs: int):
        super().__init__()

        self.num_outputs = num_outputs
        self.weight = nn.Parameter(
            torch.empty(self.NUM_INPUTS, num_outputs, dtype=torch.float32)
        )

        self.reset_parameters()

    def merged_weight(self) -> torch.Tensor:
        return self.weight

    @torch.no_grad()
    def coalesce(self) -> None:
        pass  # no virtual weights

    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        pass  # no virtual weights

    @torch.no_grad()
    def init_weights(self, num_psqt_buckets: int, nnue2score: float) -> None:
        """Pawn pairs carry no material, so PSQT columns are zero."""
        L1 = self.num_outputs - num_psqt_buckets
        for i in range(num_psqt_buckets):
            self.weight[:, L1 + i] = 0.0

    @torch.no_grad()
    def get_export_weights(self) -> torch.Tensor:
        return self.weight.data.clone()

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        self.weight.data.copy_(export_weight)

    def clip_weights(self, quantization) -> None:
        """int8 export like threats -> same quantization-safe clamp range."""
        self.weight.data.clamp_(
            quantization.min_threat_weight, quantization.max_threat_weight
        )
