import torch
from torch import nn

from .input_feature import InputFeature


class PassedPawns(InputFeature):
    """Passed-pawn inputs (rubicon-alea-v3 graft).

    One feature per passed pawn: 96 slots (48 own + 48 enemy; squares on
    ranks 2-7 oriented per perspective and file-mirrored with the king like
    HalfKA/threats/PawnPair). "Passed" = no enemy pawn on the same or
    adjacent files ahead of the pawn AND no own pawn directly ahead on the
    same file (a rear doubled pawn is not a passer).

    The passed property is the relational signal HalfKA cannot represent
    directly; interactions with blockers / kings / chains are learnable
    through the SFNNv13 pairwise-multiplied L1 against the HalfKA and
    PawnPair blocks, so v1 is deliberately square-only (no flags).
    Extraction lives in the C++ data loader ("PassedPawns").
    """

    HASH = 0x50535344  # "PSSD" — must match the engine-side hash (passed_pawns.h)
    FEATURE_NAME = "PassedPawns"
    INPUT_FEATURE_NAME = "PassedPawns"
    MAX_ACTIVE_FEATURES = 16  # all 16 pawns passed (theoretical bound)

    NUM_INPUTS = 96
    NUM_REAL_FEATURES = 96
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
        """Passed pawns carry no material (the pawn itself is already in the
        HalfKA PSQT), so PSQT columns are zero."""
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
