import torch
from torch import nn

from .input_feature import InputFeature


class CandidatePassers(InputFeature):
    """Candidate-passer inputs (rubicon-alea v4-slot graft).

    One feature per candidate passed pawn: 96 slots (48 own + 48 enemy;
    squares oriented per perspective and file-mirrored with the king like
    HalfKA/threats/PawnPair/PassedPawns). A pawn of color C on sq is a
    candidate passer iff (1) it is NOT already passed (the PassedPawns block
    covers those), (2) its file is semi-open ahead (no pawn of either color
    on the same file strictly ahead), and (3) majority count: own pawns on
    adjacent files behind-or-level >= enemy pawns on adjacent files strictly
    ahead (helpers vs sentinels).

    Index math identical to PassedPawns (pawn_id, o-8): candidates are
    pawns, oriented ranks 2-7 (in fact 2-6).

    Deliberately square-only, pawn-event-only: interactions are learnable
    through the SFNNv13 pairwise-multiplied L1 against the HalfKA and
    PawnPair blocks. Extraction lives in the C++ data loader
    ("CandidatePassers").
    """

    HASH = 0x43414E44  # "CAND" — must match the engine-side hash (candidate_passers.h)
    FEATURE_NAME = "CandidatePassers"
    INPUT_FEATURE_NAME = "CandidatePassers"
    MAX_ACTIVE_FEATURES = 16  # every pawn a candidate (theoretical bound)

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
        """A candidate-passer feature carries no material, so PSQT columns are zero."""
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
