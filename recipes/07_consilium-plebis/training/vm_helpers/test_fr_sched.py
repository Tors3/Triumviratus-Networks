# Check the schedule of the restart from F's end: a PiecewiseLR restored at F's final step keeps its count, rises to
# 0.5% of P's peak in one epoch and falls back to 0.004% in ten.
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/c2/nnue-pytorch")
import torch
from model.optimizers.rangerlite_wrapper import PiecewiseLR, parse_lr_pieces, lr_pieces_fraction
OLD = "7628:0.1494:1:lin;87722:1:0.6:cos;38140:0.6:0.6:lin;95350:0.6:0.00016:cos"
NEW = OLD + ";3814:0.00016:0.02:lin;38140:0.02:0.00016:cos"
LR, GS, PEAK_P = 1.4142e-4, 228840, 5.6569e-4
def opt():
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=LR)
o = opt(); s = PiecewiseLR(o, parse_lr_pieces(OLD))
for _ in range(GS):
    o.step(); s.step()
print("end of F: last_epoch", s.last_epoch, f"lr {o.param_groups[0]['lr']:.4e}")
o2 = opt(); s2 = PiecewiseLR(o2, parse_lr_pieces(NEW))
o2.load_state_dict(o.state_dict()); s2.load_state_dict(s.state_dict())
new = parse_lr_pieces(NEW)
for k in range(0, 12):
    target = GS + k * 3814 - (1 if k == 11 else 0)
    while s2.last_epoch < target:
        o2.step(); s2.step()
    lr = o2.param_groups[0]["lr"]
    print(f"+{k:2d} epochs (step {target}): lr {lr:.4e} = {lr / PEAK_P * 100:6.3f}% of P's peak",
          f"expected {LR * lr_pieces_fraction(new, target):.4e}")
