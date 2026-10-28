# Check the tail schedule: a PiecewiseLR restored at the second cycle's final step keeps its count and holds 0.004% of
# P's peak for the five tail epochs.
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/c2/nnue-pytorch")
import torch
from model.optimizers.rangerlite_wrapper import PiecewiseLR, parse_lr_pieces, lr_pieces_fraction
OLD = ("7628:0.1494:1:lin;87722:1:0.6:cos;38140:0.6:0.6:lin;95350:0.6:0.00016:cos;"
       "7628:0.00016:0.4:lin;11442:0.4:0.16:cos;19070:0.16:0.16:lin;57210:0.16:0.00016:cos")
NEW = OLD + ";19070:0.00016:0.00016:lin"
LR, GS, PEAK_P = 1.4142e-4, 324190, 5.6569e-4
def opt():
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=LR)
o = opt(); s = PiecewiseLR(o, parse_lr_pieces(OLD))
for _ in range(GS):
    o.step(); s.step()
print("end of the second cycle: last_epoch", s.last_epoch, f"lr {o.param_groups[0]['lr']:.4e}",
      f"= {o.param_groups[0]['lr'] / PEAK_P * 100:.4f}% of P's peak")
o2 = opt(); s2 = PiecewiseLR(o2, parse_lr_pieces(NEW))
o2.load_state_dict(o.state_dict()); s2.load_state_dict(s.state_dict())
new = parse_lr_pieces(NEW)
for target in (GS, GS + 1, GS + 3814, GS + 2 * 3814, GS + 4 * 3814, GS + 19070 - 1, GS + 19070):
    while s2.last_epoch < target:
        o2.step(); s2.step()
    lr = o2.param_groups[0]["lr"]
    print(target, f"lr {lr:.4e} = {lr / PEAK_P * 100:.4f}% of P's peak", f"expected {LR * lr_pieces_fraction(new, target):.4e}")
