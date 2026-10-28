#!/bin/bash
# 26/10/2026: at the end of P (epoch 549, max_epochs=550) wait for the trainer to exit, then export the final last.ckpt
# and write the md5 of the checkpoint for the download. After the restart of 26/10 the run writes to version_6: the
# newest last.ckpt is taken, as export_only.sh does.
until tail -c 6000 /root/train_c2.log | tr "\r" "\n" | grep -q -E "max_epochs=550|Epoch 549 \(Train\): \[train_loss_epoch"; do sleep 60; done
echo "END SEEN $(date +%H:%M:%S)"
for i in $(seq 1 30); do
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break
  sleep 30
done
echo "GPU FREE OR TIMEOUT $(date +%H:%M:%S)"
sleep 30
/root/valfix/export_only.sh
L=$(find /root/c2/run/P -name last.ckpt -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
md5sum "$L" > /root/valfix/final_ckpt.md5
echo "FINAL DONE $(date +%H:%M:%S)"
