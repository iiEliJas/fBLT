import copy
import json
import os

BASE = "configs/ablations/baseline_p4.json"
OUT = "configs/ablations"

base = json.load(open(BASE))


def cfg(tag, phase, mutate):
    c = copy.deepcopy(base)
    c["tag"] = tag
    c["phase"] = phase
    mutate(c)
    path = os.path.join(OUT, f"{tag}.json")
    with open(path, "w") as f:
        json.dump(c, f, indent=2)
        f.write("\n")
    print(path)


def set_ngram(sizes, vocab):
    def m(c):
        c["model"]["encoder"]["ngram"]["enabled"] = len(sizes) > 0
        c["model"]["encoder"]["ngram"]["sizes"] = sizes
        c["model"]["encoder"]["ngram"]["vocab_size"] = vocab
    return m


def set_placement(placement, pooling_init=True):
    def m(c):
        c["cross_attention"]["placement"] = placement
        c["cross_attention"]["pooling_init"] = pooling_init
    return m


def set_depth(enc_layers, dec_layers):
    def m(c):
        c["model"]["encoder"]["num_layers"] = enc_layers
        c["model"]["decoder"]["num_layers"] = dec_layers
    return m


# ---------------------------------------------------------------- 
# ngram
# vocab capped at 200k (dense SGD cost), see docs/ablations.md deviations.
cfg("ngram_none", "5.1_ngram", lambda c: (
    c["model"]["encoder"]["ngram"].update({"enabled": False}),
    c["train"].update({"ent_warmup_steps": 800})))
for sizes, name in ([[3, 4, 5], "s345"], [[6, 7, 8], "s678"],
                    [[3, 4, 5, 6, 7, 8], "all"]):
    for v in ([50000, 100000, 200000] if name != "all" else [50000, 100000]):
        cfg(f"ngram_{name}_v{v // 1000}k", "5.1_ngram",
            set_ngram(sizes, v))

# ----------------------------------------------------------------
# xattn
# "both" + pooling MEAN is the baseline itself
cfg("xattn_none", "5.2_xattn", set_placement("none"))
cfg("xattn_encoder_all", "5.2_xattn", set_placement("encoder_all"))
cfg("xattn_encoder_last", "5.2_xattn", set_placement("encoder_last"))
cfg("xattn_decoder_all", "5.2_xattn", set_placement("decoder_all"))
cfg("xattn_decoder_first", "5.2_xattn", set_placement("decoder_first"))
cfg("xattn_encoder_all_poolmax", "5.2_xattn", set_placement("encoder_all", False))
cfg("xattn_encoder_last_poolmax", "5.2_xattn", set_placement("encoder_last", False))
cfg("xattn_both_poolmax", "5.2_xattn", set_placement("both", False))

# ----------------------------------------------------------------
# depth
cfg("depth_enc1_dec9", "5.3_depth", set_depth(1, 9))
cfg("depth_enc3_dec7", "5.3_depth", set_depth(3, 7))
cfg("depth_enc5_dec5", "5.3_depth", set_depth(5, 5))
cfg("depth_enc9_dec1", "5.3_depth", set_depth(9, 1))

# ----------------------------------------------------------------
# patch
# thresholds from post-warmup entropy stats (mean 4.49, sd 0.77);
# P(fire) = 1/target_patch_len -> thr = mean + z*sd
cfg("patch_t4", "5.5_patch", lambda c: (
    c["patcher"].update({"rule": "global", "threshold_global": 5.01}),))
cfg("patch_t6", "5.5_patch", lambda c: (
    c["patcher"].update({"rule": "global", "threshold_global": 5.24}),))
cfg("patch_t8", "5.5_patch", lambda c: (
    c["patcher"].update({"rule": "global", "threshold_global": 5.37}),))
cfg("patch_whitespace", "5.5_patch", lambda c: (
    c["patcher"].update({"rule": "whitespace", "max_patch_length": 32}),))
cfg("patch_strided4", "5.5_patch", lambda c: (
    c["patcher"].update({"rule": "fixed:4"}),))

print("done")
