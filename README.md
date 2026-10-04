# GreenSplit – code for "An Adaptive Spatio-Temporal Framework for Carbon-Aware and Privacy-Preserving DNN Partitioning"

    pip install numpy scipy matplotlib pandas torch torchvision mlxtend scikit-learn
    python greensplit/experiments.py          # main simulator: Tables 3-9, 11, Figs 1,3,5-11 -> results/results.json
    python greensplit/extra_analysis.py       # throttle analysis, energy numbers, decision time, Fig 4 (synthetic)
    python greensplit/privacy_ttest.py        # synthetic cosine-similarity t-test
    python greensplit/thermal_sensitivity.py  # Sec. 6.7: thermal/battery/DVFS sensitivity (Fig 13)
    python greensplit/awc_study.py            # Sec. 6.9: AWC on unseen randomised trajectories (Fig 14)
    python greensplit/privacy_real.py         # Sec. 6.5(E): small MobileNetV2 trained on real MNIST digits (mlxtend), real-activation attacks (Fig 12, ~5 min)
    git clone --depth 1 https://github.com/EliSchwartz/imagenet-sample-images data/imagenet_sample   # 1000 real ImageNet val images (not redistributed here)
    python greensplit/privacy_imagenet.py     # Sec. 6.5(D): PRETRAINED ImageNet MobileNetV2 (data/, BSD-2) on 1000 real images (Fig 12b, ~20 min, 8 GB RAM)
    python paper/build_main.py <old main.tex> <new main.tex>   # regenerates every number in the manuscript and asserts the claims

`GreenSplit_torch_check.ipynb` compares analytic tensor sizes/MACs with torchvision and re-runs the scripts above.
`original/` holds the originally supplied scripts, unmodified. The AWC studies use the reduced model of original/awc_model_core.py and are
not comparable with greensplit/sim.py. Decision timing (results/extra.json) is machine dependent.

`data/` holds the pretrained MobileNetV2 checkpoint and model class from github.com/ericsun99/MobileNet-V2-Pytorch (BSD-2 licence, see data/MobileNetV2_LICENSE).
The ImageNet images are from github.com/EliSchwartz/imagenet-sample-images and are not redistributed; check their terms before reuse.
