# vMF models

# circles
# 67044527
# python main.py -q vmf_product --vmf-dim 2 -v vmf_poe -p uniform_hypershperical --epochs 2000
# 91139835
python main.py -q vmf_product --vmf-dim 2 -v vmf_poe -p uniform_hypershperical -x mnist_mcar --epochs 5000
# spheres
# 67241136
# python main.py -q vmf_product --vmf-dim 3 -v vmf_poe -p uniform_hypershperical --epochs 2000
# 91336444
python main.py -q vmf_product --vmf-dim 3 -v vmf_poe -p uniform_hypershperical -x mnist_mcar --epochs 5000
# 4-sphere
# 67634354
# python main.py -q vmf_product --vmf-dim 5 -v vmf_poe -p uniform_hypershperical --epochs 2000
# 91729662
python main.py -q vmf_product --vmf-dim 5 -v vmf_poe -p uniform_hypershperical -x mnist_mcar --epochs 5000
# 5-sphere
# 67830963
# python main.py -q vmf_product --vmf-dim 6 -v vmf_poe -p uniform_hypershperical --epochs 2000
# 91926271
python main.py -q vmf_product --vmf-dim 6 -v vmf_poe -p uniform_hypershperical -x mnist_mcar --epochs 5000

# Gaussian PoE models
# 77089584
# python main.py --epochs 2000
# 77591932
python main.py -x mnist_mcar --epochs 5000

# MMVAE
# 68548139
# python main.py -v gaussian_moe -q diag_gaussian_mixture -o mmvae_quadratic --epochs 2000

# IWAE Gaussian PoE.
# 13474357
# python main.py -o dreg_iwae --epochs 2000
# 54994305
python main.py -o dreg_iwae -x mnist_mcar --epochs 5000


# EBM PoE
# 46032319
python main.py -q ebm -v ebm -o iwae --K 1 --ebm-samples 3 --epochs 2000 --theta-dim 20
# 16284639
python main.py -q ebm -v ebm -o iwae --K 1 --ebm-samples 3 --epochs 5000 --theta-dim 20 -x mnist_mcar
