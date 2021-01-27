
# 2-spheres
# 99088720
python main.py -q vmf_product --vmf-dim 3 -v vmf_poe -p uniform_hypershperical --epochs 1000
# 13488496
python main.py -q vmf_product --vmf-dim 3 -v vmf_poe -p uniform_hypershperical -x mnist_mcar --epochs 2000

# 5-sphere
# 99678547
python main.py -q vmf_product --vmf-dim 6 -v vmf_poe -p uniform_hypershperical --epochs 1000
# 14078323
python main.py -q vmf_product --vmf-dim 6 -v vmf_poe -p uniform_hypershperical -x mnist_mcar --epochs 2000

# Gaussian PoE models
# 93282000
python main.py --epochs 1000
# 78321648
python main.py -x mnist_mcar --epochs 2000

# MMVAE
# 66259403
python main.py -v gaussian_moe -q diag_gaussian_mixture -o mmvae_quadratic --epochs 1000

# IWAE Gaussian PoE.
# 45031893
python main.py -o dreg_iwae --epochs 1000
# 56671221
python main.py -o dreg_iwae -x mnist_mcar --epochs 2000


# EBM PoE
# 02655425
python main.py -q ebm -v ebm -o iwae --K 1 --ebm-samples 5 --epochs 1000 --theta-dim 20
# 72907745
python main.py -q ebm -v ebm -o iwae --K 1 --ebm-samples 5 --epochs 2000 --theta-dim 20 -x mnist_mcar
