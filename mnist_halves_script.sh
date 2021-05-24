# MNIST HALVES

##################
# STANDARD ELBOS #
##################
# unstructured Gaussian: 97430565
time python main.py --unstructured-encoder=True --likelihood=bernoulli --epochs=500 --mll-freq=50
# unstructured 4-vMF: 31292580
time python main.py --unstructured-encoder=True --vmf-dim=4 --n-vmfs=5 --likelihood=bernoulli --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=500 --mll-freq=50
# Gaussian MoE: 08518830
time python main.py --likelihood=bernoulli --variational-strategy=gaussian_moe --variational-posterior=diag_gaussian_mixture --objective=iwae --K=1 --epochs=500 --mll-freq=50
# Gaussian PoE: 78814576
time python main.py --likelihood=bernoulli --objective=elbo --mll-freq=50 --epochs=500
# EBM PoE:
python main.py --likelihood=bernoulli --objective=iwae --K=1 --variational-strategy=loc_scale_ebm --variational-posterior=loc_scale_ebm --epochs=500 --mll-freq=50
# 2-vMF PoE: 24107545
time python main.py --vmf-dim=2 --n-vmfs=10 --likelihood=bernoulli -objective=elbo --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=500 --mll-freq=50
# 4-vMF PoE: 15429103
time python main.py --vmf-dim=4 --n-vmfs=5 --likelihood=bernoulli -objective=elbo --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=500 --mll-freq=50
# 10-vMF PoE:
time python main.py --vmf-dim=10 --n-vmfs=2 --likelihood=bernoulli -objective=elbo --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=500 --mll-freq=50


#############
# MVAE ELBO #
#############
# EBM PoE: 82101143
python main.py --likelihood=bernoulli --objective=mvae_elbo --K=0 --variational-strategy=loc_scale_ebm --variational-posterior=loc_scale_ebm --epochs=500 --mll-freq=50


##############
# IWAE ELBOS #
##############
# EBM
time python main.py --likelihood=bernoulli --objective=iwae --variational-strategy=loc_scale_ebm --variational-posterior=loc_scale_ebm --epochs=500 --mll-freq=50


###################
# DReG IWAE ELBOS #
###################
# unstructured Gaussian
time python main.py --unstructured-encoder=True --likelihood=bernoulli --objective=dreg_iwae --epochs=500 --mll-freq=50
# unstructured 4-vMF
time python main.py --unstructured-encoder=True --vmf-dim=4 --n-vmfs=5 --likelihood=bernoulli -objective=dreg_iwae --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=500 --mll-freq=50
# Gaussian MoE
time python main.py --likelihood=bernoulli --variational-strategy=gaussian_moe --variational-posterior=diag_gaussian_mixture --objective=dreg_iwae
# Gaussian PoE
time python main.py --likelihood=bernoulli --objective=dreg_iwae --epochs=500 --mll-freq=50
# # EBM
# time python main.py --likelihood=bernoulli --objective=dreg_iwae --variational-strategy=loc_scale_ebm --variational-posterior=loc_scale_ebm --epochs=500 --mll-freq=50
# 2-vMF PoE
time python main.py --vmf-dim=2 --n-vmfs=10 --likelihood=bernoulli -objective=dreg_iwae --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=500 --mll-freq=50
# 4-vMF PoE
time python main.py --vmf-dim=4 --n-vmfs=5 --likelihood=bernoulli -objective=dreg_iwae --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=500 --mll-freq=50
# 10-vMF PoE
time python main.py --vmf-dim=10 --n-vmfs=2 --likelihood=bernoulli -objective=dreg_iwae --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=500 --mll-freq=50

###################
# MMVAE Objective #
###################
# Gaussian MoE
time python main.py --likelihood=bernoulli --variational-strategy=gaussian_moe --variational-posterior=diag_gaussian_mixture --objective=mmvae_elbo
