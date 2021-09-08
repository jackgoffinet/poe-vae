# MNIST HALVES

# Some MNIST Pixels...
time python main.py --dataset=mnist_pixels --likelihood=bernoulli --objective=elbo --mll-freq=25 --epochs=500

##################
# STANDARD ELBOS #
##################
# unstructured Gaussian: 75803683, -126.520485
time python main.py --unstructured-encoder=True --likelihood=bernoulli --epochs=500 --mll-freq=50
# unstructured 4-vMF: 10583202, -106.37919
time python main.py --unstructured-encoder=True --vmf-dim=4 --n-vmfs=5 --likelihood=bernoulli --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=500 --mll-freq=50
# Gaussian MoE: 85712300, -124.09139
time python main.py --likelihood=bernoulli --variational-strategy=gaussian_moe --variational-posterior=diag_gaussian_mixture --objective=iwae --K=1 --epochs=500 --mll-freq=50
# Gaussian PoE: 57056622, -104.91585
time python main.py --likelihood=bernoulli --objective=elbo --mll-freq=50 --epochs=500
# EBM PoE: 88118629, -103.06324
python main.py --likelihood=bernoulli --objective=iwae --K=1 --variational-strategy=loc_scale_ebm --variational-posterior=loc_scale_ebm --epochs=500 --mll-freq=50
# 2-vMF PoE: 03267095, -102.39493
time python main.py --vmf-dim=2 --n-vmfs=10 --likelihood=bernoulli -objective=elbo --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=500 --mll-freq=50
# 4-vMF PoE: 94588653, -96.908295
time python main.py --vmf-dim=4 --n-vmfs=5 --likelihood=bernoulli -objective=elbo --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=500 --mll-freq=50
# 10-vMF PoE: 67597591, -99.63157
time python main.py --vmf-dim=10 --n-vmfs=2 --likelihood=bernoulli -objective=elbo --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=500 --mll-freq=50


##############
# MVAE ELBOS #
##############
# unstructured Gaussian: 19276282, -119.61201
time python main.py --objective=mvae_elbo --K=0 --unstructured-encoder=True --likelihood=bernoulli --epochs=500 --mll-freq=50
# unstructured 4-vMF: 92320889
time python main.py --objective=mvae_elbo --K=0 --unstructured-encoder=True --vmf-dim=4 --n-vmfs=5 --likelihood=bernoulli --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=500 --mll-freq=50
# Gaussian MoE:
time python main.py --objective=mvae_elbo --K=0 --likelihood=bernoulli --variational-strategy=gaussian_moe --variational-posterior=diag_gaussian_mixture --epochs=500 --mll-freq=50
# Gaussian PoE: 31396677, -128.54623
time python main.py --objective=mvae_elbo --K=0 --likelihood=bernoulli --mll-freq=50 --epochs=500
# EBM PoE: 89594472, -131.55229
python main.py --objective=mvae_elbo --K=0 --likelihood=bernoulli --variational-strategy=loc_scale_ebm --variational-posterior=loc_scale_ebm --epochs=500 --mll-freq=50
# 2-vMF PoE: 01420262, -101.36428
time python main.py --objective=mvae_elbo --K=0 --vmf-dim=2 --n-vmfs=10 --likelihood=bernoulli -objective=elbo --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=500 --mll-freq=50
# 4-vMF PoE: 89937340, -99.71951
time python main.py --objective=mvae_elbo --K=0 --vmf-dim=4 --n-vmfs=5 --likelihood=bernoulli -objective=elbo --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=500 --mll-freq=50
# 10-vMF PoE: 65750758, -95.45661
time python main.py --objective=mvae_elbo --K=0 --vmf-dim=10 --n-vmfs=2 --likelihood=bernoulli -objective=elbo --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=500 --mll-freq=50

printf '\7'

##################
# STANDARD ELBOS #
##################
# unstructured Gaussian: 13814567, -145.25188
time python main.py --train-m=0.9 --unstructured-encoder=True --likelihood=bernoulli --epochs=500 --mll-freq=50
# unstructured 4-vMF: 46759078, -131.52562
time python main.py --train-m=0.9 --unstructured-encoder=True --vmf-dim=4 --n-vmfs=5 --likelihood=bernoulli --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=500 --mll-freq=50
# Gaussian MoE: 26082480, -136.75092
time python main.py --train-m=0.9 --likelihood=bernoulli --variational-strategy=gaussian_moe --variational-posterior=diag_gaussian_mixture --objective=iwae --K=1 --epochs=500 --mll-freq=50
# Gaussian PoE: 95329650, -128.73068
time python main.py --train-m=0.9 --likelihood=bernoulli --objective=elbo --mll-freq=50 --epochs=500
# EBM PoE: 26653801, -133.36168
python main.py --train-m=0.9 --likelihood=bernoulli --objective=iwae --K=1 --variational-strategy=loc_scale_ebm --variational-posterior=loc_scale_ebm --epochs=500 --mll-freq=50
# 2-vMF PoE: 39705115, -123.20199
time python main.py --train-m=0.9 --vmf-dim=2 --n-vmfs=10 --likelihood=bernoulli -objective=elbo --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=500 --mll-freq=50
# 4-vMF PoE: 31026673, -122.624565
time python main.py --train-m=0.9 --vmf-dim=4 --n-vmfs=5 --likelihood=bernoulli -objective=elbo --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=500 --mll-freq=50
# 10-vMF PoE: 04297755, -121.56015
time python main.py --train-m=0.9 --vmf-dim=10 --n-vmfs=2 --likelihood=bernoulli -objective=elbo --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=500 --mll-freq=50


##############
# MVAE ELBOS #
##############
# unstructured Gaussian: 57287166, -141.05986
time python main.py --train-m=0.9 --objective=mvae_elbo --K=0 --unstructured-encoder=True --likelihood=bernoulli --epochs=500 --mll-freq=50
# unstructured 4-vMF: 34512509
time python main.py --train-m=0.9 --objective=mvae_elbo --K=0 --unstructured-encoder=True --vmf-dim=4 --n-vmfs=5 --likelihood=bernoulli --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=500 --mll-freq=50
# Gaussian MoE:
time python main.py --train-m=0.9 --objective=mvae_elbo --K=0 --likelihood=bernoulli --variational-strategy=gaussian_moe --variational-posterior=diag_gaussian_mixture --epochs=500 --mll-freq=50
# Gaussian PoE: 69669705, -142.09937
time python main.py --train-m=0.9 --objective=mvae_elbo --K=0 --likelihood=bernoulli --mll-freq=50 --epochs=500
# EBM PoE: 28129644, -148.82947
python main.py --train-m=0.9 --objective=mvae_elbo --K=0 --likelihood=bernoulli --variational-strategy=loc_scale_ebm --variational-posterior=loc_scale_ebm --epochs=500 --mll-freq=50
# 2-vMF PoE: 37858282, -123.53773
time python main.py --train-m=0.9 --objective=mvae_elbo --K=0 --vmf-dim=2 --n-vmfs=10 --likelihood=bernoulli -objective=elbo --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=500 --mll-freq=50
# 4-vMF PoE: 26375360, -118.8995
time python main.py --train-m=0.9 --objective=mvae_elbo --K=0 --vmf-dim=4 --n-vmfs=5 --likelihood=bernoulli -objective=elbo --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=500 --mll-freq=50
# 10-vMF PoE: 02450922, -122.63696
time python main.py --train-m=0.9 --objective=mvae_elbo --K=0 --vmf-dim=10 --n-vmfs=2 --likelihood=bernoulli -objective=elbo --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=500 --mll-freq=50

printf '\7'
