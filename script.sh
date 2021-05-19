# Does AR-ELBO improve things?

# # Standard ELBO Gaussian, 67740735, -73.2987
# time python main.py --likelihood=bernoulli --objective=elbo --K=0 --epochs=250 --save-model=True --mll-freq=50

# # MVAE ELBO Gaussian, 53154631, -92.5031
# time python main.py --likelihood=bernoulli --objective=mvae_elbo --K=0 --epochs=250 --save-model=True --mll-freq=50

# # AR-ELBO Gaussian, 83138210, -71.54
# time python main.py --likelihood=bernoulli --objective=ar_elbo  --epochs=250 --save-model=True --mll-freq=50


# # Standard Elbo EBM, 10007655, -73.35971
# time python main.py --likelihood=bernoulli --objective=iwae --K=1 --variational-strategy=loc_scale_ebm --variational-posterior=loc_scale_ebm --epochs=250 --save-model=True --mll-freq=50

# # MVAE ELBO EBM, 11483498, -93.72
# time python main.py --likelihood=bernoulli --objective=mvae_elbo --K=0 --variational-strategy=loc_scale_ebm --variational-posterior=loc_scale_ebm --epochs=250 --save-model=True --mll-freq=50

# # AR-ELBO EBM, 12750484, -71.11
# time python main.py --likelihood=bernoulli --objective=ar_elbo --K=0 --variational-strategy=loc_scale_ebm --variational-posterior=loc_scale_ebm --epochs=250 --save-model=True --mll-freq=50


# vMF d=2, n=10, Standard ELBO, 24107545
time python main.py --vmf-dim=2 --n-vmfs=10 --likelihood=bernoulli -objective=elbo --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=250 --save-model=True --mll-freq=50

# vMF d=2, n=10, IWAE ELBO, 99146269, -68.48391
time python main.py --vmf-dim=2 --n-vmfs=10 --likelihood=bernoulli -objective=iwae --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=250 --save-model=True --mll-freq=50

# vMF d=4, n=5, Standard ELBO, 15429103, -73.840546
time python main.py --vmf-dim=4 --n-vmfs=5 --likelihood=bernoulli -objective=elbo --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=250 --save-model=True --mll-freq=50

# vMF d=4, n=5, IWAE ELBO, 90467827,  -72.0801
time python main.py --vmf-dim=4 --n-vmfs=5 --likelihood=bernoulli -objective=iwae --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=250 --save-model=True --mll-freq=50

# vMF d=5, n=4, Standard ELBO, 95833839, -75.29439
time python main.py --vmf-dim=5 --n-vmfs=4 --likelihood=bernoulli -objective=elbo --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=250 --save-model=True --mll-freq=50

# vMF d=5, n=4, IWAE ELBO, 70872563, -72.45864
time python main.py --vmf-dim=5 --n-vmfs=4 --likelihood=bernoulli -objective=iwae --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=250 --save-model=True --mll-freq=50

# vMF d=10, n=2, Standard ELBO, 88569113, -73.48438
time python main.py --vmf-dim=10 --n-vmfs=2 --likelihood=bernoulli -objective=elbo --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=250 --save-model=True --mll-freq=50

# vMF d=10, n=2, IWAE ELBO, 63869981, -72.83443
time python main.py --vmf-dim=10 --n-vmfs=2 --likelihood=bernoulli -objective=iwae --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=250 --save-model=True --mll-freq=50


# IWAE ELBO Gaussian, 54377588, -72.76325
time python main.py --likelihood=bernoulli --objective=iwae --epochs=250 --save-model=True --mll-freq=50

# IWAE Elbo EBM, 82101143, -72.63761
time python main.py --likelihood=bernoulli --objective=iwae --variational-strategy=loc_scale_ebm --variational-posterior=loc_scale_ebm --epochs=250 --save-model=True --mll-freq=50
