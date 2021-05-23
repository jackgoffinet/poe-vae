# MNIST HALVES
# Standard ELBO unstructured Gaussian
time python main.py --unstructured-encoder=True --likelihood=bernoulli --objective=elbo --K=0 --epochs=500 --mll-freq=50
# Standard ELBO Gaussian PoE
time python main.py --likelihood=bernoulli --objective=elbo --K=0 --epochs=500 --mll-freq=50
# Standard ELBO EBM
time python main.py --dataset=mnist_pixels --likelihood=bernoulli --objective=iwae --K=1 --variational-strategy=loc_scale_ebm --variational-posterior=loc_scale_ebm --epochs=500 --mll-freq=50
# Standard ELBO 2-vMF
time python main.py --vmf-dim=2 --n-vmfs=10 --likelihood=bernoulli -objective=elbo --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=500 --mll-freq=50
# Standard ELBO 4-vMF
time python main.py --vmf-dim=4 --n-vmfs=5 --likelihood=bernoulli -objective=elbo --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=500 --mll-freq=50
# Standard ELBO 10-vMF
time python main.py --vmf-dim=10 --n-vmfs=2 --likelihood=bernoulli -objective=elbo --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=500 --mll-freq=50





# # # Standard ELBO Gaussian, 67740735, -73.2987
# time python main.py --likelihood=bernoulli --objective=elbo --K=0 --epochs=250 --mll-freq=50
# # 33381969, -103.96381
# time python main.py --dataset=mnist_pixels --likelihood=bernoulli --objective=elbo --K=0 --epochs=250 --mll-freq=50


# # IWAE ELBO Gaussian, 54377588, -72.76325
# time python main.py --likelihood=bernoulli --objective=iwae --epochs=250 --mll-freq=50
#
# time python main.py --dataset=mnist_pixels --likelihood=bernoulli --objective=iwae --epochs=250 --mll-freq=50


# # MVAE ELBO Gaussian, 53154631, -92.74215
# time python main.py --likelihood=bernoulli --objective=mvae_elbo --K=0 --epochs=250 --mll-freq=50
#
# time python main.py --dataset=mnist_pixels --likelihood=bernoulli --objective=mvae_elbo --epochs=250 --mll-freq=50



# # AR-ELBO Gaussian, 83138210, -71.54
# time python main.py --likelihood=bernoulli --objective=ar_elbo  --epochs=250 --mll-freq=50
# # 47772702, -112.43655
# time python main.py --dataset=mnist_pixels --ar_step_size=128 --likelihood=bernoulli --objective=ar_elbo  --epochs=250 --mll-freq=50



# # Standard Elbo EBM, 10007655, -73.35971
# time python main.py --likelihood=bernoulli --objective=iwae --K=1 --variational-strategy=loc_scale_ebm --variational-posterior=loc_scale_ebm --epochs=250 --mll-freq=50
#
# time python main.py --dataset=mnist_pixels --likelihood=bernoulli --objective=iwae --K=1 --variational-strategy=loc_scale_ebm --variational-posterior=loc_scale_ebm --epochs=250 --mll-freq=50



# # IWAE Elbo EBM, 82101143, -72.63761
# time python main.py --likelihood=bernoulli --objective=iwae --variational-strategy=loc_scale_ebm --variational-posterior=loc_scale_ebm --epochs=250 --mll-freq=50
#
# time python main.py --dataset=mnist_pixels --likelihood=bernoulli --objective=iwae --variational-strategy=loc_scale_ebm --variational-posterior=loc_scale_ebm --epochs=250 --mll-freq=50


# # MVAE ELBO EBM, 11483498, -90.36852
# time python main.py --likelihood=bernoulli --objective=mvae_elbo --K=0 --variational-strategy=loc_scale_ebm --variational-posterior=loc_scale_ebm --epochs=250 --mll-freq=50
#
# time python main.py --dataset=mnist_pixels --likelihood=bernoulli --objective=mvae_elbo --variational-strategy=loc_scale_ebm --variational-posterior=loc_scale_ebm --epochs=250 --mll-freq=50



# # vMF d=2, n=10, Standard ELBO, 24107545,   -75.91017
# time python main.py --vmf-dim=2 --n-vmfs=10 --likelihood=bernoulli -objective=elbo --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=250 --mll-freq=50

# # vMF d=2, n=10, IWAE ELBO, 99146269, -74.55919
# time python main.py --vmf-dim=2 --n-vmfs=10 --likelihood=bernoulli -objective=iwae --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=250 --mll-freq=50


# # vMF d=4, n=5, Standard ELBO, 15429103, -75.27895
# time python main.py --vmf-dim=4 --n-vmfs=5 --likelihood=bernoulli -objective=elbo --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=250 --mll-freq=50
# # 78711041, -102.99369
# time python main.py --dataset=mnist_pixels --vmf-dim=4 --n-vmfs=5 --likelihood=bernoulli -objective=elbo --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=250 --mll-freq=50


# # vMF d=4, n=5, IWAE ELBO, 90467827, -74.07313
# time python main.py --vmf-dim=4 --n-vmfs=5 --likelihood=bernoulli -objective=iwae --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=250 --mll-freq=50
#
# time python main.py --dataset=mnist_pixels --vmf-dim=4 --n-vmfs=5 --likelihood=bernoulli -objective=iwae --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=250 --mll-freq=50

#
# time python main.py --dataset=mnist_pixels --vmf-dim=4 --n-vmfs=5 --likelihood=bernoulli -objective=mvae_elbo --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=250 --mll-freq=50



# # vMF d=5, n=4, Standard ELBO, 95833839,
# time python main.py --vmf-dim=5 --n-vmfs=4 --likelihood=bernoulli -objective=elbo --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=250 --mll-freq=50

# # vMF d=5, n=4, IWAE ELBO, 70872563, -73.24464
# time python main.py --vmf-dim=5 --n-vmfs=4 --likelihood=bernoulli -objective=iwae --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=250 --mll-freq=50

# # vMF d=10, n=2, Standard ELBO, 88569113,
# time python main.py --vmf-dim=10 --n-vmfs=2 --likelihood=bernoulli -objective=elbo --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=250 --mll-freq=50

# # vMF d=10, n=2, IWAE ELBO, 63869981, -72.07257
# time python main.py --vmf-dim=10 --n-vmfs=2 --likelihood=bernoulli -objective=iwae --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=250 --mll-freq=50

# # vMF d=10, n=2, MVAE ELBO, 63869981, NaN loss!
# time python main.py --vmf-dim=10 --n-vmfs=2 --likelihood=bernoulli -objective=mvae_elbo --variational-strategy=vmf_poe --variational-posterior=vmf_product --prior=uniform_hyperspherical --epochs=250 --mll-freq=50
