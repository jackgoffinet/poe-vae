# 2D MNIST Halves toy example.

time python main.py --latent-dim=2 --likelihood=bernoulli --objective=elbo --epochs=100 --save-model=True

time python main.py --latent-dim=2 --likelihood=bernoulli --objective=iwae --K=1 --variational-strategy=loc_scale_ebm --variational-posterior=loc_scale_ebm --epochs=100 --save-model=True
