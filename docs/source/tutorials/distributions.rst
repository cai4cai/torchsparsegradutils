Distributions Tutorial
======================

This tutorial covers the sparse multivariate normal distributions in torchsparsegradutils.

Basic Usage
-----------

Creating Sparse Multivariate Normal Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from torchsparsegradutils.distributions import SparseMultivariateNormal

   # Create a simple tridiagonal precision matrix
   n = 10

   # Main diagonal
   i_main = torch.arange(n)
   j_main = torch.arange(n)
   v_main = torch.full((n,), 2.0)

   # Super and sub diagonals
   i_super = torch.arange(n-1)
   j_super = torch.arange(1, n)
   v_super = torch.full((n-1,), -0.5)

   i_sub = torch.arange(1, n)
   j_sub = torch.arange(n-1)
   v_sub = torch.full((n-1,), -0.5)

   # Combine indices and values
   indices = torch.stack([
       torch.cat([i_main, i_super, i_sub]),
       torch.cat([j_main, j_super, j_sub])
   ])
   values = torch.cat([v_main, v_super, v_sub])

   precision = torch.sparse_coo_tensor(indices, values, (n, n))
   mean = torch.zeros(n)

   # Create distribution
   dist = SparseMultivariateNormal(
       loc=mean,
       precision_matrix=precision,
       param='precision_LDL'
   )

   print(f"Distribution event shape: {dist.event_shape}")
   print(f"Distribution batch shape: {dist.batch_shape}")

Sampling from the Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Sample from the distribution
   num_samples = 1000
   samples = dist.sample((num_samples,))
   print(f"Samples shape: {samples.shape}")  # Should be (1000, 10)

   # Compute sample statistics
   sample_mean = samples.mean(dim=0)
   sample_cov = torch.cov(samples.T)

   print(f"Sample mean: {sample_mean}")
   print(f"Sample covariance diagonal: {torch.diag(sample_cov)}")

Log Probability Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compute log probabilities
   log_probs = dist.log_prob(samples)
   print(f"Log probabilities shape: {log_probs.shape}")  # Should be (1000,)
   print(f"Mean log probability: {log_probs.mean()}")

Different Parameterizations
----------------------------

Precision Matrix Parameterizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # LDL parameterization (numerically stable)
   dist_ldl = SparseMultivariateNormal(
       loc=mean,
       precision_matrix=precision,
       param='precision_LDL'
   )

   # LL^T parameterization
   dist_ll = SparseMultivariateNormal(
       loc=mean,
       precision_matrix=precision,
       param='precision_LL'
   )

   # Compare samples
   samples_ldl = dist_ldl.sample((100,))
   samples_ll = dist_ll.sample((100,))

   print(f"LDL samples std: {samples_ldl.std(dim=0).mean()}")
   print(f"LL samples std: {samples_ll.std(dim=0).mean()}")

Covariance Matrix Parameterization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Convert precision to covariance (for demonstration)
   # Note: This is expensive for large matrices!
   covariance = torch.inverse(precision.to_dense()).to_sparse()

   dist_cov = SparseMultivariateNormal(
       loc=mean,
       covariance_matrix=covariance,
       param='covariance_LL'
   )

   samples_cov = dist_cov.sample((100,))

Gradient Computation
--------------------

Reparameterized Sampling
~~~~~~~~~~~~~~~~~~~~~~~~

For gradient-based optimization, use `rsample`:

.. code-block:: python

   # Enable gradients
   mean_param = torch.zeros(n, requires_grad=True)
   precision_values = precision.values().clone().requires_grad_(True)

   # Reconstruct precision matrix with gradients
   precision_grad = torch.sparse_coo_tensor(
       precision.indices(),
       precision_values,
       precision.shape
   )

   dist_grad = SparseMultivariateNormal(
       loc=mean_param,
       precision_matrix=precision_grad,
       param='precision_LDL'
   )

   # Reparameterized sampling
   samples_grad = dist_grad.rsample((50,))

   # Compute some loss
   loss = samples_grad.mean()
   loss.backward()

   print(f"Mean gradient: {mean_param.grad}")
   print(f"Precision gradient norm: {precision_values.grad.norm()}")

Parameter Learning Example
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Example: Learn parameters from data
   true_mean = torch.randn(n)
   true_samples = SparseMultivariateNormal(
       loc=true_mean,
       precision_matrix=precision,
       param='precision_LDL'
   ).sample((500,))

   # Initialize learnable parameters
   learned_mean = torch.zeros(n, requires_grad=True)
   learned_prec_values = torch.ones_like(precision.values(), requires_grad=True)

   optimizer = torch.optim.Adam([learned_mean, learned_prec_values], lr=0.01)

   for epoch in range(100):
       optimizer.zero_grad()

       # Create distribution with current parameters
       learned_precision = torch.sparse_coo_tensor(
           precision.indices(),
           learned_prec_values,
           precision.shape
       )

       learned_dist = SparseMultivariateNormal(
           loc=learned_mean,
           precision_matrix=learned_precision,
           param='precision_LDL'
       )

       # Negative log likelihood loss
       nll = -learned_dist.log_prob(true_samples).mean()
       nll.backward()

       optimizer.step()

       if epoch % 20 == 0:
           print(f"Epoch {epoch}, NLL: {nll.item():.4f}")

   print(f"True mean: {true_mean}")
   print(f"Learned mean: {learned_mean.detach()}")

Batched Operations
------------------

Working with Batch Dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create batched distributions
   batch_size = 5

   # Different means for each batch element
   batch_means = torch.randn(batch_size, n)

   # Same precision matrix for all batch elements
   batch_dist = SparseMultivariateNormal(
       loc=batch_means,
       precision_matrix=precision,  # Broadcasted
       param='precision_LDL'
   )

   print(f"Batch distribution shape: {batch_dist.batch_shape}")
   print(f"Batch distribution event shape: {batch_dist.event_shape}")

   # Sample from batched distribution
   batch_samples = batch_dist.sample((100,))  # Shape: (100, 5, 10)
   print(f"Batch samples shape: {batch_samples.shape}")

   # Log probabilities
   batch_log_probs = batch_dist.log_prob(batch_samples)
   print(f"Batch log probs shape: {batch_log_probs.shape}")  # (100, 5)

Advanced Applications
---------------------

Gaussian Process Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Sparse GP example using precision matrix
   def create_gp_precision(x_train, lengthscale=1.0, noise=0.1):
       """Create sparse precision matrix for GP."""
       n = len(x_train)

       # Create sparse kernel matrix (example: only nearby points)
       indices = []
       values = []

       for i in range(n):
           for j in range(max(0, i-5), min(n, i+6)):  # Local connections
               dist = torch.norm(x_train[i] - x_train[j])
               kernel_val = torch.exp(-0.5 * (dist / lengthscale) ** 2)
               if i == j:
                   kernel_val += noise  # Add noise to diagonal
               if kernel_val > 1e-6:  # Threshold for sparsity
                   indices.append([i, j])
                   values.append(kernel_val)

       indices = torch.tensor(indices).T
       values = torch.tensor(values)
       covariance = torch.sparse_coo_tensor(indices, values, (n, n))

       # Convert to precision (approximate)
       return torch.sparse_coo_tensor(
           covariance.indices(),
           1.0 / (covariance.values() + 1e-6),
           covariance.shape
       )

   # Example usage
   x_train = torch.randn(50, 2)  # 50 training points in 2D
   y_train = torch.randn(50)

   gp_precision = create_gp_precision(x_train)
   gp_dist = SparseMultivariateNormal(
       loc=torch.zeros(50),
       precision_matrix=gp_precision,
       param='precision_LDL'
   )

   # Sample from GP prior
   gp_samples = gp_dist.sample((10,))
   print(f"GP samples shape: {gp_samples.shape}")

Sparse VAE Latent Space
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Variational Autoencoder with sparse prior
   class SparseVAE:
       def __init__(self, latent_dim):
           # Create sparse precision for structured latent space
           # Example: chain graph structure
           indices = []
           values = []

           # Chain connections
           for i in range(latent_dim - 1):
               indices.extend([[i, i+1], [i+1, i]])
               values.extend([1.0, 1.0])

           # Self-connections
           for i in range(latent_dim):
               indices.append([i, i])
               values.append(2.0)

           indices = torch.tensor(indices).T
           values = torch.tensor(values)
           self.prior_precision = torch.sparse_coo_tensor(
               indices, values, (latent_dim, latent_dim)
           )

           self.prior = SparseMultivariateNormal(
               loc=torch.zeros(latent_dim),
               precision_matrix=self.prior_precision,
               param='precision_LDL'
           )

       def kl_divergence(self, posterior_mean, posterior_precision):
           """Compute KL divergence between posterior and sparse prior."""
           posterior = SparseMultivariateNormal(
               loc=posterior_mean,
               precision_matrix=posterior_precision,
               param='precision_LDL'
           )

           # Sample-based KL estimation
           z_samples = posterior.rsample((100,))
           kl = (posterior.log_prob(z_samples) - self.prior.log_prob(z_samples)).mean()
           return kl

   # Usage
   vae = SparseVAE(latent_dim=20)

   # Example posterior parameters
   post_mean = torch.randn(20, requires_grad=True)
   post_prec_values = torch.ones_like(vae.prior_precision.values(), requires_grad=True)
   post_precision = torch.sparse_coo_tensor(
       vae.prior_precision.indices(),
       post_prec_values,
       vae.prior_precision.shape
   )

   kl_loss = vae.kl_divergence(post_mean, post_precision)
   print(f"KL divergence: {kl_loss}")

Performance Tips
----------------

1. **Choose appropriate parameterization**:
   - Use `precision_LDL` for numerical stability
   - Use `precision_LL` when you need strict positive definiteness
   - Use `covariance_LL` when working with covariance directly

2. **Batch operations** when possible for better performance

3. **Use `rsample()`** instead of `sample()` for gradient-based learning

4. **Keep precision matrices sparse** - avoid converting to dense

.. code-block:: python

   # Good: Keep operations sparse
   samples = dist.rsample((1000,))
   log_probs = dist.log_prob(samples)

   # Avoid: Converting to dense
   dense_precision = precision.to_dense()  # Memory intensive!

Next Steps
----------

- Learn about :doc:`backends` for GPU acceleration
- Explore :doc:`optimization_examples` for ML applications
- Check out :doc:`basic_operations` for core sparse operations
