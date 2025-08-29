Optimization Examples
=====================

This tutorial demonstrates how to use torchsparsegradutils in machine learning and optimization applications.

Sparse Gaussian Process Regression
-----------------------------------

Using sparse precision matrices for scalable Gaussian processes.

Basic Sparse GP
~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import matplotlib.pyplot as plt
   from torchsparsegradutils import sparse_generic_solve
   from torchsparsegradutils.distributions import SparseMultivariateNormal

   class SparseGP:
       def __init__(self, inducing_points, kernel_func, noise_var=0.01):
           self.inducing_points = inducing_points
           self.kernel_func = kernel_func
           self.noise_var = noise_var
           self.m = len(inducing_points)

       def build_sparse_precision(self, x_train):
           """Build sparse precision matrix using inducing points."""
           n = len(x_train)

           # Only connect points to nearby inducing points
           indices = []
           values = []

           for i, x_i in enumerate(x_train):
               for j, u_j in enumerate(self.inducing_points):
                   dist = torch.norm(x_i - u_j)
                   if dist < 2.0:  # Sparsity threshold
                       k_val = self.kernel_func(x_i, u_j)
                       if k_val > 1e-4:
                           indices.append([i, j])
                           values.append(k_val)

           # Build sparse kernel matrix
           if indices:
               indices = torch.tensor(indices).T
               values = torch.tensor(values)
               K_sparse = torch.sparse_coo_tensor(indices, values, (n, self.m))

               # Approximate precision: (K + σ²I)^{-1} ≈ sparse approximation
               K_T = K_sparse.transpose(0, 1)
               precision_approx = torch.sparse.mm(K_T, K_sparse)

               # Add noise term
               diag_indices = torch.stack([torch.arange(self.m), torch.arange(self.m)])
               diag_values = torch.full((self.m,), self.noise_var)
               noise_term = torch.sparse_coo_tensor(diag_indices, diag_values, (self.m, self.m))

               return precision_approx + noise_term
           else:
               # Fallback identity
               eye_indices = torch.stack([torch.arange(n), torch.arange(n)])
               eye_values = torch.ones(n)
               return torch.sparse_coo_tensor(eye_indices, eye_values, (n, n))

       def predict(self, x_train, y_train, x_test):
           """Sparse GP prediction."""
           precision = self.build_sparse_precision(x_train)

           # Solve for GP weights
           weights = sparse_generic_solve(precision, y_train.unsqueeze(-1), method='cg')

           # Predict at test points (simplified)
           predictions = []
           for x_t in x_test:
               pred = 0.0
               for i, x_i in enumerate(x_train):
                   pred += weights[i] * self.kernel_func(x_t, x_i)
               predictions.append(pred)

           return torch.tensor(predictions)

   # Example usage
   def rbf_kernel(x1, x2, lengthscale=1.0):
       return torch.exp(-0.5 * torch.norm(x1 - x2)**2 / lengthscale**2)

   # Generate synthetic data
   x_train = torch.linspace(-5, 5, 50).unsqueeze(-1)
   y_train = torch.sin(x_train.squeeze()) + 0.1 * torch.randn(50)

   # Inducing points
   inducing_points = torch.linspace(-6, 6, 20).unsqueeze(-1)

   # Create and train GP
   gp = SparseGP(inducing_points, rbf_kernel)

   # Predictions
   x_test = torch.linspace(-6, 6, 100).unsqueeze(-1)
   y_pred = gp.predict(x_train, y_train, x_test)

   print(f"Prediction shape: {y_pred.shape}")

Sparse Variational Autoencoder
-------------------------------

Using structured sparse priors in VAE latent spaces.

Sparse VAE Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch.nn as nn
   import torch.nn.functional as F
   from torchsparsegradutils.distributions import SparseMultivariateNormal

   class SparseVAE(nn.Module):
       def __init__(self, input_dim, latent_dim, hidden_dim=128):
           super().__init__()
           self.latent_dim = latent_dim

           # Encoder
           self.encoder = nn.Sequential(
               nn.Linear(input_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, hidden_dim),
               nn.ReLU(),
           )
           self.fc_mu = nn.Linear(hidden_dim, latent_dim)
           self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

           # Decoder
           self.decoder = nn.Sequential(
               nn.Linear(latent_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, input_dim),
               nn.Sigmoid()
           )

           # Sparse prior structure (chain graph)
           self.register_buffer('prior_precision', self._build_chain_precision(latent_dim))

       def _build_chain_precision(self, dim):
           """Build chain-structured precision matrix."""
           indices = []
           values = []

           # Chain connections: each variable connects to neighbors
           for i in range(dim):
               # Self-connection
               indices.append([i, i])
               values.append(2.0)

               # Neighbor connections
               if i > 0:
                   indices.append([i, i-1])
                   indices.append([i-1, i])
                   values.extend([-0.5, -0.5])

           indices = torch.tensor(indices).T
           values = torch.tensor(values)
           return torch.sparse_coo_tensor(indices, values, (dim, dim))

       def encode(self, x):
           h = self.encoder(x)
           mu = self.fc_mu(h)
           logvar = self.fc_logvar(h)
           return mu, logvar

       def reparameterize(self, mu, logvar):
           std = torch.exp(0.5 * logvar)
           eps = torch.randn_like(std)
           return mu + eps * std

       def decode(self, z):
           return self.decoder(z)

       def forward(self, x):
           mu, logvar = self.encode(x)
           z = self.reparameterize(mu, logvar)
           return self.decode(z), mu, logvar

       def sparse_kl_loss(self, mu, logvar):
           """KL divergence with sparse prior."""
           batch_size = mu.shape[0]

           # Convert logvar to precision (diagonal approximation)
           posterior_precision_diag = torch.exp(-logvar)

           # Create diagonal precision matrices for posterior
           total_kl = 0.0
           for i in range(batch_size):
               # Diagonal precision matrix for this sample
               diag_indices = torch.stack([torch.arange(self.latent_dim),
                                         torch.arange(self.latent_dim)])
               posterior_precision = torch.sparse_coo_tensor(
                   diag_indices,
                   posterior_precision_diag[i],
                   (self.latent_dim, self.latent_dim)
               )

               # Create distributions
               posterior = SparseMultivariateNormal(
                   loc=mu[i],
                   precision_matrix=posterior_precision,
                   param='precision_LL'
               )

               prior = SparseMultivariateNormal(
                   loc=torch.zeros(self.latent_dim),
                   precision_matrix=self.prior_precision,
                   param='precision_LL'
               )

               # Monte Carlo KL estimate
               z_samples = posterior.rsample((10,))
               kl_i = (posterior.log_prob(z_samples) - prior.log_prob(z_samples)).mean()
               total_kl += kl_i

           return total_kl / batch_size

   # Training example
   def train_sparse_vae(model, data_loader, epochs=10, device='cpu'):
       model = model.to(device)
       optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

       model.train()
       for epoch in range(epochs):
           total_loss = 0
           for batch_idx, (data, _) in enumerate(data_loader):
               data = data.view(-1, 784).to(device)  # Flatten MNIST

               optimizer.zero_grad()
               recon_batch, mu, logvar = model(data)

               # Reconstruction loss
               recon_loss = F.binary_cross_entropy(recon_batch, data, reduction='sum')

               # Sparse KL divergence
               kl_loss = model.sparse_kl_loss(mu, logvar)

               loss = recon_loss + kl_loss
               loss.backward()
               optimizer.step()

               total_loss += loss.item()

               if batch_idx % 100 == 0:
                   print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

       return model

   # Example usage
   # Assuming MNIST data loader is available
   # model = SparseVAE(input_dim=784, latent_dim=20)
   # trained_model = train_sparse_vae(model, train_loader)

Graph Neural Networks with Sparse Operations
---------------------------------------------

Using sparse matrix operations for efficient GNN computation.

Sparse GCN Layer
~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch.nn as nn
   from torchsparsegradutils import sparse_mm

   class SparseGCNLayer(nn.Module):
       def __init__(self, in_features, out_features):
           super().__init__()
           self.linear = nn.Linear(in_features, out_features)
           self.reset_parameters()

       def reset_parameters(self):
           nn.init.xavier_uniform_(self.linear.weight)

       def forward(self, x, adj_matrix):
           """
           Args:
               x: Node features [N, in_features]
               adj_matrix: Sparse adjacency matrix [N, N]
           """
           # Linear transformation
           h = self.linear(x)  # [N, out_features]

           # Message passing with sparse matrix multiplication
           # This preserves sparsity in gradients
           output = sparse_mm(adj_matrix, h)  # [N, out_features]

           return output

   class SparseGCN(nn.Module):
       def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
           super().__init__()

           self.layers = nn.ModuleList()

           # First layer
           self.layers.append(SparseGCNLayer(input_dim, hidden_dim))

           # Hidden layers
           for _ in range(num_layers - 2):
               self.layers.append(SparseGCNLayer(hidden_dim, hidden_dim))

           # Output layer
           self.layers.append(SparseGCNLayer(hidden_dim, output_dim))

       def forward(self, x, adj_matrix):
           for i, layer in enumerate(self.layers):
               x = layer(x, adj_matrix)
               if i < len(self.layers) - 1:  # No activation on last layer
                   x = F.relu(x)
           return x

   # Example: Node classification
   def create_graph_data(num_nodes=1000, num_features=64, num_classes=7):
       """Create synthetic graph data."""
       # Random node features
       x = torch.randn(num_nodes, num_features)

       # Create sparse adjacency matrix (random graph)
       edge_prob = 0.01  # 1% edge density
       edges = torch.rand(num_nodes, num_nodes) < edge_prob
       edges = edges | edges.T  # Make symmetric
       edges.fill_diagonal_(False)  # No self-loops

       # Convert to sparse tensor
       indices = torch.nonzero(edges, as_tuple=False).T
       values = torch.ones(indices.shape[1])
       adj_matrix = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))

       # Add self-loops and normalize
       adj_matrix = adj_matrix + torch.sparse_coo_tensor(
           torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)]),
           torch.ones(num_nodes),
           (num_nodes, num_nodes)
       )

       # Random labels
       labels = torch.randint(0, num_classes, (num_nodes,))

       return x, adj_matrix, labels

   # Training example
   def train_gcn():
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

       # Create data
       x, adj_matrix, labels = create_graph_data()
       x, adj_matrix, labels = x.to(device), adj_matrix.to(device), labels.to(device)

       # Create model
       model = SparseGCN(input_dim=64, hidden_dim=128, output_dim=7).to(device)
       optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
       criterion = nn.CrossEntropyLoss()

       # Training split
       train_mask = torch.zeros(len(labels), dtype=torch.bool)
       train_mask[:int(0.6 * len(labels))] = True

       model.train()
       for epoch in range(200):
           optimizer.zero_grad()

           # Forward pass
           out = model(x, adj_matrix)
           loss = criterion(out[train_mask], labels[train_mask])

           # Backward pass
           loss.backward()
           optimizer.step()

           if epoch % 50 == 0:
               model.eval()
               with torch.no_grad():
                   pred = out.argmax(dim=1)
                   train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
               print(f'Epoch {epoch}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}')
               model.train()

   # Run training
   # train_gcn()

Sparse Neural ODEs
------------------

Using sparse solvers for Neural ODE integration.

Sparse NODE Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch.nn as nn
   from torchsparsegradutils import sparse_generic_solve

   class SparseODEFunc(nn.Module):
       def __init__(self, dim, sparse_structure=None):
           super().__init__()
           self.dim = dim

           if sparse_structure is None:
               # Default: tridiagonal structure
               sparse_structure = self._build_tridiagonal(dim)

           self.register_buffer('sparse_indices', sparse_structure['indices'])
           self.sparse_weights = nn.Parameter(torch.randn(sparse_structure['nnz']))

       def _build_tridiagonal(self, dim):
           """Build tridiagonal sparsity pattern."""
           indices = []

           # Main diagonal
           for i in range(dim):
               indices.append([i, i])

           # Super and sub diagonals
           for i in range(dim - 1):
               indices.append([i, i + 1])     # Super diagonal
               indices.append([i + 1, i])     # Sub diagonal

           indices = torch.tensor(indices).T
           return {'indices': indices, 'nnz': len(indices[0])}

       def forward(self, t, y):
           # Create sparse weight matrix
           W = torch.sparse_coo_tensor(
               self.sparse_indices,
               self.sparse_weights,
               (self.dim, self.dim)
           )

           # Sparse matrix-vector multiplication
           dydt = torch.sparse.mv(W, y) + torch.tanh(y)
           return dydt

   class SparseNeuralODE(nn.Module):
       def __init__(self, ode_func):
           super().__init__()
           self.ode_func = ode_func

       def forward(self, y0, t):
           """Simple Euler integration (replace with better integrator)."""
           dt = t[1] - t[0]
           y = y0

           trajectory = [y0]
           for i in range(len(t) - 1):
               dydt = self.ode_func(t[i], y)
               y = y + dt * dydt
               trajectory.append(y)

           return torch.stack(trajectory, dim=0)

   # Training example
   def train_sparse_node():
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

       # Create synthetic time series data
       t_data = torch.linspace(0, 1, 20)
       y_true = torch.sin(2 * torch.pi * t_data).unsqueeze(-1) * torch.exp(-t_data).unsqueeze(-1)
       y_true = y_true.expand(-1, 10)  # 10-dimensional system

       # Initialize model
       ode_func = SparseODEFunc(dim=10)
       model = SparseNeuralODE(ode_func).to(device)
       optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

       t_data, y_true = t_data.to(device), y_true.to(device)

       for epoch in range(1000):
           optimizer.zero_grad()

           # Forward pass
           y_pred = model(y_true[0], t_data)
           loss = F.mse_loss(y_pred, y_true)

           loss.backward()
           optimizer.step()

           if epoch % 100 == 0:
               print(f'Epoch {epoch}, Loss: {loss.item():.6f}')

       return model

   # Example usage
   # trained_node = train_sparse_node()

Sparse Transformer Attention
-----------------------------

Using sparse attention mechanisms for efficiency.

Sparse Attention Layer
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class SparseAttention(nn.Module):
       def __init__(self, d_model, n_heads, sparsity_pattern='local'):
           super().__init__()
           self.d_model = d_model
           self.n_heads = n_heads
           self.d_k = d_model // n_heads

           self.W_q = nn.Linear(d_model, d_model)
           self.W_k = nn.Linear(d_model, d_model)
           self.W_v = nn.Linear(d_model, d_model)
           self.W_o = nn.Linear(d_model, d_model)

           self.sparsity_pattern = sparsity_pattern

       def create_sparse_mask(self, seq_len):
           """Create sparse attention mask."""
           if self.sparsity_pattern == 'local':
               # Local attention: each token attends to k neighbors
               k = 8  # Window size
               indices = []
               values = []

               for i in range(seq_len):
                   for j in range(max(0, i - k//2), min(seq_len, i + k//2 + 1)):
                       indices.append([i, j])
                       values.append(1.0)

               indices = torch.tensor(indices).T
               values = torch.tensor(values)
               return torch.sparse_coo_tensor(indices, values, (seq_len, seq_len))

           elif self.sparsity_pattern == 'strided':
               # Strided attention pattern
               stride = 4
               indices = []
               values = []

               for i in range(seq_len):
                   # Local connections
                   for j in range(max(0, i-2), min(seq_len, i+3)):
                       indices.append([i, j])
                       values.append(1.0)

                   # Strided connections
                   for j in range(i % stride, seq_len, stride):
                       if abs(i - j) > 2:  # Avoid duplicates with local
                           indices.append([i, j])
                           values.append(1.0)

               indices = torch.tensor(indices).T
               values = torch.tensor(values)
               return torch.sparse_coo_tensor(indices, values, (seq_len, seq_len))

       def forward(self, x):
           batch_size, seq_len, d_model = x.shape

           # Linear projections
           Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
           K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
           V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

           # Create sparse attention mask
           sparse_mask = self.create_sparse_mask(seq_len).to(x.device)

           # Sparse attention computation
           attention_output = []
           for head in range(self.n_heads):
               q_head = Q[:, head]  # [batch_size, seq_len, d_k]
               k_head = K[:, head]
               v_head = V[:, head]

               # Compute attention scores
               scores = torch.bmm(q_head, k_head.transpose(-2, -1)) / (self.d_k ** 0.5)

               # Apply sparse mask efficiently
               # Only compute attention for non-zero positions
               mask_indices = sparse_mask.coalesce().indices()
               mask_values = sparse_mask.coalesce().values()

               sparse_scores = torch.sparse_coo_tensor(
                   mask_indices,
                   scores[:, mask_indices[0], mask_indices[1]].mean(dim=0),  # Simplified
                   (seq_len, seq_len)
               )

               # Sparse softmax approximation
               sparse_attention = torch.sparse.softmax(sparse_scores, dim=1)

               # Apply attention to values
               head_output = sparse_mm(sparse_attention, v_head.mean(dim=0))  # Simplified
               attention_output.append(head_output)

           # Concatenate heads
           attention_output = torch.cat(attention_output, dim=-1)

           # Final linear projection
           return self.W_o(attention_output.unsqueeze(0).expand(batch_size, -1, -1))

Performance Monitoring
----------------------

.. code-block:: python

   def benchmark_sparse_operations():
       """Benchmark different sparse operations."""
       import time

       sizes = [1000, 5000, 10000]
       densities = [0.001, 0.01, 0.1]

       results = {}

       for n in sizes:
           for density in densities:
               # Create test matrix
               nnz = int(n * n * density)
               indices = torch.randint(0, n, (2, nnz))
               values = torch.randn(nnz)
               A = torch.sparse_coo_tensor(indices, values, (n, n))
               b = torch.randn(n, 1)

               # Benchmark sparse solve
               start = time.time()
               try:
                   x = sparse_generic_solve(A, b, method='cg', tol=1e-6)
                   solve_time = time.time() - start
                   residual = torch.norm(torch.sparse.mm(A, x) - b)
               except:
                   solve_time = float('inf')
                   residual = float('inf')

               results[f'n={n}_d={density}'] = {
                   'solve_time': solve_time,
                   'residual': residual.item() if residual != float('inf') else float('inf')
               }

       return results

   # results = benchmark_sparse_operations()
   # for key, value in results.items():
   #     print(f"{key}: {value['solve_time']:.4f}s, residual: {value['residual']:.2e}")

These examples demonstrate how torchsparsegradutils can be integrated into various machine learning workflows, providing both computational efficiency and maintaining gradient flow for end-to-end training.

Next Steps
----------

- Explore :doc:`backends` for GPU acceleration of these examples
- Learn about :doc:`linear_solvers` for choosing optimal algorithms
- Check out :doc:`distributions` for probabilistic modeling applications
