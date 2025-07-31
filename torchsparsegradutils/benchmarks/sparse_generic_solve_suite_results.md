# Sparse solve methods benchmark

| layout   | algo                    | index_dt   | value_dt   |      N |   fwd_time_s |   fwd_mem_MB |   bwd_time_s |   bwd_mem_MB |
|:---------|:------------------------|:-----------|:-----------|-------:|-------------:|-------------:|-------------:|-------------:|
| coo      | sparse generic cg       | int32      | float32    | 123440 |        0.199 |        183.1 |            0 |  3.63736e+07 |
| coo      | sparse generic bicgstab | int32      | float32    | 123440 |       29.206 |        192.1 |            0 |  3.63736e+07 |
| coo      | sparse generic minres   | int32      | float32    | 123440 |        0.148 |        194.1 |            0 |  3.63736e+07 |
| coo      | jax default             | int32      | float32    | 123440 |        1.801 |        174.1 |            0 |  3.63736e+07 |
| coo      | jax cg                  | int32      | float32    | 123440 |        0.327 |        174.1 |            0 |  3.63736e+07 |
| coo      | jax bicgstab            | int32      | float32    | 123440 |        0.328 |        174.1 |            0 |  3.63736e+07 |
| csr      | sparse generic cg       | int32      | float32    | 123440 |        0.002 |        153.9 |            0 |  3.63736e+07 |
| csr      | sparse generic bicgstab | int32      | float32    | 123440 |       22.162 |        154.3 |            0 |  3.63736e+07 |
| csr      | sparse generic minres   | int32      | float32    | 123440 |        0.123 |        156.4 |            0 |  3.63736e+07 |
| csr      | jax default             | int32      | float32    | 123440 |        0.391 |        149.9 |            0 |  3.63736e+07 |
| csr      | jax cg                  | int32      | float32    | 123440 |        0.412 |        149.9 |            0 |  3.63736e+07 |
| csr      | jax bicgstab            | int32      | float32    | 123440 |        0.333 |        149.9 |            0 |  3.63736e+07 |
