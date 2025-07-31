# sparse_mm vs torch.sparse.mm vs dense.mm benchmark

| layout   | algo      | index_dt   | value_dt   |      N |   fwd_time_s |   fwd_mem_MB |   bwd_time_s |   bwd_mem_MB |
|:---------|:----------|:-----------|:-----------|-------:|-------------:|-------------:|-------------:|-------------:|
| coo      | sparse.mm | int32      | float32    | 123440 |            0 |        228.5 |        0     |      58132.5 |
| coo      | sparse_mm | int32      | float32    | 123440 |            0 |        228.5 |        0.002 |        406.1 |
| coo      | dense.mm  | int32      | float32    | 123440 |            0 |      58132.5 |        0     |          0   |
| csr      | sparse.mm | int32      | float32    | 123440 |            0 |        180.1 |        0.002 |        382.9 |
| csr      | sparse_mm | int32      | float32    | 123440 |            0 |        180.1 |        0.001 |        408.3 |
| csr      | dense.mm  | int32      | float32    | 123440 |            0 |      58132.5 |        0     |          0   |
| coo      | sparse.mm | int32      | float64    | 123440 |            0 |        279.8 |        0     |     116255   |
| coo      | sparse_mm | int32      | float64    | 123440 |            0 |        279.8 |        0.002 |        508.8 |
| coo      | dense.mm  | int32      | float64    | 123440 |            0 |     116255   |        0     |          0   |
| csr      | sparse.mm | int32      | float64    | 123440 |            0 |        231.4 |        0.001 |        434.7 |
| csr      | sparse_mm | int32      | float64    | 123440 |            0 |        231.4 |        0.001 |        486.1 |
| csr      | dense.mm  | int32      | float64    | 123440 |            0 |     116255   |        0     |          0   |
| coo      | sparse.mm | int64      | float32    | 123440 |            0 |        253.6 |        0     |      58132.5 |
| coo      | sparse_mm | int64      | float32    | 123440 |            0 |        253.6 |        0.001 |        431.3 |
| coo      | dense.mm  | int64      | float32    | 123440 |            0 |      58132.5 |        0     |          0   |
| csr      | sparse.mm | int64      | float32    | 123440 |            0 |        205.3 |        0.001 |        408.1 |
| csr      | sparse_mm | int64      | float32    | 123440 |            0 |        205.3 |        0.001 |        433.5 |
| csr      | dense.mm  | int64      | float32    | 123440 |            0 |      58132.5 |        0     |          0   |
| coo      | sparse.mm | int64      | float64    | 123440 |            0 |        305   |        0     |     116255   |
| coo      | sparse_mm | int64      | float64    | 123440 |            0 |        305   |        0.002 |        533.9 |
| coo      | dense.mm  | int64      | float64    | 123440 |            0 |     116255   |        0     |          0   |
| csr      | sparse.mm | int64      | float64    | 123440 |            0 |        256.6 |        0.001 |        459.9 |
| csr      | sparse_mm | int64      | float64    | 123440 |            0 |        256.6 |        0.001 |        511.2 |
| csr      | dense.mm  | int64      | float64    | 123440 |            0 |     116255   |        0     |          0   |
