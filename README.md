# WeightGrad: Geo-Distributed Data Analysis Using Quantization for Faster Convergence and Better Accuracy

**WeightGrad: Geo-Distributed Data Analysis Using Quantization for Faster Convergence and Better Accuracy**<br>
Paper published at SIGKDD'20.<br>
Paper Link: https://dl.acm.org/doi/10.1145/3394486.3403097<br>
If you find this repository helpful, please cite the following:
```text
@inproceedings{10.1145/3394486.3403097,
  author = {Akter, Syeda Nahida and Adnan, Muhammad Abdullah},
  title = {WeightGrad: Geo-Distributed Data Analysis Using Quantization for Faster Convergence and Better Accuracy},
  year = {2020},
  isbn = {9781450379984},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3394486.3403097},
  doi = {10.1145/3394486.3403097},
  abstract = {High network communication cost for synchronizing weights and gradients in geo-distributed data analysis consumes the benefits of advancement in computation and optimization techniques. Many quantization methods for weight, gradient or both have been proposed in recent years where weight-quantized model suffers from error related to weight dimension and gradient-quantized method suffers from slow convergence rate by a factor related to the gradient quantization resolution and gradient dimension. All these methods have been proved to be infeasible in terms of distributed training across multiple data centers all over the world. Moreover recent studies show that communicating over WANs can significantly degrade DNN model performance by upto 53.7x because of unstable and limited WAN bandwidth. Our goal in this work is to design a geo-distributed Deep-Learning system that (1) ensures efficient and faster communication over LAN and WAN and (2) maintain accuracy and convergence for complex DNNs with billions of parameters. In this paper, we introduce WeightGrad which acknowledges the limitations of quantization and provides loss-aware weight-quantized networks with quantized gradients for local convergence and for global convergence it dynamically eliminates insignificant communication between data centers while still guaranteeing the correctness of DNN models. Our experiments on our developed prototypes of WeightGrad running across 3 Amazon EC2 global regions and on a cluster that emulates EC2 WAN bandwidth show that WeightGrad provides 1.06% gain in top-1 accuracy, 5.36x speedup over baseline and 1.4x-2.26x over the four state-of-the-art distributed ML systems.},
  booktitle = {Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery &amp; Data Mining},
  pages = {546–556},
  numpages = {11},
  keywords = {deep neural networks, geo-distributed datasets, quantization, data analytics},
  location = {Virtual Event, CA, USA},
  series = {KDD '20}
}
```
