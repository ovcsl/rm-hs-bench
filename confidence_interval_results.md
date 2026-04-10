### Bootstrap Confidence Intervals for Precision and Recall Differences on the Lay Consensus Dataset

This document reports the 95% bootstrap confidence intervals (based on 10,000 resamples) for the performance differences observed on the lay consensus dataset. The values represent the change in precision and recall when using the rulemapping approach compared to two baselines: zero-context and long-context. Positive values indicate an improvement over the respective baseline.


| Model         | Metric    | Difference vs Zero-Context (95% CI) | Difference vs Long-Context (95% CI) |
| ------------- | --------- | ----------------------------------- | ----------------------------------- |
| Mistral Large | Precision | +0.504 [0.428, 0.583]               | +0.368 [0.270, 0.468]               |
| Mistral Large | Recall    | +0.000 [-0.071, 0.071]              | +0.200 [0.108, 0.298]               |
| GPT-4o        | Precision | +0.454 [0.379, 0.534]               | +0.486 [0.412, 0.565]               |
| GPT-4o        | Recall    | -0.075 [-0.155, 0.000]              | -0.150 [-0.233, -0.075]             |
| GPT-o1        | Precision | +0.459 [0.385, 0.538]               | +0.444 [0.369, 0.523]               |
| GPT-o1        | Recall    | -0.175 [-0.261, -0.096]             | -0.175 [-0.261, -0.096]             |
| Deepseek V3   | Precision | +0.200 [0.151, 0.252]               | +0.117 [0.071, 0.165]               |
| Deepseek V3   | Recall    | -0.087 [-0.153, -0.029]             | -0.013 [-0.068, 0.041]              |
| GPT-OSS-120B  | Precision | +0.418 [0.310, 0.528]               | +0.386 [0.279, 0.493]               |
| GPT-OSS-120B  | Recall    | -0.449 [-0.564, -0.338]             | -0.437 [-0.551, -0.326]             |
| GPT-5 mini    | Precision | +0.408 [0.165, 0.624]               | +0.440 [0.196, 0.655]               |
| GPT-5 mini    | Recall    | -0.837 [-0.915, -0.750]             | -0.862 [-0.934, -0.781]             |
