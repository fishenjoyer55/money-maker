Naive attempt at training a PyTorch neural network on ~20,000 earnings reports. Unfortunately not very successful lmao
- No clear earnings pattern emerged when training with 7 parameters. The model got trapped in simply predicting the average to minimize loss, resulting in standard loss of ~20,000 * variance
- Training with more parameters may help, but the data is inherently messy, and also a lot of the reports lack the less-common metrics, making those unusable
- Possible next steps: don't use a basic neural network and rely just on historical data, hop on live data instead
