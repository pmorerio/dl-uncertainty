## Combining Aleatoric and Epistemic Uncertainty in One Model
Section 3. of the paper: A Kendall, Y Gal, “**What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?**”, NIPS 2017 [arXiv](https://arxiv.org/abs/1703.04977)

  - Combines the loss of eq (8) and dropout at test time.

### Code

Train an autoencoder:

`
python main.py --mode train
`

Test it:

`
python main.py --mode test --checkpoint model/model0
`

Visualize TensorBoard logs:

`
tensorboard --logdir logs/
`

###  Results

*Real Images*

![images](./pics/combined_real.png)

*Generated images*

![images](./pics/combined_generated.png)

*Aleatoric Uncertainty*

![images](./pics/combined_aleatoric.png)

*Epistemic Uncertainty*

![images](./pics/combined_epistemic.png)

*L2 reconstruction error*

![images](./pics/combined_rec_error_L2.png)

