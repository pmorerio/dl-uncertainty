## Epistemic Uncertainty
Should be implemented with dropout at test time. I am sampling 20 nets here.

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

![images](./pics/epistemic_real.png)

*Generated images*

![images](./pics/epistemic_generated.png)

*Variance*

![images](./pics/epistemic_uncertainty.png)

