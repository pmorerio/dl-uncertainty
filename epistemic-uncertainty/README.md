## Epistemic Uncertainty
Should be implemented with dropout at teat time.

### Code

Download MNIST:

`
./download.sh
`

Rescale and save in a python dictionary (possibly resize):

`
python prepro.py
`

Train an autoencoder:

`
python main.py --mode train
`

Test it:

`
python main.py --mode test --checkpoint model/model
`

Visualize TensorBoard logs:

`
tensorboard --logdir logs/
`

###  Results

*Real Images*
![images](./pics/0.png)

*Generated images*
![images](./pics/1.png)

*Log Variance*
![images](./pics/3.png)

