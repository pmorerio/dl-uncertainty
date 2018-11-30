## Combining Aleatoric and Epistemic Uncertainty in One Model
Section 3. of the paper: A Kendall, Y Gal, “**What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?**”, NIPS 2017 [arXiv](https://arxiv.org/abs/1703.04977) 

Combines the loss of eq (8) and dropout at test time to model both uncertainties jointly

My observations below (provided the code is doing what it is intended to):
  
  - Epistemic and aleatoric uncertainties seem to capture the same phenomenon when modeled separately. 
  - Possibly they capture the main source of uncertainty, which is the one related to contours reconstruction.
  - Modeling the two uncertainties together yields very different results than modeling separately.
  - The two uncertainties seem to capture extremely complementary information when modeled together.
  - Weird enough, epistemic uncertainty is maximal on the background. The uncertainty related to digits contours is pheraps already captured by the aleatoric uncertainty branch and there is nothing left to be modeled but background uncertainty.
    
  
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

*Real and Reconstructed Images*

![images](./pics/combined_real.png)
![images](./pics/combined_generated.png)

*Aleatoric Uncertainty, Epistemic Uncertainty and L2 reconstruction error*

![images](./pics/combined_aleatoric.png)
![images](./pics/combined_epistemic.png)
![images](./pics/combined_rec_error_L2.png)


