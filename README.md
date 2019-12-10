# Vector-based navigation using grid-like representations in artificial agents

DeepMind released work [published in Nature](https://www.nature.com/articles/s41586-018-0102-6) about emergence of grid-cell-like behavior in supervised learning experiments using LSTMs to perform path integration. This work is an important stepping stone in understanding the relationship between AI and the brain. 

In this class project, I attempt to use their approach for other experiments. In particular:
* What is the role of dropout? What if the layer size is reduced instead of applying dropout? Would grid-cells emerge if we stop the training of LSTMs early (is it due to learning or the linear bottleneck)? 
* Would grid cells emerge in 3D path integration?
* In real-life rat experiments, grid-cells do not emerge when the infant rat develops in a spherical environment deprived of boundaries and geometric references, distal spacial cues, and experience with environmental anchoring. 
* What about the case of learning path integration in pseudo-infinite space (dataset is sparse of the space)
* What about tasks involving remembering odors in relation to places?

Work in this area is important to strengthen the parallel between brain behavior and emergent behavior in AI. 

# Results

While I have been able to generate path trajectories and reproduce results in the paper, I have yet to build a dataset as clean as the one released in the code. 

One interesting result I got is I was able to generate extremely crisp and grid-like cells that are more triangular than the ones released in the paper while only using half the neurons in the linear layer and only 20% dropout (instead of 50%). Even more surprising was that the LSTM layer developed similar grid cells which was not the case in the paper. The grids did not develop different scales however. This might suggest grid cells emerge from the pressure of having few resources that are not enough to represent space in an inefficient fashion and therefore converge towards grid representation which encodes space more efficiently. The next step is to remove dropout altogether and try with fewer units. 
## linear layer 128 unit 20% dropout
![linear layer 128 unit 20% dropout](https://github.com/salimmj/grid-cells/blob/master/assets/20%25dropout.PNG)
## LSTM layer 128 unit 20% dropout
![LSTM layer 128 unit 20% dropout](https://github.com/salimmj/grid-cells/blob/master/assets/lstm20dropout.png)
3D experiment TBD!
