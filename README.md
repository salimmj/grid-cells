# Vector-based navigation using grid-like representations in artificial agents

DeepMind released work [published in Nature](https://www.nature.com/articles/s41586-018-0102-6) about emergence of grid-cell-like behavior in supervised learning experiments using LSTMs to perform path integration. This work is an important stepping stone in understanding the relationship between AI and the brain. 

In this class project, I attempt to use their approach for other experiments. In particular:
* What is the role of dropout? What if the layer size is reduced instead of applying dropout? Would grid-cells emerge if we stop the training of LSTMs early (is it due to learning or the linear bottleneck)? 
* Would grid cells emerge in 3D path integration?
* In real-life rat experiments, grid-cells do not emerge when the infant rat develops in a spherical environment deprived of boundaries and geometric references, distal spacial cues, and experience with environmental anchoring. 
* What about the case of learning path integration in pseudo-infinite space (dataset is sparse of the space)
* What about tasks involving remembering odors in relation to places?
