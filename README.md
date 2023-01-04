<p align="center">
  <img src=resources/camel.png width="50%">
</p>
  
<p align = "center">
  <b>Ryan Eveloff, Denghui Chen</b>
</p>
  
# CλHMML: Custom Lambda HMM Library

CλHMML (aka cahmml, pronounced camel) is a lightweight library meant to simplify complex Hidden Markov Models. We provide two abstract classes, <code>Observation</code> and <code>State</code>, which when implemented can run seamlessly in a parallelized HMM structure built on NumPy matrices.

## Motivation

<details>
  
During our research into multimodal genetic HMMs, we found that the majority of plug and play HMMs available require the input of a single transition matrix $T$ and a single, finite-library emission matrix $E$. In our case, we required a scalable, multi-sample HMM architecture that could operate with a Bayesian model at each timestep. After asking our colleagues, we found that many labs simply recreate the boilerplate code necessary for running an HMM each time they require it for their research. In the effort of saving time and making HMMs a simple and efficient interface for unsupervised language modeling, we created **CλHMML**.

</details>
  
## Installation

We have provided a wheel in `dist` that can be used to install via pip.

```bash
pip3 install dist/*.whl
```

## Usage

### Importing CλHMML

<details>

```python

from cahmml import hmm
```

If necessary, you can also import the utilities for CλHMML via <code>cahmml.util</code>, though it is unnecessary and generally not useful.

</details>
  
### Initializing an HMM

<details>
  
#### State Abstract Class
  
An implementation of <code>hmm.State</code> requires 2 functions to be completed:
  - <code>transition_probability</code>
  - <code>emission_probability</code>
  
```python
# State class
class MyState(hmm.State):
  
  def emission_probability(self,obs:Iterable[Observation],t:int,hyperparameters:dict = {}) -> np.ndarray:
    return P(obs|self,t,hyperparameters)
  
  def transition_probability(self,next:"State",obs:Iterable[Observation],t:int,hyperparameters:dict = {}) -> np.ndarray:
    return P(next|self,obs,t,hyperparameters)
```
#### Observation Abstract Class
  
An implementation of <code>hmm.Observation</code> requires nothing to be completed and serves as a modable passthrough class for <code>hmm.State</code>. You can even use built-in classes like <code>int</code> or <code>str</code>! In the case below, we use a simple <code>str</code> wrapper.
  
```python
# Observation Class
class myObservation(hmm.Observation):
  
  def __init__(self,value:str):
    self.v = value
```

#### Filling Samples with Observations

Pass in a sample_id and an iterable of <code>hmm.Observation</code> to create a sample.

```python
   # Given list[Observation] obs
   myFirstSample = hmm.Sample("first sample!",obs)
```
  ,,
</details>

### Running an HMM

<details>
  
  Assuming you've already implemented <code>hmm.State</code> and <code>hmm.Observation</code>, running Viterbi on your HMM with a given input is convenient and fast!
  
  ```python
  # Given list[hmm.State] states, list[hmm.Sample] samples, and list[float] initial_probs
  model = hmm.HMM(states)
  model.fit(samples,initial_probs)
  pred_states = model.viterbi()
  ```
  
  **Note:** Advanced users can specify hyperparameters for each function via <code>e_hparams</code> and <code>t_hparams</code>!
  
  This code will yield an array corresponding to the Viterbi-predicted state of each sample at each observation.
  
</details>

### Addendum: Complexity Analysis

<details>

Filling $T$ and $E$ runs in $\mathcal{O}(m \cdot n \cdot s \cdot f)$ time, where $m$ is the number of samples, $n$ is the number of observations, $s$ is the number of states, and $f$ is the maximum runtime of <code>transition_probability</code> and <code>emission_probability</code>. NumPy parallelization allows **Viterbi** runtime to scale linearly with the number of observations, or $\mathcal{O}(n)$.

More anecdotally, we expect a run of 100 states, 100 samples, 1,000,000 observations, and constant time $T$ and $E$ functions to run in less than an hour with consumer-grade hardware.

</details>
  
## Testing

Coverage reports are available in our test branch; for simple HMM testing, we validated output using <code>hmmlearn</code> by scikitlearn. For complex HMM testing, we used small, hand-reproducible examples.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
