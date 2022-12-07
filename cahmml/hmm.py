from typing import Iterable
from abc import ABC,abstractmethod
import numpy as np
from rich.progress import track
from . import util as hu

class Observation(ABC):
    """Observation for use in HMM. Note that while this is fully abstract, it should have some properties!
    """ 
    pass

class Sample:

    def __init__(self,sample_id:str,observations:Iterable[Observation]):
        """Constructor for Sample class. Wraps Iterable of Observations.

        Args:
            observations (Iterable[Observation]): Observations of this Sample
            sample_id (str): ID of this Sample
        """        
        self.sample_id = sample_id
        self.obs = observations

        # Any validations we need to assume before loading into an HMM
        self._validate()

    def _validate(self):
        """Internal validation layer for Sample class

        Raises:
            hu.HMMValidationError: Sample fails validation layer
        """        
        self.n_obs = 0
        self.obs_type = None

        for o in self.obs:
            self.n_obs += 1

            # Assertions
            try:
                assert o is not None
            except AssertionError:
                raise hu.HMMValidationError("Observation cannot be NoneType")
            
            if self.obs_type is None:
                self.obs_type = type(o)
            else:
                try:
                    assert self.obs_type == type(o)
                except AssertionError:
                    raise hu.HMMValidationError(f"Observation types {self.obs_type} and {type(o)} do not match")

    def __sizeof__(self) -> int:
        """Number of Observations for this Sample

        Returns:
            int: Number of Observations
        """        
        return self.n_obs

    def __repr__(self) -> str:
        """Readable output for Sample

        Returns:
            str: Sample ID, number of Observations, type of Observations
        """        
        return f"{self.sample_id} with {self.n_obs} observations of type {self.obs_type}"

class State(ABC):
    
    @abstractmethod
    def emission_probability(self,obs:Iterable[Observation],t:int,hyperparameters:dict = {}) -> np.ndarray:     
        """Core emission function for the HMM, pass in extra information via kwargs

        Args:
            obs (Iterable[Observation]): Observations for which to compute probability
            t (int): Timepoint index
            hyperparameters (dict, optional): Any hyperparameters you'll want passed in later. Defaults to {}.

        Returns:
            float: log10(P(obs|self)) x |SAMPLES|
        """        
        pass

    @abstractmethod    
    def transition_probability(self,next:"State",obs:Iterable[Observation],t:int,hyperparameters:dict = {}) -> np.ndarray:
        """Core transition function for the HMM, pass in extra infromation via kwargs

        Args:
            next (State): destination State for which to compute probability
            obs (Iterable[Observations]): Observations for each sample
            t (int): Timepoint index
            hyperparameters (dict, optional): Any hyperparameters you'll want passed in later. Defaults to {}.

        Returns:
            float: log10(P(next|self)) x |SAMPLES|
        """        
        pass

class HMM:
    
    def __init__(self,states:Iterable[State]):      
        """Constructor for HMM

        Args:
            states (Iterable[State]): |States| states, no need to replicate
        """        
        self.states = states
        self.samples = None
        self.T = None
        self.E = None
        self.initial_probabilities = None

    def fit(self,samples:Iterable[Sample],initial_probabilities:Iterable[float],e_hparams:dict = {},t_hparams:dict = {}):
        """Fits the HMM with the given samples, pass extra information to functions with respective kwargs

        Args:
            samples (Iterable[Sample]): Samples to fit HMM
            initial_probabilities (Iterable[float]): Initial probabilities to seed transition matrix
            e_hparams (dict, optional): Hyperparameters to be passed to emission function. Defaults to {}.
            t_hparams (dict, optional): Hyperparameters to be passed to transition function. Defaults to {}.
        """

        self.samples = samples
        self.initial_probabilities = initial_probabilities
        # Any validations we need to assume before fitting
        self._validate()

        # Initialize the matrices
        self.initial_probabilities = np.log10(np.array(self.initial_probabilities))
        self.E = np.zeros([self.n_samples,self.n_states,self.n_obs])
        self.T = np.zeros([self.n_samples,self.n_states,self.n_states,self.n_obs])

        # TODO May want to parallelize eventually
        for o,obs in track(self.sample_iterator(),total=self.n_obs,description="Fitting"):
            for i,s in enumerate(self.states):
                self.E[:,i,o] = s.emission_probability(obs,o,e_hparams)
                for j,t in enumerate(self.states):
                    self.T[:,i,j,o] = s.transition_probability(t,obs,o,t_hparams)

    def sample_iterator(self):
        """Internal iterator to be passed to emission_probability and transition_probability

        Yields:
            List[Observation]: Observations for each Sample at a given timepoint
        """        
        sample_iters = [iter(sample.obs) for sample in self.samples]
        for i in range(self.n_obs):
            yield i,[next(sample_iter) for sample_iter in sample_iters]

    def _validate(self):
        self.n_states = 0
        self.n_samples = 0
        self.n_obs = 0
        self.obs_type = None
        self.state_type = None
        self.sample_type = None

        # First, check states
        for s in self.states:
            self.n_states += 1

            # State can't be None
            try:
                assert s is not None
            except AssertionError:
                raise hu.HMMValidationError("State cannot be NoneType")

            # State types have to match
            if self.state_type is None:
                self.state_type = type(s),
            else:
                try:
                    assert isinstance(s,self.state_type)
                except AssertionError:
                    raise hu.HMMValidationError(f"State types {self.state_type} and {type(s)} do not match")

        # Next, check samples
        for s in self.samples:
            self.n_samples += 1

            # Sample can't be None
            try:
                assert s is not None
            except AssertionError:
                raise hu.HMMValidationError("Sample cannot be NoneType")

            # Sample types have to match
            if self.sample_type is None:
                self.sample_type = type(s)
                self.obs_type = s.obs_type
                self.n_obs = s.n_obs
            else:
                try:
                    assert isinstance(s,self.sample_type)
                except AssertionError:
                    raise hu.HMMValidationError(f"Sample types {self.sample_type} and {type(s)} do not match")

            # Samples have to have same number of observations
            try:
                assert self.n_obs == s.n_obs
            except AssertionError:
                raise hu.HMMValidationError(f"Samples have differing number of observations ({self.n_obs} vs. {s.n_obs})")

            # Sample type must also match State function!

        # Lastly, make sure we have the correct number of states for initial probabilities
        try:
            assert len(self.initial_probabilities) == self.n_states
        except AssertionError:
            raise hu.HMMValidationError(f"Initial probabilities shape does not match number of states ({len(self.initial_probabilities)} vs {self.n_states})")
        
    def viterbi(self) -> np.ndarray:
        """Run Viterbi algorithm on the HMM

        Raises:
            hu.HMMValidationError: Raised if fit() was not called first

        Returns:
            np.ndarray: |Samples| x |Observations| matrix with state predictions
        """ 

        # Validate that we've already fit samples
        try:
            assert self.samples is not None
        except AssertionError:
            raise hu.HMMValidationError("Call fit() before viterbi()")

        # Instantiate T1 and T2
        T1 = np.zeros([self.n_samples,self.n_states,self.n_obs])
        T2 = np.zeros_like(T1,dtype=int)
        T1[:,:,0] = self.initial_probabilities + self.E[:, :, 0]

        # Populate Viterbi
        for o in track(range(1,self.n_obs),description="Populating"):

            # This line is very confusing, so here's more of a description
            # Line 1: T1[k,j-1]
            # Line 2: Aki
            # Line 3: Biy
            tmp = np.repeat(T1[:,:,o-1,np.newaxis],self.n_states,axis=-1) + self.T[:,:,:,o]

            # Max and argmax, respectively
            T1[:,:,o] = tmp.max(axis=1) + self.E[:,:,o]
            T2[:,:,o] = tmp.argmax(axis=1)

        # Backtrack to find the best path
        bt_ptr = T1[:,:,-1].argmax(axis=1)
        bt = np.zeros([self.n_samples,self.n_obs],dtype=int)
        bt[:,-1] = bt_ptr
        
        for o in track(range(self.n_obs-2,-1,-1),description="Backtracking"):
            # We need to index like this to satisfy numpy's "advanced" indexing
            bt_ptr = T2[np.arange(T2.shape[0]),np.array(bt_ptr),np.array([o]*self.n_samples)]
            bt[:,o] = bt_ptr
            
        # Return states as an array
        return bt