from typing import Iterable
from abc import ABC,abstractmethod
from inspect import getfullargspec
import numpy as np
import hmm_util as hu

class Observation(ABC):
    """Observation for use in HMM. Note that while this is fully abstract, it should have some properties!
    """ 
    pass

class Sample:

    def __init__(self,observations:Iterable[Observation],sample_id:str):
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
    def emission_probability(self,obs:Observation,**kwargs) -> float:
        """Core emission function for the HMM, pass in extra information via kwargs

        Args:
            obs (Observation): Observation for which to compute probability

        Returns:
            float: P(obs|self)
        """        
        pass

    @abstractmethod    
    def transition_probability(self,next:"State",**kwargs) -> float:
        """Core transition function for the HMM, pass in extra infromation via kwargs

        Args:
            next (State): State for which to compute probability

        Returns:
            float: P(next|self)
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

    def fit(self,samples:Iterable[Sample],initial_probabilities:Iterable[float]):
        """Fits the HMM with the given samples

        Args:
            samples (Iterable[Sample]): Samples to fit HMM
            initial_probabilities (Iterable[float]): Initial probabilities to seed transition matrix 
        """        

        self.samples = samples
        self.initial_probabilities = initial_probabilities
        # Any validations we need to assume before fitting
        self._validate()

        self.T = np.array([self.n_samples,self.n_states,self.n_states,self.n_obs])
        self.E = np.array([self.n_samples,self.n_states,self.n_obs])

        # TODO: Ryan

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
                    assert self.state_type == type(s)
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
                    assert self.sample_type == type(s)
                except AssertionError:
                    raise hu.HMMValidationError(f"Sample types {self.sample_type} and {type(s)} do not match")

            # Samples have to have same number of observations
            try:
                assert self.n_obs == s.n_obs
            except AssertionError:
                raise hu.HMMValidationError(f"Samples have differing number of observations ({self.n_obs} vs. {s.n_obs})")

            # Sample type must match State function
            try:
                # If there's an error on the line below, add type hinting to your abstract class extension
                state_obs_requirement = getfullargspec(self.state_type).annotations["obs"]
                assert state_obs_requirement == type(self.obs_type)
            except AssertionError:
                raise hu.HMMValidationError(f"Observation type from Sample must match required State argument")
            
        # Lastly, make sure we have the correct number of states for initial probabilities
        try:
            assert len(self.initial_probabilities) == self.n_states
        except AssertionError:
            raise hu.HMMValidationError(f"Initial probabilities shape does not match number of states ({len(self.initial_probabilities)} vs {self.n_states})")
        
    def EM(self) -> np.ndarray:
        """Run Expectation-Maximization on the HMM

        Returns:
            np.ndarray: |Samples| x |Observations| matrix with state predictions
        """

        # Validate that we've already fit samples
        try:
            assert self.samples is not None
        except AssertionError:
            raise hu.HMMValidationError("Call fit() before EM()")

        # TODO: Den