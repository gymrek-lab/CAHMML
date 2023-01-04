import pytest
import sys

sys.path.append("../cahmml")
from cahmml import hmm as h
from cahmml import util as hu


class ObsExample(h.Observation):
    """Example Observation class for testing."""

    def __init__(self, obs_eg: str):
        self.obs_eg = obs_eg

    def __repr__(self):
        return self.obs_eg


class TestSample:

    sample_id = "foo"
    n_obs = h.np.random.randint(100)
    obs_options = [ObsExample("obs1"), ObsExample("obs2"), ObsExample("obs3")]
    obs = h.np.random.choice(obs_options, n_obs)

    def test_valid_sample(self):
        sample = h.Sample(self.sample_id, self.obs)

        assert sample.sample_id == self.sample_id
        assert sample.obs_type == type(self.obs[0])
        assert sample.n_obs == self.n_obs
        assert len(sample) == self.n_obs
        assert (
            f"{sample}"
            == f"{self.sample_id} with {self.n_obs} observations of type {type(self.obs_options[0])}"
        )

    def test_invalid_sample(self):
        with pytest.raises(hu.HMMValidationError):
            h.Sample(0, self.obs)

        with pytest.raises(hu.HMMValidationError):
            h.Sample(self.sample_id, [])

        with pytest.raises(hu.HMMValidationError):
            h.Sample(self.sample_id, [None, None, None])

        with pytest.raises(hu.HMMValidationError):
            h.Sample(self.sample_id, self.obs.tolist() + [1])


class StateExample(h.State):
    """Example State class for testing."""

    def __init__(self, state_eg: str):
        self.state_eg = state_eg

    def __repr__(self):
        return self.state_eg

    def emission_probability(
        self, obs: h.Iterable[ObsExample], t: int, hyperparameters: dict = {}
    ):
        probs = []
        for o in obs:
            if self.state_eg == "state1":
                if o.obs_eg == "obs1":
                    probs.append(0.7)
                elif o.obs_eg == "obs2":
                    probs.append(0.001)
                elif o.obs_eg == "obs3":
                    probs.append(0.299)
            elif self.state_eg == "state2":
                if o.obs_eg == "obs1":
                    probs.append(0.099)
                elif o.obs_eg == "obs2":
                    probs.append(0.9)
                elif o.obs_eg == "obs3":
                    probs.append(0.001)
            elif self.state_eg == "state3":
                if o.obs_eg == "obs1":
                    probs.append(0.001)
                elif o.obs_eg == "obs2":
                    probs.append(0.199)
                elif o.obs_eg == "obs3":
                    probs.append(0.8)
        return h.np.log10(h.np.array(probs))

    def transition_probability(
        self,
        next: "StateExample",
        obs: h.Iterable[ObsExample],
        t: int,
        hyperparameters: dict = {},
    ):
        probs = []
        for o in obs:
            if self.state_eg == "state1":
                if next.state_eg == "state1":
                    probs.append(0.8)
                elif next.state_eg == "state2":
                    probs.append(0.1)
                elif next.state_eg == "state3":
                    probs.append(0.1)
            elif self.state_eg == "state2":
                if next.state_eg == "state1":
                    probs.append(0.2)
                elif next.state_eg == "state2":
                    probs.append(0.7)
                elif next.state_eg == "state3":
                    probs.append(0.1)
            elif self.state_eg == "state3":
                if next.state_eg == "state1":
                    probs.append(0.1)
                elif next.state_eg == "state2":
                    probs.append(0.3)
                elif next.state_eg == "state3":
                    probs.append(0.6)
        return h.np.log10(h.np.array(probs))


class TestHMM:

    states = [StateExample("state1"), StateExample("state2"), StateExample("state3")]
    obs_options = [ObsExample("obs1"), ObsExample("obs2"), ObsExample("obs3")]

    def init_test_helper(self, model):
        assert h.np.array_equal(self.states, model.states)
        assert model.samples == None
        assert model.T == None
        assert model.E == None
        assert model.initial_probabilities == None

    def fit_test_helper(self, samples, obs, initial_prob, model):
        assert h.np.array_equal(samples, model.samples)
        assert h.np.array_equal(
            h.np.log10(h.np.array(initial_prob)), model.initial_probabilities
        )
        assert model.n_states == len(self.states)
        assert model.n_samples == len(samples)
        assert model.n_obs == len(obs)
        assert model.obs_type == type(self.obs_options[0])
        assert model.state_type == type(self.states[0])
        assert model.sample_type == type(samples[0])

    def test_valid_hmm(self):
        ## single sample
        samples = []
        obs_list = []
        obs = h.np.array([self.obs_options[j] for j in [0, 2, 0, 2, 2, 1]])
        samples.append(h.Sample(str(len(samples)), obs))
        obs_list.append(obs)

        # test HMM __init__
        model = h.HMM(self.states)
        self.init_test_helper(model)

        # test HMM fit
        initial_prob = [0.6, 0.2, 0.2]
        model.fit(samples, initial_prob)
        self.fit_test_helper(samples, obs_list[0], initial_prob, model)

        # test HMM sample_iterator
        expected_i = 0
        for obs_i, obs_l in model.sample_iterator():
            assert expected_i == obs_i
            expected_i += 1
            assert obs_l == [obs_list[i][obs_i] for i in range(model.n_samples)]

        # test HMM viterbi
        # validated with hmmlearn https://github.com/hmmlearn/hmmlearn
        X = model.viterbi()
        assert h.np.array_equal(X, [[0, 0, 0, 2, 2, 1]])

        ## multiple samples
        samples = []
        obs_list = []
        obs = h.np.array([self.obs_options[j] for j in [0, 2, 0, 2, 2, 1]])
        samples.append(h.Sample(str(len(samples)), obs))
        obs_list.append(obs)
        obs = h.np.array([self.obs_options[j] for j in [0, 2, 0, 2, 2, 2]])
        samples.append(h.Sample(str(len(samples)), obs))
        obs_list.append(obs)
        obs = h.np.array([self.obs_options[j] for j in [0, 2, 1, 2, 2, 1]])
        samples.append(h.Sample(str(len(samples)), obs))
        obs_list.append(obs)

        # test HMM __init__
        model = h.HMM(self.states)
        self.init_test_helper(model)

        # test HMM fit
        initial_prob = [0.6, 0.2, 0.2]
        model.fit(samples, initial_prob)
        self.fit_test_helper(samples, obs_list[0], initial_prob, model)

        # test HMM sample_iterator
        expected_i = 0
        for obs_i, obs_l in model.sample_iterator():
            assert expected_i == obs_i
            expected_i += 1
            assert obs_l == [obs_list[i][obs_i] for i in range(model.n_samples)]

        # test HMM viterbi
        # validated with hmmlearn https://github.com/hmmlearn/hmmlearn
        X = model.viterbi()
        assert h.np.array_equal(
            X, [[0, 0, 0, 2, 2, 1], [0, 0, 0, 2, 2, 2], [0, 2, 2, 2, 2, 1]]
        )

    def test_invalid_hmm(self):
        ## single sample
        samples = []
        obs = h.np.array([self.obs_options[j] for j in [0, 2, 0, 2, 2, 1]])
        samples.append(h.Sample(str(len(samples)), obs))
        initial_prob = [0.6, 0.2, 0.2]

        # test HMM with empty states
        model = h.HMM([])
        with pytest.raises(hu.HMMValidationError):
            model.fit(samples, initial_prob)

        # test HMM with NoneType state
        model = h.HMM([None])
        with pytest.raises(hu.HMMValidationError):
            model.fit(samples, initial_prob)

        # test HMM with state types non-match
        model = h.HMM(self.states + [1])
        with pytest.raises(hu.HMMValidationError):
            model.fit(samples, initial_prob)

        # test HMM with empty samples
        model = h.HMM(self.states)
        with pytest.raises(hu.HMMValidationError):
            model.fit([], initial_prob)

        # test HMM with NoneType sample
        model = h.HMM(self.states)
        with pytest.raises(hu.HMMValidationError):
            model.fit([None], initial_prob)

        # test HMM with sample types non-match
        model = h.HMM(self.states)
        with pytest.raises(hu.HMMValidationError):
            model.fit(samples + [1], initial_prob)

        # test HMM with initial probability and states non-match
        model = h.HMM(self.states)
        with pytest.raises(hu.HMMValidationError):
            model.fit(samples, initial_prob[: len(initial_prob) - 1])

        # test HMM with invalid initial probability
        model = h.HMM(self.states)
        with pytest.raises(hu.HMMValidationError):
            model.fit(samples, [1.6, 0.2, 0.2])

        ## multiple samples
        samples = []
        obs = h.np.array([self.obs_options[j] for j in [0, 2, 0, 2, 2, 1]])
        samples.append(h.Sample(str(len(samples)), obs))
        obs = h.np.array([self.obs_options[j] for j in [0, 2, 0, 2, 2, 2]])
        samples.append(h.Sample(str(len(samples)), obs))
        obs = h.np.array([self.obs_options[j] for j in [0, 2, 1, 2, 2, 1]])
        samples.append(h.Sample(str(len(samples)), obs))
        initial_prob = [0.6, 0.2, 0.2]

        # test HMM with empty states
        model = h.HMM([])
        with pytest.raises(hu.HMMValidationError):
            model.fit(samples, initial_prob)

        # test HMM with NoneType state
        model = h.HMM([None])
        with pytest.raises(hu.HMMValidationError):
            model.fit(samples, initial_prob)

        # test HMM with state types non-match
        model = h.HMM(self.states + [1])
        with pytest.raises(hu.HMMValidationError):
            model.fit(samples, initial_prob)

        # test HMM with empty samples
        model = h.HMM(self.states)
        with pytest.raises(hu.HMMValidationError):
            model.fit([], initial_prob)

        # test HMM with NoneType sample
        model = h.HMM(self.states)
        with pytest.raises(hu.HMMValidationError):
            model.fit([None], initial_prob)

        # test HMM with sample types non-match
        model = h.HMM(self.states)
        with pytest.raises(hu.HMMValidationError):
            model.fit(samples + [1], initial_prob)

        # test HMM with initial probability and states non-match
        model = h.HMM(self.states)
        with pytest.raises(hu.HMMValidationError):
            model.fit(samples, initial_prob[: len(initial_prob) - 1])

        # test HMM with invalid initial probability
        model = h.HMM(self.states)
        with pytest.raises(hu.HMMValidationError):
            model.fit(samples, [1.6, 0.2, 0.2])

        # test HMM with samples having different number of observations
        obs = h.np.array([self.obs_options[j] for j in [0, 2, 1, 2, 2]])
        samples.append(h.Sample(str(len(samples)), obs))
        model = h.HMM(self.states)
        with pytest.raises(hu.HMMValidationError):
            model.fit(samples, initial_prob)
