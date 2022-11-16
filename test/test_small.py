import sys
sys.path.append("../src")

import hmm as h
import hmm_util as hu

class Clothing(h.Observation):
    
    def __init__(self,clothing:str):
        self.clothing = clothing

    def __repr__(self):
        return self.clothing

class Weather(h.State):

    def __init__(self,weather:str):
        self.weather = weather

    def __repr__(self):
        return self.weather

    def emission_probability(self,obs:h.Iterable[Clothing],hyperparameters:dict = {}):
        probs = []
        for o in obs:
            if self.weather == "Hot":
                if o.clothing == "Shorts":
                    probs.append(0.7)
                elif o.clothing == "Jeans":
                    probs.append(0.2)
                elif o.clothing == "Parka":
                    probs.append(0.1)
            elif self.weather == "Pleasant":
                if o.clothing == "Shorts":
                    probs.append(0.3)
                elif o.clothing == "Jeans":
                    probs.append(0.4)
                elif o.clothing == "Parka":
                    probs.append(0.3)
            elif self.weather == "Cold":
                if o.clothing == "Shorts":
                    probs.append(0.1)
                elif o.clothing == "Jeans":
                    probs.append(0.1)
                elif o.clothing == "Parka":
                    probs.append(0.8)
        return h.np.log10(h.np.array(probs))

    def transition_probability(self,next:"Weather",obs:h.Iterable[Clothing],hyperparameters:dict = {}):
        probs = []
        for o in obs:
            if self.weather == "Hot":
                if next.weather == "Hot":
                    probs.append(0.5)
                elif next.weather == "Pleasant":
                    probs.append(0.4)
                elif next.weather == "Cold":
                    probs.append(0.1)
            elif self.weather == "Pleasant":
                if next.weather == "Hot":
                    probs.append(0.3)
                elif next.weather == "Pleasant":
                    probs.append(0.4)
                elif next.weather == "Cold":
                    probs.append(0.3)
            elif self.weather == "Cold":
                if next.weather == "Hot":
                    probs.append(0.1)
                elif next.weather == "Pleasant":
                    probs.append(0.5)
                elif next.weather == "Cold":
                    probs.append(0.4)
        return h.np.log10(h.np.array(probs))

if __name__ == "__main__":
    states = [
        Weather("Hot"),
        Weather("Pleasant"),
        Weather("Cold"),
    ]
    observation_options = [
        Clothing("Shorts"),
        Clothing("Jeans"),
        Clothing("Parka")
    ]
    n_samples = 88
    n_obs = 1000000

    samples = []
    for i in range(n_samples):
        samples.append(h.Sample(i,h.np.random.choice(observation_options,n_obs)))

    model = h.HMM(states)
    model.fit(samples,[1/3]*len(states))