# Author: Lois I Omotara
# ModelPuck.py
import random
import numpy as np
import math
from sentence_transformers import SentenceTransformer,util
from scipy.special import softmax
from scipy import stats

class ModelPuck :
    def __init__(self,choices):
        self.p =[0,0]
        self.v = [0,0]
        self.a = [0.0]
        self.F = [0,0]
        self.m = 2
        self.dt = 1
        self.choices = choices

    def get_choicep(self, choice):
        # define position of choices here
        choice_positions = [[5,0], [0,0],[5,5],[0,5]]
        return choice_positions[self.choices.index(choice)]

    def get_dt(self):
        return self.dt

    def update_F(self,mass_m,mass_p,dists):
        G = 6.6743 * 10^-11
        invsqr_dists = 1/(dists)^2
        return -G*mass_m*mass_p*(invsqr_dists)

    def update_a(self):
        self.a = self.F/ self.m

    def update_v(self):
        self.v = self.v + self.a*self.dt

    def update_p(self):
        self.p = self.p + self.v*self.dt

    def update(self):
        self.update_F()
        self.update_a()
        self.update_v()
        self.update_p()
        return 1
