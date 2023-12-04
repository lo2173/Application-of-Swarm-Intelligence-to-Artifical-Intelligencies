# Author Lois I. Omotara
# algorithm.py
import random
import numpy as np
import math
from sentence_transformers import SentenceTransformer,util
from scipy.special import softmax
from scipy import stats
import model_puck
# right now will organize object for each chatbot puck
# for each puck I need its current position, update function forn new position, and dist from choosing puck
# where to put all the choices, may need to go into a more general class
# create another class for the c
class ChatbotPuck():
    def __init__(self,convic_words, convic_freq, model):
        # right now convm limited to 3 word max
        self.model = model
        self.allwords = convic_words
        self.word = convic_words[convic_freq.index(max(convic_freq))]
        self.simm= self.update_sim()
        self.choice = self.model.choices[self.simm.index(max(self.simm))]
        self.convm = softmax(convic_freq)
        self.dt = model.get_dt()
        self.p= [0,0]
        self.F = [0,0]
        self.v = [0,0]
        self.a = [0,0]
        self.m = 1
        self.dist = math.dist(self.p,self.model.p)

    # ------ calculations-------------
    def calc_F(self,threshold):
        choice_p = self.model.get_choicep(self.choice)
        angle = math.atan(choice_p[1]/choice_p[0])
        # might dampen this a little bit (t0 make it interesting??)
        magn = 2*self.m*(threshold-self.v(self.dt))/(self.dt*self.dt)
        return [magn*math.cos(angle),magn*math.sin(angle)]

    #------- get functions -----------
    def get_position(self):
        return self.p

    def get_velocity(self):
        return self.v

    def get_acceleration(self):
        return self.a

    # ------ update functions---------
    # make sure it is done in this order: update_F, update_a, update_v, update_c

    def update_sim(self):
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        e_botword = model.encode(self.word)
        sim_matrix = [0]*len(self.model.choices)
        for o in self.model.choices:
            e_o = model.encode(o)
            sim_matrix[o] = util.cos_sim(e_botword, e_o)
        return sim_matrix

    def update_F(self):
        # switch with a probability based on current conviction
        if random.random()<self.convm[self.allwords.index(self.word)] :
            # switch to another word with probability based on conviction
            new_index = stats.rv_discrete(values=(np.arrange(3),self.convm))
            self.word = self.allwords[new_index]
            self.simm = self.update_sim()
            self.choice = self.model.choices[self.simm.index(max(self.simm))]
        # should make more sophisticated but will use range(max-min) for now
        # will need the current distance from the choice puck as well as the direction of the choice
        urgcare = max(self.simm) - min(self.simm)
        # affect how close we want to be to puck- could have a standed want_d that is scaled up or down
        # based on how much we care, it would be scaled higher the less we care
        k = 0.8
        basethresh = 1
        if urgcare < 0.2:
            return self.F * k
        else:
            threshold = basethresh(1-urgcare)
            if self.dist < threshold:
                return self.calc_F(threshold)
            else:
                return self * k


    def update_a(self):
        self.a = self.F/ self.m

    def update_v(self):
        self.v = self.v + self.a*self.dt

    def update_p(self):
        self.p = self.p + self.v*self.dt
        self.dist = math.dist(self.p,self.model.p)

