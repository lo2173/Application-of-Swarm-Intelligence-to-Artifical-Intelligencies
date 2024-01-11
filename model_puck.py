# Author: Lois I Omotara
# ModelPuck.py
import random
import numpy as np
import math
from sentence_transformers import SentenceTransformer,util
import chatbot_puck
from scipy.special import softmax
from scipy import stats
import scipy
import matplotlib.pyplot as plt
import time
# needs terminiation condition
class ModelPuck :
    def __init__(self,chatbots=None, choices = None,debug=True):
        self.p = np.array([0,0])
        self.v = np.array([0,0])
        self.a = np.array([0.0])
        self.F = np.array([0,0])
        self.m = 2
        self.dt = 0.1
        self.debug = debug
        self.chatbots = chatbots
        self.choices = choices
        self.choicep  = np.array([[-5,-5], [5,5],[-5,5],[5,-5]])
        self.choiced = np.array([math.sqrt(50)]*4)
        self.haltd = 0.1
        self.halt = False

    # ------- calculations----------
    def calc_choicesd(self):
        self.choiced = scipy.spatial.distance.cdist([self.p], self.choicep)

    # ------ set functions------------
    def set_chatbots(self,chatbots):
        self.chatbots = chatbots

    def set_choices(self,choices):
        self.choices = choices
    # ------ get functions -----------
    def get_choicep(self, choice):
        # define position of choices here, corners of simulation
        choice_direction = np.array([[-1,-1],[1,1],[-1,1],[1,-1]])
        if self.debug == False:
            c = self.choices.index(choice)
            return self.choicep[c], choice_direction[c]
        else:
            return self[choice], choice_direction[choice]

    def get_dt(self):
        return self.dt

    def get_m(self):
        return self.m

    def get_p(self):
        return self.p

    def get_v(self):
        return self.v

    def get_a(self):
        return self.a


    #-------- update functions --------
    def update_F(self):
        # revisit gravitational stuff
        sum = 0
        for cp in self.chatbots:
            sum = sum+((abs(cp.F) * cp.dirn)/math.dist(self.p,cp.p))
        self.F = sum
        # print("model F: ",self.F)
    # def update_F(self):
    #     sum = np.array([0,0])
    #     G = 6.6743 * (10**-11)
    #     for cb in self.chatbots:
    #         cp = cb.get_p()
    #         base = (G*(self.m)*(cb.get_m()))/(math.dist(cp,self.p))**(3/2)
    #         # print("BASE: ",base)
    #         # base = base *10**10
    #         sum= sum+((abs(self.p - cp))*base)*cb.dirn
    #         # print("DIFF: ", self.p-cp)
    #     # adjust to use chatbots param
    #     self.F = sum
    #     # print("d",math.dist(cp,self.p))
    #     # print("F",self.F)

    def update_a(self):
        self.a = self.F/ self.m

    def update_v(self):
        self.v = self.v + self.a*self.dt

    def update_p(self):
        self.p = (self.p + self.v*self.dt)
        if self.p[0]>5:
            self.p[0] = 5
        if self.p[0] < -5:
            self.p[0] = -5
        if self.p[1]>5:
            self.p[1] = 5
        if self.p[1] < -5:
            self.p[1] = -5

    def update(self,start):
        self.calc_choicesd()
        # print('DIST = ',np.min(self.choiced))
        if(np.min(self.choiced) <= 0.99):
            end = time.time()
            print('elapsed time: ',end-start)
            print("DECISION = ", self.choices[np.argmin(self.choiced)])
            plt.text(0,1,'decision: '+self.choices[np.argmin(self.choiced)])
            self.halt = True
            # exit()
            return self.choices[np.argmin(self.choiced)], end-start

        self.update_F()
        self.update_a()
        self.update_v()
        self.update_p()
        # print('model F: ',self.F)
        # print('model p: ',self.p)

        return -1
