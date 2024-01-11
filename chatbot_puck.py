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
    def __init__(self,name, convic_words=None, convic_freq=None, model=None,debug = False):
        self.flip = 1
        self.name = name
        self.invar = [0,0]
        self.m = 1
        self.p = np.array([1,1])
        self.v = np.array([0,0])
        self.a = np.array([0.0])
        self.dt = 0.1
        self.change = 1
        self.debug = debug
        self.model = model
        self.dist = math.dist(self.p,self.model.get_p())
        self.allwords = convic_words
        # self.word = convic_words[convic_freq.index(max(convic_freq))]
        # self.choice = self.model.choices[self.simm.index(max(self.simm))]
        # self.dampF = 0.1
        self.simm = None
        self.word =  None
        self.choice = None
        self.choose(convic_words,convic_freq)
        # print(self.name,' choice: ',self.choice)
        self.convm = softmax(convic_freq)
        # print(self.name," convic: ",self.convm)
        _,self.dirn = self.model.get_choicep(self.choice)
        self.F = self.dirn
        if debug == True:
            self.choice = 2
        # right now convm limited to 3 word max

    #--------------------------------
    def choose(self,convic_words,convic_freq):
        if len(set(convic_freq)) == 1:
            self.word = convic_words[convic_freq.index(max(convic_freq))]
            self.simm = self.update_sim()
            self.choice = self.model.choices[self.simm.index(max(self.simm))]
            return
        convic_freq = np.array(convic_freq)
        words_indices = np.where(convic_freq == max(convic_freq))[0]
        maxindex = -1
        maxsimm = -1
        for i in words_indices:
            newsimm = self.update_sim(word = convic_words[i])
            if max(newsimm) > maxsimm:
                maxindex = i
                maxsimm = max(newsimm)
        self.word = convic_words[maxindex]
        self.simm = self.update_sim()
        self.choice = self.model.choices[self.simm.index(max(self.simm))]

    # ------ changing position-------
    def change_choice(self):
        choice_p = 0
        # if self.debug == False:
        choice_p,self.dirn = self.model.get_choicep(self.choice)
        # print(self.name ,' choice: ', self.choice )
        # diff = self.p + choice_p
        # self.p = self.p - diff/self.change
        # self.p = (abs(self.p) - 0.5)*-self.dirn
        self.p = self.dirn
        self.change = 0
    # ------ calculations-------------
    # def calc_F(self,threshold):
    #     magn = 2*self.m*(threshold-(self.v*self.dt))/(self.dt*self.dt)
    #     self.F = self.dirn * (abs(magn)*self.convm[self.allwords.index(self.word)])*self.dampF
        # return [magn,magn]

    #------- get functions -----------

    def get_p(self):
        return self.p

    def get_v(self):
        return self.v

    def get_a(self):
        return self.a


    # ------ update functions---------
    # make sure it is done in this order: update_F, update_a, update_v, update_c

    def update_sim(self,word = None):
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        if word is None: word = self.word
        e_botword = model.encode(word)
        sim_matrix = [0]*len(self.model.choices)
        i = 0
        for o in self.model.choices:
            e_o = model.encode(o)
            sim_matrix[i] = util.cos_sim(e_botword, e_o)
            i = i+1
        return sim_matrix

    def update_F(self):
        conv = self.convm[self.allwords.index(self.word)]
        if(self.debug == True):
            self.F = self.dirn * abs(self.F) * 0.8
            return
        if(math.dist(self.p,self.model.p)>2): self.flip = -self.flip
        if(random.random()<=conv):
            self.F = self.dirn*self.flip
            return
        # switch with a probability based on current conviction
        urgcare = (max(self.simm) - min(self.simm))[0][0].numpy()
        # print(self.name, " urgcare: ", urgcare)
        if (random.random()>conv+0.1) & (urgcare < 0.7):
            # switch to another word with probability based on conviction
            new_index = stats.rv_discrete(values=(np.arange(len(self.convm)),self.convm)).rvs(size = 1)[0]
            # ?print('NEW_INDEX: ',new_index)
            self.word = self.allwords[new_index]
            self.simm = self.update_sim()
            new_choice = self.model.choices[self.simm.index(max(self.simm))]
            if new_choice == self.choice:
                self.F = self.dirn * (abs(self.F) * 0.8)*self.flip
                return
            self.choice = new_choice
            self.change = 2
            self.F = abs(self.F) * 0.6
            return
        self.F = self.dirn * (abs(self.F) * 0.8)*self.flip
    def update_a(self):
        self.a = self.F/ self.m

    def update_v(self):
        self.v = self.v + self.a*self.dt

    def update_p(self):
        self.p = self.p + self.v*self.dt
        if self.p[0]>5:
            self.p[0] = 5
        if self.p[0] < -5:
            self.p[0] = -5
        if self.p[1]>5:
            self.p[1] = 5
        if self.p[1] < -5:
            self.p[1] = -5

    #  ----- Func Update ------------------
    def reset(self):
        self.p = np.array([1,1])
        self.v = np.array([0,0])
        self.a = np.array([0.0])
        self.F = np.array([0,0])

    def update(self):
        self.update_F()
        # print(self.name ,'F: ', self.F)
        if self.model.halt:
            self.p = [0,0]
            return
        if(self.change > 0):
            self.change_choice()
        else:
            self.update_a()
            self.update_v()
            self.update_p()
            self.dist = math.dist(self.p,self.model.get_p())
            # print(self.name," F: ",self.F)
            # print(self.name, "F: ",self.F)
            # print("Am: ",self.a)
            # print('F: ',self.F)
            # print("Vm: ",self.v)


        return 1

