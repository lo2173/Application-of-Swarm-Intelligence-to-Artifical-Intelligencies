# Author: Lois I Omotara
# test.py
# client class to create animation
# 12/8 goals: solidify current directioning system, introduce word embeddings, done with basic slides
import numpy as np
import time
import matplotlib.pyplot as plt
import math
import chatbot_puck as cp
import model_puck as mp
from matplotlib import animation
from functools import partial
# used code from: https://pythonalgos.com/2021/12/20/create-animations-with-matplotlib-animation-funcanimation/
# gotta use FuncAnimation to do the animation here(may have to write a seperate class)
def main():
    fig, ax = plt.subplots()
    ax.set_ylim([-6,6])
    ax.set_xlim([-6,6])
    #----- simul data -----------
    choices = ['darkness','shadows','screams','lost']
    mp1 = mp.ModelPuck(choices=choices,debug=False)
    cgpt_w = ['darkness','dread','silence','whispers','creak','beneath','shadows','eerie']
    cgpt_freq = [2,2,1,1,1,1,1]
    koala_w = ['darkness','shadows','ominous','alone','haunting','midnight','unsettling','chilling','whispers','shivers']
    koala_freq =[1]*10
    claud_w = ['screams','midnight','shadows','glimpse','lost']
    claud_freq =[1,3,3,2,1]
    perp_w = ['nightmare']
    perp_freq = [10]
    hug_w = ['darkness', 'shadows', 'nightfall', 'solitude', 'whispers', 'unseen', 'veil', 'hushed', 'echoes']
    hug_freq = [2,1,1,1,1,1,1,1,1]
    pi_w = ['darkness']
    pi_freq = [10]
    mp1 = mp.ModelPuck(choices=choices,debug=False)
    cgpt = cp.ChatbotPuck(model=mp1,name = 'cgpt',convic_words=cgpt_w,convic_freq=cgpt_freq,debug=False)
    claud = cp.ChatbotPuck(model =mp1,name = "claud",convic_words=claud_w, convic_freq=claud_freq,debug=False)
    koala = cp.ChatbotPuck(model=mp1,name='koala',convic_words=koala_w,convic_freq=koala_freq)
    mp1.set_chatbots([cgpt,claud,koala])
    #--------set stage-----------
    # choice1, = plt.plot(-5,-5)
    plt.text(-5,-5,'darkness')
    # choice2, = plt.plot(5,5, 'ro')
    plt.text(5,5,'shadows')
    # choice3, = plt.plot(-5,5, 'ro')
    plt.text(-5,5,'screams')
    # choice4, = plt.plot(5,-5, 'ro')
    plt.text(5,-5,'lost')
    cgptbot, = plt.plot(1,1,'bo')
    claudbot, = plt.plot(-1,-1,'go')
    # perbot, = plt.plot(1,-1, 'mo')
    koalabot, = plt.plot(1,-1,'mo')
    puck, = plt.plot(0,0,'ko')
    start = time.time()
    # ------ run animation---------
    def func(i,mp1):
        if(i>0):
            cgpt.update()
            claud.update()
            # perplex.update()
            koala.update()
            mp1.update(start)
        cgptbot.set_data(cgpt.get_p())
        claudbot.set_data(claud.get_p())
        # perbot.set_data(perplex.get_p())
        koalabot.set_data(koala.get_p())
        puck.set_data(mp1.get_p())
        # print('DOT: ',dot3.get_data())
        return puck,
    ani = animation.FuncAnimation(fig, partial(func,mp1 = mp1), frames=np.arange(0,100), interval=.01)
    # ani.save('./finalone.gif',writer =animation.PillowWriter(fps=30))
    plt.show()

if __name__ == "__main__":
    main()
