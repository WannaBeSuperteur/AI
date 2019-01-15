import math
import random

# read data from file
def getData():
    # get data
    file = open('QLearning.txt', 'r')
    read = file.readlines()
    file.close

    states = int(read[0]) # number of states
    rewards = [] # reward data (also state data)
    moves = [] # move data
    
    for i in range(1, len(read)):
        row = read[i].split('\n')[0].split(' ')

        # append
        if i <= states:
            row[1] = int(row[1])
            rewards.append(row)
        else:
            row.append(0)
            moves.append(row)
        
    return (rewards, moves)

# lookup table -> return index
def lookup(table, coln, value):
    for i in range(len(table)):
        if table[i][coln] == value: return i
    return -1

# do Q Learning
def QLearning(rewards, moves):
    # print
    print('Rewards:')
    for i in range(len(rewards)): print(rewards[i])
    
    times = 0
    lastModifiedTimes = 0 # times when action table was lastly changed
    while 1:
        # round info
        times += 1
        doRandom = 10/(times+10) # probability of searching randomly
        print('')
        print('----------------<<<< ROUND ' + str(times) + ' >>>>----------------')

        # do Q learning until the last state
        state = 0
        while state != len(rewards)-1:

            # print
            symbol = rewards[state][0] # symbol of state
            print('')
            print('symbol of state: ' + symbol)
            for i in range(len(moves)): print(moves[i])

            # find all possible actions
            nextstate = []
            for i in range(len(moves)):
                if moves[i][0] == symbol: nextstate.append(moves[i])
            print('possible next moves: ' + str(nextstate))

            # decide next action
            nextActionIndex = -1 # index of next action
            # RANDOM
            if random.random() < doRandom:
                print('decide RANDOM')
                next_ = random.randint(0, len(nextstate)-1)
                state = lookup(rewards, 0, nextstate[next_][1]) # update state
                nextActionIndex = next_
            # BEST state
            else:
                print('decide BEST')
                bestVal = 0
                bestIndex = -1
                for i in range(len(nextstate)):
                    if nextstate[i][2] > bestVal:
                        bestVal = nextstate[i][2]
                        bestIndex = i

                # best of next state (same value as best choice)
                # choose best next state randomly
                bestNextState = []
                for i in range(len(nextstate)):
                    if nextstate[i][2] == bestVal:
                        bestNextState.append(nextstate[i])
                selectBestNext = random.randint(0, len(bestNextState)-1)
                
                state = lookup(rewards, 0, bestNextState[selectBestNext][1]) # update state
                nextActionIndex = i

            # get immediate reward r(s, a)
            r = rewards[state][1]

            # find new state s' -> y*max(a')(Q(s', a'))
            symbol = rewards[state][0]
            maxReward = 0 # next action that has max reward 
            for i in range(len(moves)):
                if moves[i][0] == symbol:
                    if moves[i][2] > maxReward: maxReward = moves[i][2]

            # update state
            value = r + 0.5 * maxReward
            if nextstate[nextActionIndex][2] != value:
                lastModifiedTimes = times
                nextstate[nextActionIndex][2] = value
                print('update ' + str(nextstate[nextActionIndex][0]) + ' to '
                  + str(nextstate[nextActionIndex][1]) + ': ' + str(value))

            print('symbol of state : ' + symbol)
            print('immediate reward: ' + str(r))
            print('max reward      : ' + str(maxReward))

        # break when converged
        if lastModifiedTimes * 2 + 10 < times: break
        
(rewards, moves) = getData()
QLearning(rewards, moves)
