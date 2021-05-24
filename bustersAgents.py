from __future__ import print_function

import random, util
from sys import maxsize

# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu). :D


from builtins import range
from builtins import object
import util
from game import *
from keyboardAgents import KeyboardAgent
import inference
import busters

class NullGraphics(object):
    "Placeholder for graphics"
    def initialize(self, state, isBlue = False):
        pass
    def update(self, state):
        pass
    def pause(self):
        pass
    def draw(self, state):
        pass
    def updateDistributions(self, dist):
        pass
    def finish(self):
        pass

class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """
    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent(object):
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__(self, index=0, inference="ExactInference", ghostAgents=None, observeEnable=True, elapseTimeEnable=True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable
        self.switch = 1

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True


    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        #for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        #self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        x = self.getClassifierStatus(gameState)
        tree = self.weka.predict("./RandomTree.model", x.copy(), "./training_set_c.arff")
        lwl = self.weka.predict("./LWL.model", x.copy(), "./training_set_c.arff")
        seed = random.random()
        if seed < self.umbral_confianza:
            best_move = tree
        else:
            best_move = lwl
        moves = ['North', 'South', 'West', 'East']
        best_move = best_move.tolist()
        moves, best_move = self.eliminate_illegal(gameState.getLegalPacmanActions(), moves, best_move)
        return moves[best_move.index(max(best_move))]

    def eliminate_illegal(self, legal, moves, chances):
        removed = []
        new_chances = []
        new_moves = []
        for i in range(len(moves)):
            if moves[i] in legal:
                new_chances.append(chances[i])
                new_moves.append(moves[i])
        return new_moves, new_chances

    def getClassifierStatus(self, gameState):

        classifierStatus = []
        #Pacman position
        for i in range(2):
            classifierStatus.append(str(gameState.getPacmanPosition()[i]))
        #Pacman legal moves
        moves = {'North', 'South', 'West', 'East'}
        for move in moves:
            if move in gameState.getLegalPacmanActions():
                classifierStatus.append('1')
            else:
                classifierStatus.append('0')
        #Ghost positions
        for i in range(len(gameState.getGhostPositions())):
            if gameState.data.ghostDistances[i]==None:
                for j in range(2):
                    classifierStatus.append(str(-1))
            else :
                for j in range(2):
                    classifierStatus.append(str(gameState.getGhostPositions()[i][j]))
        #Ghost distances                    
        for each in gameState.data.ghostDistances:
            if each == None:
                classifierStatus.append(str(-1))
            else:
                classifierStatus.append(str(each))
        #Score
        classifierStatus.append(str(gameState.getScore()))
        #DistanceFood
        classifierStatus.append("-1") if str(gameState.getDistanceNearestFood()) == 'None' else classifierStatus.append(str(gameState.getDistanceNearestFood()))
        #RemainingFood
        classifierStatus.append(str(gameState.getNumFood()))
        return classifierStatus

    def printLineData(self, gameState, step):

        ghostPositions = ""
        for i in range(len(gameState.getGhostPositions())):
            if gameState.data.ghostDistances[i]==None:
                ghostPositions += str(-1) + "," + str(-1) + ","
            else :
                ghostPositions += str(gameState.getGhostPositions()[i][0]) + "," + str(gameState.getGhostPositions()[i][1]) + ","

        ghostDistances = ""
        for each in gameState.data.ghostDistances:
            if each == None:
                ghostDistances += str(-1)
            else:
                ghostDistances += str(each)
            ghostDistances += ","
        ghostDirections = ""
        for i in range(len(gameState.getGhostDirections())):
            if gameState.data.ghostDistances[i]==None:
                ghostDirections += "\'" + "Dead" + "\'" + ","
            else: 
                ghostDirections += "\'" + str(gameState.getGhostDirections()[i]) + "\'" + ","
        for i in range(4-len(gameState.getGhostDirections())):
            ghostDirections+= "\'" + "Dead" + "\'" + ","

        moves = {'North', 'South', 'West', 'East', 'Stop'}
        legalActions = ""
        for move in moves:
            if move in gameState.getLegalPacmanActions():
                legalActions+='1,' 
            else:
                legalActions+='0,'

        distNearestFood = '-1' if str(gameState.getDistanceNearestFood()) == 'None' else str(gameState.getDistanceNearestFood())


        if step == 0: 
            next_state = ''
        else: 
            next_state =  ''.join(str(gameState.getPacmanPosition()[0]) + "," + str(gameState.getPacmanPosition()[1]) +
            ","+ legalActions +
            ghostPositions + ghostDistances + ghostDirections +
            str(gameState.getScore()) +
            ","+ distNearestFood +
            ","+ str(gameState.getNumFood()) +
            ","+ "\'" + str(gameState.data.agentStates[0].getDirection()) + "\'" + "\n")

        current_state = ''.join(str(gameState.getPacmanPosition()[0]) + "," + str(gameState.getPacmanPosition()[1]) +
        "," +  str(gameState.getNumAgents() - 1) +
        ","+ legalActions +
        ghostPositions + ghostDistances + ghostDirections +
        str(gameState.getScore()) +
        ","+ distNearestFood +
        ","+ str(gameState.getNumFood()) +
        ","+ "\'" + str(gameState.data.agentStates[0].getDirection()) + "\'" + ",")

        return next_state + current_state

################################################################
#                         PRÁCTICA 2                           #
################################################################

class QLearningAgent(BustersAgent):
    """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    def computePosition(self, qState):

        hash = {'':0, 'East':1, 'West':2, 'North':3, 'South':6}
        atr1 = qState[0]
        return hash[atr1[0]] + hash[atr1[1]] - 1 + 8 * int(qState[1])
        # return state[0]+state[1]*

    def getQState(self, gameState):
        """
        Generates a Q table state from the gamestate
        -> Direccion para el fantasma mas cercano
        --> Bucle indice fantasma mas cercano
        --> Comparar x, y con pacman
        """
        qState = []
        # Atributo 1
        i = 0
        nearestLivingGhost = -1
        while i < len(gameState.getLivingGhosts()) - 1:
            if gameState.getLivingGhosts()[i + 1] == True:
                currentGhostDistance = gameState.data.ghostDistances[i]
                if (nearestLivingGhost == -1 or currentGhostDistance < gameState.data.ghostDistances[
                    nearestLivingGhost]):
                    nearestLivingGhost = i
            i += 1

        # Get nearest food
        nearestGhostPositions = gameState.getGhostPositions()[nearestLivingGhost]
        nearestObj = nearestGhostPositions

        row_i = 0
        col_i = 0
        if (gameState.getNumFood() > 0):
            minDistance = 900000
            pacmanPosition = gameState.getPacmanPosition()
            for i in range(gameState.data.layout.width):
                for j in range(gameState.data.layout.height):
                    if gameState.hasFood(i, j):
                        foodPosition = i, j
                        distance = util.manhattanDistance(pacmanPosition, foodPosition)
                        if distance < minDistance:
                            minDistance = distance
                            row_i = i
                            col_i = i

            nearestGhostDistance = gameState.data.ghostDistances[nearestLivingGhost]
            if nearestGhostDistance < minDistance:
                nearestObj = nearestGhostPositions
            else:
                nearestObj = [row_i, col_i]

        actualPosition = gameState.getPacmanPosition()

        x_axis = ''
        if nearestObj[0] > actualPosition[0]:
            x_axis = 'East'
        elif nearestObj[0] < actualPosition[0]:
            x_axis = 'West'

        y_axis = ''
        if nearestObj[1] > actualPosition[1]:
            y_axis = 'North'
        elif nearestObj[1] < actualPosition[1]:
            y_axis = 'South'

        qState.append((x_axis, y_axis))
        # Atributo 2
        touchWall = False
        legals = gameState.getLegalPacmanActions()
        if ((x_axis not in legals) and x_axis != '') or ((y_axis not in legals) and y_axis != ''):
            touchWall = True
        qState.append(touchWall)

        return qState

    def rewardFunction(self):
        reward = 0
        atr1 = self.lastQState[0]
        if atr1[0] != '' and atr1[0] != self.lastAction and self.lastAction != 'North' and self.lastAction != 'South':
            reward -= 1
        elif atr1[1] != '' and atr1[1] != self.lastAction and self.lastAction != 'West' and self.lastAction != 'East':
            reward -= 1
        else:
            reward += 5

        latr2 = self.lastQState[1]
        atr2 = self.currentQState[1]

        if latr2:
            if atr2:
                print('Sigue tocando!')
                reward+=10
            else:
                print('Se separó del muro! Mal!')
                reward-=10
        return reward

    # alpha = 0.2 epsilon = 0.05
    def __init__(self, alpha=0.1, epsilon=0.4, gamma=0.8, numTraining = 10,  index=0, inference="ExactInference", ghostAgents=None, observeEnable=True, elapseTimeEnable=True):

        BustersAgent.__init__(self,index,inference, ghostAgents, observeEnable, elapseTimeEnable)
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0

        self.lastState = None
        self.lastQState = None

        self.lastAction = None

        self.currentState = None
        self.currentQState = None

        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.numTraining = int(numTraining)
        self.index = 0  # This is always Pacman

        self.actions = {"North": 0, "East": 1, "South": 2, "West": 3}
        self.table_file = open("qtable.txt", "r+")
        self.q_table = self.readQtable()

    def registerInitialState(self, state):
        if self.episodesSoFar == 0:
            print('Beginning %d episodes of Training' % (self.numTraining))
        if os.path.exists("qtable.txt"):
            if self.switch == 1:
                self.table_file = open("qtable.txt", "r+")
                self.q_table = self.readQtable()
                self.switch = 0
        else:
            self.table_file = open("qtable.txt", "w+")

    def observationFunction(self, gameState):
        """
            This is where we ended up after our last action.
            The simulation should somehow ensure this is called
        """
        self.currentState = gameState
        self.updateQStates()
        if not self.lastState is None:
            reward = self.currentState.getScore() - self.lastState.getScore()
            self.update(reward)
        return gameState

    def updateQStates(self):
        if self.lastState is not None:
            self.lastQState = self.getQState(self.lastState)
        if self.currentState is not None:
            self.currentQState = self.getQState(self.currentState)



    def update(self, scoreDiff):

        # TRACE for transition and position to update. Comment the following lines if you do not want to see that trace
        print("Diferencia puntuacion: "+str(scoreDiff))
        custom = self.rewardFunction()
        print("Custom reward: "+str(custom))
        reward = scoreDiff + custom
        print("Update Q-table with transition: ", self.lastQState, self.lastAction, self.currentQState, reward)
        position = self.computePosition(self.lastQState)
        action_column = self.actions[self.lastAction]
        print("Corresponding Q-table cell to update:", position, action_column)
        
        
        "*** YOUR CODE HERE ***"

        if len(self.lastState.getLivingGhosts()) == 0:
            self.q_table[position][action_column] = (1-self.alpha) * self.getQValue(self.lastQState,self.lastAction) + self.alpha * reward;
        else:
            self.q_table[position][action_column] = (1-self.alpha) * self.getQValue(self.lastQState,self.lastAction) + self.alpha * (reward + self.discount*self.getValue() );
        # TRACE for updated q-table. Comment the following lines if you do not want to see that trace
        print("Q-table:")
        self.printQtable()





    def getQValue(self, qState, action):

        position = self.computePosition(qState)
        action_column = self.actions[action]

        return self.q_table[position][action_column]

    def getValue(self):

        if len(self.currentState.getLegalPacmanActions())==0:
          return 0
        return max(self.q_table[self.computePosition(self.currentQState)])

    def readQtable(self):
        table = self.table_file.readlines()
        q_table = []

        for i, line in enumerate(table):
            row = line.split()
            row = [float(x) for x in row]
            q_table.append(row)

        return q_table

    def writeQtable(self):
        self.table_file.seek(0)
        self.table_file.truncate()
        for line in self.q_table:
            for item in line:
                self.table_file.write(str(item)+" ")
            self.table_file.write("\n")

    def printQtable(self):
        for line in self.q_table:
            print(line)
        print("\n")    

##  RW Qtable}

## {Bellman stuff

    ##{ Pipe game loop and Bellman 
    def getAction(self, gameState):

        legalActions = gameState.getLegalPacmanActions()
        legalActions.remove('Stop')

        qState = self.getQState(gameState)
        # Pick Action
        action = None

        if len(legalActions) != 0:
            flip = util.flipCoin(self.epsilon)
            if flip:
                action = random.choice(legalActions)
            else:
                action = self.getPolicy(qState, legalActions)

        self.lastState = gameState
        self.lastAction = action
        return action

    def getPolicy(self, qState, legalActions):
        if len(legalActions)==0:
          return None

        best_actions = [legalActions[0]]
        best_value = self.getQValue(qState, legalActions[0])
        for action in legalActions:
            value = self.getQValue(qState, action)
            if value == best_value:
                best_actions.append(action)
            if value > best_value:
                best_actions = [action]
                best_value = value

        return random.choice(best_actions)

    def __del__(self):
        "Destructor. Invokation at the end of each episode"
        if os.path.exists("qtable.txt"):
            self.writeQtable()
            self.table_file.close()

    def final(self, state):
        self.writeQtable()




################################################################
#                         FIN PRÁCTICA 2                       #
################################################################

class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index = 0, inference = "KeyboardInference", ghostAgents = None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)

from distanceCalculator import Distancer
from game import Actions
from game import Directions
import random, sys

'''Random PacMan Agent'''
class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        ##print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table
        
    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move

        
class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i+1]]
        return Directions.EAST


