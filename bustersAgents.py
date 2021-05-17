from __future__ import print_function
from wekaI import Weka
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

    def __init__( self, index = 0, inference = "ExactInference", ghostAgents = None, observeEnable = True, elapseTimeEnable = True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable
        self.weka = Weka()
        self.weka.start_jvm()
        # Para escoger cual clasificador se aplicará
        self.umbral_confianza = .6

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
    def __init__(self, alpha=1.0, epsilon=0.05, gamma=0.8, numTraining = 10):
        "Initialize Q-values"
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.numTraining = int(numTraining)

        self.index = 0  # This is always Pacman

        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.numTraining = int(numTraining)

        self.legalActions = []
        self.nextLegalActions = []
        self.actions = {"north":0, "east":1, "south":2, "west":3, "exit":4}
        self.table_file = open("qtable.txt", "r+")
        self.q_table = self.readQtable()
        self.epsilon = 0.05

    def update(self, gameState, action, nextGameState):
        """
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        You should do your Q-Value update here

        Good Terminal state -> reward 1
        Bad Terminal state -> reward -1
        Otherwise -> reward 0

        Q-Learning update:

        # Funcion de actualizacion no determinista: (aplpha es la tasa de aprendizaje [0,1]
        # alpha nos permite modular la agresividad a la hora de actualizar la tabla Q
        if terminal_state:
        Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + 0)
        else:
        Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + self.discount * max a' Q(nextState, a'))

        """
        qState = self.getQState(gameState)
        self.legalActions = gameState.getLegalPacmanActions()
        qNextState = self.getQState(nextGameState)
        self.nextLegalActions = nextGameState.getLegalPacmanActions()

        # TRACE for transition and position to update. Comment the following lines if you do not want to see that trace
        reward = self.rewardFunction(qState, action, qNextState)
        print("Update Q-table with transition: ", qState, action, qNextState, reward)
        position = self.computePosition(qState)
        action_column = self.actions[action]
    
        print("Corresponding Q-table cell to update:", position, action_column)
        position = self.computePosition(qState)
        
        
        "*** YOUR CODE HERE ***"

        if qState == (3,2) or qState == (3,1):
            self.q_table[position][action_column] = (1-self.alpha) * self.getQValue(qState,action) + self.alpha * reward;
        else:
            self.q_table[position][action_column] = (1-self.alpha) * self.getQValue(qState,action) + self.alpha * (reward + self.discount*self.computeValueFromQValues(qNextState) );
        # TRACE for updated q-table. Comment the following lines if you do not want to see that trace
        print("Q-table:")
        self.printQtable()

    def rewardFunction(self, state, action, nextState):
        """
        --> state: (arriba..., )
        -> + reward si se acerca
        -> ++ reward si se come un fantasma (<vivos)
        -> ++ reward si se come algo 
        """

        # state[0] == atributo 1 (dir ghost mas cercano)
        reward = 0
        atr1 = state[0]
        if atr1[0] != "" and atr1[0] != action:
            reward -= 1
        elif atr1[1] != "" and atr1[1] != action:
            reward -= 1
                    
        return reward

### {gameState -> qState
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
            if gameState.getLivingGhosts()[i+1] == True:
                currentGhostDistance = gameState.data.ghostDistances[i]
                if(nearestLivingGhost == -1 or currentGhostDistance < gameState.data.ghostDistances[nearestLivingGhost]) :
                    nearestLivingGhost = i
            i += 1
        nearestGhostPositions = gameState.getGhostPositions()[nearestLivingGhost]
        actualPosition = gameState.getPacmanPosition()

        x_axis = ''
        if nearestGhostPositions[0] > actualPosition[0]:
            x_axis ='east'
        elif nearestGhostPositions[0] < actualPosition[0]:
            x_axis ='west'

        y_axis = ''
        if nearestGhostPositions[1] > actualPosition[1]:
            y_axis ='north'
        elif nearestGhostPositions[1] < actualPosition[1]:
            y_axis ='south'
        
        
        qState.append((x_axis, y_axis))
        # Atributo 2

        # Devuelve el estado
        return qState

### gameState -> qState}

### {qState -> qTable[i]
    def computePosition(self, qState):
        """
        Compute the row of the qtable for a given state.
        For instance, the state (3,1) is the row 7
        """
        hash = {'':0, 'east':1, 'west':2, 'north':3, 'south':6}
        atr1 = qState[0]
        return hash[atr1[0]] + hash[atr1[1]] - 1
        # return state[0]+state[1]*
### qState -> qTable[i]}

##  {RW Qtable
    def readQtable(self):
        "Read qtable from disc"
        table = self.table_file.readlines()
        q_table = []

        for i, line in enumerate(table):
            row = line.split()
            row = [float(x) for x in row]
            q_table.append(row)

        return q_table

    def writeQtable(self):
        "Write qtable to disc"
        self.table_file.seek(0)
        self.table_file.truncate()
        for line in self.q_table:
            for item in line:
                self.table_file.write(str(item)+" ")
            self.table_file.write("\n")

    def printQtable(self):
        "Print qtable"
        for line in self.q_table:
            print(line)
        print("\n")    

##  RW Qtable}

## {Bellman stuff

    ##{ Pipe game loop and Bellman 
    def getAction(self, gameState):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """
        self.legalActions = gameState.getLegalPacmanActions()
        qState = self.getQState(gameState)
        # Pick Action
        action = None

        if len(self.legalActions) == 0:
             return action

        flip = util.flipCoin(self.epsilon)

        if flip:
            return random.choice(self.legalActions)
        return self.getPolicy(qState)

    ## Pipe game loop and Bellman}

    ## {Q(S,A)
    def getQValue(self, qState, action):

        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        position = self.computePosition(qState)
        action_column = self.actions[action]

        return self.q_table[position][action_column]
    ## Q(S,A)}

    ## {max Q(S,A)
    def computeValueFromQValues(self, qNextState):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        if len(self.nextLegalActions)==0:
          return 0
        return max(self.q_table[self.computePosition(qNextState)])
    ## max Q(S,A)}

    ## {argmax Q(S,A)
    def computeActionFromQValues(self, qState):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
      #  legalActions = self.getLegalActions(state) # get legal actions.
        if len(self.legalActions)==0:
          return None

        best_actions = [self.legalActions[0]]
        best_value = self.getQValue(qState, self.legalActions[0])
        for action in self.legalActions:
            value = self.getQValue(qState, action)
            if value == best_value:
                best_actions.append(action)
            if value > best_value:
                best_actions = [action]
                best_value = value

        return random.choice(best_actions)

    def getPolicy(self, qState):
        "Return the best action in the qtable for a given state"
        return self.computeActionFromQValues(qState)

    ## argmax Q(S,A)}

## Bellman stuff}

## {Misc.
    def __del__(self):
        "Destructor. Invokation at the end of each episode"
        self.writeQtable()
        self.table_file.close()
## Misc.}

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


