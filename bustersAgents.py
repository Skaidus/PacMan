from __future__ import print_function
from wekaI import Weka
import random
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
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from builtins import range
from builtins import object
import util
from game import Agent
from game import Directions
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


previousLivingGhosts = -1

class BasicAgentAA(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0
        
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
        #print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print("---------------- TICK ", self.countActions, " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())
        # Pacman direction
        print("Pacman direction: ", gameState.data.agentStates[0].getDirection())
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts())
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ", [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood())
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ", gameState.getDistanceNearestFood())
        # Map walls
        print("Map:")
        print( gameState.getWalls())
        # Score
        print("Score: ", gameState.getScore())
    
    
    def chooseAction(self, gameState):
        global previousLivingGhosts
        # Inicializa el contador de fantasmas vivos en caso de que sea el primer turno
        if previousLivingGhosts == -1: previousLivingGhosts = gameState.getLivingGhosts().count(True)
        self.countActions = self.countActions + 1
        self.printInfo(gameState)
        legal = gameState.getLegalActions(0) ##Legal position from the pacman

        # Si solo hay una opción legal, devuelve esa
        if len(legal) == 2: return legal[0]
        # Escoge un primer fantasma vivo arbitrario para empezar a buscar el más cercano
        nearestLivingGhost = -1
        livingGhosts = gameState.getLivingGhosts().count(True)
        i = 0
        # Itera para hallar el fantasma más cercano a Pacman 
        while i < len(gameState.getLivingGhosts()) - 1:
            if gameState.getLivingGhosts()[i+1] == True:
                currentGhostDistance = gameState.data.ghostDistances[i]
                if(nearestLivingGhost == -1 or currentGhostDistance < gameState.data.ghostDistances[nearestLivingGhost]) :
                    nearestLivingGhost = i
            i += 1
        # Almacena el resultado
        nearestGhostPositions = gameState.getGhostPositions()[nearestLivingGhost]
        actualPosition = gameState.getPacmanPosition()
        prevMove = gameState.data.agentStates[0].getDirection()
        # Escoge un movimiento legal arbitrario
        best_move = legal[0]
        i = 0
        # Itera para asegurarse que cumple las condiciones de no ser el movimiento inverso del anterior o que se quede parado
        while (Directions.REVERSE[best_move] == prevMove and livingGhosts == previousLivingGhosts) or best_move == Directions.STOP:
            i+=1
            best_move = legal[i]
        # Itera sobre todos los legales con objetivo de encontrar uno que deje a Pacman a menor distancia y cumpla las restricciones
        for move in legal:
            if (Directions.REVERSE[move] != prevMove or livingGhosts != previousLivingGhosts) and move != Directions.STOP:
                if util.manhattanDistance(self.applyDirection( actualPosition, best_move), nearestGhostPositions) > util.manhattanDistance(self.applyDirection(actualPosition,move), nearestGhostPositions):
                    best_move = move

        # Actualiza el contador de fantasmas vivos
        previousLivingGhosts = livingGhosts
        return best_move

    # Funcion auxiliar para predecir la posicion de pacman al aplicar una dirección
    def applyDirection(self, xy, direction):
        if direction == Directions.SOUTH:
            return [xy[0], xy[1]-1]
        elif direction == Directions.EAST:
            return [xy[0]+1, xy[1]]
        elif direction == Directions.WEST:
            return [xy[0]-1, xy[1]]
        else: 
            return [xy[0], xy[1]+1]

    
