# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if action == 'Stop':
            return -10000
        score =successorGameState.getScore() - currentGameState.getScore()
        closestFoodDistance= float('inf')
        #caclulate nearest food
        closestFood= None



        for food in newFood.asList():
            distance=manhattanDistance(food,newPos)
            if distance<closestFoodDistance:
                closestFoodDistance=distance
                closestFood= food
        #calculate nearest ghost
        if closestFood:
            score -= manhattanDistance(closestFood,newPos) * 13/29

        ghostFactor= float('inf')
        for ghost in newGhostStates:
            ghostPos= ghost.configuration.pos
            distance= manhattanDistance(ghostPos,newPos)
            timer=ghost.scaredTimer
            if timer==0 and  ghostFactor<distance:
                ghostFactor=distance
            elif ghostFactor<(distance*0.6):
                ghostFactor=distance*0.6

        if ghostFactor<5:
            score -=(5-ghostFactor)*2000
        else:
            score-= 200/ghostFactor

        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        legelActions= gameState.getLegalActions(0)
        successorStates= [gameState.generateSuccessor(0,action) for action in legelActions]
        minimaxValues= [self.computeValue(state,0,1) for state in successorStates]
        maxVal=max(minimaxValues)
        for i, val in enumerate(minimaxValues):
            if val==maxVal:
                return legelActions[i]

        return legelActions[0]

    def computeValue(self,gameState,currDepth,index):
        if(gameState.isWin() or gameState.isLose() or currDepth==self.depth):
            return scoreEvaluationFunction(gameState)
        elif index==0:
            return self.maxValue(gameState,currDepth)
        else:
            return self.miniValue(gameState,currDepth,index)

    def maxValue(self,gameState,currDepth):
        successorStates = [gameState.generateSuccessor(0, action) for action in gameState.getLegalActions(0)]
        scores=[self.computeValue(state,currDepth,1) for state in successorStates];
        return max(scores)

    def miniValue(self,gameState,depth,agentIndex):
        successorStates=[gameState.generateSuccessor(agentIndex,action) for action in gameState.getLegalActions(agentIndex)]
        ghostCount=gameState.getNumAgents()-1;
        if agentIndex==ghostCount:
            scores= [self.computeValue(state,depth+1,0) for state in successorStates]
        else:
            scores=[self.computeValue(state,depth,agentIndex+1) for state in successorStates]
        return min(scores)
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        score= float('-inf')
        alpha= float('-inf')
        beta= float('inf')
        maxAction= ''
        for action in gameState.getLegalActions(0):
            nextState= gameState.generateSuccessor(0,action)
            nextVal= self.computeValue(nextState,0,1,alpha,beta)
            if nextVal>score:
                score= nextVal
                maxAction= action
            alpha= max(alpha,score)
        return maxAction

    def computeValue(self,gameState,currDepth,index,alpha,beta):
        if(gameState.isWin() or gameState.isLose() or currDepth==self.depth):
            return scoreEvaluationFunction(gameState)
        elif index==0:
            return self.alphaBetaMaxValue(gameState,currDepth,alpha,beta)
        else:
            return self.alphaBetaMinValue(gameState,currDepth,index,alpha,beta)


    def alphaBetaMinValue(self,gameState,depth,agentIndex,alpha,beta):
        v= float('inf')
        for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            ghostCount = gameState.getNumAgents() - 1;
            if agentIndex == ghostCount:
                v= min(v,self.computeValue(successorState,depth+1,0,alpha,beta))
            else:
                v = min(v,self.computeValue(successorState, depth, agentIndex + 1,alpha,beta))
            if v<alpha:
                return v
            beta=min(beta,v)
        return v

    def alphaBetaMaxValue(self, gameState, depth, alpha, beta):
        v = float('-inf')
        for action in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, action)
            v= max(v,self.computeValue(successorState,depth,1,alpha, beta))
            if v>beta:
                return v
            alpha = max(alpha, v)
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        legelActions = gameState.getLegalActions(0)
        successorStates = [gameState.generateSuccessor(0, action) for action in legelActions]
        expectiValues = [self.computeValue(state, 0, 1) for state in successorStates]
        maxVal = max(expectiValues)
        for i, val in enumerate(expectiValues):
            if val == maxVal:
                return legelActions[i]

        return legelActions[0]

    def computeValue(self, gameState, currDepth, index):
        if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
            return self.evaluationFunction(gameState)
        elif index == 0:
            return self.expectiMax(gameState, currDepth)
        else:
            return self.probablity(gameState, currDepth, index)

    def probablity(self, gameState, depth, agentIndex):
        successorStates = [gameState.generateSuccessor(agentIndex, action) for action in
                           gameState.getLegalActions(agentIndex)]
        numGhosts = gameState.getNumAgents() - 1
        if agentIndex == numGhosts:
            probs = [self.computeValue(state, depth + 1,0) for state in successorStates]
        else:
            probs = [self.computeValue(state, depth, agentIndex + 1) for state in successorStates]
        return sum(probs) / len(probs)

    def expectiMax(self, gameState, depth):
        emaxScore = float('-inf')
        for action in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, action)
            emaxScore = max(emaxScore, self.computeValue(successorState, depth, 1))
        return emaxScore
