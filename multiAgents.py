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
import math

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
        return successorGameState.getScore()

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
        
        #max-agent function, to find pacman's move
        def max_value(gameState, depth):
            
            #check if terminal state
            if gameState.isWin() or gameState.isLose():
                return gameState.getScore()
        
            #set current max value to -inf
            v = -math.inf
            
            #set best action to standing still
            best_action = Directions.STOP
        
            #pacman has agent index zero
            current_agent = 0
        
            #get actions for pacman
            actions = gameState.getLegalActions(current_agent)
            
            
            for a in actions:
                #call min-agent function 
                score = min_value(gameState.generateSuccessor(current_agent, a), depth, 1) #1 refers to first ghost
                if score > v:
                    v = score
                    best_action = a
                    
            #if at root node level, return the best action
            if depth == 0:
                return best_action
            
            #if not at root node, return the score for a min-agent to use it
            else:
                return v
            
        #min-agent function, to find ghosts' moves
        def min_value(gameState, depth, agent):
            
            #check if terminal state
            if gameState.isWin() or gameState.isLose():
                return gameState.getScore()
            
            #increment agent index
            agent_next = agent + 1
            
            #if we have been through all agents, want to reset agent to refer to pacman
            #e.g. if we have pacman and 2 ghosts, numAgents = 3. Ghosts are then indexed by 1 and 2

            if agent == gameState.getNumAgents() - 1:
                agent_next = 0 
    
            #get actions for current ghost
            ghost_actions = gameState.getLegalActions(agent)
            
            #set v to minus infinity
            v = math.inf
            
            #iterate through all actions
            for a in ghost_actions:
                if agent_next == 0: #pacman has next turn
                    if depth == self.depth - 1:
                        score = self.evaluationFunction(gameState.generateSuccessor(agent, a))
                    else:
                        #recursively call the max-agent function in one "layer" (depth) down in the tree
                        score = max_value(gameState.generateSuccessor(agent, a), depth + 1)
                else:
                    score = min_value(gameState.generateSuccessor(agent, a), depth, agent_next)
                
                #update v if necessary
                if score < v:
                    v = score
            return v
    
        
        #this is the only "running" code in get_action
        return max_value(gameState, 0)
        
        
        
        
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def max_value(gameState, depth, alpha, beta):
            
            #check if terminal state
            if gameState.isWin() or gameState.isLose():
                return gameState.getScore()
        
            #set current max value to -inf
            v = -math.inf
            
            #initialize best action to standing still
            best_action = Directions.STOP
        
            #pacman has agent index zero
            current_agent = 0
        
            #get actions for pacman
            actions = gameState.getLegalActions(current_agent)
            for a in actions:
                score = min_value(gameState.generateSuccessor(current_agent, a), depth, 1, alpha, beta)
                if score > v:
                    v = score
                    best_action = a
                
                #store the alpha value for later use
                alpha = max(alpha, v)
                
                #if find a min_score atleast as high as beta, we prune the rest of the possible actions, as the min will not choose this path anyway
                if v > beta:
                    return v
            if depth == 0:
                return best_action
            else:
                return v
            
                
        def min_value(gameState, depth, agent, alpha, beta):
            if gameState.isWin() or gameState.isLose():
                return gameState.getScore()
        
            agent_next = agent + 1
        
            if agent == gameState.getNumAgents() - 1:
                agent_next = 0 
    
            ghost_actions = gameState.getLegalActions(agent)
            v = math.inf
            
            for a in ghost_actions:
                if agent_next == 0:
                    if depth == self.depth - 1:
                        score = self.evaluationFunction(gameState.generateSuccessor(agent, a))
                    else:
                        score = max_value(gameState.generateSuccessor(agent, a), depth + 1, alpha, beta)
                else:
                    score = min_value(gameState.generateSuccessor(agent, a), depth, agent_next, alpha, beta)
                if score < v:
                    v = score
                    
                #store the minimum value for later use
                beta = min(beta, v)
                
                #if one of the actions lead to a v smaller than alpha, the max-agent one level up will not choose the action leading to the current min-node anyway
                #just return v and prune rest of possible actions
                if v < alpha:
                    return v
            return v
    
        #when depth=0, max_value returns best action
        return max_value(gameState, 0, alpha = -math.inf, beta = math.inf)
        

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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
