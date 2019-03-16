package agent;

public class Agent {
	/*
	 * Initialise QTable: 
	 * [obs1] = Cart Position 
	 * [obs2] = Cart Velocity 
	 * [obs3] = Pole Angle 
	 * [obs4] = Pole Velocity At Tip
	 * 
	 * [act] = action 1 or 0
	 * 
	 */

	//private static int num = 10;
	private static int cartPosition = 1;
	private static int cartVelocity = 1;
	private static int poleAngle	= 12;
	private static int poleVelocity = 24;

	private float alpha = 0.1f;
	private float gamma = 1.0f;
	private float epsilon = 0.1f;

	private int numActions = 2;

	float[][][][][] qTable = new float[cartPosition][cartVelocity][poleAngle][poleVelocity][numActions];

	public Agent(int numActions) {
		this.numActions = numActions;
		initialiseQvalues();
	}

	public Agent(int numActions, float alpha, float gamma, float epsilon) {
		this.numActions = numActions;
		this.alpha = alpha;
		this.gamma = gamma;
		this.epsilon = epsilon;
		initialiseQvalues();
	}

	private void initialiseQvalues() {
		for (int i = 0; i < cartPosition; i++) {
			for (int j = 0; j < cartVelocity; j++) {
				for (int x = 0; x < poleAngle; x++) {
					for (int y = 0; y < poleVelocity; y++) {
						for (int z = 0; z < numActions; z++) {
							qTable[i][j][x][y][z] = 0;
						}
					}
				}
			}
		}
	}

	// update the Q value
	public void updateQValue(int[] previousState, int selectedAction, int[] currentState, float reward) {
		
		//System.out.println("previousState: " + previousState[0] +","+ previousState[1]+","+ previousState[2]+","+ previousState[3] 
				//+" selectedAction:"+selectedAction+"\ncurrentState:  "+currentState[0] +"," + currentState[1] +"," + currentState[2] +"," + currentState[3] + " reward: "+ reward+ " epsilon: "+ epsilon+ " alpha: "+ alpha);
		// implementation of Q-learning TD update rule
		float oldQ = qTable[previousState[0]][previousState[1]][previousState[2]][previousState[3]][selectedAction];
		float maxQ = getMaxQValue(currentState);
		float newQ = oldQ + alpha * (reward + gamma * maxQ - oldQ);
		qTable[previousState[0]][previousState[1]][previousState[2]][previousState[3]][selectedAction] = newQ;
	}

	private float getMaxQValue(int[] state) {
		// return the Q value of the most valuable action for a particular state
		int maxIndex = getMaxValuedAction(state);
		return qTable[state[0]][state[1]][state[2]][state[3]][maxIndex];
	}

	private int getMaxValuedAction(int[] state) {
		// greedy action selection implementation
		// return the index of the most valuable action for a particular state
		int maxIndex = -1;
		float maxValue = -Float.MAX_VALUE;
		for (int action = 0; action < numActions; action++) {
			if (qTable[state[0]][state[1]][state[2]][state[3]][action] > maxValue) {
				maxIndex = action;
				maxValue = qTable[state[0]][state[1]][state[2]][state[3]][action];
			}
		}
		return maxIndex;
	}

	public int selectAction(int[] state) {
		// epsilon-greedy action selection strategy implementation
		int selectedAction = -1;
		double randomValue = Math.random();

		// select a random action with probability epsilon
		// else select the most valuable action
		if (randomValue < epsilon) {
			selectedAction = selectRandomAction();
		}
		else {
			selectedAction = getMaxValuedAction(state);
		}

		return selectedAction;
	}

	public int selectRandomAction() {
		// select a random action between 0 and (numActions-1)
		return (int) (Math.random() * numActions);
	}

	// getters and setters
	public int getCartPosition() {
		return cartPosition;
	}

	public int getCartVelocity() {
		return cartVelocity;
	}

	public int getPoleAngle() {
		return poleAngle;
	}

	public int getPoleVelocity() {
		return poleVelocity;
	}

	public void setEpsilon(float epsilon) {
		this.epsilon = epsilon;		
	}

	public void setAlpha(float alpha) {
		this.alpha = alpha;
	}

}
