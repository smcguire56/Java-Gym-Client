package agent;

public class Agent {
	/*
	 * Initialise QTable:
	 * 
	 * [obs1] = Cart Position 
	 * [obs2] = Cart Velocity 
	 * [obs3] = Pole Angle 
	 * [obs4] = Pole Velocity At Tip
	 * 
	 * [action] = action 1 or 0
	 * 
	 */
	private float alpha = 0.3f;
	private float gamma = 1f;
	private float epsilon = 0.3f;

	private final int cartPosition = 10;
	private final int cartVelocity = 5;
	private final int poleAngle = 20;
	private final int poleVelocity = 25;

	private int numActions = 2;

	// Q-Table of size from buckets and actions above
	float[][][][][] qTable = new float[cartPosition][cartVelocity][poleAngle][poleVelocity][numActions];

	// New Agent constructor
	public Agent(int numActions) {
		this.numActions = numActions;
	}

	// New Agent constructor
	public Agent(int numActions, float alpha, float gamma, float epsilon) {
		this.numActions = numActions;
		this.alpha = alpha;
		this.gamma = gamma;
		this.epsilon = epsilon;
	}

	// update the Q value
	public void updateQValue(int[] previousState, int selectedAction, int[] currentState, float reward) {
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
			// if Q-Value here is greater than maxValue
			if (qTable[state[0]][state[1]][state[2]][state[3]][action] > maxValue) {
				// Assign maxIndex to the action
				maxIndex = action;
				// Assign maxValue to that Q-Value
				maxValue = qTable[state[0]][state[1]][state[2]][state[3]][action];
			}
		}
		// Return better action
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
		} else {
			selectedAction = getMaxValuedAction(state);
		}

		return selectedAction;
	}

	public int selectRandomAction() {
		// Selects a random action; either 0 (Left) or 1 (Right)
		return (int) (Math.random() * numActions);
	}

	// Getters and Setters for each variable
	public float getAlpha() {
		return alpha;
	}

	public void setAlpha(float alpha) {
		this.alpha = alpha;
	}

	public float getGamma() {
		return gamma;
	}

	public void setGamma(float gamma) {
		this.gamma = gamma;
	}

	public float getEpsilon() {
		return epsilon;
	}

	public void setEpsilon(float epsilon) {
		this.epsilon = epsilon;
	}

	public int getNumActions() {
		return numActions;
	}

	public void setNumActions(int numActions) {
		this.numActions = numActions;
	}

	public float[][][][][] getqTable() {
		return qTable;
	}

	public void setqTable(float[][][][][] qTable) {
		this.qTable = qTable;
	}

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

}
