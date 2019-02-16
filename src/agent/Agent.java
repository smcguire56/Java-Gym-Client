package agent;

public class Agent {
	/*
	 * Initialise QTable: 
	 * [obs1] = Cart Position 
	 * [obs2] = Cart Velocity 
	 * [obs3] =Pole Angle 
	 * [obs4] = Pole Velocity At Tip
	 * 
	 * [act] = action 1 or 0
	 * 
	 */

	private static int cartPosition = 49;
	private static int cartVelocity = 21;
	private static int poleAngle = 85;
	private static int poleVelocity = 21;
	private static int actions = 2;

	private float alpha = 0.10f;
	private float gamma = 1.0f;
	private float epsilon = 0.10f;
	private int numActions;
	
	float[][][][][] qTable = new float[cartPosition][cartVelocity][poleAngle][poleVelocity][actions];


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

		
		System.out.print("Setting Q table to 0 ");

		for (int i = 0; i < cartPosition; i++) {
			for (int j = 0; j < cartVelocity; j++) {
				for (int x = 0; x < poleAngle; x++) {
					for (int y = 0; y < poleVelocity; y++) {
						for (int z = 0; z < actions; z++) {
							qTable[i][j][x][y][z] = 0;
						}
					}
				}
			}
		}
		
		/*System.out.println("Printing Q Table: ");
		for (int i = 0; i < cartPosition; i++) {
			for (int j = 0; j < cartVelocity; j++) {
				for (int x = 0; x < poleAngle; x++) {
					for (int y = 0; y < poleVelocity; y++) {
						for (int z = 0; z < actions; z++) {
							System.out.print(" "+qTable[i][j][x][y][z]);
						}
					}
					System.out.println();
				}
				System.out.println();
			}
			System.out.println();
		}*/
	}
	
	// 
	public void updateQValue(int[] previousState, int selectedAction, int[] currentState, float reward) {
		// implementation of Q-learning TD update rule (see slides #26, #27 )
		float oldQ = qTable[previousState[0]][previousState[1]][previousState[2]][previousState[3]][selectedAction];
		float maxQ = getMaxQValue(currentState);
		float newQ = oldQ + alpha * (reward + gamma * maxQ - oldQ);
		// 0,0 is 0, 0 1 is 1 and so on.
		// action is 1 2 3 4 direction
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
		// epsilon-greedy action selection strategy implementation (see slide #23)
		int selectedAction = -1;
		double randomValue = Math.random();

		//System.out.println("Agent: selecting action, epsilon=" + epsilon + ", randomValue=" + randomValue);
	

		if (randomValue < epsilon) { // select a random action with probability epsilon
			selectedAction = selectRandomAction();
			//System.out.println("Agent: selected action " + selectedAction + " at random");
			
		} else { // else select the most valuable action
			selectedAction = getMaxValuedAction(state);
			//System.out.println("Agent: selected action " + selectedAction + " greedily");
			
		}
		return selectedAction;	
	}
	
	public int selectRandomAction() {
		// select a random action between 0 and (numActions-1)
		return (int) (Math.random() * numActions);
	}

}
