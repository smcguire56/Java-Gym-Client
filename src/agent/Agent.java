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

	private float[][][][][] qTable = new float[49][21][100][100][2];

	// agent parameters, default values
	private float alpha = 0.1f; // Slide #28, learning rate
	private float gamma = 0.9f; // Slide #28, discount factor
	private float epsilon = 0.1f; // Slide #23, exploration rate

	private float[][][][] numStates;
	private int numActions;

	public Agent() {
		super();
	}

	public Agent(float numStates[][][][], int numActions, float alpha, float gamma, float epsilon) {
		this.numStates = numStates;
		this.numActions = numActions;
		this.alpha = alpha;
		this.gamma = gamma;
		this.epsilon = epsilon;
		initialiseQvalues();
	}

	private void initialiseQvalues() {

		// qTable = new
		// float[numStates[0][0][0][1]][numStates[0][0][1][0]][numStates[0][1][0][0]][numStates[1][0][0][0]][numActions];
		qTable = new float[49][21][100][100][2];

		System.out.print("Q table: ");

		float[][][][][] qTable = new float[2][3][3][4][5];
		// print array in rectangular form
		for (int i = 0; i < qTable.length; i++) {
			for (int j = 0; j < qTable[i].length; j++) {
				for (int x = 0; x < 3; x++) {
					for (int y = 0; y < 3; y++) {
						for (int z = 0; z < 3; z++) {
							System.out.print("Q table: " + qTable[i][j][x][y]);
						}
					}
				}
			}
			System.out.println("");
		}
	}

	// set q values initially to 0
	/*
	 * for (int i = 0; i < numStates[0][0][0][1]; i++) { for (int j = 0; j <
	 * numActions; j++) { qTable[i][j][0][0][0] = 0.0f; } }
	 */

	public float[][][][][] getqTable() {
		return qTable;
	}

	public void setqTable(float[][][][][] qTable) {
		this.qTable = qTable;
	}

}
