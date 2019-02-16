package agent;

import javaclient.GymJavaHttpClient;
import javaclient.StepObject;

public class Environment {
	private int numEpisodes = 50;

	private static int cartPosition = 49;
	private static int cartVelocity = 21;
	private static int poleAngle = 85;
	private static int poleVelocity = 21;

	private Agent agent;
	private int[] currentAgentState = new int[] { 1, 1, 1, 1 }; // agent's current x,y position (state vector)
	private int[] previousAgentState = new int[] { -1, -1, -1, -1 };
	private int numActions = 2;

	private float alpha = 0.10f;
	private float gamma = 1.0f;
	private float epsilon = 0.10f;

	float[][][][] array = new float[cartPosition][cartVelocity][poleAngle][poleVelocity];

	int numBucketsObs0 = 2200; // CartPosition (-2.4 to 2.4) = -24 + 24 + 1 = 49 different buckets
	int numBucketsObs1 = 210; // CartVelocity (-inf to inf) = -10 to 10 + 1 = 21
	int numBucketsObs2 = 8000; // PoleAngle (-41.8 to 41.8) = -42 + 42 = 84 + 1 = 85
	int numBucketsObs3 = 210; // PoleVelocity (-inf to inf) = -10 to 10 + 1 = 21

	// obsValue is the initial observed value
	// minValue is the minimum that observation can be
	// maxValue is the maximum that observation can be
	// numBuckets is how many buckets that observation has
	public int getBucketIndex(float obsValue, float minValue, float maxValue, float numBuckets) {
		int bucketIndex = 0;

		if (minValue < 1) {
			maxValue += Math.abs(minValue);
			obsValue += Math.abs(minValue);
			minValue = 0;
		}

		float bucketIncrement = maxValue / numBuckets;
		
		float bucket = obsValue / bucketIncrement;
		
		bucketIndex = (int) bucket;
/*		System.out.println("obsValue: "+obsValue);
		System.out.println("minValue: "+minValue);
		System.out.println("maxValue: "+maxValue);
		System.out.println("numBuckets: "+numBuckets);
		System.out.println("bucketIncrement: "+bucketIncrement);
		System.out.println("bucket: "+bucket);
		System.out.println("bucketIndex: "+bucketIndex);*/
		
		return bucketIndex;
	}

	public void runExperiments(String id, Object obs) {
		setupAgent();

		for (int i = 0; i < numEpisodes; i++) {
			System.out.println("Episode: " + i);
			doEpisode(i, id, obs);
		}

	}

	private void setupAgent() {
		agent = new Agent(numActions, alpha, gamma, epsilon);

	}

	private void doEpisode(int currentEpisode, String id, Object obs) {
		int action;
		int counter = 0;
		int max_steps = 2000;
		float reward = 0;
		boolean done = false;

		try {
			obs = GymJavaHttpClient.resetEnv(id);
		} catch (Exception e) {
			System.out.println("cannot reset environment");
		}

		for (int j = 0; j < max_steps; j++) {
			// get next action based on q table
			// action = agent.selectAction(currentAgentState);
			action = agent.selectRandomAction();

			StepObject step;
			try {
				step = GymJavaHttpClient.stepEnv(id, action, true, false);
				obs = step.observation;
				done = step.done;
				reward = step.reward;
				counter++;
			} catch (Exception e) {
				System.out.println("EOF");
				break;
			}

			String stringValues = obs.toString();
			stringValues = stringValues.replaceAll("\\[", "").replaceAll("\\]", "");
			String[] integerStrings = stringValues.split(",");

			//System.out.println(integerStrings[0]+" "+ integerStrings[1]+" "+ integerStrings[2]+" "+ integerStrings[3]);

			currentAgentState[0]  = getBucketIndex(Float.parseFloat(integerStrings[0]), (float)-24 , (float)24, (float)numBucketsObs0);
			currentAgentState[1]  = getBucketIndex(Float.parseFloat(integerStrings[1]), (float)-10 , (float)10, (float)numBucketsObs1);
			currentAgentState[2]  = getBucketIndex(Float.parseFloat(integerStrings[2]), (float)-42 , (float)42, (float)numBucketsObs2);
			currentAgentState[3] = getBucketIndex(Float.parseFloat(integerStrings[3]), (float)-10 , (float)10, (float)numBucketsObs3);
			
			System.out.println(currentAgentState[0] + " " + currentAgentState[1] + " " + currentAgentState[2] + " " + currentAgentState[3]);
			
			currentAgentState[0] = 1;
			// update q value table here
			/*
			 * System.out.println("Action: " + action); System.out.println("obs: " +
			 * obs.toString()); System.out.println("done: " + done);
			 * System.out.println("counter: " + counter); System.out.println("reward: " +
			 * reward); System.out.println(GymJavaHttpClient.listEnvs());
			 */

			if (done) {
				break;
			}
		}
	}

	/*
	 * private int obsToAction(Object obs) { JSONObject obj = new JSONObject();
	 * Iterator<Object> iter = ((JSONArray) obs).iterator(); double sum = 0; while
	 * (iter.hasNext()) { sum += (double) iter.next(); }
	 * 
	 * if (sum > 0) { return 1; } else { return 0; } }
	 */

}
