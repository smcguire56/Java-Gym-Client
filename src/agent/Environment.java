package agent;

// ArrayList is used for storing the rewards
import java.util.ArrayList;

// Import HTTP classes
import javaclient.*;

// Class Environment stores everything about what the Agent interacts with
public class Environment {

	// the number of episodes you want it to run
	private int numEpisodes = 4000;
	// the number of times you want the experiments to loop the number of episodes
	private int loopExperiments = 1;

	// The agents information is stored in Environment
	private Agent agent;
	private int[] currentAgentState = new int[] { -1, -1, -1, -1 }; // agent's current state of agent
	private int[] previousAgentState = new int[] { -1, -1, -1, -1 };// agent's previous state of agent
	private int numActions = 2;

	// Alpha is the learning rate
	private final float alpha = 0.2f;
	// The minimum value we want Alpha to be
	private final float minAlpha = 0.2f;
	// Epsilon is the balance between Exploration & Exploitation
	private final float epsilon = 0.3f;
	// The lowest possible value for Epsilon
	private final float minEpsilon = 0.01f;
	// Gamma is the Discount Factor
	private final float gamma = 1f;

	// The rate at which we want Epsilon & Alpha to decay at
	private final float epsilonDecayRate = 0.999f;
	private final float alphaDecayRate = 0.999f;

	// Booleans which enable/ disable if we are decaying Alpha or Epsilon
	private boolean epsilonDecays = true;
	private boolean alphaDecays = false;

	// Array list which we use for the CSV file, keeping track of the total reward
	// from each Episode
	private ArrayList<ArrayList<Float>> totalRewardAllRuns = new ArrayList<ArrayList<Float>>();

	// The minimum and maximum values for each observation, used for the Bucket
	// sorting function.
	private final float cartPosMin = -5f;
	private final float cartPosMax = 5f;
	private final float cartVelMin = -4f;
	private final float cartVelMax = 4f;
	private final float poleAngleMin = -2f;
	private final float poleAngleMax = 2f;
	private final float poleVelMin = -10f;
	private final float poleVelMax = 10f;

	// Runs the Experiments, taking in the ID and the initial Observation from the
	// Runner
	public void runExperiments(String id, Object obs) {

		// Loops over how many times we run the experiment
		for (int j = 1; j <= loopExperiments; j++) {

			// Local array list for storing total rewards
			ArrayList<Float> totalRewardEpisode = new ArrayList<Float>();

			// Pass in agents values
			agent = new Agent(numActions, alpha, gamma, epsilon);

			// Resetting average reward
			float averageReward = 0;

			// Runs Episode
			for (int i = 1; i <= numEpisodes; i++) {

				// Episode reward returned from doEpside function
				float epReward = doEpisode(id, obs);

				// Append the reward to calculate the average reward
				averageReward += epReward;

				// Every 100 episodes calculate the average value for the last 100 episodes
				if (i % 100 == 0) {

					averageReward /= 100;
					System.out.println("Average Reward after " + i + ": " + averageReward);
					averageReward = 0;
				}

				// Append to the list of rewards
				totalRewardEpisode.add(epReward);
				System.out.println(i + ": " + epReward);

			}
			// Append to the total reward list for every episode
			totalRewardAllRuns.add(totalRewardEpisode);
			// resets the Environment
			resetEnvironment();
		}
	}

	// runs the individual episode
	private float doEpisode(String id, Object obs) {
		// Local variables for interacting with the Python Server
		int action;
		int max_steps = 1000;
		float reward = 0;
		boolean done = false;
		float episodeReward = 0;

		// Reset the Environment
		obs = GymJavaHttpClient.resetEnv(id);

		convertObsToState(obs);
		// Loop until max steps is reached (is 200 from server anyways)
		for (int j = 0; j < max_steps; j++) {

			// get next action based on Q-Table
			action = agent.selectAction(currentAgentState);

			// Create a new Step Object for the episode
			StepObject step;
			try {
				// Agent does an action here based on the action above
				step = GymJavaHttpClient.stepEnv(id, action, true, true);
				// Read in the step object
				obs = step.observation;
				done = step.done;
				reward = step.reward;
				// Append the Episode reward here.
				episodeReward += reward;

			} catch (Exception e) {
				System.out.println("EOF");
				break;
			}

			// Calculates the previous state of the agent, done by copying the current state
			for (int i = 0; i < 4; i++) {
				previousAgentState[i] = currentAgentState[i];
			}

			// Similar to above, this time calculating the next State
			convertObsToState(obs);

			// update q value table here
			agent.updateQValue(previousAgentState, action, currentAgentState, reward);

			if (done) {
				break;
			}
		}

		decayEpsilon();
		decayAlpha();
		return episodeReward;
	}

	private int[] convertObsToState(Object obs) {
		// convert Object received to a string
		String ObsToString = obs.toString();
		// Remove the [] brackets from the string
		ObsToString = ObsToString.replaceAll("\\[", "").replaceAll("\\]", "");
		// Convert the comma separated String into string array
		String[] observations = ObsToString.split(",");

		// Get the current state of the Agent
		// Using the values from the observation array,
		// the minimum value, the maximum value and the number of buckets for that
		// observation
		// This is repeated for each observation.
		currentAgentState[0] = getBucketIndex(Float.parseFloat(observations[0]), cartPosMin, cartPosMax,
				(float) agent.getCartPosition());
		currentAgentState[1] = getBucketIndex(Float.parseFloat(observations[1]), cartVelMin, cartVelMax,
				(float) agent.getCartVelocity());
		currentAgentState[2] = getBucketIndex(Float.parseFloat(observations[2]), poleAngleMin, poleAngleMax,
				(float) agent.getPoleAngle());
		currentAgentState[3] = getBucketIndex(Float.parseFloat(observations[3]), poleVelMin, poleVelMax,
				(float) agent.getPoleVelocity());

		// return the state based on the observations
		return currentAgentState;

	}

	// obsValue is the initial observed value
	// minValue is the minimum that observation can be
	// maxValue is the maximum that observation can be
	// numBuckets is how many buckets that observation has
	public int getBucketIndex(float obsValue, float minValue, float maxValue, float numBuckets) {
		// The value returned, is the bucket
		int bucketIndex = 0;

		// Shift the values to start at 0
		if (minValue < 0) {
			maxValue += Math.abs(minValue);
			obsValue += Math.abs(minValue);
			minValue = 0;
		}

		// Calculate the bucket increment value
		float bucketIncrement = maxValue / numBuckets;

		float bucket = obsValue / bucketIncrement;

		// Round to the nearest bucket (whole number)
		bucketIndex = (int) bucket;

		// Return the bucket value
		return bucketIndex;
	}

	// Resets environment
	private void resetEnvironment() {
		String id = GymJavaHttpClient.createEnv("CartPole-v0");
		GymJavaHttpClient.resetEnv(id);
	}

	// Decay epsilon value
	public void decayEpsilon() {
		// Check the boolean if we are decaying
		if (epsilonDecays) {
			// Using the agent's epsilon value
			if (agent.getEpsilon() >= minEpsilon) {
				// Set the Agents epsilon value to the new decayed value
				agent.setEpsilon(agent.getEpsilon() * epsilonDecayRate);
			} else {
				// Set to the minimum value
				agent.setEpsilon(minEpsilon);
				epsilonDecays = false;
			}
		}
	}

	// Decay Alpha value
	public void decayAlpha() {
		// Check the boolean if we are decaying
		if (alphaDecays) {
			// Using the agent's Alpha value
			if (agent.getAlpha() >= minAlpha) {
				// Set the Agents Alpha value to the new decayed value
				agent.setAlpha(agent.getAlpha() * alphaDecayRate);
			} else {
				// Set to the minimum value
				agent.setAlpha(minAlpha);
				alphaDecays = false;
			}

		}
	}

	// Returns the total reward list
	public ArrayList<ArrayList<Float>> getTotalRewardEpisode() {
		return totalRewardAllRuns;
	}

}
