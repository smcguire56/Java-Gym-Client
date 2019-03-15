package agent;

import java.util.ArrayList;

import javaclient.GymJavaHttpClient;
import javaclient.StepObject;

public class Environment {
	private int numEpisodes = 10;
	private int loopEpisode = 2;

	private Agent agent;
	private int[] currentAgentState = new int[] { -1, -1, -1, -1 }; // agent's current state of agent
	private int[] previousAgentState = new int[] { -1, -1, -1, -1 };// agent's previous state of agent

	private int numActions = 2;

	private float alpha = 0.20f;
	private float gamma = 1.0f;
	private float epsilon = 0.10f;
	private float epsilonDecayRate = 0.9999f;
	private boolean epsilonDecays = true;

	 private ArrayList<ArrayList<Float>> totalRewardAllRuns = new ArrayList<ArrayList<Float>>();

	private final float cartPosMin = -10f;
	private final float cartPosMax = 10f;
	private final float cartVelMin = -10f;
	private final float cartVelMax = 10f;
	private final float poleAngleMin = -20f;
	private final float poleAngleMax = 20f;
	private final float poleVelMin = -10f;
	private final float poleVelMax = 10f;

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
		// System.out.println("obs: "+ obsValue + " min: "+ minValue + " max: " +
		// maxValue + " num" + numBuckets +" bucketIndex " + bucketIndex);

		return bucketIndex;
	}

	public void runExperiments(String id, Object obs) {
		// run 10x times

		for (int j = 0; j < loopEpisode; j++) {
			 ArrayList<Float> totalRewardEpisode = new ArrayList<Float>();
			
			setupAgent();
			
			for (int i = 1; i <= numEpisodes; i++) {
				System.out.println("Episode: " + i);
				float epReward = doEpisode(id, obs);
				totalRewardEpisode.add(epReward);
				System.out.println("reward: " +epReward);
			}
			totalRewardAllRuns.add(totalRewardEpisode);
			resetEnvironment();
		}
		//System.out.println(totalRewardAllRuns);
	}

	private void resetEnvironment() {
		String id = GymJavaHttpClient.createEnv("CartPole-v0");
		GymJavaHttpClient.resetEnv(id);
	}

	private void setupAgent() {
		agent = new Agent(numActions, alpha, gamma, epsilon);

	}

	private float doEpisode(String id, Object obs) {
		int action;
		int max_steps = 2000;
		float reward = 0;
		boolean done = false;
		float episodeReward = 0;

		try {
			obs = GymJavaHttpClient.resetEnv(id);
			String ObsToString = obs.toString();
			ObsToString = ObsToString.replaceAll("\\[", "").replaceAll("\\]", "");
			String[] observations = ObsToString.split(",");
			
			currentAgentState[0] = getBucketIndex(Float.parseFloat(observations[0]), cartPosMin, cartPosMax,
					(float) agent.getCartPosition());
			currentAgentState[1] = getBucketIndex(Float.parseFloat(observations[1]), cartVelMin, cartVelMax,
					(float) agent.getCartVelocity());
			currentAgentState[2] = getBucketIndex(Float.parseFloat(observations[2]), poleAngleMin, poleAngleMax,
					(float) agent.getPoleAngle());
			currentAgentState[3] = getBucketIndex(Float.parseFloat(observations[3]), poleVelMin, poleVelMax,
					(float) agent.getPoleVelocity());
		} catch (Exception e) {
			System.out.println("cannot reset environment");
		}

		for (int j = 0; j < max_steps; j++) {

			// get next action based on q table
			action = agent.selectAction(currentAgentState);
			// action = agent.selectRandomAction();

			StepObject step;
			try {
				step = GymJavaHttpClient.stepEnv(id, action, true, true);
				obs = step.observation;
				done = step.done;
				// save reward to float for each episode, then to a array list, print to a text
				// file.
				reward = step.reward;
				episodeReward += reward;

			} catch (Exception e) {
				System.out.println("EOF");
				break;
			}

			for (int i = 0; i < 4; i++) {
				previousAgentState[i] = currentAgentState[i];
			}

			// currentAgentState set to next state
			String ObsToString = obs.toString();
			ObsToString = ObsToString.replaceAll("\\[", "").replaceAll("\\]", "");
			String[] observations = ObsToString.split(",");

			currentAgentState[0] = getBucketIndex(Float.parseFloat(observations[0]), cartPosMin, cartPosMax,
					(float) agent.getCartPosition());
			currentAgentState[1] = getBucketIndex(Float.parseFloat(observations[1]), cartVelMin, cartVelMax,
					(float) agent.getCartVelocity());
			currentAgentState[2] = getBucketIndex(Float.parseFloat(observations[2]), poleAngleMin, poleAngleMax,
					(float) agent.getPoleAngle());
			currentAgentState[3] = getBucketIndex(Float.parseFloat(observations[3]), poleVelMin, poleVelMax,
					(float) agent.getPoleVelocity());

			// update q value table here
			agent.updateQValue(previousAgentState, action, currentAgentState, reward);

			if (done) {
				break;
			}
		}
		decayEpsilon();
		return episodeReward;
	}

	public void decayEpsilon() {
		if (epsilonDecays) {
			epsilon = epsilon * epsilonDecayRate;
			agent.setEpsilon(epsilon);
		}
	}

	public ArrayList<ArrayList<Float>> getTotalRewardEpisode() {
		return totalRewardAllRuns;
	}

}
