package agent;

import java.util.ArrayList;

import javaclient.GymJavaHttpClient;
import javaclient.StepObject;

public class Environment {
	private int numEpisodes = 1000;
	private int loopEpisode = 4;

	private Agent agent;
	private int[] currentAgentState = new int[] { -1, -1, -1, -1 }; // agent's current state of agent
	private int[] previousAgentState = new int[] { -1, -1, -1, -1 };// agent's previous state of agent

	private int numActions = 2;

	private final float alpha = 0.2f;
	private final float minAlpha = 0.1f;
	private final float epsilon = 0.3f;
	private final float minEpsilon = 0.01f;
	
	private final float gamma = 1.0f;
	private float epsilonDecayRate = 0.999f;
	private float alphaDecayRate = 0.999f;
	private boolean epsilonDecays = true;
	private boolean alphaDecays = false;

	private ArrayList<ArrayList<Float>> totalRewardAllRuns = new ArrayList<ArrayList<Float>>();

	private final float cartPosMin = -5f;
	private final float cartPosMax = 5f;
	private final float cartVelMin = -4f;
	private final float cartVelMax = 4f;
	private final float poleAngleMin = -1.8f;
	private final float poleAngleMax = 1.8f;
	private final float poleVelMin = -10f;
	private final float poleVelMax = 10f;

	// obsValue is the initial observed value
	// minValue is the minimum that observation can be
	// maxValue is the maximum that observation can be
	// numBuckets is how many buckets that observation has
	public int getBucketIndex(float obsValue, float minValue, float maxValue, float numBuckets) {
		int bucketIndex = 0;

		if (minValue < 0) {
			maxValue += Math.abs(minValue);
			obsValue += Math.abs(minValue);
			minValue = 0;
		}

		float bucketIncrement = maxValue / numBuckets;

		float bucket = obsValue / bucketIncrement;

		bucketIndex = (int) bucket;

		return bucketIndex;
	}

	public void runExperiments(String id, Object obs) {

		for (int j = 1; j <= loopEpisode; j++) {
			ArrayList<Float> totalRewardEpisode = new ArrayList<Float>();

			setupAgent();
			float averageReward = 0;

			for (int i = 1; i <= numEpisodes; i++) {

				// System.out.printf("Episode: %d, Inner: %d\n" ,j, i);
				float epReward = doEpisode(id, obs);

				averageReward += epReward;

				if (i % 100 == 0) {

					averageReward /= 100;
					System.out.println("Average Reward after " + i + ": " + averageReward);
					averageReward = 0;
				}
				
				totalRewardEpisode.add(epReward);
				System.out.println(i+": "+epReward);

			}

			totalRewardAllRuns.add(totalRewardEpisode);
			resetEnvironment();
		}
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
		int max_steps = 1000;
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
		decayAlpha();
		return episodeReward;
	}

	public void decayEpsilon() {
		if (epsilonDecays) {
			if (agent.getEpsilon() >= minEpsilon) {
				agent.setEpsilon(agent.getEpsilon() * epsilonDecayRate);
			} else {
				agent.setEpsilon(minEpsilon);
				epsilonDecays = false;
			}
		}
	}

	public void decayAlpha() {
		if (alphaDecays) {
			if (agent.getAlpha() >= minAlpha) {
				agent.setAlpha(agent.getAlpha() * alphaDecayRate);
			} else {
				agent.setAlpha(minAlpha);
				alphaDecays = false;
			}

		}
	}

	public ArrayList<ArrayList<Float>> getTotalRewardEpisode() {
		return totalRewardAllRuns;
	}

}
