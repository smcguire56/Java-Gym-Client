package agent;

import java.util.ArrayList;

import javaclient.GymJavaHttpClient;
import javaclient.StepObject;

public class Environment {
	private int numEpisodes = 5000;

	private Agent agent;
	private int[] currentAgentState = new int[] { -1, -1, -1, -1 }; // agent's current state of agent
	private int[] previousAgentState = new int[] { -1, -1, -1, -1 };// agent's previous state of agent

	private int numActions = 2;

	private float alpha = 0.10f;
	private float gamma = 1.0f;
	private float epsilon = 0.10f;

	private ArrayList <Float> bufferedList = new ArrayList<Float>();
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
		
		return bucketIndex;
	}

	public void runExperiments(String id, Object obs) {
		setupAgent();
		// reset agent position
		currentAgentState[0] = agent.getCartPosition() / 2;
		currentAgentState[1] = agent.getCartVelocity() / 2;
		currentAgentState[2] = agent.getPoleAngle() / 2;
		currentAgentState[3] = agent.getPoleVelocity() / 2;
		for (int i = 1; i <= numEpisodes; i++) {
			System.out.println("Episode: " + i);
			doEpisode(id, obs);
		}
		for (int i = 0; i < numEpisodes; i++) {
			System.out.println(bufferedList.get(i));
		}
	}

	private void setupAgent() {
		agent = new Agent(numActions, alpha, gamma, epsilon);

	}

	private void doEpisode(String id, Object obs) {
		int action;
		int max_steps = 2000;
		float reward = 0;
		boolean done = false;
		float episodeReward = 0;
		
		
		try {
			obs = GymJavaHttpClient.resetEnv(id);
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
				// save reward to float for each episode, then to a array list, print to a text file.
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
			currentAgentState[0] = getBucketIndex(Float.parseFloat(observations[0]), (float) -5, (float) 5,(float) agent.getCartPosition());
			currentAgentState[1] = getBucketIndex(Float.parseFloat(observations[1]), (float) -5, (float) 5,(float) agent.getCartVelocity());
			currentAgentState[2] = getBucketIndex(Float.parseFloat(observations[2]), (float) -5, (float) 5,(float) agent.getPoleAngle());
			currentAgentState[3] = getBucketIndex(Float.parseFloat(observations[3]), (float) -5, (float) 5,(float) agent.getPoleVelocity());
			

			// update q value table here
			agent.updateQValue(previousAgentState, action, currentAgentState, reward);
			
			if (done) {
				break;
			}
		}
		bufferedList.add(episodeReward);		
	}
	

}
