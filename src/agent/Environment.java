package agent;

import java.util.Iterator;

import org.json.JSONArray;
import org.json.JSONObject;

import javaclient.GymJavaHttpClient;
import javaclient.StepObject;

public class Environment {
	private int numEpisodes = 5;
	private int numActions = 2;

	private static int CartPosition = 49;
	private static int CartVelocity = 21;
	private static int PoleAngle = 100;
	private static int PoleVelocity = 100;

	private float alpha = 0.10f;
	private float gamma = 1.0f;
	private float epsilon = 0.10f;

	float[][][][] array = new float[49][21][100][100];

	public void runExperiments(String id, Object obs) {
		setupAgent();
		for (int i = 0; i < numEpisodes; i++) {
			System.out.println("Episode: " + numEpisodes);
			doEpisode(id, obs);
		}

	}

	private void doEpisode(String id, Object obs) {
		int action;
		int counter = 0;
		int max_steps = 2000;
		float reward = 0;
		boolean done = false;

		obs = GymJavaHttpClient.resetEnv(id);

		for (int j = 0; j < max_steps; j++) {
			// get next action based on q table 
			action = obsToAction(obs);
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
			
			// update q value table here 

			System.out.println("Action: " + action);
			System.out.println("obs: " + obs.toString());
			System.out.println("done: " + done);
			System.out.println("counter: " + counter);
			System.out.println("reward: " + reward);
			System.out.println(GymJavaHttpClient.listEnvs());

			if (done) {
				break;
			}
		}
	}

	private int obsToAction(Object obs) {
		JSONObject obj = new JSONObject();
		Iterator<Object> iter = ((JSONArray) obs).iterator();
		double sum = 0;
		while (iter.hasNext()) {
			sum += (double) iter.next();
		}

		if (sum > 0) {
			return 1;
		} else {
			return 0;
		}
	}

	private void setupAgent() {
		new Agent(getNumStates(), numActions, alpha, gamma, epsilon);

	}

	private float[][][][] getNumStates() {
		// CartPosition * CartVelocity * PoleAngle * PoleVelocity;
		array = new float[49][21][10][10];
		
		for (int i = 0; i < 49; i++) {
			for (int j = 0; j < 21; j++) {
				for (int k = 0; k < 10; k++) {
					for (int l = 0; l < 10; l++) {
						array[i][j][k][l] = 0;
					}
				}
			}
		}
		
		for (int i = 0; i < 49; i++) {
			for (int j = 0; j < 21; j++) {
				for (int k = 0; k < 10; k++) {
					for (int l = 0; l < 10; l++) {
						System.out.print(array[i][j][k][l]);
					}
					System.out.println();
				}
				System.out.println();
			}
			System.out.println();
		}		
		
		System.out.println(array);
		
		
		return array;

	}

}
