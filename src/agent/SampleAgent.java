package agent;

import javaclient.*;

import java.awt.List;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.Map;

import org.json.JSONArray;
import org.json.JSONObject;

public class SampleAgent {

	static int action;
	static int counter = 0;
	static int episode_count = 1000;
	static int max_steps = 2000;
	static float reward = 0;
	static boolean done = false;
	static int scoreRequirement = 50;

	public static void main(String[] args) {

		GymJavaHttpClient.baseUrl = "http://127.0.0.1:5000"; // this is the default value, but just showing that you can
		// change it

		String id = GymJavaHttpClient.createEnv("CartPole-v0"); // create an environment

		Object actionSpace = GymJavaHttpClient.actionSpace(id);

		// Do this if not a standard attribute
		System.out.println(actionSpace.getClass().getName()); // helpful to know how to deal with
		System.out.println(actionSpace); // helpful to see format of object

		Object obs = GymJavaHttpClient.resetEnv(id); // reset the environment (get initial observation)
		System.out.println(obs.getClass().getName());// see what observation looks like to work with it
		System.out.println(obs.toString());

		// randomGames(id, obs);
		initialPopulation(id, obs);

	}

	@SuppressWarnings("unchecked")
	private static void initialPopulation(String id, Object obs) {
		int score = 0;
		Map<ArrayList<Double>, ArrayList<Integer>> trainingData = new LinkedHashMap<ArrayList<Double>, ArrayList<Integer>>();

		ArrayList<Integer> scores = new ArrayList<Integer>();

		ArrayList<Integer> AcceptedScores = new ArrayList<Integer>();
		
		for (int i = 0; i < episode_count; i++) {
			obs = GymJavaHttpClient.resetEnv(id);

			score = 0;
			Map<ArrayList<Object>, Integer> gameMemory = new LinkedHashMap<ArrayList<Object>, Integer>();

			ArrayList<Object> prevObservation = new ArrayList<Object>();

			for (int j = 0; j < max_steps; j++) {

				action = obsToAction(obs);
				StepObject step = GymJavaHttpClient.stepEnv(id, action, true, false);
				obs = step.observation;
				done = step.done;
				reward = step.reward;

				if (!prevObservation.isEmpty()) {
					gameMemory.put(prevObservation, action);
				}
				prevObservation.add(obs);
				score += reward;

				counter++;
				System.out.println("Action: " + action);
				System.out.println("obs: " + obs.toString());
				System.out.println("done: " + done);
				System.out.println("counter: " + counter);
				System.out.println("reward: " + reward);
				System.out.println(GymJavaHttpClient.listEnvs());

				if (done) {
					break;
				}

				if (score >= scoreRequirement) {
					AcceptedScores.add(score);
					
					gameMemory.forEach((k, v) -> {
					    ArrayList<Integer> output = new ArrayList<Integer>();
						if(v == 1) {
							//output = [0,1];
							//output.add(1);
						}
						else if (v == 0) {
							//output.add(1,0);

						}
						
						trainingData.put(null, output);
					});
				}
			}

		}



	}

	private static void randomGames(String id, Object obs) {

		for (int i = 0; i < episode_count; i++) {

			obs = GymJavaHttpClient.resetEnv(id);

			for (int j = 0; j < max_steps; j++) {

				action = obsToAction(obs);
				StepObject step = GymJavaHttpClient.stepEnv(id, action, true, true);
				obs = step.observation;
				done = step.done;
				reward = step.reward;
				counter++;
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
	}

	/**
	 * Do a policy where you add the values in observation and return 1 or 0 based
	 * on the sum.
	 * 
	 * @param obs
	 * @return
	 */
	public static int obsToAction(Object obs) {
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
}
