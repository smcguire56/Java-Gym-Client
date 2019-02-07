package agent;

import javaclient.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.json.JSONArray;
import org.json.JSONObject;

public class Runner {
	/*
	 * Adapted from: https://github.com/Ryan-Amaral/working-gym-java-client
	 */
	public static void main(String[] args) {
		System.out.println("running main");

		GymJavaHttpClient.baseUrl = "http://127.0.0.1:5000";

		String id = GymJavaHttpClient.createEnv("CartPole-v0");

		Object actionSpace = GymJavaHttpClient.actionSpace(id);

		System.out.println(actionSpace.getClass().getName());

		Object obs = GymJavaHttpClient.resetEnv(id);

		System.out.println(obs.getClass().getName());
		System.out.println(obs.toString());
		
		Environment e = new Environment();
		System.out.println("running experiments");
		e.runExperiments(id, obs);
		//randomGames(id, obs);
		//initialPopulation(id, obs);

	}
	
	

	private static void initialPopulation(String id, Object obs) {

		int action;
		int episode_count = 100;
		int max_steps = 1000;
		float reward = 0;
		boolean done = false;
		int scoreRequirement = 50;
		int score = 0;

		ArrayList<Integer> AcceptedScores = new ArrayList<Integer>();

		for (int i = 0; i < episode_count; i++) {
			score = 0;
			Map<Object, Integer> gameMemory = new LinkedHashMap<Object, Integer>();

			Object prevObservation = new Object();

			for (int j = 0; j < max_steps; j++) {

				action = obsToAction(obs);

				StepObject step;
				try {
					step = GymJavaHttpClient.stepEnv(id, action, true, true);
					obs = step.observation;
					done = step.done;
					reward = step.reward;
				} catch (Exception e) {
					break;
				}

				if (prevObservation != null) {
					gameMemory.put(prevObservation, action);
				}
				prevObservation = obs;
				score += reward;

				if (done) {
					break;
				}

				/*System.out.println("Action: " + action);
				System.out.println("obs: " + obs.toString());
				System.out.println("done: " + done);
				System.out.println("reward: " + reward);
				System.out.println(GymJavaHttpClient.listEnvs());*/

				if (score >= scoreRequirement) {
					AcceptedScores.add(score);
					gameMemory.forEach((k, v) -> {
						// converting to one-hot
						int[] output = new int[2];
						if (v == 1) {
							output[0] = 0;
							output[1] = 1;
						} else if (v == 0) {
							output[0] = 1;
							output[1] = 0;
						}

						System.out.println(k.toString() + " " + Arrays.toString(output));
					});
				}
			}
			try {
				obs = GymJavaHttpClient.resetEnv(id);
			} catch (Exception e) {
				break;
			}

		} // for loop

		System.out.println("Average accepted score: " + mean(AcceptedScores));
		System.out.println("Median score for accepted scores:" + median(AcceptedScores));
		System.out.println("Counter: " + AcceptedScores.size());
	}

	public static double mean(List<Integer> marks) {
		Integer sum = 0;
		if (!marks.isEmpty()) {
			for (Integer mark : marks) {
				sum += mark;
			}
			return sum.doubleValue() / marks.size();
		}
		return sum;
	}

	public static double median(List<Integer> sets) {
		if (sets.isEmpty()) {
			return 0;
		}
		int middle = sets.size() / 2;
		middle = middle > 0 && middle % 2 == 0 ? middle - 1 : middle;
		return sets.get(middle);
	}

	private static void randomGames(String id, Object obs) {
		int action;
		int counter = 0;
		int episode_count = 100;
		int max_steps = 2000;
		float reward = 0;
		boolean done = false;

		for (int i = 0; i < episode_count; i++) {

			obs = GymJavaHttpClient.resetEnv(id);

			for (int j = 0; j < max_steps; j++) {

				action = obsToAction(obs);
				StepObject step;
				try {
					step = GymJavaHttpClient.stepEnv(id, action, true, false);
					obs = step.observation;
					done = step.done;
					reward = step.reward;
					counter++;
				} catch (Exception e) {
					// e.printStackTrace();
					System.out.println("EOF");
					break;
				}

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
