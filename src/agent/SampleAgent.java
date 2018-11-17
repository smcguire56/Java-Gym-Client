package agent;

import javaclient.*;

import java.util.Iterator;

import org.json.JSONArray;
import org.json.JSONObject;

public class SampleAgent {

	public static void main(String[] args) {

		GymJavaHttpClient.baseUrl = "http://127.0.0.1:5000"; // this is the default value, but just showing that you can
																// change it

		String id = GymJavaHttpClient.createEnv("CartPole-v0"); // create an environment

		Object actionSpace = GymJavaHttpClient.actionSpace(id);

		// Do this if not a standard attribute
		System.out.println(actionSpace.getClass().getName()); // helpful to know how to deal with
		System.out.println(actionSpace); // helpful to see format of object
		// int numActions = ((JSONObject)actionSpace).getInt("n");

		// but we have method to get action space size from action space object
		// there is only 2 in this example left or right 1 or 0
		int numActions = GymJavaHttpClient.actionSpaceSize((JSONObject) actionSpace);
		
		
		int action; // action for agent to do
		Object obs = GymJavaHttpClient.resetEnv(id); // reset the environment (get initial observation)
		System.out.println(obs.getClass().getName());// see what observation looks like to work with it
		System.out.println(obs.toString());

		int counter = 0;
		int episode_count = 1000;
		int max_steps = 2000;
		float reward = 0;
		boolean done = false;

		for (int i = 0; i < episode_count; i++) {
			obs = GymJavaHttpClient.resetEnv(id);

			for (int j = 0; j < max_steps; j++) {
				//System.out.println("numActions: " + numActions);
				// get random number 1 or 0 
				//action = (int) Math.round(Math.random());
				
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
