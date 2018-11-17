package javaclient;

/**
 * Information that is returned from a step in the environment.
 * https://gym.openai.com/docs/#observations
 * 
 * @author Ryan Amaral - (ryan-amaral on GitHub)
 */
public class StepObject {
	public Object observation;
	public float reward;
	public boolean done;
	public Object info;

	public StepObject(Object observation, float reward, boolean done, Object info) {
		this.observation = observation;
		this.reward = reward;
		this.done = done;
		this.info = info;
	}

}
