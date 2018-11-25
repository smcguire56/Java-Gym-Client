package javaclient;

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
