/**
 * Agent which have location as Double 2D must extend this
 * */ 

package sim.engine;
import sim.util.Int2D;
/** include index and set,get location method with Double2D
 * 
 * */

public abstract class AgentInt
    {
		public Int2D loc;
		public int index;
		public abstract Int2D getLocation();
		public abstract void setLocation(Int2D location);
    }
