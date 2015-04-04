/*
  Copyright 2006 by Sean Luke and George Mason University
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/

package sim.app.flockers;
import sim.engine.*;
import sim.field.continuous.*;
import sim.util.*;
import ec.util.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;



// it extend AgentDouble which handle position as Double2D, set and get method
public class Flocker extends AgentDouble implements sim.portrayal.Orientable2D
    {
    private static final long serialVersionUID = 1;

	private final int LOC_X = 0;
	private final int LOC_Y = 1;
	private final int DX = 2;
	private final int DY = 3;

    public Continuous2D flockers;
    public Flockers theFlock;
    public boolean dead = false;
//this flocker index= normal index *4  
    
    public Flocker(Double2D location) { loc = location; }

//overiide AgentDouble here  
	@Override
	public Double2D getLocation()
	{
		
		return new Double2D(theFlock.data[offset].get(index+LOC_X),theFlock.data[offset].get(index+LOC_Y));
	}
//DataBuffer.asDoubleBuffer().put(y*width +x, val );	
//DataBuffer.asDoubleBuffer().get(y*width +x);
	@Override
	public void setLocation(Double2D location)
	{
//		loc = location;
		theFlock.data[offset].put(index+LOC_X, location.x);
		theFlock.data[offset].put(index+LOC_Y, location.y);
	
	}
    
    public double getOrientation() { return orientation2D(); }
    public boolean isDead() { return dead; }
    public void setDead(boolean val) { dead = val; }
    
    public void setOrientation2D(double val)
        {
      //  lastd = new Double2D(Math.cos(val),Math.sin(val));
			theFlock.data[offset].put(index+DX, Math.cos(val));
			theFlock.data[offset].put(index+DY, Math.sin(val));
        }
    
    public double orientation2D()
        {
			final double dx = theFlock.data[offset].get(index+DX);
			final double dy = theFlock.data[offset].get(index+DY);
        if (dx == 0 && dy == 0) return 0;
			return Math.atan2(dy, dx);
        }
    
   
 /*       
        public void test_speed()
        {
			Bag b;
			Bag c;
				long start_time  = System.nanoTime();
					b = flockers.getNeighborsExactlyWithinDistance(loc, theFlock.neighborhood, true);
				long end_time = System.nanoTime();
				System.out.println("numobj : " +b.numObjs + " time " + (end_time - start_time));
			
			flock.index_list = Collections.synchronizedList(new ArrayList<int[]>());
	
			start_time = System.nanoTime();
			c = flockers.getNeighborsWithinDistance(loc, theFlock.neighborhood, true, false, null);
			for(int i=0;i<c.numObjs;i++)
				flock.index_list.add(new int[]{this.index,((Flocker)(c.objs[i])).index});
				end_time = System.nanoTime();
				System.out.println("numobj more : "+ c.numObjs + " time " + (end_time - start_time));
				
				for (int i=0; i < flock.index_list.size(); i++)
					System.out.println(flock.index_list.get(i)[0] + "  " + flock.index_list.get(i)[1]);
		
		
				
				flock.index_list = Collections.synchronizedList(new ArrayList<int[]>());
				start_time = System.nanoTime();
				flockers.getNeighborsWithinDistance(index, loc, theFlock.neighborhood, true, false,flock.index_list);
				end_time = System.nanoTime();
				System.out.println("numobj more next type:  time " + (end_time - start_time));
				for (int i=0; i < flock.index_list.size(); i++)
					System.out.println(flock.index_list.get(i)[0] + "  " + flock.index_list.get(i)[1]);



				start_time = System.nanoTime();
				c = flockers.getNeighborsWithinDistance(loc, theFlock.neighborhood, true, false, null);
				for(int i=0;i<c.numObjs;i++)
				{
					Flocker other = (Flocker)(c.objs[i]);
					flock.index_list.add(new int[]{this.index,((Flocker)(c.objs[i])).index});
				}
				end_time = System.nanoTime();
				System.out.println("numobj more with type caster : "+ c.numObjs + " time " + (end_time - start_time));

		}
    */    
 
    }
