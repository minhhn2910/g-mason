/*
  Copyright 2006 by Sean Luke and George Mason University
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/

package sim.app.heatbugsMultiGPU;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.JCuda.*;
import java.util.Arrays;
import jcuda.jcurand.*;
import jcuda.runtime.JCuda;
import java.io.*;
import jcuda.*;
import jcuda.driver.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;

import sim.engine.*;
import sim.field.grid.*;
import sim.util.*;

public /*strictfp*/ class HeatBugs extends SimState
    {
    private static final long serialVersionUID = 1;

    public double minIdealTemp = 17000;
    public double maxIdealTemp = 31000;
    public double minOutputHeat = 6000;
    public double maxOutputHeat = 10000;

    public double evaporationRate = 0.993;
    public double diffusionRate = 1.0;
    public static final double MAX_HEAT = 32000;
    public double randomMovementProbability = 0.1;

    public int gridHeight;
    public int gridWidth;
    public int bugCount;
    
    HeatBug[] bugs;
	
	
	public int numDevices = 4;
	public HeatBugsGPUContext[] contexts;

	public Pointer locationPointer;
	public IntBuffer location;
	
	public double[] idealTemp;// = new double[bugCount];
    public double[] heatOutput;// = new double[bugCount];
    
    
    public double getMinimumIdealTemperature() { return minIdealTemp; }
    public void setMinimumIdealTemperature( double temp ) { if( temp <= maxIdealTemp ) minIdealTemp = temp; }
    public double getMaximumIdealTemperature() { return maxIdealTemp; }
    public void setMaximumIdealTemperature( double temp ) { if( temp >= minIdealTemp ) maxIdealTemp = temp; }
    public double getMinimumOutputHeat() { return minOutputHeat; }
    public void setMinimumOutputHeat( double temp ) { if( temp <= maxOutputHeat ) minOutputHeat = temp; }
    public double getMaximumOutputHeat() { return maxOutputHeat; }
    public void setMaximumOutputHeat( double temp ) { if( temp >= minOutputHeat ) maxOutputHeat = temp; }
    public double getEvaporationConstant() { return evaporationRate; }
    public void setEvaporationConstant( double temp ) { if( temp >= 0 && temp <= 1 ) evaporationRate = temp; }
    public Object domEvaporationConstant() { return new Interval(0.0,1.0); }
    public double getDiffusionConstant() { return diffusionRate; }
    public void setDiffusionConstant( double temp ) { if( temp >= 0 && temp <= 1 ) diffusionRate = temp; }
    public Object domDiffusionConstant() { return new Interval(0.0, 1.0); }
    public double getRandomMovementProbability() { return randomMovementProbability; }
        
    public double[] getBugXPos() {
        try
            {
            double[] d = new double[bugs.length];
            for(int x=0;x<bugs.length;x++)
                {
                d[x] = ((Int2D)(buggrid.getObjectLocation(bugs[x]))).x;
                }
            return d;
            }
        catch (Exception e) { return new double[0]; }
        }
    
    public double[] getBugYPos() {
        try
            {
            double[] d = new double[bugs.length];
            for(int x=0;x<bugs.length;x++)
                {
					d[x] = ((Int2D)(buggrid.getObjectLocation(bugs[x]))).y;
                }
            return d;
            }
        catch (Exception e) { return new double[0]; }
        }


    public void setRandomMovementProbability( double t )
        {
        if (t >= 0 && t <= 1)
            {
            randomMovementProbability = t;
            for( int i = 0 ; i < bugCount ; i++ )
                if (bugs[i]!=null)
                    bugs[i].setRandomMovementProbability( randomMovementProbability );
            }
        }
    public Object domRandomMovementProbability() { return new Interval(0.0, 1.0); }
        
    public double getMaximumHeat() { return MAX_HEAT; }

    // we presume that no one relies on these DURING a simulation
    public int getGridHeight() { return gridHeight; }
    public void setGridHeight(int val) { if (val > 0) gridHeight = val; }
    public int getGridWidth() { return gridWidth; }
    public void setGridWidth(int val) { if (val > 0) gridWidth = val; }
    public int getBugCount() { return bugCount; }
    public void setBugCount(int val) { if (val >= 0) bugCount = val; }
    
    public DoubleGrid2D valgrid;
  //  public DoubleGrid2D valgrid2;
    public SparseGrid2D buggrid;
    

    /** Creates a HeatBugs simulation with the given random number seed. */
    public HeatBugs(long seed)
        {
       this(seed, 10000, 10000, 1000000);
        }
        
    public HeatBugs(long seed, int width, int height, int count)
        {
        super(seed);
        gridWidth = width; gridHeight = height; bugCount = count;
        createGrids();
        }

    protected void createGrids()
        {
        bugs = new HeatBug[bugCount];
        valgrid = new DoubleGrid2D(gridWidth, gridHeight,0);

        buggrid = new SparseGrid2D(gridWidth, gridHeight);      
        }
    
 //   ThreadedDiffuser diffuser = null;
        
    /** Resets and starts a simulation */
    public void start()
        {
        super.start();  // clear out the schedule
    
		long locationSize = bugCount*2*Sizeof.INT;
		locationPointer = new Pointer();
		cudaHostAlloc(locationPointer, locationSize, cudaHostAllocPortable);
		location = locationPointer.getByteBuffer(0, locationSize).order(ByteOrder.nativeOrder()).asIntBuffer();


		idealTemp = new double[bugCount];
		heatOutput = new double[bugCount];    
        // make new grids
        createGrids();
        
        contexts = new HeatBugsGPUContext[numDevices];
        
		System.out.println("num Agent " + bugCount + "  size "+ gridWidth+" x " + gridHeight);
        // Schedule the heat bugs -- we could instead use a RandomSequence, which would be faster
        // But we spend no more than 3% of our total runtime in the scheduler max, so it's not worthwhile
        for(int x=0;x<bugCount;x++)
            {
			idealTemp[x] = random.nextDouble() * (maxIdealTemp - minIdealTemp) + minIdealTemp;
			heatOutput[x] = random.nextDouble() * (maxOutputHeat - minOutputHeat) + minOutputHeat;

            //~ loc_x[x] = random.nextInt(gridWidth);
            //~ loc_y[x] = random.nextInt(gridHeight);
     
            bugs[x] = new HeatBug(idealTemp[x],heatOutput[x],randomMovementProbability);           
            bugs[x].index = x;
            bugs[x].TheHeatBug = this;

            bugs[x].setLocation(new Int2D (random.nextInt(gridWidth),random.nextInt(gridHeight)));
            buggrid.InitObjectLocation(bugs[x],bugs[x].getLocation().x,bugs[x].getLocation().y);           
      //      schedule.scheduleRepeating(bugs[x]);
	//		if(x == 15 || x ==28 )
		//		System.out.println(x + " ==> output " + heatOutput[x] + " ideal ==> " + idealTemp[x]);
            }
      
      								//~ System.out.println();
						//~ //		System.out.println(schedule.getSteps()+ " ==> " );
								//~ for(int i=0;i<bugCount;i++)
								//~ {
								//~ 
									//~ System.out.print(" ("+i +": "+ loc_y[i]+ ","+loc_x[i]+")");
								//~ }
								
      //~ System.out.println();System.out.println();
      //~ 
      
         schedule.currentPhase = 2;               
    
                            

        schedule.scheduleRepeating(schedule.EPOCH, 1, new Steppable()
		{
			public void step(SimState state) { return; }
		});
  
  
  
        AsynchronousSteppable s = new AsynchronousSteppable()
		{
			boolean shouldQuit = false;
			int gpu_step = 0;
			int schedule_gpuSteps = 0;

			Object[] lock = new Object[0]; // an array is a unique, serializable object

			protected void run(boolean resuming)
			{
				boolean quit = false;
				if (!resuming)
				{	
					cuda_init();
					
					schedule_gpuSteps = schedule.getgpuSteps();
					
					System.out.println("Async agent started");
					// we’re starting fresh -- set up here if you have to
					
				}
				else // (resuming)
				{

					cuda_init();
					
					schedule_gpuSteps = schedule.getgpuSteps();
					
					System.out.println("Async agent resumed");
			// 		we’re starting fresh -- set up here if you have to
				}
				while(!quit)
				{
					if(schedule.currentPhase == 2)  //GPU phases
					{

						for(int i =0;i<numDevices;i++)
						{
							contexts[i].GPUHeatBug();
							contexts[i].eventRecord();
						}
						
						
						for(int i =0;i<numDevices;i++)
						{
							contexts[i].eventSync();
							contexts[i].sync_valgrid(contexts);
							contexts[i].sync_best_location_peer(contexts);
							contexts[i].eventRecord();
						}

						for(int i=0; i<numDevices ; i++)
						{
							contexts[i].eventSync();	
							contexts[i].GPUDiffuse();			
							contexts[i].GPUCollectData();
							contexts[i].eventRecord();
						}	
						
						for(int i=0; i<numDevices ; i++)
							contexts[i].eventSync();
						
						gpu_step ++;
						if(gpu_step == schedule_gpuSteps) //time to copy back and wake up mason's Schedule
						{	
							//~ start_time = System.nanoTime();
							for(int i=0; i<numDevices ; i++)
								contexts[i].copy_back(locationPointer, valgrid.DataPointer);
								gpu_step = 0;
								schedule.return_to_CPU();
						 }
						 		 
				for(int i=0; i<numDevices ; i++)
				{
					contexts[i].sync_data();
					contexts[i].sync_location_peer(contexts);
					contexts[i].sync_valgrid(contexts);
					contexts[i].eventRecord();
				}
				for(int i=0; i<numDevices ; i++)
					contexts[i].eventSync();

					}
						synchronized(lock) { quit = shouldQuit; shouldQuit = false; }
					}
				//quit here
				System.out.println("Free cuda mem");
				cudaDeviceReset();
				System.out.println("Cleanup success");
				
				}
				protected void halt(boolean pausing) { synchronized(lock) { shouldQuit = true; } }
			};
				
			s.step(this);   

        }
    
    public void stop()
        {
         }
    
    public void cuda_init()
    {
               JCudaDriver.setExceptionsEnabled(true);
                JCuda.setExceptionsEnabled(true);
                JCurand.setExceptionsEnabled(true);
				cudaDeviceReset();
				cuInit(0);
				
				int numCells = (gridHeight*gridWidth)/numDevices;
		//		int numBugs = bugCount/numDevices;
				for(int i = 0;i <numDevices ;i++)
				{
					contexts[i] = new HeatBugsGPUContext(i,i);
					contexts[i].InitHeatBugsConstant(gridWidth,gridHeight,i*gridHeight/numDevices,(i+1)*gridHeight/numDevices -1, MAX_HEAT, randomMovementProbability, evaporationRate,diffusionRate, bugCount,numDevices);
					contexts[i].InitModule(1234,numCells,i*(gridWidth*gridHeight)/numDevices);
					contexts[i].InitMemory(locationPointer, idealTemp, heatOutput);
					contexts[i].InitKernelParameters();
					System.out.println("device " + contexts[i].index + " begin " + contexts[i].begin_row + " endrow " + contexts[i].end_row  + " gridoffset "+ contexts[i].gridOffset + " numCells " +contexts[i].numCells); 
					
				}

	}
    
    /** This little function calls Runtime.getRuntime().availableProcessors() if it's available,
        else returns 1.  That function is nonexistent in Java 1.3.1, but it exists in 1.4.x.
        So we're doing a little dance through the Reflection library to call the method tentatively!
        The value returned by Runtime is the number of available processors on the computer.  
        If you're only using 1.4.x, then all this is unnecessary -- you can just call
        Runtime.getRuntime().availableProcessors() instead. */
    public static int availableProcessors()
        {
        Runtime runtime = Runtime.getRuntime();
        try { return ((Integer)runtime.getClass().getMethod("availableProcessors", (Class[])null).
                invoke(runtime,(Object[])null)).intValue(); }
        catch (Exception e) { return 1; }  // a safe but sometimes wrong assumption!
        }
        
    
    
    public static void main(String[] args)
        {
        doLoop(HeatBugs.class, args);
        System.exit(0);
        }    
    }
    
    
    
    
    
