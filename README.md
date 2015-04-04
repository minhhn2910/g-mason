## g-mason
An optimized version for MASON, enabling Multi-GPU computing for Multi Agent Simulations

### Requirement:
  1. JCuda library [http://www.jcuda.org/]
  2. Java JDK
  3. Cuda [https://developer.nvidia.com/about-cuda]

### Setup Instruction:
1. Please make sure you have a proper configuration of JCuda and Cuda. To do this, please refer to [http://www.jcuda.org/tutorial/TutorialIndex.html#GeneralSetup]

2. Set the classpath according to the libraries you intend to use with MASON.
The sample setting for the environment variables is provided in docs/sample-config.sh

3. Use the makefile for building the project. 

4. To play around with three examples with Visualization as default (if you dont want to see visualization, just disable it in the GUI):
  
  $ start/mason.sh
  
  Or to start example models directly without GUI:
  
  $ java sim.app.flockers.Flockers

  $ java sim.app.heatbugs.Heatbugs
  
  $ java sim.app.students.Students
  
  For GUI:
  
  $ java sim.app.flockers.FlockersWithUI
  
  $ java sim.app.heatbugs.HeatBugsWithUI
  
  $ java sim.app.students.StudentsWithUI

5. Explore the power of Multi-GPU with your own simulation model. please place your file into sim.app package so the makefile can work properly

### Additional API

  it is described in docs/Additional-API.txt
