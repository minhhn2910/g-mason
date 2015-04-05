export CUDA_HOME=/usr/local/cuda-6.5
export PATH=$PATH:$CUDA_HOME:$CUDA_HOME/bin
export JCUDA=/home/user/jcuda
export LD_PRELOAD=/usr/lib64/libcuda.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$JCUDA

export JAVA_HOME=/usr/lib/jvm/java-1.7.0-openjdk.x86_64
export MASON_LIB=/home/user/g-mason/libraries
export CLASSPATH=:/home/user:$JCUDA/jcublas-0.6.5.jar:$JCUDA/jcuda-0.6.5.jar:$JCUDA/jcurand-0.6.5.jar:$JCUDA/jcusparse-0.6.5.jar:/home/user/g-mason:$MASON_LIB/itext-1.2.jar:$MASON_LIB/jcommon-1.0.16.jar:$MASON_LIB/jfreechart-1.0.13.jar:$MASON_LIB/jmf.jar:$MASON_LIB/portfolio.jar:/home/user/jcuda
export PATH=$JAVA_HOME/bin:$PATH
