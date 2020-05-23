# Read Variables
THREAD_TILE_M=$1
THREAD_TILE_N=$2
WARP_TILE_M=$3
WARP_TILE_N=$4
THREADBLOCK_TILE_M=$5
THREADBLOCK_TILE_N=$6
THREADBLOCK_TILE_K=$7
LOAD_K=$8
SPLIT_K=$9
M=${10}
N=${11}
K=${12}




# Write config file
CONFIG=./src/model/src/config.h

echo "/*" >$CONFIG
echo " * config.h" >>$CONFIG
echo " *" >>$CONFIG
echo " *  Created on: $(date)" >>$CONFIG
echo " *      Author: Automatically generated" >>$CONFIG
echo " */" >>$CONFIG
echo "#ifndef CONFIG_H_" >>$CONFIG
echo "#define CONFIG_H_" >>$CONFIG

echo "// This STAYS ////////////////" >>$CONFIG
echo "#define M $M				//" >>$CONFIG
echo "#define N $N				//" >>$CONFIG
echo "#define K $K				//" >>$CONFIG
echo "#define TYPE float			//" >>$CONFIG
echo "//////////////////////////////" >>$CONFIG

echo "// THIS ARE THE ARGUMENTS ////////////" >>$CONFIG
echo "#define THREADBLOCK_TILE_M $THREADBLOCK_TILE_M		//" >>$CONFIG
echo "#define THREADBLOCK_TILE_N $THREADBLOCK_TILE_N		//" >>$CONFIG
echo "#define THREADBLOCK_TILE_K $THREADBLOCK_TILE_K	//" >>$CONFIG
echo "#define LOAD_K $LOAD_K	//" >>$CONFIG
echo "									//" >>$CONFIG
echo "#define WARP_TILE_M $WARP_TILE_M				//" >>$CONFIG
echo "#define WARP_TILE_N $WARP_TILE_N				//" >>$CONFIG
echo "									//" >>$CONFIG
echo "#define THREAD_TILE_M $THREAD_TILE_M				//" >>$CONFIG
echo "#define THREAD_TILE_N $THREAD_TILE_N				//" >>$CONFIG
echo "									//" >>$CONFIG
echo "#define SPLIT_K $SPLIT_K					//" >>$CONFIG
echo "#define CORRECTNESS_TEST" >>$CONFIG
echo "#define BENCHMARK" >>$CONFIG
echo "//////////////////////////////////////" >>$CONFIG
echo "#endif /* CONFIG_H_ */" >>$CONFIG

time_file=../time.txt
# compile program
cd ./src/model/Release
make clean >/dev/null 2>/dev/null
make >/dev/null 2>/dev/null



if [ $? -ne 0 ]; then
  echo "999999999999" > $time_file
  exit 1
fi

# run program

./cuCOSMA

if [ $? -ne 0 ]; then

  echo "999999999999" > $time_file
  exit 1
fi



