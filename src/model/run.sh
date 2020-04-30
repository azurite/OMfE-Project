# Read Variables
THREADBLOCK_TILE_M=$1
THREADBLOCK_TILE_N=$2
THREADBLOCK_WARP_TILE_K=$3
WARP_TILE_M=$4
WARP_TILE_N=$5
SPLIT_K=$6

#THREADBLOCK_TILE_M=128
#THREADBLOCK_TILE_N=128
#THREADBLOCK_WARP_TILE_K=8
#WARP_TILE_M=64
#WARP_TILE_N=32
#SPLIT_K=1

# Write config file
CONFIG=./src/model/src/config.h

echo "/*" > $CONFIG
echo " * config.h" >>$CONFIG
echo " *" >>$CONFIG
echo " *  Created on: $(date)" >>$CONFIG
echo " *      Author: Automatically generated" >>$CONFIG
echo " */" >>$CONFIG
echo "#ifndef CONFIG_H_" >>$CONFIG
echo "#define CONFIG_H_" >>$CONFIG

echo "// This STAYS ////////////////" >>$CONFIG
echo "#define M 499				//" >>$CONFIG
echo "#define N 2377				//" >>$CONFIG
echo "#define K 5857				//" >>$CONFIG
echo "#define TYPE float			//" >>$CONFIG
echo "#define RUNS 50 			//" >>$CONFIG
echo "//////////////////////////////" >>$CONFIG

echo "// THIS ARE THE ARGUMENTS ////////////" >>$CONFIG
echo "#define THREADBLOCK_TILE_M $THREADBLOCK_TILE_M		//" >>$CONFIG
echo "#define THREADBLOCK_TILE_N $THREADBLOCK_TILE_N		//" >>$CONFIG
echo "#define THREADBLOCK_WARP_TILE_K $THREADBLOCK_WARP_TILE_K	//" >>$CONFIG
echo "									//" >>$CONFIG
echo "#define WARP_TILE_M $WARP_TILE_M				//" >>$CONFIG
echo "#define WARP_TILE_N $WARP_TILE_N				//" >>$CONFIG
echo "									//" >>$CONFIG
echo "#define SPLIT_K $SPLIT_K					//" >>$CONFIG
echo "//////////////////////////////////////" >>$CONFIG
echo "#endif /* CONFIG_H_ */" >>$CONFIG

# compile program
cd ./src/model/Release
make clean > /dev/null 2> /dev/null
make > /dev/null 2> /dev/null

if [ $? -eq 0 ]; then
  echo "good"
else
  time_file=../time.txt
  echo "999999999999" > $time_file
  exit 1
fi



# run program

./OMfE



