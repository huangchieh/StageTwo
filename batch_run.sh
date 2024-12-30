dataPath=/scratch/phys/project/sin/AFM_Hartree_DB/AFM_sims/striped
for basedata in Water-bilayer # Water-Au111
do # Loop 0 ----Start
    echo Dataset $basedata
    mkdir -p $basedata 
    cd $basedata 
    # Loop 1  ----Start
    for CYCLEGANDATA in PPAFM2Exp_CoAll # Loop G model trained on different datasets
    do
	for LAMBDA1 in 0 10 20 30 40 50 60 70 80 90 100 
	do
		for LAMBDA2 in 0 0.1 1 10
		for LAMBDA2 in  1 
		do
			for EPOCH in latest # Loop G model trained on different epoc
			do
    				#for case in Ref L30 L50 L100 Ref-L50
    				#do
				case=${CYCLEGANDATA}_L${LAMBDA1}_L${LAMBDA2}_E${EPOCH}
    				echo  Case $case 
    				mkdir -p $case 
    				cd $case 
    				rsync -av ../../Template/ ./

    				function run_batch {
    				    folder=$1
    				    subBatch=$2
    				    after=$3

    				    cd $folder
    				    if [[ $basedata == "Water-bilayer" ]]; then
    				        tarPrefix="Water-bilayer"
    				    else
    				        tarPrefix="Water"
    				    fi
    				    sed -i "s|JOBSUFFIX|${case}|g" $subBatch
    				    sed -i "s|DATA_DIR|${dataPath}/${basedata}-FB|g" $subBatch

    				    subatomnumber=79 # Au atomic number because we use Au(111) surface
    				    if [[ $folder == "train_posnet" ]]; then
    				        sed -i "s|SUBATOMICNUMBER|${subatomnumber}|g" fit_posnet.py
    				    fi

    				    if [[ $folder == "train_graphnet" ]]; then
    				            sed -i "s|SUBATOMICNUMBER|${subatomnumber}|g" fit_graphnet.py
    				    fi

    				    if [[ $folder == "test_combined" ]]; then
    				            sed -i "s|SUBATOMICNUMBER|${subatomnumber}|g" test_graphnet.py
    				    fi 

    				    if [[ $case != *-* ]]; then # No mixed dataset  
    				        sed -i "s|URLS_TRAIN|${basedata}_FB_${CYCLEGANDATA}_L${LAMBDA1}_L${LAMBDA2}_E${EPOCH}/${tarPrefix}-K-{1..10}_train_{0..31}.tar|g" $subBatch
    				        sed -i "s|URLS_VAL|${basedata}_FB_${CYCLEGANDATA}_L${LAMBDA1}_L${LAMBDA2}_E${EPOCH}/${tarPrefix}-K-{1..10}_val_{0..3}.tar|g" $subBatch
    				        sed -i "s|CASE|${basedata}_FB_${CYCLEGANDATA}_L${LAMBDA1}_L${LAMBDA2}_E${EPOCH}|g" $subBatch # For test combined
    				        sed -i "s|URLS_TEST|${basedata}_FB_${CYCLEGANDATA}_L${LAMBDA1}_L${LAMBDA2}_E${EPOCH}/${tarPrefix}-K-{1..10}_test_{0..3}.tar|g" $subBatch
				    fi
				    # I don't care about mixed dataset for now
    				    #else
    				    #    IFS='-' read -ra parts <<< "$case"
    				    #    sed -i "s|URLS_TRAIN|${basedata}_FB_${parts[0]}/${tarPrefix}-K-{1..10}_train_{0..31}.tar::${basedata}_FB_${parts[1]}/${tarPrefix}-K-{1..10}_train_{0..31}.tar|g" $subBatch
    				    #    sed -i "s|URLS_VAL|${basedata}_FB_${parts[0]}/${tarPrefix}-K-{1..10}_val_{0..3}.tar::${basedata}_FB_${parts[1]}/${tarPrefix}-K-{1..10}_val_{0..3}.tar|g" $subBatch
    				    #    sed -i "s|URLS_TEST|${basedata}_FB_${parts[0]}/${tarPrefix}-K-{1..10}_test_{0..3}.tar::${basedata}_FB_${parts[1]}/${tarPrefix}-K-{1..10}_test_{0..3}.tar|g" $subBatch
    				    #fi
    				    
    				    # Submit job based on dependency
    				    if [[ $after == "None" ]]; then
    				        jobid=$(sbatch $subBatch | awk '{print $4}')
    				    else
    				        jobid=$(sbatch --dependency=afterok:$after $subBatch | awk '{print $4}')
    				    fi 
    				    cd .. 
    				    echo $jobid
    				    }
    				
    				# 1. Train posnet
    				jobid1=$(run_batch train_posnet batch_fit_posnet.sh None)
    				# 2. Train graphnet 
    				jobid2=$(run_batch train_graphnet batch_fit_graphnet.sh None)
    				# 3. Test combined 
    				jobid3=$(run_batch test_combined batch_test_graphnet.sh $jobid1:$jobid2)
    				# 4. Test experiments
    				jobid4=$(run_batch test_experiment batch_test_experiments.sh $jobid1:$jobid2)
    				cd .. # Exit case
    				#done # Loop 1 ----End
			done # Loop EPOCH ----End
		done # Loop LAMBDA2 ----End
	done # Loop lambda1 ----End 
    done # Loop G ----End
    cd ..
done # Loop 0 ----End
