dataPath=/scratch/phys/project/sin/AFM_Hartree_DB/AFM_sims/striped
for basedata in Water-bilayer # Water-Au111
# Loop 0 ----Start
do 
    echo Dataset $basedata
    mkdir -p $basedata 
    cd $basedata 
    # Loop 1  ----Start
    #for case in Ref L30 L50 L100 Ref-L50
    cases=$(cat ../cases.txt)
    for case in $cases
    do
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
            sed -i "s|URLS_TRAIN|${basedata}_FB_${case}/${tarPrefix}-K-{1..10}_train_{0..31}.tar|g" $subBatch
            sed -i "s|URLS_VAL|${basedata}_FB_${case}/${tarPrefix}-K-{1..10}_val_{0..3}.tar|g" $subBatch
            sed -i "s|URLS_TEST|${basedata}_FB_${case}/${tarPrefix}-K-{1..10}_test_{0..3}.tar|g" $subBatch
        else
            IFS='-' read -ra parts <<< "$case"
            sed -i "s|URLS_TRAIN|${basedata}_FB_${parts[0]}/${tarPrefix}-K-{1..10}_train_{0..31}.tar::${basedata}_FB_${parts[1]}/${tarPrefix}-K-{1..10}_train_{0..31}.tar|g" $subBatch
            sed -i "s|URLS_VAL|${basedata}_FB_${parts[0]}/${tarPrefix}-K-{1..10}_val_{0..3}.tar::${basedata}_FB_${parts[1]}/${tarPrefix}-K-{1..10}_val_{0..3}.tar|g" $subBatch
            sed -i "s|URLS_TEST|${basedata}_FB_${parts[0]}/${tarPrefix}-K-{1..10}_test_{0..3}.tar::${basedata}_FB_${parts[1]}/${tarPrefix}-K-{1..10}_test_{0..3}.tar|g" $subBatch
        fi
        
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
    done # Loop 1 ----End
    cd ..
done # Loop 0 ----End
