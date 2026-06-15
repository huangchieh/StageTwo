# Obtain the model list 
declare -a model_list  # Declare an array to hold the models
model_list+=("Ref")  # Append the original model to the array
model_list+=("Ref_Pure_C9")  # Append the original model to the array
model_list+=("PPAFM2Exp_CoAll_L20_L1_Elatest_Only_C7")
basedata=Water-bilayer
run_suffix=issue34
run_root=issue34
prediction_b_dataflag=${basedata}_FB_PPAFM2Exp_CoAll_L20_L1_Elatest
mkdir -p ${run_root}
for CYCLEGANDATA in PPAFM2Exp_CoAll
do
for LAMBDA1 in  20 #10 
do
for LAMBDA2 in  1 #10
do 
for EPOCH in latest
do 
for C in  1 # Different test
do
case=${CYCLEGANDATA}_L${LAMBDA1}_L${LAMBDA2}_E${EPOCH}_C${C}
model=${case}
model_list+=("$model")  # Append model path to the array
done done done done done

# Forword pass to get the prediction a, b, c for all trained models
for model in "${model_list[@]}"
do
	echo "$model"
	run_model=${run_root}/${model}_${run_suffix}
	optimal_posnet_model=../${basedata}/${model}/train_posnet/best_model.pth
	optimal_graphnet_model=../${basedata}/${model}/train_graphnet/best_model.pth
	rsync -a EvaTemplate_issue34/ ${run_model}/
	cp ${optimal_posnet_model} ${run_model}/train_posnet/
	cp ${optimal_graphnet_model} ${run_model}/train_graphnet/
	for Prediction in Prediction_a Prediction_b Prediction_c
	do
		cd ${run_model}/${Prediction}
		if [ "${Prediction}" = "Prediction_b" ]; then
			dataflag=${prediction_b_dataflag}
			# Force export of all available val batches for Prediction_b outputs.
			sed -i 's/^pred_batches:.*/pred_batches: 9999/' config.yaml
		elif [[ "${model}" == Ref_Pure_C* ]]; then
			dataflag=${basedata}_FB_Ref
		else
			dataflag=${basedata}_FB_${model}
		fi
		sed -i "s/DATAFLAG/${dataflag}/g" config.yaml # It only changes the config.yaml in which DATAFLAG is present
		job=${basedata}_FB_${model}
		sed -i "s/JOBNAME/${job}/g" 01_run.sh 
		sbatch 01_run.sh
		cd -
	done
done

# Real Distribution P 
#mkdir -p P 
#cp Original/Prediction_1/predictions/*_ref.xyz P # This is the top layer of water 


