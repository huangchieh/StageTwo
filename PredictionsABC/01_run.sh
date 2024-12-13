# Obtain the model list 
declare -a model_list  # Declare an array to hold the models
model_list+=("Ref")  # Append the original model to the array
basedata=Water-bilayer
for CYCLEGANDATA in PPAFM2Exp_CoAll
do
for LAMBDA1 in 0 10 20 40 50 60 70 80 90 100
do
for LAMBDA2 in 0 0.1 1 10
do 
for EPOCH in latest
do 
case=${CYCLEGANDATA}_L${LAMBDA1}_L${LAMBDA2}_E${EPOCH}
model=${case}
model_list+=("$model")  # Append model path to the array
done done done done

# Forword pass to get the prediction a, b, c for all trained models
for model in "${model_list[@]}"
do
	echo "$model"
	optimal_posnet_model=../${basedata}/${model}/train_posnet/best_model.pth
	optimal_graphnet_model=../${basedata}/${model}/train_graphnet/best_model.pth
	rsync -a EvaTemplate/ ${model}/
	cp ${optimal_posnet_model} ${model}/train_posnet/
	cp ${optimal_graphnet_model} ${model}/train_graphnet/
	for Prediction in Prediction_a Prediction_b Prediction_c
	do
		cd ${model}/${Prediction}
		dataflag=${basedata}_FB_${model}
		sed -i "s/DATAFLAG/${dataflag}/g" config.yaml # It only changes the config.yaml in which DATAFLAG is present
		job=${dataflag}
		sed -i "s/JOBNAME/${job}/g" 01_run.sh 
		sbatch 01_run.sh
		cd -
	done
done

# Real Distribution P 
#mkdir -p P 
#cp Original/Prediction_1/predictions/*_ref.xyz P # This is the top layer of water 


