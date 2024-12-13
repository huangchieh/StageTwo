# This script is used to organize the structure predictions

# Create a folder to store the structure predictions
outFolder=BatchOutStructues
mkdir -p $outFolder

declare -a model_list  # Declare an array to hold the models
model_list+=("Ref")  # Append the original model to the array
# Find all the folders with name starting with "PPAFM2Exp_CoAll_" and append them to the array
model_list+=($(ls -d PPAFM2Exp_CoAll_*))

# Loop through the models
for model in "${model_list[@]}"
do 
	echo "Processing $model"
	sourceFolder_a=$model/Prediction_a
	sourceFolder_b=$model/Prediction_b
	sourceFolder_c=$model/Prediction_c
	PredictionFolder_a=$outFolder/${sourceFolder_a}
	PredictionFolder_b=$outFolder/${sourceFolder_b}
	PredictionFolder_c=$outFolder/${sourceFolder_c}
	mkdir -p ${PredictionFolder_a}
	mkdir -p ${PredictionFolder_b}
	mkdir -p ${PredictionFolder_c}
	cp $sourceFolder_a/predictions/*_pred.xyz ${PredictionFolder_a}
	cp $sourceFolder_b/predictions/*_pred.xyz ${PredictionFolder_b}
	cp $sourceFolder_c/predictions/*.xyz ${PredictionFolder_c}
done

