## Prepare two clean models 
# cp -r ../Water-bilayer/Ref Original
# cp -r ../Water-bilayer/L50 NewModel

# mkdir -p Original/train_posnet
# cp ../Water-bilayer/Ref/train_posnet/best_model.pth  Original/train_posnet
# mkdir -p Original/train_graphnet
# cp ../Water-bilayer/Ref/train_graphnet/best_model.pth  Original/train_graphnet
# 
# mkdir -p NewModel/train_posnet
# cp ../Water-bilayer/L50/train_posnet/best_model.pth  NewModel/train_posnet
# mkdir -p NewModel/train_graphnet
# cp ../Water-bilayer/L50/train_graphnet/best_model.pth  NewModel/train_graphnet

# Real Distribution P 
mkdir -p P 
cp Original/Prediction_1/predictions/*_ref.xyz P

# # Prediction 1 2 3 
# mkdir -p Q/Prediction_1 Q/Prediction_2 Q/Prediction_3
# cp Original/Prediction_1/predictions/*_pred.xyz Q/Prediction_1
# cp Original/Prediction_2/predictions/*_pred.xyz Q/Prediction_2
# cp Original/Prediction_3/predictions_Original/*_mol.xyz Q/Prediction_3
# 
# # Prediction a b c
# mkdir -p Q/Prediction_a Q/Prediction_b Q/Prediction_c
# cp NewModel/Prediction_a/predictions/*_pred.xyz Q/Prediction_a
# cp NewModel/Prediction_b/predictions/*_pred.xyz Q/Prediction_b
# cp NewModel/Prediction_c/predictions_NewModel/*_mol.xyz Q/Prediction_c


