GPU=7

DATASET="rialto_bridge"
KEYWORD="rialto bridge"
DATASET_PATH=data/$DATASET
LMBD=20

python download_data.py --dataset $DATASET --keyword $KEYWORD

colmap feature_extractor \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images 
        
colmap exhaustive_matcher \
    --database_path $DATASET_PATH/database.db 

mkdir $DATASET_PATH/sparse

colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse 
    
for CLUSTER_ID in 0 1 2 3 4
do
    colmap model_converter \
        --input_path $DATASET_PATH/sparse/$CLUSTER_ID \
        --output_path $DATASET_PATH/sparse/$CLUSTER_ID \
        --output_type TXT

    # sleep 5

    python align_images.py \
        --dataset $DATASET \
        --cluster_id $CLUSTER_ID

    # sleep 5

    CUDA_VISIBLE_DEVICES=$GPU python generate_timelapse_2.py \
        --dataset $DATASET \
        --cluster_id $CLUSTER_ID \
        --lmbd $LMBD

done




