# The project folder must contain a folder "images" with all the images.
# DATASET_PATH=data/south-building
# DATASET_PATH=data/rome_colosseum
DATASET_PATH=data/trevi_fountain_all
# DATASET_PATH=data/briksdalsbreen
# DATASET_PATH=data/statue_of_liberty

# colmap feature_extractor \
#     --database_path $DATASET_PATH/database.db \
#     --image_path $DATASET_PATH/images \
#     --SiftExtraction.use_gpu 0 
    

# sequential_matcher

colmap sequential_matcher \
    --database_path $DATASET_PATH/database.db \
    --SiftMatching.use_gpu 0
    
# colmap exhaustive_matcher \
#     --database_path $DATASET_PATH/database.db \
#     --SiftMatching.use_gpu 0

# mkdir $DATASET_PATH/sparse

# colmap mapper \
#     --database_path $DATASET_PATH/database.db \
#     --image_path $DATASET_PATH/images \
#     --output_path $DATASET_PATH/sparse

# colmap model_converter \
#     --input_path $DATASET_PATH/sparse/0 \
#     --output_path $DATASET_PATH/sparse/0 \
#     --output_type TXT







# $ mkdir $DATASET_PATH/dense

# $ colmap image_undistorter \
#     --image_path $DATASET_PATH/images \
#     --input_path $DATASET_PATH/sparse/0 \
#     --output_path $DATASET_PATH/dense \
#     --output_type COLMAP \
#     --max_image_size 2000

# $ colmap patch_match_stereo \
#     --workspace_path $DATASET_PATH/dense \
#     --workspace_format COLMAP \
#     --PatchMatchStereo.geom_consistency true

# $ colmap stereo_fusion \
#     --workspace_path $DATASET_PATH/dense \
#     --workspace_format COLMAP \
#     --input_type geometric \
#     --output_path $DATASET_PATH/dense/fused.ply

# $ colmap poisson_mesher \
#     --input_path $DATASET_PATH/dense/fused.ply \
#     --output_path $DATASET_PATH/dense/meshed-poisson.ply

# $ colmap delaunay_mesher \
#     --input_path $DATASET_PATH/dense \
#     --output_path $DATASET_PATH/dense/meshed-delaunay.ply