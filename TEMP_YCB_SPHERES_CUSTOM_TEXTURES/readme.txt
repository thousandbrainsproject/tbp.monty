Included are the necessary YCB config and model files for the reskinned baseball and softballs. The contents of the ycb_1.2-configs and ycb1.2-meshes can be copied and pasted directly into the respective folders under data/habitat/versioned_data/ycb_1.2

Included also are the resultant .csv from the evaluation runs of the custom dataset with and without LBP enabled. The relevant experiment-level configurations to obtain these
results are: 
'_lbp_eval_blender_textures_dist_agent' - eval of model trained with lbp features extracted and using lbp features for evidence
'_lbp_eval_blender_textures_dist_agent_lbp0weight' - eval of model trained with lbp features extracted but not using lbp features for evidence
'_lbp_eval_blender_textures_no_lbp' - eval of model trained without lbp features
'_lbp_train_blender_textures_dist_agent' - train model of custom object dataset with lbp feature extraction
'_lbp_train_blender_textures_no_lbp' - train model of custom object dataset without lbp feature extraction

Note, I misspoke for the LBP parameters used for this specific collection of results. The actual LBP setup used were:
LBP Algorithm = Local Ternary Pattern
Neighborhood = 8
Pixel radius = 1.0
Threshold = 5.0
Bitstring Encoding Scheme = uniform (not 'ror' as I had thought-- that was on an older version of this branch')