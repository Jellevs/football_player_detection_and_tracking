Install: 
1. uv sync
2. uv pip install pytorch with cuda


Create folders inside jersey_number directory:
- parseq: clone parseq https://github.com/baudm/parseq
- pose: clone ViTPose https://github.com/ViTAE-Transformer/ViTPose (not needed anymore)
- reid: clone centroid reid https://github.com/mikwieczorek/centroids-reid (some import issue you have to fix)


put model weights inside weights folder


python sn-trackeval/scripts/run_mot_challenge.py --BENCHMARK SNMOT --SPLIT_TO_EVAL after_merge --GT_FOLDER C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\evaluation\gt --TRACKERS_FOLDER C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\evaluation\trackers --TRACKERS_TO_EVAL after_merge --SEQMAP_FILE C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\sequence\SNMOT-test.txt --METRICS HOTA CLEAR Identity --DO_PREPROC False --USE_PARALLEL False --TRACKER_SUB_FOLDER data --SKIP_SPLIT_FOL True


# Run after merge tracklets for evaluation
python sn-trackeval/scripts/run_mot_challenge.py `
--BENCHMARK SNMOT `
--SPLIT_TO_EVAL SNMOT-test `
--GT_FOLDER C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\evaluation\gt\SNMOT-test `
--TRACKERS_FOLDER C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\evaluation\trackers `
--TRACKERS_TO_EVAL after_merge `
--SEQMAP_FILE C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\sequence\SNMOT-test.txt `
--METRICS HOTA CLEAR Identity `
--DO_PREPROC False `
--USE_PARALLEL False `
--TRACKER_SUB_FOLDER data `
--SKIP_SPLIT_FOL True

# Run baseline tracklets for evaluation
python sn-trackeval/scripts/run_mot_challenge.py `
--BENCHMARK SNMOT `
--SPLIT_TO_EVAL SNMOT-test `
--GT_FOLDER C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\evaluation\gt\SNMOT-test `
--TRACKERS_FOLDER C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\evaluation\trackers `
--TRACKERS_TO_EVAL baseline `
--SEQMAP_FILE C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\sequence\SNMOT-test.txt `
--METRICS HOTA CLEAR Identity `
--DO_PREPROC False `
--USE_PARALLEL False `
--TRACKER_SUB_FOLDER data `
--SKIP_SPLIT_FOL True
