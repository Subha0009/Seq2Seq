Required: python3.8 or above, tensorflow 2.4.1, transformers.
 
1. Machine Translation:
	Data/ contains two notebooks to download, tokenize, and prepare the WMT 2014 En-De and En-Fr training data

	To train, use "python3 train_MT.py $arg1 $arg2 $arg3 $arg4 $arg5 $arg6 $arg7 $arg8 $arg9 $arg10 $arg11"
	where
	$arg1: model config {random/full}-{1/2}
	$arg2: batch size
	$arg3: maxlen
	$arg4: lr_max
	$arg5: warmup steps
	$arg6: TPU name
	$arg7: TPU zone
	$arg8: cloud project name
	$arg9: task name {en-de/en-fr}
	$arg10: checkpoint path (in cloud storage bucket)
	$arg11: directory of tfrecords files
	
	To test, use "python3 train_MT.py $arg1 $arg2 $arg3 $arg4"
	where 
	$arg1: model config {random/full}-{1/2}
	$arg2: task name {en-de/en-fr}
	$arg3: checkpoint path
	$arg4: checkpoint index

2. Text Classification:
	IMDB data for text classification is downloaded automatically. For AGnews, download .csv files from 
        "https://www.kaggle.com/amananandrai/ag-news-classification-dataset"
	Training and testing are done within the same script. To run, use
	"python3 text_classification.py $arg1 $arg2 $arg3 $arg4 $arg5 $arg6 $arg7 $arg8 $arg9"
	where
	$arg1: model config {random/full}-{1/2}
	$arg2: batch size
	$arg3: maxlen
	$arg4: lr_max
	$arg5: warmup steps
	$arg6: TPU name
	$arg7: TPU zone
	$arg8: cloud project name
	$arg9: task name {imdb/agnews}

3. Long sequence classification:
	Download IMDB movie review dataset from "https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
	Clone Github repo containing listops dataset: "https://github.com/nyu-mll/spinn.git"
	In Data/ run "python3 listops_tfrecord_writer.py directory_to_save_tfrecords"
        and "python3 charIMDB_tfrecord_writer.py directory_to_save_tfrecords"
	Training and testing done in same script. Run 
	"python3 long_sequence.py $arg1 $arg2 $arg3 $arg4 $arg5 $arg6 $arg7 $arg8 $arg9 $arg10"
	where
	$arg1: model config {random/full}-{1/2}
	$arg2: batch size
	$arg3: maxlen
	$arg4: lr_max
	$arg5: warmup steps
	$arg6: TPU name
	$arg7: TPU zone
	$arg8: cloud project name
	$arg9: task name {imdb/listops}
	$arg10: directory containing the tfrecords data
