process MERGE_DATASET{

    container "alessiovignoli3/model-check:generate_fasta"

    input:
    path(prediction_file)
    path(metadata_file)
    path(gct_file)

    output:
    path("merged_dataset.tsv"), emit: merged_dataset

    script:
    """
    launch_merge_datasets.py -p ${prediction_file} -m ${metadata_file} -g ${gct_file} -o merged_dataset.tsv 
    """



}