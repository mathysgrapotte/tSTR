process MAKE_PREDICTIONS {

    container 'alessiovignoli3/model-check:dataload_training'

    input:
    path(model_weights)
    path(model_architecture)
    path(fasta)

    output:
    path("predictions.tsv"), emit: predictions

    script:
    """
    launch_make_predictions.py -mw $model_weights -ma $model_architecture -f $fasta  -o predictions.tsv
    """
}