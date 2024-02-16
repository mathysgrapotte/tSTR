include {MAKE_PREDICTIONS} from '../modules/make_predictions.nf'

workflow PREDICT {

    take:
    model_weights
    model_architecture
    fasta

    main:
    MAKE_PREDICTIONS(model_weights, model_architecture, fasta)
    predictions = MAKE_PREDICTIONS.out.predictions

    emit:
    predictions
}