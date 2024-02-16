include { MERGE_DATASET } from '../modules/merge_dataset.nf'

workflow MERGE_TO_MAKE_PLOT_DATA {
    take:
    prediction_file
    metadata_file   
    gct_file

    main:
    MERGE_DATASET(prediction_file, metadata_file, gct_file)
    plot_data = MERGE_DATASET.out.merged_dataset

    emit:
    plot_data



}